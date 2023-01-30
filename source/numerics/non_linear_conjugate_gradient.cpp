/********************************************************************************
  Copyright (C) 2019 - 2022 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lifex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with lifex.  If not, see <http://www.gnu.org/licenses/>.
********************************************************************************/

/**
 * @file
 *
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/numerics/non_linear_conjugate_gradient.hpp"
#include "source/numerics/numbers.hpp"

#include <boost/io/ios_state.hpp>

#include <algorithm>

namespace lifex::utils
{
  template <class VectorType>
  void
  NonLinearConjugateGradient<VectorType>::declare_parameters(
    ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    {
      params.declare_entry_selection(
        "Descent update method",
        "PR",
        "FR|PR|HS",
        "Method for the selection of the descent direction. Fletcher-Reeves, "
        "Polak-Ribiere or Hestenes-Stiefel.");

      params.declare_entry("Log frequency",
                           "1",
                           Patterns::Integer(1),
                           "Only print convergence info every n iterations.");

      params.enter_subsection("Stopping criterion");
      {
        params.declare_entry("Maximum number of iterations",
                             "1000",
                             Patterns::Integer(0));
        params.declare_entry("Loss reduction tolerance",
                             "1e-6",
                             Patterns::Double(0));
        params.declare_entry("Relative increment tolerance",
                             "0",
                             Patterns::Double(0));
      }
      params.leave_subsection();

      params.enter_subsection("Line search");
      {
        params.declare_entry("Initial step length",
                             "1.0",
                             Patterns::Double(0),
                             "Initial step length for the first iteration of "
                             "the method. Subsequent iterations choose an "
                             "initial step length based on previous ones.");

        params.declare_entry("Step length reduction factor",
                             "0.5",
                             Patterns::Double(0, 1),
                             "If a step results in increased loss function, it "
                             "is reduced by this factor.");

        params.declare_entry(
          "Loss improvement factor",
          "1e-4",
          Patterns::Double(0, 1),
          "Minimum improvement of the loss function between iterations.");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  template <class VectorType>
  void
  NonLinearConjugateGradient<VectorType>::parse_parameters(ParamHandler &params)
  {
    params.parse();

    params.enter_subsection_path(prm_subsection_path);
    {
      prm_descent_update_method = params.get("Descent update method");
      prm_log_frequency         = params.get_integer("Log frequency");

      params.enter_subsection("Stopping criterion");
      {
        prm_n_max_iterations =
          params.get_integer("Maximum number of iterations");
        prm_tolerance_loss_reduction =
          params.get_double("Loss reduction tolerance");
        prm_tolerance_increment =
          params.get_double("Relative increment tolerance");
      }
      params.leave_subsection();

      params.enter_subsection("Line search");
      {
        prm_linesearch_initial_step_length =
          params.get_double("Initial step length");
        prm_linesearch_step_reduction =
          params.get_double("Step length reduction factor");
        prm_linesearch_loss_improvement =
          params.get_double("Loss improvement factor");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  template <class VectorType>
  void
  NonLinearConjugateGradient<VectorType>::initialize(VectorType *sol_owned_,
                                                     VectorType *sol_)
  {
    Assert(sol_ != nullptr, ExcNotInitialized());
    // Ghost elements exist only if run in parallel.
    Assert(sol_->has_ghost_elements() || (mpi_size == 1),
           ExcParallelNonGhosted());

    Assert(sol_owned_ != nullptr, ExcNotInitialized());
    Assert(!sol_owned_->has_ghost_elements(), ExcGhostsPresent());

    sol_owned   = sol_owned_;
    sol         = sol_;
    initialized = true;

    loss_gradient.reinit(*sol_owned);
    loss_gradient_old.reinit(*sol_owned);
    descent.reinit(*sol_owned);
    tmp_owned.reinit(*sol_owned);
    tmp.reinit(*sol);
  }

  template <class VectorType>
  void
  NonLinearConjugateGradient<VectorType>::solve(
    const std::function<double(const VectorType &)> &loss_fun,
    const std::function<void(const VectorType &, VectorType &)>
      &loss_gradient_fun)
  {
    Assert(initialized, ExcNotInitialized());

    TimerOutput::Scope timer_section(timer_output, prm_subsection_path);

    const boost::io::ios_all_saver iostream_backup(pcout.get_stream());

    double alpha = prm_linesearch_initial_step_length;
    double beta  = 0.0;

    double loss_0 = 0.0;

    double loss     = loss_fun(*sol);
    double loss_old = loss;
    double incr     = 0.0;

    bool converged_loss = false;
    bool converged_incr = false;

    n_iter = 0;

    while (n_iter < prm_n_max_iterations && !converged_loss && !converged_incr)
      {
        // Compute the loss gradient at current position.
        loss_gradient_old = loss_gradient;
        loss_gradient_fun(*sol, loss_gradient);
        loss_gradient *= -1.0;

        // Safeguard against exactly null gradient.
        if (loss_gradient.l2_norm() <= 0)
          {
            pcout << "\tNLCG iteration " << std::setw(4) << (n_iter + 1) << "/"
                  << prm_n_max_iterations << std::endl;
            pcout << "\t\t[NLCG] step length:           " << std::scientific
                  << std::setprecision(5) << 0.0 << " *" << std::endl;
            break;
          }

        // Select the descent direction.
        if (n_iter == 0)
          {
            descent = loss_gradient;
          }
        else
          {
            alpha *= (loss_gradient_old * descent);

            if (prm_descent_update_method == "PR" ||
                prm_descent_update_method == "HS")
              {
                tmp_owned = loss_gradient;
                tmp_owned -= loss_gradient_old;
              }

            if (prm_descent_update_method == "FR")
              beta = loss_gradient.norm_sqr() / loss_gradient_old.norm_sqr();
            else if (prm_descent_update_method == "PR")
              beta = std::max((loss_gradient * tmp_owned) /
                                loss_gradient_old.norm_sqr(),
                              0.0);
            else // if (prm_descent_update_method == "HS")
              beta = -loss_gradient.norm_sqr() / (descent * tmp_owned);

            descent.sadd(beta, loss_gradient);
            alpha /= loss_gradient * descent;
          }

        // Perform line search to determine step length (and contextually
        // compute the new value of the loss function).
        {
          loss_old = loss;

          const double t =
            -prm_linesearch_loss_improvement * (loss_gradient * descent);

          alpha /= prm_linesearch_step_reduction;

          do
            {
              alpha *= prm_linesearch_step_reduction;

              tmp_owned = *sol_owned;
              tmp_owned.add(alpha, descent);
              tmp  = tmp_owned;
              loss = loss_fun(tmp);
            }
          while (std::isnan(loss) ||
                 loss > loss_old - alpha * prm_linesearch_loss_improvement * t);
        }

        // Update the solution.
        sol_owned->add(alpha, descent);
        *sol = *sol_owned;

        incr = std::abs(loss - loss_old);

        if (n_iter == 0)
          loss_0 = loss_old;

        ++n_iter;

        converged_loss = loss < prm_tolerance_loss_reduction * loss_0;
        converged_incr = (0 <= incr && incr < prm_tolerance_increment * loss_0);

        if (n_iter == 1 || n_iter % prm_log_frequency == 0 || converged_loss ||
            converged_incr)
          {
            pcout << "\tNLCG iteration " << std::setw(4) << n_iter << "/"
                  << prm_n_max_iterations << std::endl;
            pcout << "\t\t[NLCG] alpha:                 " << std::scientific
                  << std::setprecision(5) << alpha << std::endl;
            pcout << "\t\t[NLCG] step length:           " << std::scientific
                  << std::setprecision(5) << std::abs(alpha) * descent.l2_norm()
                  << std::endl;
            pcout << "\t\t[NLCG] loss:                  " << std::scientific
                  << std::setprecision(5) << loss << std::endl;
            pcout << "\t\t[NLCG] loss reduction:        " << std::scientific
                  << std::setprecision(5) << loss / loss_0
                  << (converged_loss ? " *" : "") << std::endl;
            pcout << "\t\t[NLCG] relative increment:    " << std::scientific
                  << std::setprecision(5) << incr / loss_0
                  << (converged_incr ? " *" : "") << std::endl;
          }
      }

    pcout.set_condition(mpi_rank == 0);
  }

  /// Explicit instantiation.
  template class NonLinearConjugateGradient<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class NonLinearConjugateGradient<LinAlg::MPI::BlockVector>;
} // namespace lifex::utils
