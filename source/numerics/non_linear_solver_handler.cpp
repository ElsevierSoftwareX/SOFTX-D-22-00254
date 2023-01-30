/********************************************************************************
  Copyright (C) 2020 - 2022 by the lifex authors.

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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/numerics/non_linear_solver_handler.hpp"
#include "source/numerics/numbers.hpp"

#include <boost/io/ios_state.hpp>

#include <algorithm>
#include <limits>

namespace lifex::utils
{
  template <class VectorType>
  NonLinearSolverHandler<VectorType>::NonLinearSolverHandler(
    const std::string &subsection)
    : CoreModel(subsection)
    , initialized(false)
    , n_iterations(0)
    , n_iterations_linear(0)
    , n_iterations_linear_min(0)
    , n_iterations_linear_max(0)
    , n_iterations_linear_tot(0)
    , n_iterations_linear_avg(0)
    , time_solve(0.0)
  {
    static_assert(
      is_any_v<VectorType, LinAlg::MPI::Vector, LinAlg::MPI::BlockVector>,
      "NonLinearSolverHandler: template parameter not allowed.");
  }

  template <class VectorType>
  void
  NonLinearSolverHandler<VectorType>::declare_parameters(
    ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    {
      params.set_verbosity(VerbosityParam::Minimal);
      {
        params.declare_entry(
          "Linearized",
          "false",
          Patterns::Bool(),
          "If true, the following parameters will be ignored: the max. number "
          "of "
          "iterations will implicitly be set to 1 and tolerances to 0.");

        params.declare_entry("Maximum number of iterations",
                             "50",
                             Patterns::Integer(0));

        params.declare_entry("Absolute residual tolerance",
                             "1e-8",
                             Patterns::Double(0));

        params.declare_entry("Relative residual tolerance",
                             "1e-8",
                             Patterns::Double(0));

        params.declare_entry("Relative increment tolerance",
                             "1e-10",
                             Patterns::Double(0));
      }
      params.reset_verbosity();
      params.set_verbosity(VerbosityParam::Full);
      {
        params.declare_entry("Jacobian assembly lag",
                             "1",
                             Patterns::Integer(1),
                             "Assemble Jacobian every n-th iteration.");

        params.declare_entry_selection(
          "Acceleration scheme",
          "None",
          std::string("None | ") +
            utils::FixedPointAcceleration<
              LinAlg::MPI::Vector>::Factory::get_registered_keys_prm(),
          "Acceleration scheme to be used for relaxation.");
      }
      params.reset_verbosity();
    }
    params.leave_subsection_path();

    utils::FixedPointAcceleration<VectorType>::Factory::
      declare_children_parameters(params,
                                  prm_subsection_path +
                                    " / Fixed point / Relaxation");
  }

  template <class VectorType>
  void
  NonLinearSolverHandler<VectorType>::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      prm_linearized = params.get_bool("Linearized");

      prm_max_iterations =
        prm_linearized ? 1 : params.get_integer("Maximum number of iterations");

      prm_absolute_res_tol =
        prm_linearized ? 0 : params.get_double("Absolute residual tolerance");

      prm_relative_res_tol =
        prm_linearized ? 0 : params.get_double("Relative residual tolerance");

      prm_relative_incr_tol =
        prm_linearized ? 0 : params.get_double("Relative increment tolerance");

      prm_jacobian_lag = params.get_integer("Jacobian assembly lag");

      prm_acceleration_type = params.get("Acceleration scheme");
    }
    params.leave_subsection_path();

    if (prm_acceleration_type == "None")
      {
        fixed_point_acceleration = nullptr;
      }
    else
      {
        fixed_point_acceleration = utils::FixedPointAcceleration<VectorType>::
          Factory::parse_child_parameters(params,
                                          prm_acceleration_type,
                                          prm_subsection_path +
                                            " / Fixed point / Relaxation");
      }
  }

  template <class VectorType>
  void
  NonLinearSolverHandler<VectorType>::initialize(VectorType *sol_owned_,
                                                 VectorType *sol_)
  {
    Assert(sol_ != nullptr, ExcNotInitialized());
    // Ghost elements exist only if run in parallel.
    Assert(sol_->has_ghost_elements() || (mpi_size == 1),
           ExcParallelNonGhosted());

    Assert(sol_owned_ != nullptr, ExcNotInitialized());
    Assert(!sol_owned_->has_ghost_elements(), ExcGhostsPresent());

    initialized = true;

    sol_owned = sol_owned_;
    sol       = sol_;

    incr.reinit(*sol_owned);
  }

  template <class VectorType>
  bool
  NonLinearSolverHandler<VectorType>::solve(
    const std::function<double(const bool &assemble_jac)> &assemble_fun,
    const std::function<
      std::tuple<double, double, unsigned int>(const bool &assemble_prec,
                                               VectorType &incr)> &solve_fun)
  {
    AssertThrow(initialized, ExcNotInitialized());

    TimerOutput::Scope timer_section(timer_output, prm_subsection_path);
    timer_solve.restart();

    // Reset counters.
    n_iterations_linear_min = std::numeric_limits<unsigned int>::max();
    n_iterations_linear_max = 0;
    n_iterations_linear_tot = 0;
    time_solve              = 0.0;

    double initial_res = 0;

    // Pre-allocate variables.
    double norm_incr;
    double norm_res;
    double norm_sol;

    double norm_incr_old     = 0;
    double norm_incr_old_old = 0;

    bool res_converged     = false;
    bool res_rel_converged = false;
    bool incr_converged    = false;

    bool reassemble;

    // Backup stream flags and manipulators.
    const boost::io::ios_all_saver iostream_backup(pcout.get_stream());

    if (fixed_point_acceleration != nullptr)
      fixed_point_acceleration->reset(*sol_owned);

    for (n_iterations = 0; n_iterations < prm_max_iterations; ++n_iterations)
      {
        reassemble = ((n_iterations % prm_jacobian_lag) == 0);

        norm_res = assemble_fun(reassemble);
        if (n_iterations == 0)
          initial_res = norm_res;

        if (norm_res < prm_absolute_res_tol)
          {
            res_converged = true;
          }

        if (norm_res / initial_res < prm_relative_res_tol)
          {
            res_rel_converged = true;
          }

        if (!prm_linearized)
          {
            if (n_iterations == 0)
              {
                pcout << std::endl;
              }
            pcout << "\tNewton iteration " << std::setw(2) << n_iterations + 1
                  << "/" << prm_max_iterations << std::flush;
          }

        if (!res_converged && !res_rel_converged)
          {
            std::tie(norm_sol, norm_incr, n_iterations_linear) =
              solve_fun(reassemble, incr);

            n_iterations_linear_tot += n_iterations_linear;
            n_iterations_linear_min =
              std::min(n_iterations_linear_min, n_iterations_linear);
            n_iterations_linear_max =
              std::max(n_iterations_linear_max, n_iterations_linear);

            // sol = sol - incr.
            sol_owned->add(-1, incr);
            if (fixed_point_acceleration != nullptr)
              {
                *sol_owned =
                  fixed_point_acceleration->get_next_element(*sol_owned);
              }
            *sol = *sol_owned;

            if (norm_incr <= prm_relative_incr_tol * norm_sol)
              {
                incr_converged = true;
              }
          }
        else
          {
            norm_sol = norm_incr = 0;

            pcout << std::endl;
          }

        // Print current iteration.
        if (!prm_linearized)
          {
            pcout << "\t\t[Newton] absolute residual:     " << std::scientific
                  << std::setprecision(5) << norm_res
                  << (res_converged ? " *" : "") << std::endl;

            if (!utils::is_zero(initial_res))
              {
                pcout << "\t\t[Newton] relative residual:     "
                      << std::scientific << std::setprecision(5)
                      << norm_res / initial_res
                      << (res_rel_converged ? " *" : "") << std::endl;
              }

            if (!res_converged && !res_rel_converged)
              {
                pcout << "\t\t[Newton] solution norm:         "
                      << std::scientific << std::setprecision(5) << norm_sol
                      << std::endl;

                pcout << "\t\t[Newton] absolute increment:    "
                      << std::scientific << std::setprecision(5) << norm_incr
                      << std::endl;

                if (!utils::is_zero(norm_sol))
                  {
                    pcout << "\t\t[Newton] relative increment:    "
                          << std::scientific << std::setprecision(5)
                          << norm_incr / norm_sol
                          << (incr_converged ? " *" : "") << std::endl;
                  }

                if (n_iterations >= 2 && !utils::is_zero(norm_incr) &&
                    !utils::is_zero(norm_incr_old) &&
                    !utils::is_zero(norm_incr_old_old))
                  {
                    const double p_estimate_num =
                      std::log(norm_incr / norm_incr_old);

                    const double p_estimate_den =
                      std::log(norm_incr_old / norm_incr_old_old);

                    if (!utils::is_zero(p_estimate_den))
                      {
                        pcout << "\t\t[Newton] estimated conv. order: "
                              << std::fixed << std::setprecision(5)
                              << p_estimate_num / p_estimate_den << std::endl;
                      }
                  }
              }
          }

        if (res_converged || res_rel_converged || incr_converged ||
            prm_linearized || n_iterations == prm_max_iterations - 1)
          {
            break;
          }

        norm_incr_old_old = norm_incr_old;
        norm_incr_old     = norm_incr;
      }

    n_iterations += !res_converged;

    if (n_iterations != 0)
      n_iterations_linear_avg = n_iterations_linear_tot / n_iterations;

    timer_solve.stop();
    time_solve = timer_solve.wall_time();

    return (res_converged || res_rel_converged || incr_converged ||
            prm_linearized);
  }

  template <class VectorType>
  void
  NonLinearSolverHandler<VectorType>::declare_entries_csv(
    CSVWriter &        csv_writer,
    const std::string &prefix) const
  {
    csv_prefix = prefix + "_";

    csv_writer.declare_entries({
      csv_prefix + "iterations_nonlinear",
      csv_prefix + "iterations_linear_min",
      csv_prefix + "iterations_linear_max",
      csv_prefix + "iterations_linear_tot",
      csv_prefix + "iterations_linear_avg",
      csv_prefix + "solving_time",
    });
  }

  template <class VectorType>
  void
  NonLinearSolverHandler<VectorType>::set_entries_csv(
    CSVWriter &csv_writer) const
  {
    csv_writer.set_entries(
      {{csv_prefix + "iterations_nonlinear", n_iterations},
       {csv_prefix + "iterations_linear_min", n_iterations_linear_min},
       {csv_prefix + "iterations_linear_max", n_iterations_linear_max},
       {csv_prefix + "iterations_linear_tot", n_iterations_linear_tot},
       {csv_prefix + "iterations_linear_avg", n_iterations_linear_avg},
       {csv_prefix + "solving_time", time_solve}});
  }

  std::function<
    Vector<double>(const double, const double, const Vector<double> &)>
  id_minus_tau_J_inverse_quasinewton(
    const std::function<Vector<double>(const double, const Vector<double> &)>
      &                   f,
    const Vector<double> &y,
    const double &        step_rel,
    const double &        step_min)
  {
    auto id_minus_tau_J_inverse =
      [&](const double t, const double tau, const Vector<double> &res) {
        Vector<double> rhs = f(t, y);
        size_t         n   = rhs.size();

        FullMatrix<double> J(n, n);

        Vector<double> y_new;
        Vector<double> rhs_new;

        double h_i;

        for (size_t i = 0; i < n; ++i)
          {
            // y_new = [y(0), y(1), ..., y(i) + h_i, ..., y(n - 1)]^T.
            y_new = y;
            h_i   = std::max(step_min, std::abs(y(i)) * step_rel);
            y_new(i) += h_i;

            rhs_new = f(t, y_new);

            for (size_t j = 0; j < n; ++j)
              {
                J(j, i) = (rhs_new(j) - rhs(j)) / h_i;
              }
          }

        // Compute I - tau * J.
        FullMatrix<double> I_m_tau_J = IdentityMatrix(n);
        I_m_tau_J.add(-tau, J);

        // Compute inverse (I - tau * J)^(-1).
        FullMatrix<double> I_m_tau_J_inv(n, n);
        I_m_tau_J_inv.invert(I_m_tau_J);

        // Compute sol = (I - tau * J)^(-1) * res.
        Vector<double> sol(n);
        I_m_tau_J_inv.vmult(sol, res);

        return sol;
      };

    return id_minus_tau_J_inverse;
  }

  template <class VectorType>
  void
  InexactNonLinearSolverHandler<VectorType>::declare_parameters(
    ParamHandler &params) const
  {
    NonLinearSolverHandler<VectorType>::declare_parameters(params);

    params.set_verbosity(VerbosityParam::Full);
    params.enter_subsection_path(this->prm_subsection_path);
    {
      params.declare_entry_selection("Newton-Krylov method",
                                     "Newton",
                                     "Newton|PowerLaw|EisenstatWalker",
                                     "The Newton-Krylov method to be used.");

      params.enter_subsection("Power law reduction");
      {
        params.declare_entry("Alpha", "0.5", Patterns::Double(0, 1));
        params.declare_entry("Beta", "1.0", Patterns::Double(0));
        params.declare_entry("Gamma", "1.0", Patterns::Double(0));
      }
      params.leave_subsection();

      params.enter_subsection("Eisenstat-Walker method");
      {
        params.declare_entry("Max relative tolerance",
                             "1e-1",
                             Patterns::Double(0));
        params.declare_entry("Gamma", "0.5", Patterns::Double(0, 1));
        params.declare_entry("Alpha", "2.0", Patterns::Double(0));
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
    params.reset_verbosity();
  }

  template <class VectorType>
  void
  InexactNonLinearSolverHandler<VectorType>::parse_parameters(
    ParamHandler &params)
  {
    NonLinearSolverHandler<VectorType>::parse_parameters(params);

    params.enter_subsection_path(this->prm_subsection_path);

    const std::string tmp_string = params.get("Newton-Krylov method");

    if (tmp_string == "Newton")
      {
        prm_method = Method::Newton;
      }
    else if (tmp_string == "PowerLaw")
      {
        prm_method = Method::PowerLawReduction;
      }
    else // if (tmp_string == "EisenstatWalker")
      {
        prm_method = Method::EisenstatWalker;
      }

    if (prm_method == Method::PowerLawReduction)
      {
        params.enter_subsection("Power law reduction");
        {
          prm_power_law_alpha = params.get_double("Alpha");
          prm_power_law_beta  = params.get_double("Beta");
          prm_power_law_gamma = params.get_double("Gamma");
        }
        params.leave_subsection();
      }

    else if (prm_method == Method::EisenstatWalker)
      {
        params.enter_subsection("Eisenstat-Walker method");
        {
          prm_ew_tolerance_max = params.get_double("Max relative tolerance");
          prm_ew_gamma         = params.get_double("Gamma");
          prm_ew_alpha         = params.get_double("Alpha");
        }
        params.leave_subsection();
      }

    params.leave_subsection_path();
  }

  template <class VectorType>
  bool
  InexactNonLinearSolverHandler<VectorType>::solve(
    const std::function<double(const bool &assemble_jac)> &assemble_fun,
    const std::function<
      std::tuple<double, double, unsigned int>(const bool &assemble_prec,
                                               VectorType &incr)> &solve_fun)
  {
    // We define new assemble_fun and solve_fun that wrap the ones provided as
    // argument, adding some solver-related functionality.

    auto assemble_fun_wrapper =
      [&assemble_fun, &solve_fun, this](const bool &assemble_jac) {
        residual_norm_old = residual_norm;
        residual_norm     = assemble_fun(assemble_jac);
        return residual_norm;
      };

    auto solve_fun_wrapper =
      [&assemble_fun, &solve_fun, this](const bool &assemble_prec,
                                        VectorType &incr) {
        // Compute the new relative tolerance based on the chosen method.
        if (prm_method == Method::PowerLawReduction)
          {
            rel_tolerance = std::pow(prm_power_law_alpha,
                                     prm_power_law_beta * this->n_iterations +
                                       prm_power_law_gamma);
          }

        else if (prm_method == Method::EisenstatWalker)
          {
            if (this->n_iterations == 0)
              rel_tolerance = prm_ew_tolerance_max;
            else
              {
                const double rel_tolerance_min =
                  prm_ew_gamma * std::pow(rel_tolerance, prm_ew_alpha);

                rel_tolerance =
                  prm_ew_gamma *
                  std::pow(residual_norm / residual_norm_old, prm_ew_alpha);

                if (rel_tolerance_min > 0.1)
                  rel_tolerance = std::max(rel_tolerance, rel_tolerance_min);

                if (rel_tolerance > prm_ew_tolerance_max)
                  rel_tolerance = prm_ew_tolerance_max;
              }
          }

        // We actually change the tolerances only if the method is not vanilla
        // Newton.
        if (prm_method != Method::Newton)
          {
            pcout << "\n\t  Newton-Krylov: using relative tolerance "
                  << std::scientific << rel_tolerance << std::endl;

            linear_solver_handler.set_reduction_control(
              linear_solver_handler.get_max_steps(),
              linear_solver_handler.get_tolerance(),
              rel_tolerance);
          }

        return solve_fun(assemble_prec, incr);
      };

    return NonLinearSolverHandler<VectorType>::solve(assemble_fun_wrapper,
                                                     solve_fun_wrapper);
  }

  /// Explicit instantiation.
  template class NonLinearSolverHandler<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class NonLinearSolverHandler<LinAlg::MPI::BlockVector>;

  /// Explicit instantiation.
  template class InexactNonLinearSolverHandler<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class InexactNonLinearSolverHandler<LinAlg::MPI::BlockVector>;

} // namespace lifex::utils
