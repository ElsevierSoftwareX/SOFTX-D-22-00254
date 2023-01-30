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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#ifndef LIFEX_UTILS_LINEAR_SOLVER_HANDLER_HPP_
#define LIFEX_UTILS_LINEAR_SOLVER_HANDLER_HPP_

#include "source/core_model.hpp"

#include "source/numerics/preconditioner_handler.hpp"

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>

#include <boost/io/ios_state.hpp>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /// @brief Helper class to declare a linear solver
  ///
  /// The linear solver type and options can be selected at runtime by setting
  /// the proper parameters in the specified parameter file subsection.
  ///
  /// Set of allowed solvers can be specified during construction choosing among
  /// these string options:
  /// - CG;
  /// - GMRES;
  /// - BiCGStab.
  template <class VectorType>
  class LinearSolverHandler : public CoreModel
  {
  public:
    /// Constructor.
    ///
    /// @param [in] subsection      parameter subsection path;
    /// @param [in] solver_set_     set of allowed solvers;
    /// @param [in] default_solver_ default solver.
    LinearSolverHandler(const std::string &          subsection,
                        const std::set<std::string> &solver_set_,
                        const std::string &          default_solver_);

    /// Override of @ref lifex::CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref lifex::CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Method to solve a linear system with the type selected
    /// in the parameter file. The flag store_iterations is related to the @ref get_n_iterations method, where the iterations are stored only when this flag is set to true (default). If no iterations have been stored and @ref get_n_iterations is called, an error is raised.
    template <class MatrixType, class PreconditionerType>
    void
    solve(const MatrixType &        A,
          VectorType &              x,
          const VectorType &        b,
          const PreconditionerType &preconditioner,
          const bool &              store_iterations = true);

    /// Get linear solver type.
    std::string
    get_type() const
    {
      return prm_linear_solver_type;
    }

    /// Get max iterations.
    unsigned int
    get_max_steps() const
    {
      return prm_reduction_control_init.max_steps();
    }

    /// Get tolerance.
    double
    get_tolerance() const
    {
      return prm_reduction_control_init.tolerance();
    }

    /// Get reduction.
    double
    get_reduction() const
    {
      return prm_reduction_control_init.reduction();
    }

    /// Getter method for @ref n_iterations.
    const unsigned int &
    get_n_iterations() const
    {
      AssertThrow(
        iterations_stored,
        ExcMessage(
          "get_n_iterations was called but there are no iterations stored. "
          "LinearSolverHandler::solve() must be called with \"store_iterations"
          " = true\" at least once for iterations to be stored."));

      return n_iterations;
    }

    /// Set reduction control parameters.
    void
    set_reduction_control(const unsigned int &max_steps,
                          const double &      tolerance,
                          const double &      reduction)
    {
      prm_reduction_control_init.set_max_steps(max_steps);
      prm_reduction_control_init.set_tolerance(tolerance);
      prm_reduction_control_init.set_reduction(reduction);
    }

    /// Print a string with number of iterations performed and solver type.
    void
    print_iteration_log(const unsigned int &n_iterations) const
    {
      pcout << "\t" << std::setw(5) << n_iterations << " "
            << prm_linear_solver_type << " iterations" << std::endl;
    }

    /// Print the convergence history.
    void
    print_iteration_history(const SolverControl &reduction_control) const
    {
      const std::vector<double> residuals =
        reduction_control.get_history_data();

      // Backup stream flags and manipulators.
      const boost::io::ios_all_saver iostream_backup(pcout.get_stream());

      pcout << "\t\t[Linear solver]" << std::endl;
      for (unsigned int k = 0; k < residuals.size(); ++k)
        {
          if ((k % prm_log_frequency) == 0 || k == residuals.size() - 1)
            {
              pcout << "\t\t\titeration:\t" << k << ", residual:\t"
                    << std::scientific << residuals[k] << std::endl;
            }
        }
    }

  private:
    /// Set of allowed solvers.
    std::set<std::string> solver_set;

    /// Default solver.
    std::string default_solver;

    /// Current number of linear solver iterations.
    unsigned int n_iterations;

    /// Flag that keeps track if there are iterations stored.
    bool iterations_stored;

    /// @name Parameters read from file.
    /// @{

    /// Initial reduction control.
    ReductionControl prm_reduction_control_init;

    /// Log frequency.
    unsigned int prm_log_frequency;

    /// Type of solver selected from parameter file.
    std::string prm_linear_solver_type;

    /// Additional GMRES data.
    typename SolverGMRES<VectorType>::AdditionalData prm_data_gmres;
    /// Additional BiCGStab data.
    typename SolverBicgstab<VectorType>::AdditionalData prm_data_bicgstab;
    /// Additional FMGRES data.
    typename SolverFGMRES<VectorType>::AdditionalData prm_data_fgmres;

    /// @}
  };

  template <class VectorType>
  template <class MatrixType, class PreconditionerType>
  void
  LinearSolverHandler<VectorType>::solve(
    const MatrixType &        A,
    VectorType &              x,
    const VectorType &        b,
    const PreconditionerType &preconditioner,
    const bool &              store_iterations)
  {
    unsigned int       n_iterations_temp = 0;
    TimerOutput::Scope timer_section(timer_output, prm_subsection_path);

    // Reset the reduction control at each solve call to avoid accumulating
    // history data.
    ReductionControl reduction_control(prm_reduction_control_init);

    // Create and use solver.
    try
      {
        if (prm_linear_solver_type == "CG")
          {
            SolverCG<VectorType> solver(reduction_control);
            solver.solve(A, x, b, preconditioner);
          }
        else if (prm_linear_solver_type == "GMRES")
          {
            SolverGMRES<VectorType> solver(reduction_control, prm_data_gmres);
            solver.solve(A, x, b, preconditioner);
          }
#if defined(LIN_ALG_PETSC)
        else if (prm_linear_solver_type == "PETSc::GMRES")
          {
            if constexpr (std::is_same_v<PreconditionerType,
                                         PETScWrappers::PreconditionerBase>)
              {
                PETScWrappers::SolverGMRES solver(reduction_control, mpi_comm);
                solver.solve(A, x, b, preconditioner);
              }
            else if constexpr (std::is_same_v<PreconditionerType,
                                              PreconditionerHandler>)
              {
                PETScWrappers::SolverGMRES solver(reduction_control, mpi_comm);
                const std::shared_ptr<PETScWrappers::PreconditionerBase>
                  prec_ptr = preconditioner.get_preconditioner_base();
                solver.solve(A, x, b, *prec_ptr);
              }
            else
              {
                AssertThrow(false,
                            ExcMessage("PETSc::GMRES linear solver called with "
                                       "an unsupported preconditioner type."));
              }
          }
#endif
        else if (prm_linear_solver_type == "BiCGStab")
          {
            SolverBicgstab<VectorType> solver(reduction_control,
                                              prm_data_bicgstab);
            solver.solve(A, x, b, preconditioner);
          }
        else if (prm_linear_solver_type == "FGMRES")
          {
            SolverFGMRES<VectorType> solver(reduction_control, prm_data_fgmres);
            solver.solve(A, x, b, preconditioner);
          }
        else if (prm_linear_solver_type == "MinRes")
          {
            SolverMinRes<VectorType> solver(reduction_control);
            solver.solve(A, x, b, preconditioner);
          }
      }
    catch (const SolverControl::NoConvergence &exc)
      {
        n_iterations_temp = reduction_control.last_step();
        print_iteration_log(n_iterations_temp);

        if (reduction_control.log_history())
          print_iteration_history(reduction_control);

        throw;
      }

    n_iterations_temp = reduction_control.last_step();
    print_iteration_log(n_iterations_temp);

    if (reduction_control.log_history())
      print_iteration_history(reduction_control);

    if (store_iterations)
      {
        n_iterations      = n_iterations_temp;
        iterations_stored = true;
      }
  }

} // namespace lifex::utils


#endif /* LIFEX_UTILS_LINEAR_SOLVER_HANDLER_HPP_ */
