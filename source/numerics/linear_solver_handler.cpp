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

#include "source/numerics/linear_solver_handler.hpp"

namespace lifex::utils
{
  template <class VectorType>
  LinearSolverHandler<VectorType>::LinearSolverHandler(
    const std::string &          subsection,
    const std::set<std::string> &solver_set_,
    const std::string &          default_solver_)
    : CoreModel(subsection)
    , solver_set(solver_set_)
    , default_solver(default_solver_)
    , iterations_stored(false)
  {
    static_assert(is_any_v<VectorType,
                           LinAlg::MPI::Vector,
                           LinAlg::MPI::BlockVector,
                           Vector<double>,
                           LinearAlgebra::distributed::Vector<double>>,
                  "LinearSolverHandler: template parameter not allowed.");

    AssertThrow(!solver_set.empty(),
                ExcMessage("Set of allowed solvers cannot be empty."));

    const std::set<std::string> solvers_allowed = {"CG",
                                                   "GMRES",
#ifdef LIN_ALG_PETSC
                                                   "PETSc::GMRES",
#endif
                                                   "BiCGStab",
                                                   "MinRes",
                                                   "FGMRES"};

    for (const auto &solver : solver_set)
      {
        AssertThrow(solvers_allowed.find(solver) != solvers_allowed.end(),
                    ExcLifexNotImplemented());
      }

    AssertThrow(solver_set.find(default_solver) != solver_set.end(),
                ExcMessage(
                  "Default solver must be in the set of allowed solvers."));
  }


  template <class VectorType>
  void
  LinearSolverHandler<VectorType>::declare_parameters(
    ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    {
      params.set_verbosity(VerbosityParam::Minimal);
      {
        // Solver control.
        params.declare_entry("Maximum number of iterations",
                             "1000",
                             Patterns::Integer(0));

        params.declare_entry("Tolerance",
                             "1e-10",
                             Patterns::Double(0),
                             "Tolerance for the absolute l2 residual norm.");

        params.declare_entry(
          "Reduction",
          "0",
          Patterns::Double(0),
          "Reduction factor w.r.t. initial linear solver residual. If 0, then "
          "only the absolute residual norm is checked as a stopping "
          "criterion.");

        // Solver type.
        std::string solver_selection;

        for (const auto &solver : solver_set)
          {
            solver_selection += solver + " | ";
          }
        solver_selection =
          solver_selection.substr(0, solver_selection.size() - 3);

        params.declare_entry_selection("Solver type",
                                       default_solver,
                                       solver_selection);
      }
      params.reset_verbosity();
      params.set_verbosity(VerbosityParam::Full);
      {
        params.declare_entry(
          "Log history",
          "false",
          Patterns::Bool(),
          "Log each iteration step, use 'Log frequency' for skipping steps.");

        params.declare_entry(
          "Log frequency",
          "1",
          Patterns::Integer(1),
          "Set logging frequency when 'Log history' is enabled.");

        // GMRES additional data.
        if (solver_set.find("GMRES") != solver_set.end())
          {
            params.enter_subsection("GMRES");

            params.declare_entry("Right preconditioning",
                                 "true",
                                 Patterns::Bool(),
                                 "Toggle using right or left preconditioning.");

            // default value = #iter prevents restart
            params.declare_entry("Maximum number of temporary vectors",
                                 "1000",
                                 Patterns::Integer(0),
                                 "The number of iterations before restart.");

            params.declare_entry("Use default residual",
                                 "true",
                                 Patterns::Bool());

            params.declare_entry("Force re-orthogonalization",
                                 "false",
                                 Patterns::Bool(),
                                 "Force re-orthogonalization of orthonormal "
                                 "basis in every step; if "
                                 "false, the solver automatically checks for "
                                 "loss of orthogonality "
                                 "every 5 iterations.");

            params.leave_subsection();
          }

        // BiCGStab additional data.
        if (solver_set.find("BiCGStab") != solver_set.end())
          {
            params.enter_subsection("BiCGStab");
            params.declare_entry(
              "Exact residual",
              "true",
              Patterns::Bool(),
              "Toggle using exact linear system residual or an estimate.");

            params.declare_entry(
              "Breakdown",
              "1e-10",
              Patterns::Double(0),
              "A threshold telling which numbers are considered zero.");
            params.leave_subsection();
          }

        // FGMRES additional data.
        if (solver_set.find("FGMRES") != solver_set.end())
          {
            params.enter_subsection("FGMRES");

            params.declare_entry("Maximum basis size",
                                 "500",
                                 Patterns::Integer(0),
                                 "The maximum number of temporary vectors "
                                 "is (2 * max_basis_size + 1).");

            params.leave_subsection();
          }
      }
      params.reset_verbosity();
    }
    params.leave_subsection_path();
  }


  template <class VectorType>
  void
  LinearSolverHandler<VectorType>::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      // Solver control.
      prm_reduction_control_init.set_max_steps(
        params.get_integer("Maximum number of iterations"));
      prm_reduction_control_init.set_tolerance(params.get_double("Tolerance"));
      prm_reduction_control_init.set_reduction(params.get_double("Reduction"));
      prm_reduction_control_init.log_history(params.get_bool("Log history"));

      prm_log_frequency = params.get_integer("Log frequency");

      prm_reduction_control_init.log_frequency(prm_log_frequency);
      prm_reduction_control_init.log_result(true);

      if (prm_reduction_control_init.log_history())
        {
          prm_reduction_control_init.enable_history_data();
        }

      prm_linear_solver_type = params.get("Solver type");

      if (prm_linear_solver_type == "GMRES")
        {
          params.enter_subsection("GMRES");
          prm_data_gmres.max_n_tmp_vectors =
            params.get_integer("Maximum number of temporary vectors");

          prm_data_gmres.right_preconditioning =
            params.get_bool("Right preconditioning");

          prm_data_gmres.use_default_residual =
            params.get_bool("Use default residual");

          prm_data_gmres.force_re_orthogonalization =
            params.get_bool("Force re-orthogonalization");
          params.leave_subsection();
        }
      else if (prm_linear_solver_type == "BiCGStab")
        {
          params.enter_subsection("BiCGStab");
          prm_data_bicgstab.exact_residual = params.get_bool("Exact residual");
          prm_data_bicgstab.breakdown      = params.get_double("Breakdown");
          params.leave_subsection();
        }
      else if (prm_linear_solver_type == "FGMRES")
        {
          params.enter_subsection("FGMRES");
          prm_data_fgmres.max_basis_size =
            params.get_integer("Maximum basis size");
          params.leave_subsection();
        }
    }
    params.leave_subsection_path();
  }

  /// Explicit instantiation.
  template class LinearSolverHandler<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class LinearSolverHandler<Vector<double>>;

  /// Explicit instantiation.
  template class LinearSolverHandler<LinAlg::MPI::BlockVector>;

  /// Explicit instantiation.
  template class LinearSolverHandler<
    LinearAlgebra::distributed::Vector<double>>;

} // namespace lifex::utils
