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

#include "source/numerics/preconditioner_handler.hpp"

#include "examples/multidomain_heat_nonlinear/multidomain_heat_nonlinear.hpp"

#include <iomanip>
#include <set>

namespace lifex::examples
{
  ExampleMultidomainHeatNonLinear::ExampleMultidomainHeatNonLinear(
    const std::string &subsection,
    const std::string &subsection_subproblem_0_,
    const std::string &subsection_subproblem_1_)
    : CoreModel(subsection)
    , interface_tag({{1, 0}})
    , subsection_subproblem_0(subsection_subproblem_0_)
    , subsection_subproblem_1(subsection_subproblem_1_)
    , non_linear_solver(prm_subsection_path +
                        " / Monolithic / Non-linear solver")
    , linear_solver(prm_subsection_path + " / Monolithic / Linear solver",
                    {"GMRES", "BiCGStab"},
                    "GMRES")
    , preconditioner(prm_subsection_path + " / Block preconditioner",
                     system_matrix)
  {}

  void
  ExampleMultidomainHeatNonLinear::declare_parameters(
    lifex::ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Refinements",
                         "2",
                         Patterns::Integer(0),
                         "Number of refinements of the mesh.");

    params.declare_entry_selection("Scheme",
                                   "Fixed point",
                                   "Fixed point|Monolithic",
                                   "Scheme used for coupling the subdomains.");

    params.enter_subsection("Fixed point");
    {
      params.declare_entry(
        "Interface conditions",
        "Dirichlet, Neumann",
        Patterns::List(Patterns::Selection("Dirichlet|Neumann|Robin"), 2, 2),
        "Interface conditions applied to the subdomains "
        "(Dirichlet|Neumann|Robin).");

      params.enter_subsection("Relaxation");
      params.declare_entry_selection(
        "Scheme",
        utils::FixedPointRelaxation<LinAlg::MPI::Vector>::label,
        utils::FixedPointAcceleration<
          LinAlg::MPI::Vector>::Factory::get_registered_keys_prm(),
        "Scheme to be used for relaxation.");
      params.leave_subsection();

      params.declare_entry(
        "Tolerance",
        "1e-6",
        Patterns::Double(0),
        "Tolerance on the norm of the increment of interface "
        "data from second subdomain, to stop iterations.");

      params.declare_entry("Max iterations",
                           "50",
                           Patterns::Integer(0),
                           "Maximum number of iterations.");

      params.declare_entry(
        "Robin coefficients",
        "1.0, 0.5",
        Patterns::List(Patterns::Double(0)),
        "Coefficients for Robin interface conditions on the two subdomains.");
    }
    params.leave_subsection();

    params.enter_subsection("Time solver");
    {
      params.declare_entry("BDF order",
                           "1",
                           Patterns::Integer(1, 3),
                           "BDF order: 1, 2, 3.");

      params.declare_entry("Initial time",
                           "0",
                           Patterns::Double(),
                           "Initial time.");

      params.declare_entry("Final time",
                           "1",
                           Patterns::Double(),
                           "Final time.");

      params.declare_entry("Time step",
                           "1e-2",
                           Patterns::Double(0),
                           "Time step.");
    }
    params.leave_subsection();

    params.leave_subsection_path();

    non_linear_solver.declare_parameters(params);
    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);

    ExampleHeatNonLinear tmp_subproblem_0(
      0,
      1,
      0.0,
      0.1,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_0);
    tmp_subproblem_0.declare_parameters(params);

    ExampleHeatNonLinear tmp_subproblem_1(
      1,
      1,
      0.0,
      0.1,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_1);
    tmp_subproblem_1.declare_parameters(params);

    utils::FixedPointAcceleration<LinAlg::MPI::Vector>::Factory::
      declare_children_parameters(params,
                                  prm_subsection_path +
                                    " / Fixed point / Relaxation");
  }

  void
  ExampleMultidomainHeatNonLinear::parse_parameters(lifex::ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);

    prm_n_refinements = params.get_integer("Refinements");

    prm_scheme = params.get("Scheme");

    params.enter_subsection("Fixed point");
    {
      const std::vector<std::string> &interface_type_str =
        params.get_vector<std::string>("Interface conditions");

      params.enter_subsection("Relaxation");
      prm_fixed_point_acceleration_type = params.get("Scheme");
      params.leave_subsection();

      prm_tolerance  = params.get_double("Tolerance");
      prm_n_max_iter = params.get_integer("Max iterations");

      const std::vector<double> &robin_coefficients =
        params.get_vector<double>("Robin coefficients");

      for (unsigned int i = 0; i < 2; ++i)
        {
          if (interface_type_str[i] == "Dirichlet")
            prm_fixed_point_ICs[i] = InterfaceType::Dirichlet;
          else if (interface_type_str[i] == "Neumann")
            prm_fixed_point_ICs[i] = InterfaceType::Neumann;
          else if (interface_type_str[i] == "Robin")
            prm_fixed_point_ICs[i] = InterfaceType::Robin;

          prm_robin_coefficient[i] = robin_coefficients[i];
        }
    }
    params.leave_subsection();

    params.enter_subsection("Time solver");
    {
      prm_bdf_order    = params.get_integer("BDF order");
      prm_initial_time = params.get_double("Initial time");
      prm_final_time   = params.get_double("Final time");
      prm_time_step    = params.get_double("Time step");
    }
    params.leave_subsection();

    params.leave_subsection_path();

    non_linear_solver.parse_parameters(params);
    linear_solver.parse_parameters(params);
    preconditioner.parse_parameters(params);

    subproblems[0] = std::make_unique<ExampleHeatNonLinear>(
      0,
      prm_bdf_order,
      prm_initial_time,
      prm_time_step,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_0);
    subproblems[0]->parse_parameters(params);

    subproblems[1] = std::make_unique<ExampleHeatNonLinear>(
      1,
      prm_bdf_order,
      prm_initial_time,
      prm_time_step,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_1);
    subproblems[1]->parse_parameters(params);

    fixed_point_acceleration =
      utils::FixedPointAcceleration<LinAlg::MPI::Vector>::Factory::
        parse_child_parameters(params,
                               prm_fixed_point_acceleration_type,
                               prm_subsection_path +
                                 " / Fixed point / Relaxation");
  }

  void
  ExampleMultidomainHeatNonLinear::run()
  {
    create_mesh(prm_n_refinements);
    setup_system();

    pcout << utils::log::separator_section << "\nSetting up interface maps"
          << std::endl;

    setup_interface_maps();

    pcout << utils::log::separator_section << std::endl;

    if (prm_scheme == "Monolithic")
      setup_system_monolithic();

    while (subproblems[0]->time < prm_final_time)
      {
        pcout << "Timestep " << std::setw(4) << subproblems[0]->timestep_number
              << ",   time " << std::setw(13) << std::setprecision(8)
              << subproblems[0]->time;

        if (prm_scheme == "Fixed point")
          {
            pcout << std::endl;

            n_iter = 0;
            fixed_point_acceleration->reset(subproblems[1]->solution_owned);

            do
              {
                ++n_iter;
                pcout << "\tn_iter = " << n_iter << "/" << prm_n_max_iter
                      << std::endl;

                pcout << "\tsubproblem 0";
                subproblems[0]->step();

                pcout << "\tsubproblem 1";
                subproblems[1]->step();
                subproblems[1]->set_solution(
                  fixed_point_acceleration->get_next_element(
                    subproblems[1]->solution_owned));

                res_norm = utils::compute_interface_residual_norm(
                  owned_interface_dofs,
                  subproblems[0]->solution,
                  subproblems[0]->residual_no_interface,
                  *interface_maps[0],
                  subproblems[1]->solution,
                  subproblems[1]->residual_no_interface,
                  *interface_maps[1],
                  weight_dirichlet,
                  weight_neumann,
                  prm_fixed_point_ICs[0] == InterfaceType::Robin ?
                    interface_mass_matrix :
                    nullptr);
              }
            while (n_iter < prm_n_max_iter && res_norm > prm_tolerance);
          }
        else // if (prm_scheme == "Monolithic")
          {
            solve_monolithic();

            pcout << std::endl;
          }

        subproblems[0]->time_advance();
        subproblems[1]->time_advance();
      }
  }

  void
  ExampleMultidomainHeatNonLinear::create_mesh(
    const unsigned int &n_refinements)
  {
    subproblems[0]->create_mesh(n_refinements);
    subproblems[1]->create_mesh(n_refinements);
  }

  void
  ExampleMultidomainHeatNonLinear::setup_system()
  {
    subproblems[0]->setup_system();
    subproblems[1]->setup_system();
  }

  void
  ExampleMultidomainHeatNonLinear::setup_interface_maps()
  {
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Setup interface maps");

      std::tie(interface_maps, owned_interface_dofs, relevant_interface_dofs) =
        utils::compute_interface_maps(subproblems[0]->dof_handler,
                                      subproblems[1]->dof_handler,
                                      interface_tag,
                                      {});

      pcout << "Number of interface DoFs: " << owned_interface_dofs.size()
            << std::endl;
    }

    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Setup interface handlers");

      for (unsigned int subdomain = 0; subdomain < 2; ++subdomain)
        {
          const unsigned int other_subdomain = (subdomain == 0 ? 1 : 0);

          // Vectors of interface data.
          std::vector<std::shared_ptr<utils::InterfaceHandler<
            LinAlg::MPI::Vector>::InterfaceDataDirichlet>>
            interface_data_dirichlet;

          std::vector<std::shared_ptr<
            utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataNeumann>>
            interface_data_neumann;

          std::vector<std::shared_ptr<utils::InterfaceHandler<
            LinAlg::MPI::Vector>::InterfaceDataRobinNonLinear>>
            interface_data_robin;

          if (prm_fixed_point_ICs[subdomain] == InterfaceType::Dirichlet)
            interface_data_dirichlet.push_back(
              std::make_shared<utils::InterfaceHandler<
                LinAlg::MPI::Vector>::InterfaceDataDirichlet>(
                subproblems[other_subdomain]->solution,
                owned_interface_dofs,
                relevant_interface_dofs,
                interface_maps[subdomain],
                interface_maps[other_subdomain]));

          if (prm_fixed_point_ICs[subdomain] == InterfaceType::Neumann)
            interface_data_neumann.push_back(
              std::make_shared<utils::InterfaceHandler<
                LinAlg::MPI::Vector>::InterfaceDataNeumann>(
                subproblems[other_subdomain]->residual_no_interface,
                owned_interface_dofs,
                relevant_interface_dofs,
                interface_maps[subdomain],
                interface_maps[other_subdomain]));

          if (prm_fixed_point_ICs[subdomain] == InterfaceType::Robin)
            interface_data_robin.push_back(
              std::make_shared<utils::InterfaceHandler<
                LinAlg::MPI::Vector>::InterfaceDataRobinNonLinear>(
                subproblems[other_subdomain]->residual_no_interface,
                subproblems[other_subdomain]->solution,
                subproblems[subdomain]->solution,
                owned_interface_dofs,
                relevant_interface_dofs,
                interface_maps[subdomain],
                interface_maps[other_subdomain],
                std::set<types::boundary_id>({{interface_tag[subdomain]}}),
                prm_robin_coefficient[subdomain],
                ComponentMask({true})));

          subproblems[subdomain]->setup_interface(
            subproblems[other_subdomain]->dof_handler,
            interface_data_dirichlet,
            interface_data_neumann,
            interface_data_robin);
        }

      if (prm_fixed_point_ICs[0] == InterfaceType::Dirichlet)
        {
          weight_dirichlet = 1.0;
          weight_neumann   = 0.0;
        }
      else if (prm_fixed_point_ICs[0] == InterfaceType::Neumann)
        {
          weight_dirichlet = 0.0;
          weight_neumann   = 1.0;
        }
      else if (prm_fixed_point_ICs[0] == InterfaceType::Robin)
        {
          weight_dirichlet = prm_robin_coefficient[0];
          weight_neumann   = 1.0;

          // For Robin conditions, we use the mass matrix to weigh the
          // Dirichlet residual.
          interface_mass_matrix =
            std::make_shared<LinAlgTrilinos::Wrappers::SparseMatrix>();
          compute_interface_mass_matrix(*interface_maps[0],
                                        owned_interface_dofs,
                                        relevant_interface_dofs,
                                        subproblems[0]->dof_handler,
                                        {interface_tag[0]},
                                        *interface_mass_matrix);
        }
    }
  }

  void
  ExampleMultidomainHeatNonLinear::setup_system_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Setup monolithic system");

    // Count DoFs on the two subdomains.
    n_dofs.resize(2);
    n_dofs[0] = subproblems[0]->dof_handler.n_dofs();
    n_dofs[1] = subproblems[1]->dof_handler.n_dofs();

    const unsigned int n_total_dofs = n_dofs[0] + n_dofs[1];

    // Construct index sets of owned and relevant DoFs.
    block_owned_dofs.resize(2);
    block_owned_dofs[0] = subproblems[0]->owned_dofs;
    block_owned_dofs[1] = subproblems[1]->owned_dofs;

    block_relevant_dofs.resize(2);
    block_relevant_dofs[0] = subproblems[0]->relevant_dofs;
    block_relevant_dofs[1] = subproblems[1]->relevant_dofs;

    for (const auto &dof : *interface_maps[0])
      block_relevant_dofs[1].add_index(dof.second.get_other_subdomain_index());

    for (const auto &dof : *interface_maps[1])
      block_relevant_dofs[0].add_index(dof.second.get_other_subdomain_index());

    owned_dofs_global.set_size(n_total_dofs);
    relevant_dofs_global.set_size(n_total_dofs);
    unsigned int offset = 0;
    for (unsigned int i = 0; i < 2; ++i)
      {
        owned_dofs_global.add_indices(block_owned_dofs[i], offset);
        relevant_dofs_global.add_indices(block_relevant_dofs[i], offset);
        offset += n_dofs[i];
      }

    /// Constructs constraints for the monolithic system.
    AffineConstraints<double> constraints_0(block_relevant_dofs[0]);
    subproblems[0]->bc_handler.apply_dirichlet(constraints_0);
    constraints_0.close();

    AffineConstraints<double> constraints_1(block_relevant_dofs[1]);
    subproblems[1]->bc_handler.apply_dirichlet(constraints_1);
    constraints_1.close();

    std::tie(constraints, std::ignore) =
      utils::make_interface_constraints(constraints_0,
                                        constraints_1,
                                        *interface_maps[0],
                                        *interface_maps[1],
                                        relevant_dofs_global,
                                        n_dofs[0]);
    constraints.close();

    // Construct the sparsity pattern.
    BlockDynamicSparsityPattern dsp;
    utils::make_interface_sparsity_pattern(subproblems[0]->dof_handler,
                                           subproblems[1]->dof_handler,
                                           {n_dofs[0]},
                                           {n_dofs[1]},
                                           constraints,
                                           dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs_global,
                                               mpi_comm,
                                               relevant_dofs_global);

    // Initialize the matrices and vectors.
    utils::initialize_matrix(system_matrix, block_owned_dofs, dsp);
    system_rhs.reinit(block_owned_dofs, mpi_comm);
    solution_owned.reinit(block_owned_dofs, mpi_comm);
    solution.reinit(block_owned_dofs, block_relevant_dofs, mpi_comm);

    // Initialize non-linear solver handler.
    non_linear_solver.initialize(&solution_owned, &solution);
  }

  void
  ExampleMultidomainHeatNonLinear::assemble_monolithic(const bool &assemble_jac)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Assemble monolithic system");

    system_matrix = 0.0;
    system_rhs    = 0.0;

    for (unsigned int k = 0; k < 2; ++k)
      {
        const FiniteElement<dim> &fe            = *(subproblems[k]->fe);
        const DoFHandler<dim> &   dof_handler   = subproblems[k]->dof_handler;
        const unsigned int        dofs_per_cell = fe.dofs_per_cell;

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        unsigned int c = 0; // Cell index.

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            cell->get_dof_indices(local_dof_indices);
            subproblems[k]->assemble_cell(
              cell, c, local_dof_indices, cell_matrix, cell_rhs, assemble_jac);

            // DoF indices for the second subdomain must be shifted.
            if (k == 1)
              {
                for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                  local_dof_indices[i] += n_dofs[0];
              }

            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);

            ++c;
          }
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void
  ExampleMultidomainHeatNonLinear::solve_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Solve monolithic system");

    auto assemble_fun = [this](const bool &assemble_jacobian) {
      // Copy the solution to the subproblems.
      subproblems[0]->solution_owned = solution_owned.block(0);
      subproblems[1]->solution_owned = solution_owned.block(1);
      subproblems[0]->solution       = solution.block(0);
      subproblems[1]->solution       = solution.block(1);

      assemble_monolithic(assemble_jacobian);
      return system_rhs.l2_norm();
    };

    auto solve_fun = [this](const bool & /*assemble_preconditioner*/,
                            LinAlg::MPI::BlockVector &incr) {
      const double norm_sol = solution_owned.l2_norm();

      // Construct the preconditioner.
      preconditioner.initialize(system_matrix);
      linear_solver.solve(system_matrix, incr, system_rhs, preconditioner);
      constraints.distribute(incr);

      const double norm_incr = incr.l2_norm();

      const unsigned int n_iterations_linear = linear_solver.get_n_iterations();

      return std::make_tuple(norm_sol, norm_incr, n_iterations_linear);
    };

    const bool converged = non_linear_solver.solve(assemble_fun, solve_fun);

    AssertThrow(converged, ExcNonlinearNotConverged());

    // Copy the solution to the subproblems.
    subproblems[0]->solution_owned = solution_owned.block(0);
    subproblems[1]->solution_owned = solution_owned.block(1);
    subproblems[0]->solution       = solution.block(0);
    subproblems[1]->solution       = solution.block(1);
  }
} // namespace lifex::examples
