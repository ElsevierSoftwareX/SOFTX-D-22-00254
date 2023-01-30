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

#include "examples/multidomain_stokes/multidomain_stokes.hpp"

#include <set>

namespace lifex::examples
{
  ExampleMultidomainStokes::ExampleMultidomainStokes(
    const std::string &subsection,
    const std::string &subsection_subproblem_0_,
    const std::string &subsection_subproblem_1_)
    : CoreModel(subsection)
    , interface_tag({{1, 0}})
    , subsection_subproblem_0(subsection_subproblem_0_)
    , subsection_subproblem_1(subsection_subproblem_1_)
    , linear_solver(prm_subsection_path + " / Monolithic / Linear solver",
                    {"GMRES", "BiCGStab"},
                    "GMRES")
    , prm_fixed_point_ICs({{InterfaceType::Dirichlet, InterfaceType::Neumann}})
  {}

  void
  ExampleMultidomainStokes::declare_parameters(
    lifex::ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Viscosity", "1.0", Patterns::Double(0), "Viscosity.");

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
        "Neumann, Dirichlet",
        Patterns::List(Patterns::Selection("Dirichlet|Neumann|Robin"), 2, 2),
        "Interface conditions applied to the subdomains."
        "(Dirichlet|Neumann|Robin).");

      params.enter_subsection("Relaxation");
      params.declare_entry_selection(
        "Scheme",
        utils::FixedPointRelaxation<LinAlg::MPI::BlockVector>::label,
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

    params.leave_subsection_path();

    linear_solver.declare_parameters(params);

    ExampleStokes tmp_subproblem_0(
      0,
      1.0,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_0);
    tmp_subproblem_0.declare_parameters(params);

    ExampleStokes tmp_subproblem_1(
      1,
      1.0,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_1);
    tmp_subproblem_1.declare_parameters(params);

    utils::FixedPointAcceleration<LinAlg::MPI::BlockVector>::Factory::
      declare_children_parameters(params,
                                  prm_subsection_path +
                                    " / Fixed point / Relaxation");
  }

  void
  ExampleMultidomainStokes::parse_parameters(lifex::ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);

    prm_viscosity = params.get_double("Viscosity");

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

    params.leave_subsection_path();

    linear_solver.parse_parameters(params);

    ComponentMask velocity_x(dim + 1, false);
    velocity_x.set(0, true);
    ComponentMask velocity_yz(dim + 1, false);
    velocity_yz.set(1, true);
    velocity_yz.set(2, true);
    ComponentMask velocities(dim + 1, true);
    velocities.set(dim, false);

    std::vector<utils::BC<utils::FunctionDirichlet>> dirichlet_bcs_0;
    dirichlet_bcs_0.emplace_back(
      0, std::make_shared<utils::ConstantBCFunction>(1.0, dim + 1), velocity_x);
    dirichlet_bcs_0.emplace_back(
      0, std::make_shared<utils::ZeroBCFunction>(dim + 1), velocity_yz);
    for (unsigned int i = 2; i < 6; ++i)
      dirichlet_bcs_0.emplace_back(
        i, std::make_shared<utils::ZeroBCFunction>(dim + 1), velocities);

    subproblems[0] = std::make_unique<ExampleStokes>(0,
                                                     prm_viscosity,
                                                     dirichlet_bcs_0,
                                                     subsection_subproblem_0);
    subproblems[0]->parse_parameters(params);

    std::vector<utils::BC<utils::FunctionDirichlet>> dirichlet_bcs_1;
    for (unsigned int i = 2; i < 6; ++i)
      dirichlet_bcs_1.emplace_back(
        i, std::make_shared<utils::ZeroBCFunction>(dim + 1), velocities);

    subproblems[1] = std::make_unique<ExampleStokes>(1,
                                                     prm_viscosity,
                                                     dirichlet_bcs_1,
                                                     subsection_subproblem_1);
    subproblems[1]->parse_parameters(params);

    fixed_point_acceleration =
      utils::FixedPointAcceleration<LinAlg::MPI::BlockVector>::Factory::
        parse_child_parameters(params,
                               prm_fixed_point_acceleration_type,
                               prm_subsection_path +
                                 " / Fixed point / Relaxation");
  }

  void
  ExampleMultidomainStokes::run()
  {
    create_mesh(prm_n_refinements);
    setup_system();

    pcout << utils::log::separator_section << "\nSetting up interface maps"
          << std::endl;

    setup_interface();

    pcout << utils::log::separator_section << std::endl;

    if (prm_scheme == "Fixed point")
      {
        n_iter = 0;

        fixed_point_acceleration->reset(subproblems[1]->solution_owned);

        do
          {
            pcout << "   subproblem 0";
            subproblems[0]->step(n_iter);
            subproblems[0]->compute_interface_residual();

            pcout << "   subproblem 1";
            subproblems[1]->step(n_iter);
            subproblems[1]->set_solution(
              fixed_point_acceleration->get_next_element(
                subproblems[1]->solution_owned));
            subproblems[1]->compute_interface_residual();

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

            ++n_iter;
            pcout << "Iteration " << std::setw(3) << n_iter
                  << "\tresidual = " << std::setprecision(8) << std::setw(13)
                  << res_norm << std::endl;
          }
        while (n_iter < prm_n_max_iter && res_norm > prm_tolerance);
      }
    else // if (prm_scheme == "Monolithic")
      {
        setup_system_monolithic();
        assemble_monolithic();
        solve_monolithic();

        subproblems[0]->output_results(0);
        subproblems[1]->output_results(0);
      }
  }

  void
  ExampleMultidomainStokes::create_mesh(const unsigned int &n_refinements)
  {
    subproblems[0]->create_mesh(n_refinements);
    subproblems[1]->create_mesh(n_refinements);
  }

  void
  ExampleMultidomainStokes::setup_system()
  {
    subproblems[0]->setup_system();
    subproblems[1]->setup_system();
  }

  void
  ExampleMultidomainStokes::setup_interface()
  {
    ComponentMask velocities(dim + 1, true);
    velocities.set(dim, false);

    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Setup interface maps");

      std::tie(interface_maps, owned_interface_dofs, relevant_interface_dofs) =
        utils::compute_interface_maps(subproblems[0]->dof_handler,
                                      subproblems[1]->dof_handler,
                                      interface_tag,
                                      {{0, 0}, {1, 1}, {2, 2}});

      pcout << "Number of interface DoFs: " << owned_interface_dofs.size()
            << std::endl;
    }

    if (prm_scheme == "Fixed point")
      {
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Setup handlers");

        for (unsigned int subdomain = 0; subdomain < 2; ++subdomain)
          {
            const unsigned int other_subdomain = (subdomain == 0 ? 1 : 0);

            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::BlockVector>::InterfaceDataDirichlet>>
              interface_data_dofs;

            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::BlockVector>::InterfaceDataNeumann>>
              interface_data_neumann;

            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::BlockVector>::InterfaceDataRobinLinear>>
              interface_data_robin;

            if (prm_fixed_point_ICs[subdomain] == InterfaceType::Dirichlet)
              interface_data_dofs.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::BlockVector>::InterfaceDataDirichlet>(
                  subproblems[other_subdomain]->solution,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain],
                  velocities));

            else if (prm_fixed_point_ICs[subdomain] == InterfaceType::Neumann)
              interface_data_neumann.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::BlockVector>::InterfaceDataNeumann>(
                  subproblems[other_subdomain]->residual_no_interface,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain]));

            else if (prm_fixed_point_ICs[subdomain] == InterfaceType::Robin)
              interface_data_robin.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::BlockVector>::InterfaceDataRobinLinear>(
                  subproblems[other_subdomain]->residual_no_interface,
                  subproblems[other_subdomain]->solution,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain],
                  std::set<types::boundary_id>({{interface_tag[subdomain]}}),
                  prm_robin_coefficient[subdomain],
                  velocities));

            subproblems[subdomain]->setup_interface(
              subproblems[other_subdomain]->dof_handler,
              interface_data_dofs,
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
                                          *interface_mass_matrix,
                                          velocities);
          }
      }
  }

  void
  ExampleMultidomainStokes::setup_system_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Setup monolithic system");

    // Count DoFs on each block.
    n_dofs.resize(4);
    n_dofs[0] = subproblems[0]->block_owned_dofs[0].size();
    n_dofs[1] = subproblems[0]->block_owned_dofs[1].size();
    n_dofs[2] = subproblems[1]->block_owned_dofs[0].size();
    n_dofs[3] = subproblems[1]->block_owned_dofs[1].size();

    const unsigned int n_total_dofs =
      n_dofs[0] + n_dofs[1] + n_dofs[2] + n_dofs[3];

    // Construct index sets of owned and relevant DoFs.
    block_owned_dofs.resize(4);
    block_owned_dofs[0] = subproblems[0]->block_owned_dofs[0];
    block_owned_dofs[1] = subproblems[0]->block_owned_dofs[1];
    block_owned_dofs[2] = subproblems[1]->block_owned_dofs[0];
    block_owned_dofs[3] = subproblems[1]->block_owned_dofs[1];

    block_relevant_dofs.resize(4);
    block_relevant_dofs[0] = subproblems[0]->block_relevant_dofs[0];
    block_relevant_dofs[1] = subproblems[0]->block_relevant_dofs[1];
    block_relevant_dofs[2] = subproblems[1]->block_relevant_dofs[0];
    block_relevant_dofs[3] = subproblems[1]->block_relevant_dofs[1];

    for (const auto &dof : *interface_maps[0])
      if (dof.second.get_component() < dim)
        block_relevant_dofs[2].add_index(
          dof.second.get_other_subdomain_index());
      else
        block_relevant_dofs[3].add_index(
          dof.second.get_other_subdomain_index() - n_dofs[2]);

    for (const auto &dof : *interface_maps[1])
      if (dof.second.get_component() < dim)
        block_relevant_dofs[0].add_index(
          dof.second.get_other_subdomain_index());
      else
        block_relevant_dofs[1].add_index(
          dof.second.get_other_subdomain_index() - n_dofs[0]);

    owned_dofs_global.set_size(n_total_dofs);
    relevant_dofs_global.set_size(n_total_dofs);
    unsigned int offset = 0;
    for (unsigned int i = 0; i < 4; ++i)
      {
        owned_dofs_global.add_indices(block_owned_dofs[i], offset);
        relevant_dofs_global.add_indices(block_relevant_dofs[i], offset);
        offset += n_dofs[i];
      }

    /// Constructs constraints for the two subdomains and for the global system.
    IndexSet relevant_dofs_0(n_dofs[0] + n_dofs[1]);
    relevant_dofs_0.add_indices(block_relevant_dofs[0]);
    relevant_dofs_0.add_indices(block_relevant_dofs[1], n_dofs[0]);

    AffineConstraints<double> constraints_subdomain_0(relevant_dofs_0);
    subproblems[0]->bc_handler.apply_dirichlet(constraints_subdomain_0);
    constraints_subdomain_0.close();

    IndexSet relevant_dofs_1(n_dofs[2] + n_dofs[3]);
    relevant_dofs_1.add_indices(block_relevant_dofs[2]);
    relevant_dofs_1.add_indices(block_relevant_dofs[3], n_dofs[2]);

    AffineConstraints<double> constraints_subdomain_1(relevant_dofs_1);
    subproblems[1]->bc_handler.apply_dirichlet(constraints_subdomain_1);
    constraints_subdomain_1.close();

    std::tie(constraints, std::ignore) =
      utils::make_interface_constraints(constraints_subdomain_0,
                                        constraints_subdomain_1,
                                        *interface_maps[0],
                                        *interface_maps[1],
                                        relevant_dofs_global,
                                        n_dofs[0] + n_dofs[1]);
    constraints.close();

    // Create the sparsity pattern.
    BlockDynamicSparsityPattern dsp;
    utils::make_interface_sparsity_pattern(subproblems[0]->dof_handler,
                                           subproblems[1]->dof_handler,
                                           {n_dofs[0], n_dofs[1]},
                                           {n_dofs[2], n_dofs[3]},
                                           constraints,
                                           dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs_global,
                                               mpi_comm,
                                               relevant_dofs_global);

    // Initialize matrices and vectors.
    system_matrix.reinit(block_owned_dofs, dsp, mpi_comm);
    system_rhs.reinit(block_owned_dofs, mpi_comm);
    solution_owned.reinit(block_owned_dofs, mpi_comm);
    solution.reinit(block_owned_dofs, block_relevant_dofs, mpi_comm);
  }

  /// Assemble the monolithic system.
  void
  ExampleMultidomainStokes::assemble_monolithic()
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

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            cell->get_dof_indices(local_dof_indices);
            subproblems[k]->assemble_cell(cell, cell_matrix, cell_rhs);

            // DoF indices of second subdomain must be shifted.
            if (k == 1)
              {
                for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                  local_dof_indices[i] += n_dofs[0] + n_dofs[1];
              }

            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  /// Solve the monolithic system.
  void
  ExampleMultidomainStokes::solve_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Solve monolithic system");

    // Construct the preconditioner.
    LinAlg::Wrappers::PreconditionILU prec_A_0;
    prec_A_0.initialize(system_matrix.block(0, 0));
    utils::InverseMatrix inverse_A_0(system_matrix.block(0, 0), prec_A_0);

    LinAlg::Wrappers::PreconditionILU prec_A_1;
    prec_A_1.initialize(system_matrix.block(2, 2));
    utils::InverseMatrix inverse_A_1(system_matrix.block(2, 2), prec_A_1);

    LinAlg::Wrappers::PreconditionILU prec_Mp_0;
    prec_Mp_0.initialize(subproblems[0]->pressure_mass_matrix.block(1, 1));
    utils::InverseMatrix inverse_Mp_0(
      subproblems[0]->pressure_mass_matrix.block(1, 1), prec_Mp_0);

    LinAlg::Wrappers::PreconditionILU prec_Mp_1;
    prec_Mp_1.initialize(subproblems[1]->pressure_mass_matrix.block(1, 1));
    utils::InverseMatrix inverse_Mp_1(
      subproblems[1]->pressure_mass_matrix.block(1, 1), prec_Mp_1);

    Preconditioner preconditioner(inverse_A_0,
                                  inverse_Mp_0,
                                  system_matrix.block(0, 1),
                                  inverse_A_1,
                                  inverse_Mp_1,
                                  system_matrix.block(2, 3));

    // Solve.
    linear_solver.solve(system_matrix,
                        solution_owned,
                        system_rhs,
                        preconditioner);
    constraints.distribute(solution_owned);
    solution = solution_owned;

    // Copy the solution to the subproblems.
    subproblems[0]->solution.block(0) = solution.block(0);
    subproblems[0]->solution.block(1) = solution.block(1);
    subproblems[1]->solution.block(0) = solution.block(2);
    subproblems[1]->solution.block(1) = solution.block(3);
  }
} // namespace lifex::examples
