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

#include "source/init.hpp"

#include "source/numerics/preconditioner_handler.hpp"

#include "examples/multidomain_heat/multidomain_heat.hpp"

#include <iomanip>
#include <set>

namespace lifex::examples
{
  ExampleMultidomainHeat::ExampleMultidomainHeat(
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
    , preconditioner(prm_subsection_path + " / Block Preconditioner",
                     system_matrix)
  {}

  void
  ExampleMultidomainHeat::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Transport coefficient",
                         "0, 0, 0",
                         Patterns::List(Patterns::Double(), dim, dim),
                         "Coefficient of the transport term.");

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
        Patterns::List(Patterns::Double(0), 2, 2),
        "Coefficients for Robin interface conditions on the two subdomains.");
    }
    params.leave_subsection();

    params.leave_subsection_path();

    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);

    ExampleHeat tmp_subproblem_0(0,
                                 Tensor<1, dim>(),
                                 {},
                                 subsection_subproblem_0);
    tmp_subproblem_0.declare_parameters(params);

    ExampleHeat tmp_subproblem_1(1,
                                 Tensor<1, dim>(),
                                 {},
                                 subsection_subproblem_1);
    tmp_subproblem_1.declare_parameters(params);

    utils::FixedPointAcceleration<LinAlg::MPI::Vector>::Factory::
      declare_children_parameters(params,
                                  prm_subsection_path +
                                    " / Fixed point / Relaxation");
  }

  void
  ExampleMultidomainHeat::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);

    for (unsigned int i = 0; i < dim; ++i)
      prm_transport_coefficient[i] =
        params.get_vector<double>("Transport coefficient")[i];

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
    preconditioner.parse_parameters(params);

    subproblems[0] = std::make_unique<ExampleHeat>(
      0,
      prm_transport_coefficient,
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_0);
    subproblems[0]->parse_parameters(params);

    subproblems[1] = std::make_unique<ExampleHeat>(
      1,
      prm_transport_coefficient,
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
  ExampleMultidomainHeat::run()
  {
    create_mesh(prm_n_refinements);
    setup_system();

    pcout << utils::log::separator_section << "\nSetting up interface maps"
          << std::endl;

    setup_interface_maps();

    pcout << utils::log::separator_section << std::endl;

    if (prm_scheme == "Monolithic")
      setup_system_monolithic();

    while (subproblems[0]->time < 1.0)
      {
        pcout << "Timestep " << std::setw(4) << subproblems[0]->timestep_number
              << ",   time " << std::setw(13) << std::setprecision(8)
              << subproblems[0]->time;

        if (prm_scheme == "Fixed point")
          {
            pcout << std::endl;

            fixed_point_acceleration->reset(subproblems[1]->solution_owned);

            n_iter = 0;
            do
              {
                ++n_iter;

                pcout << "   n_iter = " << n_iter << "/" << prm_n_max_iter
                      << std::endl;

                pcout << "      subdomain 0";
                subproblems[0]->step();
                subproblems[0]->compute_interface_residual();

                pcout << "      subdomain 1";
                subproblems[1]->step();
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
              }
            while (n_iter < prm_n_max_iter && res_norm > prm_tolerance);
          }
        else // if (prm_scheme == "Monolithic")
          {
            assemble_monolithic();
            solve_monolithic();
          }

        subproblems[0]->time_advance();
        subproblems[1]->time_advance();
      }
  }

  void
  ExampleMultidomainHeat::create_mesh(const unsigned int &n_refinements)
  {
    subproblems[0]->create_mesh(n_refinements);
    subproblems[1]->create_mesh(n_refinements);
  }

  void
  ExampleMultidomainHeat::setup_system()
  {
    subproblems[0]->setup_system();
    subproblems[1]->setup_system();
  }

  void
  ExampleMultidomainHeat::setup_interface_maps()
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
            LinAlg::MPI::Vector>::InterfaceDataRobinLinear>>
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
                LinAlg::MPI::Vector>::InterfaceDataRobinLinear>(
                subproblems[other_subdomain]->residual_no_interface,
                subproblems[other_subdomain]->solution,
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
  ExampleMultidomainHeat::setup_system_monolithic()
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
  }

  void
  ExampleMultidomainHeat::assemble_monolithic()
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
              cell, c, local_dof_indices, cell_matrix, cell_rhs);

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
  ExampleMultidomainHeat::solve_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Solve monolithic system");

    // Construct the preconditioner.
    preconditioner.initialize(system_matrix);

    // Solve.
    linear_solver.solve(system_matrix,
                        solution_owned,
                        system_rhs,
                        preconditioner);
    constraints.distribute(solution_owned);
    solution = solution_owned;

    // Copy the solution to the subproblems.
    subproblems[0]->solution_owned = solution_owned.block(0);
    subproblems[1]->solution_owned = solution_owned.block(1);
    subproblems[0]->solution       = solution.block(0);
    subproblems[1]->solution       = solution.block(1);
  }

  void
  ExampleMultidomainHeat::output_results()
  {
    subproblems[0]->output_results();
    subproblems[1]->output_results();
  }

} // namespace lifex::examples
