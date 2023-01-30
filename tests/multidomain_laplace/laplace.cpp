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

#include "source/io/data_writer.hpp"

#include "tests/multidomain_laplace/laplace.hpp"

#include <deal.II/grid/grid_tools.h>

namespace lifex::tests
{
  TestLaplace::TestLaplace(
    const unsigned int &  subdomain_id_,
    const unsigned int &  n_components_,
    const Tensor<1, dim> &transport_coefficient_,
    const std::vector<utils::BC<utils::FunctionDirichlet>> &dirichlet_bcs_,
    const std::string &                                     subsection)
    : CoreModel(subsection)
    , n_components(n_components_)
    , subdomain_id(subdomain_id_)
    , triangulation(prm_subsection_path, mpi_comm)
    , transport_coefficient(transport_coefficient_)
    , dirichlet_bcs(dirichlet_bcs_)
    , bc_handler(dof_handler)
    , linear_solver(prm_subsection_path + " / Linear solver",
                    {"CG", "GMRES", "BiCGStab"},
                    "CG")
    , preconditioner(prm_subsection_path + " / Preconditioner")
  {}

  void
  TestLaplace::declare_parameters(lifex::ParamHandler &params) const
  {
    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);
  }

  void
  TestLaplace::parse_parameters(lifex::ParamHandler &params)
  {
    linear_solver.parse_parameters(params);
    preconditioner.parse_parameters(params);
  }

  void
  TestLaplace::setup_interface(
    const DoFHandler<dim> &other_dof_handler,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataDirichlet>>
      interface_data_dirichlet,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataNeumann>>
      interface_data_neumann,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataRobinLinear>>
      interface_data_robin)
  {
    interface_handler =
      std::make_unique<utils::InterfaceHandler<LinAlg::MPI::Vector>>(
        other_dof_handler,
        interface_data_dirichlet,
        interface_data_neumann,
        interface_data_robin,
        std::vector<std::shared_ptr<utils::InterfaceHandler<
          LinAlg::MPI::Vector>::InterfaceDataRobinNonLinear>>());
  }

  void
  TestLaplace::create_mesh(const unsigned int &n_refinements)
  {
    triangulation.initialize_hypercube(-0.5, 0.5, true);
    triangulation.set_refinement_global(n_refinements);
    triangulation.create_mesh();

    if (subdomain_id == 1)
      {
        GridTools::rotate(M_PI,
#if LIFEX_DIM == 3
                          0,
#endif
                          triangulation.get());

        Tensor<1, dim> shift;
        shift[0] = 1;
        GridTools::shift(shift, triangulation.get());
      }
  }

  void
  TestLaplace::setup_system()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Setup system");

    const auto fe_scalar = triangulation.get_fe_lagrange(1);
    fe = std::make_unique<FESystem<dim>>(*fe_scalar, n_components);
    quadrature_formula = triangulation.get_quadrature_gauss(fe->degree + 1);
    face_quadrature_formula =
      triangulation.get_quadrature_gauss<dim - 1>(fe->degree + 1);

    // Distribute DoFs.
    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);
    pcout << "Number of DoFs in subdomain " << subdomain_id << ": "
          << dof_handler.n_dofs() << std::endl;

    owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

    bc_handler.initialize(dirichlet_bcs);

    constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints);
    constraints.close();

    // Setup sparsity.
    DynamicSparsityPattern dsp(relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs,
                                               mpi_comm,
                                               relevant_dofs);

    // Initialize matrix and vectors.
    utils::initialize_matrix(system_matrix, owned_dofs, dsp);
    system_rhs.reinit(owned_dofs, mpi_comm);
    solution.reinit(owned_dofs, relevant_dofs, mpi_comm);
    solution_owned.reinit(owned_dofs, mpi_comm);
    solution = solution_owned = 0.0;

    system_matrix_no_interface.reinit(owned_dofs, owned_dofs, dsp, mpi_comm);
    system_rhs_no_interface.reinit(owned_dofs, mpi_comm);
    residual_no_interface.reinit(owned_dofs, mpi_comm);
  }

  void
  TestLaplace::assemble_system()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Assemble system");

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    system_matrix              = 0.0;
    system_rhs                 = 0.0;
    system_matrix_no_interface = 0.0;
    system_rhs_no_interface    = 0.0;

    // Retrieve interface data.
    AssertThrow(interface_handler != nullptr, ExcNotInitialized());

    interface_handler->extract();

    // Set constraints for Dirichlet conditions.
    constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints);
    interface_handler->apply_dirichlet(constraints);
    constraints.close();

    constraints_no_interface.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints_no_interface);
    constraints_no_interface.close();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            assemble_cell(cell, cell_matrix, cell_rhs);
            cell->get_dof_indices(local_dof_indices);

            constraints_no_interface.distribute_local_to_global(
              cell_matrix,
              cell_rhs,
              local_dof_indices,
              system_matrix_no_interface,
              system_rhs_no_interface);

            if (cell->at_boundary())
              interface_handler->apply_current_subdomain(
                cell_matrix, cell_rhs, cell, *face_quadrature_formula);

            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }

    interface_handler->apply_other_subdomain(system_rhs, constraints);
    system_matrix_no_interface.compress(VectorOperation::add);
    system_rhs_no_interface.compress(VectorOperation::add);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void
  TestLaplace::assemble_cell(const DoFHandler<dim>::active_cell_iterator &cell,
                             FullMatrix<double> &cell_matrix,
                             Vector<double> &    cell_rhs)
  {
    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients |
                              update_JxW_values);
    fe_values.reinit(cell);

    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = fe_values.get_quadrature().size();

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          for (unsigned int q = 0; q < n_q_points; ++q)
            if (fe->system_to_component_index(i).first ==
                fe->system_to_component_index(j).first)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                 transport_coefficient * fe_values.shape_grad(j, q) *
                   fe_values.shape_value(i, q)) *
                fe_values.JxW(q);
      }
  }

  void
  TestLaplace::solve()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Solve");

    preconditioner.initialize(system_matrix);

    linear_solver.solve(system_matrix,
                        solution_owned,
                        system_rhs,
                        preconditioner);

    constraints.distribute(solution_owned);
    solution = solution_owned;
  }

  void
  TestLaplace::compute_interface_residual()
  {
    system_matrix_no_interface.vmult(residual_no_interface, solution_owned);
    residual_no_interface *= -1;
    residual_no_interface += system_rhs_no_interface;
  }

  void
  TestLaplace::output_results(const unsigned int &n_iter) const
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Output results");

    // Compute partition indicator for output.
    std::vector<unsigned int> partition_int(
      triangulation.get().n_active_cells());
    GridTools::get_subdomain_association(triangulation.get(), partition_int);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(partitioning, "partitioning");
    data_out.build_patches();

    utils::dataout_write_hdf5(data_out,
                              "solution_subdomain" +
                                std::to_string(subdomain_id),
                              n_iter,
                              n_iter,
                              n_iter);

    data_out.clear();
  }

  void
  TestLaplace::step()
  {
    assemble_system();
    solve();
  }
} // namespace lifex::tests
