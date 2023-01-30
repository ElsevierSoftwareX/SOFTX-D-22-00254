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

#include "source/io/data_writer.hpp"

#include "examples/multidomain_stokes/stokes.hpp"

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_tools.h>

namespace lifex::examples
{
  ExampleStokes::ExampleStokes(
    const unsigned int &                                    subdomain_id_,
    const double &                                          viscosity_,
    const std::vector<utils::BC<utils::FunctionDirichlet>> &dirichlet_bcs_,
    const std::string &                                     subsection)
    : CoreModel(subsection)
    , subdomain_id(subdomain_id_)
    , triangulation(prm_subsection_path, mpi_comm)
    , dirichlet_bcs(dirichlet_bcs_)
    , bc_handler(dof_handler)
    , viscosity(viscosity_)
    , linear_solver(prm_subsection_path + " / Linear solver",
                    {"GMRES", "BiCGStab"},
                    "GMRES")
  {}

  void
  ExampleStokes::declare_parameters(lifex::ParamHandler &params) const
  {
    linear_solver.declare_parameters(params);
  }

  void
  ExampleStokes::parse_parameters(lifex::ParamHandler &params)
  {
    linear_solver.parse_parameters(params);
  }

  void
  ExampleStokes::setup_interface(
    const DoFHandler<dim> &other_dof_handler,
    const std::vector<std::shared_ptr<utils::InterfaceHandler<
      LinAlg::MPI::BlockVector>::InterfaceDataDirichlet>>
      interface_data_dirichlet,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::BlockVector>::InterfaceDataNeumann>>
      interface_data_neumann,
    const std::vector<std::shared_ptr<utils::InterfaceHandler<
      LinAlg::MPI::BlockVector>::InterfaceDataRobinLinear>>
      interface_data_robin)
  {
    interface_handler =
      std::make_unique<utils::InterfaceHandler<LinAlg::MPI::BlockVector>>(
        other_dof_handler,
        interface_data_dirichlet,
        interface_data_neumann,
        interface_data_robin,
        std::vector<std::shared_ptr<utils::InterfaceHandler<
          LinAlg::MPI::BlockVector>::InterfaceDataRobinNonLinear>>());
  }

  void
  ExampleStokes::create_mesh(const unsigned int &n_refinements)
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
  ExampleStokes::setup_system()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Setup system");

    const auto fe_scalar_2 = triangulation.get_fe_lagrange(2);
    const auto fe_scalar_1 = triangulation.get_fe_lagrange(1);

    fe = std::make_unique<FESystem<dim>>(*fe_scalar_2, dim, *fe_scalar_1, 1);
    quadrature_formula = triangulation.get_quadrature_gauss(fe->degree + 1);
    face_quadrature_formula =
      triangulation.get_quadrature_gauss<dim - 1>(fe->degree + 1);

    // Distribute DoFs.
    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);

    // Renumber DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    // Count DoFs per block.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    pcout << "Number of DoFs in subdomain " << subdomain_id << ": "
          << dof_handler.n_dofs() << " (velocity: " << n_u
          << ", pressure: " << n_p << ")" << std::endl;

    owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);

    block_owned_dofs[0]    = owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = relevant_dofs.get_view(n_u, n_u + n_p);

    bc_handler.initialize(dirichlet_bcs);

    constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints);
    constraints.close();

    // Setup sparsity for system matrix.
    {
      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        {
          for (unsigned int d = 0; d < dim + 1; ++d)
            {
              if (c != dim || d != dim)
                coupling[c][d] = DoFTools::always;
              else
                coupling[c][d] = DoFTools::always;
            }
        }

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 mpi_comm,
                                                 relevant_dofs);

      // Initialize matrices and vectors.
      system_matrix.reinit(block_owned_dofs, dsp, mpi_comm);
      system_rhs.reinit(block_owned_dofs, mpi_comm);
      solution.reinit(block_owned_dofs, block_relevant_dofs, mpi_comm);
      solution_owned.reinit(block_owned_dofs, mpi_comm);

      system_matrix_no_interface.reinit(block_owned_dofs, dsp, mpi_comm);
      system_rhs_no_interface.reinit(block_owned_dofs, mpi_comm);
      residual_no_interface.reinit(block_owned_dofs, mpi_comm);
    }

    // Setup pressure mass matrix.
    {
      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 mpi_comm,
                                                 relevant_dofs);

      // Initialize pressure mass matrix.
      pressure_mass_matrix.reinit(block_owned_dofs, dsp, mpi_comm);

      // Assemble pressure mass matrix.
      assemble_pressure_mass_matrix();
    }
  }

  void
  ExampleStokes::assemble_system()
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

    AssertThrow(interface_handler != nullptr, ExcNotInitialized());

    interface_handler->extract();

    constraints.reinit();
    bc_handler.apply_dirichlet(constraints);

    interface_handler->apply_dirichlet(constraints);

    constraints.close();

    constraints_no_interface.reinit();
    bc_handler.apply_dirichlet(constraints_no_interface);
    constraints_no_interface.close();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            assemble_cell(cell, cell_matrix, cell_rhs);

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

    precondition_A.initialize(
      system_matrix.block(0, 0),
      LinAlg::Wrappers::PreconditionILU::AdditionalData());
    inverse_A =
      std::make_unique<utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                            LinAlg::Wrappers::PreconditionILU>>(
        system_matrix.block(0, 0), precondition_A);
  }

  void
  ExampleStokes::assemble_cell(
    const DoFHandler<dim>::active_cell_iterator &cell,
    FullMatrix<double> &                         cell_matrix,
    Vector<double> &                             cell_rhs)
  {
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula->size();

    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    fe_values.reinit(cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            grad_phi_u[k] = fe_values[velocities].gradient(k, q);
            div_phi_u[k]  = fe_values[velocities].divergence(k, q);
            phi_p[k]      = fe_values[pressure].value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = i; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
              (viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) -
               div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
              fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            cell_matrix(j, i) = cell_matrix(i, j);
      }
  }

  void
  ExampleStokes::assemble_pressure_mass_matrix()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Assemble pressure mass matrix");

    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_JxW_values);

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    AffineConstraints<double> pressure_constraints;
    pressure_constraints.reinit(relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    pressure_constraints.close();

    pressure_mass_matrix = 0.0;

    const FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell_matrix = 0.0;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe->system_to_component_index(i).first == dim)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        if (fe->system_to_component_index(j).first == dim)
                          {
                            cell_matrix(i, j) +=
                              1.0 / viscosity * fe_values.shape_value(i, q) *
                              fe_values.shape_value(j, q) * fe_values.JxW(q);
                          }
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        pressure_constraints.distribute_local_to_global(cell_matrix,
                                                        local_dof_indices,
                                                        pressure_mass_matrix);
      }

    pressure_mass_matrix.compress(VectorOperation::add);

    precondition_Mp.initialize(
      pressure_mass_matrix.block(1, 1),
      LinAlg::Wrappers::PreconditionILU::AdditionalData());
    inverse_Mp =
      std::make_unique<utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                            LinAlg::Wrappers::PreconditionILU>>(
        pressure_mass_matrix.block(1, 1), precondition_Mp);
  }

  void
  ExampleStokes::solve()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Solve system");

    Preconditioner preconditioner(*inverse_A,
                                  *inverse_Mp,
                                  system_matrix.block(0, 1));

    linear_solver.solve(system_matrix,
                        solution_owned,
                        system_rhs,
                        preconditioner);
    constraints.distribute(solution_owned);
    solution = solution_owned;
  }

  void
  ExampleStokes::compute_interface_residual()
  {
    system_matrix_no_interface.vmult(residual_no_interface, solution_owned);
    residual_no_interface *= -1;
    residual_no_interface += system_rhs_no_interface;
  }

  void
  ExampleStokes::output_results(const unsigned int &n_iter) const
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Output results");

    // Compute partition indicator for output.
    std::vector<unsigned int> partition_int(
      triangulation.get().n_active_cells());
    GridTools::get_subdomain_association(triangulation.get(), partition_int);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             {"velocity", "velocity", "velocity", "pressure"},
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(partitioning, "partitioning");
    data_out.build_patches(2);

    lifex::utils::dataout_write_hdf5(data_out,
                                     "solution_subdomain" +
                                       std::to_string(subdomain_id),
                                     n_iter,
                                     n_iter,
                                     n_iter);

    data_out.clear();
  }

  void
  ExampleStokes::step(const unsigned int &n_iter)
  {
    assemble_system();
    solve();
    output_results(n_iter);
  }
} // namespace lifex::examples
