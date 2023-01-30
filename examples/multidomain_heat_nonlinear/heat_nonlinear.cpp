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

#include "examples/multidomain_heat_nonlinear/heat_nonlinear.hpp"

#include <deal.II/grid/grid_tools.h>

#include <utility>

namespace lifex::examples
{
  ExampleHeatNonLinear::ExampleHeatNonLinear(
    const unsigned int &                                    subdomain_id_,
    const unsigned int &                                    bdf_order_,
    const double &                                          initial_time_,
    const double &                                          time_step_,
    const std::vector<utils::BC<utils::FunctionDirichlet>> &dirichlet_bcs_,
    const std::string &                                     subsection)
    : CoreModel(subsection)
    , subdomain_id(subdomain_id_)
    , triangulation(prm_subsection_path, mpi_comm)
    , non_linear_solver(prm_subsection_path + " / Non-linear solver")
    , linear_solver(prm_subsection_path + " / Linear solver",
                    {"CG", "GMRES", "BiCGStab"},
                    "CG")
    , preconditioner(prm_subsection_path + " / Preconditioner")
    , bdf_order(bdf_order_)
    , time(initial_time_)
    , time_step(time_step_)
    , dirichlet_bcs(dirichlet_bcs_)
    , bc_handler(dof_handler)
  {}

  void
  ExampleHeatNonLinear::declare_parameters(lifex::ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Diffusion coefficient",
                         "1.0",
                         Patterns::Double(0),
                         "Coefficient of the diffusion term.");

    params.declare_entry("Transport coefficient",
                         "0, 0, 0",
                         Patterns::List(Patterns::Double(), dim, dim),
                         "Coefficient of the transport term.");

    params.declare_entry("Reaction coefficient",
                         "1.0",
                         Patterns::Double(0),
                         "Coefficient of the reaction term.");

    params.leave_subsection_path();

    non_linear_solver.declare_parameters(params);
    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);
  }

  void
  ExampleHeatNonLinear::parse_parameters(lifex::ParamHandler &params)
  {
    params.parse();

    params.enter_subsection_path(prm_subsection_path);

    prm_diffusion_coefficient = params.get_double("Diffusion coefficient");

    for (unsigned int i = 0; i < dim; ++i)
      prm_transport_coefficient[i] =
        params.get_vector<double>("Transport coefficient")[i];

    prm_reaction_coefficient = params.get_double("Reaction coefficient");

    params.leave_subsection_path();

    non_linear_solver.parse_parameters(params);
    linear_solver.parse_parameters(params);
    preconditioner.parse_parameters(params);
  }

  void
  ExampleHeatNonLinear::setup_interface(
    const DoFHandler<dim> &other_dof_handler,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataDirichlet>>
      interface_data_dirichlet,
    const std::vector<std::shared_ptr<
      utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataNeumann>>
                                                          interface_data_neumann,
    const std::vector<std::shared_ptr<utils::InterfaceHandler<
      LinAlg::MPI::Vector>::InterfaceDataRobinNonLinear>> interface_data_robin)
  {
    interface_handler =
      std::make_unique<utils::InterfaceHandler<LinAlg::MPI::Vector>>(
        other_dof_handler,
        interface_data_dirichlet,
        interface_data_neumann,
        std::vector<std::shared_ptr<utils::InterfaceHandler<
          LinAlg::MPI::Vector>::InterfaceDataRobinLinear>>(),
        interface_data_robin);
  }

  void
  ExampleHeatNonLinear::create_mesh(const unsigned int &n_refinements)
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
  ExampleHeatNonLinear::setup_system()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Setup system");

    fe                 = triangulation.get_fe_lagrange(1);
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

    zero_constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(zero_constraints, true);
    zero_constraints.close();

    // Setup sparsity.
    DynamicSparsityPattern dsp(relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, zero_constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs,
                                               mpi_comm,
                                               relevant_dofs);

    // Initialize matrix and vectors.
    utils::initialize_matrix(jacobian, owned_dofs, dsp);
    residual.reinit(owned_dofs, mpi_comm);
    solution.reinit(owned_dofs, relevant_dofs, mpi_comm);
    solution_old.reinit(owned_dofs, relevant_dofs, mpi_comm);
    solution_owned.reinit(owned_dofs, mpi_comm);
    residual_no_interface.reinit(owned_dofs, mpi_comm);

    u_bdf.reinit(owned_dofs, relevant_dofs, mpi_comm);

    solution = solution_old = 0.0;

    // Initialize time advancing.
    bdf_handler.initialize(bdf_order, {solution_owned});

    // Initialize non-linear solver handler.
    non_linear_solver.initialize(&solution_owned, &solution);
  }

  void
  ExampleHeatNonLinear::assemble_system(const bool &assemble_jacobian)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Assemble system");

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    jacobian              = 0.0;
    residual              = 0.0;
    residual_no_interface = 0.0;

    unsigned int c = 0; // Cell index.

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            assemble_cell(cell,
                          c,
                          local_dof_indices,
                          cell_matrix,
                          cell_rhs,
                          assemble_jacobian);

            constraints_no_interface.distribute_local_to_global(
              cell_rhs, local_dof_indices, residual_no_interface);

            // Assemble boundary integrals.
            if (cell->at_boundary())
              interface_handler->apply_current_subdomain(
                cell_matrix, cell_rhs, cell, *face_quadrature_formula);

            if (assemble_jacobian)
              zero_constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices, jacobian, residual);
            else
              zero_constraints.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices,
                                                          residual);
            ++c;
          }
      }

    interface_handler->apply_other_subdomain(residual, zero_constraints);

    jacobian.compress(VectorOperation::add);
    residual.compress(VectorOperation::add);
    residual_no_interface.compress(VectorOperation::add);
  }

  void
  ExampleHeatNonLinear::assemble_cell(
    const DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int &                         c,
    const std::vector<types::global_dof_index> & local_dof_indices,
    FullMatrix<double> &                         cell_matrix,
    Vector<double> &                             cell_rhs,
    const bool &                                 assemble_jacobian)
  {
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula->size();

    if (c == 0)
      {
        // Copy into ghosted vector.
        u_bdf = bdf_handler.get_sol_bdf();
      }

    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    fe_values.reinit(cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double         u_bdf_loc = 0.0;
        double         u_loc     = 0.0;
        Tensor<1, dim> u_grad_loc;
        u_grad_loc = 0.0;
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            double sol_loc = solution[local_dof_indices[k]];
            u_bdf_loc +=
              u_bdf[local_dof_indices[k]] * fe_values.shape_value(k, q);
            u_loc += sol_loc * fe_values.shape_value(k, q);
            u_grad_loc += sol_loc * fe_values.shape_grad(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            if (assemble_jacobian)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    (bdf_handler.get_alpha() / time_step *
                       fe_values.shape_value(j, q) +
                     prm_diffusion_coefficient * fe_values.shape_grad(i, q) *
                       fe_values.shape_grad(j, q) +
                     prm_transport_coefficient * fe_values.shape_grad(j, q) *
                       fe_values.shape_value(i, q) +
                     prm_reaction_coefficient * 2.0 * u_loc *
                       fe_values.shape_value(i, q) *
                       fe_values.shape_value(j, q)) *
                    fe_values.JxW(q);
              }

            cell_rhs(i) +=
              ((bdf_handler.get_alpha() * u_loc - u_bdf_loc) / time_step +
               prm_diffusion_coefficient * u_grad_loc *
                 fe_values.shape_grad(i, q) +
               prm_transport_coefficient * u_grad_loc *
                 fe_values.shape_value(i, q) +
               prm_reaction_coefficient * u_loc * u_loc *
                 fe_values.shape_value(i, q)) *
              fe_values.JxW(q);

            // Add source term to the residual.
            auto f = [](const Point<dim> &p, const double &time) {
              double f_space = (p[0] < -0.25) * 0.5 *
                               (1 - cos(2 * M_PI * (p[0] + 0.5) / 0.25));
              double f_time = (0.01 < time && time < 0.51) * 0.5 *
                              (1 - cos(2 * M_PI * (time - 0.01) / 0.5));
              return 5 * f_space * f_time;
            };

            cell_rhs(i) -= f(fe_values.quadrature_point(q), time) *
                           fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
      }
  }

  void
  ExampleHeatNonLinear::solve()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Solve system");

    auto assemble_fun = [this](const bool &assemble_jacobian) {
      assemble_system(assemble_jacobian);
      return residual.l2_norm();
    };

    auto solve_fun = [this](const bool & /*assemble_preconditioner*/,
                            LinAlg::MPI::Vector &incr) {
      const double norm_sol = solution_owned.l2_norm();

      preconditioner.initialize(jacobian);
      linear_solver.solve(jacobian, incr, residual, preconditioner);

      const double norm_incr = incr.l2_norm();

      const unsigned int n_iterations_linear = linear_solver.get_n_iterations();

      return std::make_tuple(norm_sol, norm_incr, n_iterations_linear);
    };

    // Retrieve interface data.
    interface_handler->extract();

    // Apply Dirichlet BCs to initial guess.
    constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints);
    interface_handler->apply_dirichlet(constraints);
    constraints.close();

    constraints.distribute(solution_owned);
    solution = solution_owned;

    zero_constraints.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(zero_constraints, true);
    interface_handler->apply_dirichlet(zero_constraints, true);
    zero_constraints.close();

    constraints_no_interface.reinit(relevant_dofs);
    bc_handler.apply_dirichlet(constraints_no_interface, true);
    constraints_no_interface.close();

    const bool converged = non_linear_solver.solve(assemble_fun, solve_fun);

    AssertThrow(converged, ExcNonlinearNotConverged());
  }

  void
  ExampleHeatNonLinear::output_results() const
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
                              timestep_number,
                              timestep_number,
                              timestep_number);

    data_out.clear();
  }

  void
  ExampleHeatNonLinear::step()
  {
    solve();
  }

  void
  ExampleHeatNonLinear::time_advance()
  {
    solution_old = solution;

    bdf_handler.time_advance(solution_owned);
    time += time_step;
    ++timestep_number;

    output_results();
  }

  void
  ExampleHeatNonLinear::set_solution(const LinAlg::MPI::Vector &src)
  {
    solution_owned = src;
    solution       = solution_owned;
  }
} // namespace lifex::examples
