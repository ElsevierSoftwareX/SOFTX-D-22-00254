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

#include "source/numerics/laplace.hpp"
#include "source/numerics/numbers.hpp"
#include "source/numerics/tools.hpp"

#include <deal.II/numerics/matrix_tools.h>

#include <vector>

namespace lifex
{
  Laplace::Laplace(const std::string &subsection)
    : CoreModel(subsection)
    , linear_solver(prm_subsection_path + " / Linear solver",
                    {"GMRES", "CG"},
                    "CG")
    , preconditioner(prm_subsection_path + " / Preconditioner")
  {}

  void
  Laplace::declare_parameters(ParamHandler &params) const
  {
    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);
  }

  void
  Laplace::parse_parameters(ParamHandler &params)
  {
    linear_solver.parse_parameters(params);
    preconditioner.parse_parameters(params);
  }

  void
  Laplace::setup_system(const utils::MeshHandler &triangulation,
                        const unsigned int &      fe_degree)
  {
    // Setup finite element space, sparsity pattern, matrices and vectors.
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path + " / Setup system");

      fe         = triangulation.get_fe_lagrange(fe_degree);
      quadrature = triangulation.get_quadrature_gauss(fe_degree + 1);

      dof_handler.reinit(triangulation.get());
      dof_handler.distribute_dofs(*fe);

      triangulation.get_info().print(prm_subsection_path,
                                     dof_handler.n_dofs(),
                                     true);

      IndexSet owned_dofs = dof_handler.locally_owned_dofs();
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

      DynamicSparsityPattern dsp(relevant_dofs);

      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      AffineConstraints<double>(),
                                      false);

      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 mpi_comm,
                                                 relevant_dofs);

      utils::initialize_matrix(matrix_no_bcs, owned_dofs, dsp);
      utils::initialize_matrix(matrix, owned_dofs, dsp);

      rhs.reinit(owned_dofs, mpi_comm);
      sol_owned.reinit(owned_dofs, mpi_comm);
      sol.reinit(owned_dofs, relevant_dofs, mpi_comm);
    }

    // Assemble the system matrix without BCs.
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path + " / Assemble");

      const unsigned int dofs_per_cell = fe->dofs_per_cell;
      const unsigned int n_q_points    = quadrature->size();

      FEValues<dim> fe_values(*fe,
                              *quadrature,
                              update_gradients | update_JxW_values);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          fe_values.reinit(cell);
          cell_matrix = 0.0;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              for (unsigned int q = 0; q < n_q_points; ++q)
                cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);

          cell->get_dof_indices(dof_indices);
          matrix_no_bcs.add(dof_indices, cell_matrix);
        }

      matrix_no_bcs.compress(VectorOperation::add);
    }
  }

  void
  Laplace::clear_bcs()
  {
    boundary_values.clear();
  }

  void
  Laplace::apply_dirichlet_boundary(const std::set<types::boundary_id> &tags,
                                    const double &                      value)
  {
    std::map<types::boundary_id, const Function<dim, double> *> dirichlet_map;
    Functions::ConstantFunction<dim>                            function(value);

    for (const auto &tag : tags)
      dirichlet_map[tag] = &function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_map,
                                             boundary_values);
  }

  void
  Laplace::apply_dirichlet_volume(const std::set<types::material_id> &tags,
                                  const double &                      value)
  {
    std::vector<types::global_dof_index> dof_indices(fe->dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if ((!cell->is_locally_owned() && !cell->is_ghost()) ||
            !utils::contains(tags, cell->material_id()))
          continue;

        cell->get_dof_indices(dof_indices);
        for (const auto &dof : dof_indices)
          boundary_values[dof] = value;
      }
  }

  void
  Laplace::solve()
  {
    matrix.copy_from(matrix_no_bcs);

    // We reset the RHS to zero to clear out any previous boundary conditions.
    rhs = 0.0;

    MatrixTools::apply_boundary_values(
      boundary_values, matrix, sol_owned, rhs, false);

    preconditioner.initialize(matrix);
    linear_solver.solve(matrix, sol_owned, rhs, preconditioner);
    sol = sol_owned;
  }

  void
  Laplace::attach_output(DataOut<dim> &data_out, const std::string &name) const
  {
    data_out.add_data_vector(dof_handler, sol, name);
  }
} // namespace lifex
