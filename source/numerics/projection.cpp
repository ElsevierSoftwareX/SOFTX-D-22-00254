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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/core.hpp"
#include "source/quadrature_evaluation.hpp"

#include "source/numerics/projection.hpp"

#include <vector>

namespace lifex::utils
{
  ProjectionL2::ProjectionL2(const DoFHandler<dim> &dof_handler_,
                             const Quadrature<dim> &quadrature_formula_,
                             const bool &           lumping,
                             const double &         regularization_absolute,
                             const double &         regularization_relative)
    : dof_handler(dof_handler_)
    , fe(dof_handler.get_fe())
    , quadrature_formula(quadrature_formula_)
  {
    // Setup matrix and rhs.
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
                                               Core::mpi_comm,
                                               relevant_dofs);

    mat.reinit(owned_dofs, owned_dofs, dsp, Core::mpi_comm);
    rhs.reinit(owned_dofs, Core::mpi_comm);

    // Assemble matrix.
    UpdateFlags update_flags = (update_values | update_JxW_values);

    if (regularization_absolute != 0 || regularization_relative != 0)
      update_flags |= update_gradients;

    FEValues<dim> fe_values(fe, quadrature_formula, update_flags);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    double regularization_coeff;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);

            cell_matrix = 0;

            regularization_coeff =
              regularization_absolute * regularization_absolute;

            regularization_coeff += regularization_relative *
                                    regularization_relative * cell->diameter() *
                                    cell->diameter();

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        if (lumping)
                          {
                            // Only lump entries related to the same component.
                            if (fe.system_to_component_index(i).first ==
                                fe.system_to_component_index(j).first)
                              {
                                cell_matrix(i, i) +=
                                  fe_values.shape_value(i, q) *
                                  fe_values.shape_value(j, q) *
                                  fe_values.JxW(q);
                              }
                          }
                        else
                          {
                            cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                                 fe_values.shape_value(j, q) *
                                                 fe_values.JxW(q);
                          }

                        if (regularization_absolute != 0 ||
                            regularization_relative != 0)
                          {
                            cell_matrix(i, j) += regularization_coeff *
                                                 fe_values.shape_grad(i, q) *
                                                 fe_values.shape_grad(j, q) *
                                                 fe_values.JxW(q);
                          }
                      }
                  }
              }

            mat.add(dof_indices, cell_matrix);
          }
      }

    mat.compress(VectorOperation::add);

    preconditioner.initialize(mat);
  }

  template <class FunctionType>
  void
  ProjectionL2::project(const FunctionType & func,
                        LinAlg::MPI::Vector &sol_projected_owned)
  {
    static_assert(is_any_v<FunctionType,
                           QuadratureEvaluationScalar,
                           QuadratureEvaluationVector,
                           std::vector<std::vector<double>>,
                           std::vector<std::vector<Tensor<1, dim, double>>>,
                           Function<dim>>,
                  "ProjectionL2::project(): template parameter not allowed.");

    Assert(!sol_projected_owned.has_ghost_elements(), ExcGhostsPresent());

    Assert(sol_projected_owned.size() == rhs.size(),
           ExcDimensionMismatch(sol_projected_owned.size(), rhs.size()));

    Assert(sol_projected_owned.local_size() == rhs.local_size(),
           ExcDimensionMismatch(sol_projected_owned.local_size(),
                                rhs.local_size()));

    rhs = 0;

    // Assemble rhs.
    UpdateFlags update_flags = (update_values | update_JxW_values);
    if constexpr (is_any_v<FunctionType, Function<dim>>)
      {
        update_flags |= update_quadrature_points;
      }

    FEValues<dim> fe_values(fe, quadrature_formula, update_flags);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    if constexpr (is_any_v<FunctionType,
                           QuadratureEvaluationScalar,
                           QuadratureEvaluationVector>)
      {
        const_cast<FunctionType &>(func).init();
      }

    unsigned int c = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);

            if constexpr (is_any_v<FunctionType,
                                   QuadratureEvaluationScalar,
                                   QuadratureEvaluationVector>)
              {
                const_cast<FunctionType &>(func).reinit(cell);
              }

            cell_rhs = 0;

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double value_q;

                if constexpr (std::is_same_v<FunctionType,
                                             QuadratureEvaluationScalar>)
                  {
                    value_q = const_cast<FunctionType &>(func)(q);
                  }
                else if constexpr (std::is_same_v<FunctionType, Function<dim>>)
                  {
                    value_q = func.value(fe_values.quadrature_point(q));
                  }
                else if constexpr (std::is_same_v<
                                     FunctionType,
                                     std::vector<std::vector<double>>>)
                  {
                    value_q = func[c][q];
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if constexpr (std::is_same_v<FunctionType,
                                                 QuadratureEvaluationVector>)
                      {
                        value_q = const_cast<FunctionType &>(func)(
                          q)[fe.system_to_component_index(i).first];
                      }
                    else if constexpr (std::is_same_v<
                                         FunctionType,
                                         std::vector<std::vector<
                                           Tensor<1, dim, double>>>>)
                      {
                        value_q =
                          func[c][q][fe.system_to_component_index(i).first];
                      }

                    cell_rhs(i) +=
                      value_q * fe_values.shape_value(i, q) * fe_values.JxW(q);
                  }
              }

            rhs.add(dof_indices, cell_rhs);
            ++c;
          }
      }

    rhs.compress(VectorOperation::add);

    // Solve.
    // Linear solver parameters are taken from
    // the deal.II function VectorTools::project().
    ReductionControl control(5 * rhs.size(), 0, 1e-12, false, false);
    SolverCG<LinAlg::MPI::Vector> cg(control);
    cg.solve(mat, sol_projected_owned, rhs, preconditioner);
  }

  /// Explicit instantiation.
  template void
  ProjectionL2::project<>(const QuadratureEvaluationScalar &func,
                          LinAlg::MPI::Vector &sol_projected_owned);

  /// Explicit instantiation.
  template void
  ProjectionL2::project<>(const QuadratureEvaluationVector &func,
                          LinAlg::MPI::Vector &sol_projected_owned);

  /// Explicit instantiation.
  template void
  ProjectionL2::project<>(const std::vector<std::vector<double>> &func,
                          LinAlg::MPI::Vector &sol_projected_owned);

  /// Explicit instantiation.
  template void
  ProjectionL2::project<>(
    const std::vector<std::vector<Tensor<1, dim, double>>> &func,
    LinAlg::MPI::Vector &sol_projected_owned);

  /// Explicit instantiation.
  template void
  ProjectionL2::project<>(const Function<dim> &func,
                          LinAlg::MPI::Vector &sol_projected_owned);

} // namespace lifex::utils
