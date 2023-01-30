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

namespace lifex
{
  QuadratureFEMSolution::QuadratureFEMSolution(
    const LinAlg::MPI::Vector &sol_,
    const DoFHandler<dim> &    dof_handler,
    const Quadrature<dim> &    quadrature_)
    : QuadratureEvaluationFEMScalar(dof_handler, quadrature_, update_values)
    , sol(sol_)
  {
    Assert(sol.has_ghost_elements() || (Core::mpi_size == 1),
           ExcParallelNonGhosted());
  }

  double
  QuadratureFEMSolution::operator()(const unsigned int &q,
                                    const double & /*t*/,
                                    const Point<dim> & /*x_q*/)
  {
    sol_q = 0;

    for (unsigned int i = 0; i < dof_indices.size(); ++i)
      {
        sol_q += sol[dof_indices[i]] * fe_values->shape_value(i, q);
      }

    return sol_q;
  }


  QuadratureFEMGradient::QuadratureFEMGradient(
    const LinAlg::MPI::Vector &sol_,
    const DoFHandler<dim> &    dof_handler,
    const Quadrature<dim> &    quadrature_,
    const unsigned int &       component_)
    : QuadratureEvaluationFEMVector(dof_handler, quadrature_, update_gradients)
    , sol(sol_)
    , component(component_)
  {
    Assert(sol.has_ghost_elements() || (Core::mpi_size == 1),
           ExcParallelNonGhosted());
  }

  Tensor<1, dim, double>
  QuadratureFEMGradient::operator()(const unsigned int &q,
                                    const double & /*t*/,
                                    const Point<dim> & /*x_q*/)
  {
    grad_q = 0;

    for (unsigned int i = 0; i < dof_indices.size(); ++i)
      {
        if (dof_handler->get_fe().system_to_component_index(i).first ==
            component)
          grad_q += sol[dof_indices[i]] * fe_values->shape_grad(i, q);
      }

    return grad_q;
  }

  QuadratureFEMDivergence::QuadratureFEMDivergence(
    const LinAlg::MPI::Vector &sol_,
    const DoFHandler<dim> &    dof_handler,
    const Quadrature<dim> &    quadrature_)
    : QuadratureEvaluationFEMScalar(dof_handler, quadrature_, update_gradients)
    , sol(sol_)
  {
    Assert(sol.has_ghost_elements() || (Core::mpi_size == 1),
           ExcParallelNonGhosted());
  }

  double
  QuadratureFEMDivergence::operator()(const unsigned int &q,
                                      const double & /*t*/,
                                      const Point<dim> & /*x_q*/)
  {
    div_q = 0;

    for (unsigned int i = 0; i < dof_indices.size(); ++i)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            div_q +=
              sol[dof_indices[i]] * fe_values->shape_grad_component(i, q, d)[d];
          }
      }

    return div_q;
  }

} // namespace lifex
