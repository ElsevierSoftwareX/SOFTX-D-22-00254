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

#include "source/geometry/mesh_optimization.hpp"

#include <deal.II/fe/mapping_fe_field.h>

#include <vector>

namespace lifex::utils
{
  MeshOptimization::MeshOptimization(const std::string &subsection)
    : CoreModel(subsection)
    , nlcg(subsection + " / Non-linear conjugate gradient")
  {}

  void
  MeshOptimization::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    {
      params.declare_entry(
        "Minimum volume",
        "0.0",
        Patterns::Double(0),
        "Minimum element volume, as a fraction of average volume.");

      params.declare_entry(
        "Gradient penalization factor",
        "0.0",
        Patterns::Double(0),
        "Weight of the penalization to displacement gradients.");
    }
    params.leave_subsection_path();

    // Dependencies.
    {
      nlcg.declare_parameters(params);
    }
  }

  void
  MeshOptimization::parse_parameters(ParamHandler &params)
  {
    params.parse();

    params.enter_subsection_path(prm_subsection_path);
    {
      prm_min_volume      = params.get_double("Minimum volume");
      prm_weight_gradient = params.get_double("Gradient penalization factor");
    }
    params.leave_subsection_path();

    // Dependencies.
    {
      nlcg.parse_parameters(params);
    }
  }

  void
  MeshOptimization::setup_system(
    const std::shared_ptr<DoFHandler<dim>> &dof_handler_)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path + " / Initialize");

    Assert(dof_handler_->get_fe().n_components() == dim,
           ExcMessage("The DoF handler you provide must represent exactly " +
                      std::to_string(dim) + " components."));

    dof_handler = dof_handler_;

    const auto &triangulation = dof_handler->get_triangulation();

    quadrature =
      MeshHandler::get_quadrature_gauss(triangulation,
                                        dof_handler->get_fe().degree + 1);

    boundary_dofs = DoFTools::extract_boundary_dofs(*dof_handler);

    MeshInfo mesh_info(triangulation);
    mesh_info.initialize();
    average_element_volume =
      mesh_info.compute_mesh_volume() / triangulation.n_global_active_cells();
    minimum_element_volume = prm_min_volume * average_element_volume;

    pcout << "Average element volume = " << average_element_volume << std::endl;
    pcout << "Minimum element volume = " << minimum_element_volume << std::endl;

    // Initialize the position vectors.
    {
      IndexSet owned_dofs = dof_handler->locally_owned_dofs();
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(*dof_handler, relevant_dofs);

      x_owned.reinit(owned_dofs, mpi_comm);
      x.reinit(owned_dofs, relevant_dofs, mpi_comm);
      x0_owned.reinit(owned_dofs, mpi_comm);
      x0.reinit(owned_dofs, relevant_dofs, mpi_comm);
    }

    nlcg.initialize(&x_owned, &x);
  }

  void
  MeshOptimization::run(const LinAlg::MPI::Vector &x0_owned_)
  {
    // Initial position is current position vector.
    x0_owned = x0_owned_;
    x0       = x0_owned;
    x_owned  = x0_owned;
    x        = x0_owned;

    // Run the optimization algorithm.
    nlcg.solve([this](const LinAlg::MPI::Vector &x) { return compute_loss(x); },
               [this](const LinAlg::MPI::Vector &x,
                      LinAlg::MPI::Vector &      loss_gradient) {
                 compute_loss_gradient(x, loss_gradient);
               });

    // Compute and return incremental displacement.
    d_incr_owned = x_owned;
    d_incr_owned -= x0_owned;
  }

  void
  MeshOptimization::run()
  {
    VectorTools::get_position_vector(*dof_handler, x0_owned);
    run(x0_owned);
  }

  double
  MeshOptimization::compute_loss(const LinAlg::MPI::Vector &x)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Loss function evaluation");

    // Construct the mapping associated to the position vector.
    MappingFEField<dim, dim, LinAlg::MPI::Vector> mapping(*dof_handler, x);

    const FiniteElement<dim> &fe            = dof_handler->get_fe();
    const unsigned int        dofs_per_cell = fe.dofs_per_cell;
    const unsigned int        n_q_points    = quadrature->size();

    FEValues<dim> fe_values(mapping,
                            fe,
                            *quadrature,
                            update_gradients | update_jacobians |
                              update_JxW_values);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    double loss_untangling = 0.0;
    double loss_gradient   = 0.0;

    FEValuesExtractors::Vector displacement(0);

    for (const auto &cell : dof_handler->active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);

        // Untangling term.
        {
          double V_loc = 0.0;

          for (unsigned int q = 0; q < n_q_points; ++q)
            V_loc += fe_values.JxW(q);

          V_loc -= minimum_element_volume;

          if (V_loc < 0)
            {
              loss_untangling += V_loc * V_loc;
            }
        }

        // Position gradient term.
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<2, dim> &jacobian_loc = fe_values.jacobian(q);
              Tensor<2, dim>        displacement_gradient_loc;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                displacement_gradient_loc +=
                  (x[dof_indices[i]] - x0[dof_indices[i]]) *
                  fe_values[displacement].gradient(i, q);

              for (unsigned int d1 = 0; d1 < dim; ++d1)
                for (unsigned int d2 = 0; d2 < dim; ++d2)
                  {
                    double tmp = 0.0;
                    for (unsigned int d3 = 0; d3 < dim; ++d3)
                      tmp += displacement_gradient_loc[d1][d3] *
                             jacobian_loc[d3][d2];

                    loss_gradient += tmp * tmp * fe_values.JxW(q);
                  }
            }
        }
      }

    loss_untangling = Utilities::MPI::sum(loss_untangling, mpi_comm) /
                      (average_element_volume * average_element_volume);
    loss_gradient = Utilities::MPI::sum(loss_gradient, mpi_comm);

    return loss_untangling + prm_weight_gradient * loss_gradient;
  }

  void
  MeshOptimization::compute_loss_gradient(const LinAlg::MPI::Vector &x,
                                          LinAlg::MPI::Vector &loss_gradient)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Loss gradient evaluation");

    // Construct the mapping associated to the position vector.
    MappingFEField<dim, dim, LinAlg::MPI::Vector> mapping(*dof_handler, x);

    const FiniteElement<dim> &fe            = dof_handler->get_fe();
    const unsigned int        n_q_points    = quadrature->size();
    const unsigned int        dofs_per_cell = fe.dofs_per_cell;

    FEValues<dim> fe_values(mapping,
                            fe,
                            *quadrature,
                            update_gradients | update_jacobians |
                              update_inverse_jacobians | update_JxW_values);

    Vector<double>                       loss_gradient_loc(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    loss_gradient = 0.0;

    FEValuesExtractors::Vector displacement(0);
    double                     V_loc = 0.0;

    std::vector<unsigned int> components(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      components[i] = fe.system_to_component_index(i).first;

    for (const auto &cell : dof_handler->active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);

        // Compute cell volume.
        V_loc = 0.0;
        for (unsigned int q = 0; q < n_q_points; ++q)
          V_loc += fe_values.JxW(q);
        V_loc -= minimum_element_volume;

        loss_gradient_loc = 0.0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<2, dim> &jacobian_loc = fe_values.jacobian(q);

            // Untangling term.
            if (V_loc < 0.0)
              {
                const Tensor<2, dim> &jacobian_inv_loc =
                  fe_values.inverse_jacobian(q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (boundary_dofs.is_element(dof_indices[i]))
                      continue;

                    const Tensor<1, dim> &shape_grad =
                      fe_values.shape_grad(i, q);

                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                      for (unsigned int d2 = 0; d2 < dim; ++d2)
                        loss_gradient_loc[i] +=
                          2.0 * V_loc * jacobian_inv_loc[d1][components[i]] *
                          shape_grad[d2] * jacobian_loc[d2][d1] *
                          fe_values.JxW(q) /
                          (average_element_volume * average_element_volume);
                  }
              }

            // Position gradient term.
            if (prm_weight_gradient > 0)
              {
                Tensor<2, dim> displacement_gradient;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  displacement_gradient +=
                    (x[dof_indices[i]] - x0[dof_indices[i]]) *
                    fe_values[displacement].gradient(i, q);

                displacement_gradient = displacement_gradient * jacobian_loc;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (boundary_dofs.is_element(dof_indices[i]))
                      continue;

                    loss_gradient_loc[i] +=
                      prm_weight_gradient * 2.0 *
                      scalar_product(displacement_gradient,
                                     fe_values[displacement].gradient(i, q) *
                                       jacobian_loc) *
                      fe_values.JxW(q);
                  }
              }
          }

        loss_gradient.add(dof_indices, loss_gradient_loc);
      }

    loss_gradient.compress(VectorOperation::add);
  }
} // namespace lifex::utils
