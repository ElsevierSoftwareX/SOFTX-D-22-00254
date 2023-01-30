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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 */

#include "source/geometry/mesh_handler.hpp"
#include "source/geometry/mesh_info.hpp"

#include "source/numerics/bc_handler.hpp"

#include <deal.II/fe/fe_values_extractors.h>

namespace lifex::utils
{
  template <class FunctionType>
  BC<FunctionType>::BC(const types::boundary_id &           boundary_id_,
                       const std::shared_ptr<FunctionType> &function_,
                       const ComponentMask &                component_mask_)
    : boundary_id(boundary_id_)
    , function(function_)
    , component_mask(component_mask_)
  {
    static_assert(is_any_v<FunctionType, FunctionDirichlet, FunctionNeumann>,
                  "BCHandler: template parameter not allowed.");
  }

  BCHandler::BCHandler(const DoFHandler<dim> &dof_handler_)
    : dof_handler(dof_handler_)
  {}

  void
  BCHandler::initialize(
    const std::vector<BC<FunctionDirichlet>> &dirichlet_bcs_,
    const std::vector<BC<FunctionNeumann>> &  neumann_bcs_,
    const Quadrature<dim - 1> &               face_quadrature_formula,
    const std::vector<BC<FunctionDirichlet>> &dirichlet_normal_flux_bcs_,
    const std::vector<BC<FunctionDirichlet>> &dirichlet_tangential_flux_bcs_)
  {
    vec_dirichlet_owned = LinAlg::MPI::Vector();
    vec_dirichlet       = LinAlg::MPI::Vector();

    dirichlet_bcs                 = dirichlet_bcs_;
    neumann_bcs                   = neumann_bcs_;
    dirichlet_normal_flux_bcs     = dirichlet_normal_flux_bcs_;
    dirichlet_tangential_flux_bcs = dirichlet_tangential_flux_bcs_;

    Assert(neumann_bcs.size() == 0 || face_quadrature_formula.size() > 0,
           ExcMessage("You must provide a valid quadrature formula if Neumann "
                      "BCs are imposed."));

    if (neumann_bcs.size() > 0)
      fe_face_values =
        std::make_unique<FEFaceValues<dim>>(dof_handler.get_fe(),
                                            face_quadrature_formula,
                                            update_values |
                                              update_normal_vectors |
                                              update_quadrature_points |
                                              update_JxW_values);

    for (auto &bc : dirichlet_bcs)
      {
        update_bc_geometry_info(bc);
      }

    for (auto &bc : dirichlet_normal_flux_bcs)
      {
        update_bc_geometry_info(bc);
      }

    for (auto &bc : dirichlet_tangential_flux_bcs)
      {
        update_bc_geometry_info(bc);
      }
  }

  void
  BCHandler::set_time(const double &t, const bool &update_geometry_info)
  {
    for (auto &bc : dirichlet_bcs)
      {
        bc.function->set_time(t);

        if (update_geometry_info)
          update_bc_geometry_info(bc);
      }

    for (auto &bc : neumann_bcs)
      {
        bc.function->set_time(t);
      }

    for (auto &bc : dirichlet_normal_flux_bcs)
      {
        bc.function->set_time(t);

        if (update_geometry_info)
          update_bc_geometry_info(bc);
      }

    for (auto &bc : dirichlet_tangential_flux_bcs)
      {
        bc.function->set_time(t);

        if (update_geometry_info)
          update_bc_geometry_info(bc);
      }
  }

  void
  BCHandler::apply_dirichlet(AffineConstraints<double> &constraints,
                             const bool &               homogeneous_constraints)
  {
    for (const auto &bc : dirichlet_bcs)
      {
        if (homogeneous_constraints)
          {
            Functions::ZeroFunction<dim> zero_function(
              bc.function->n_components);

            VectorTools::interpolate_boundary_values(dof_handler,
                                                     {{bc.boundary_id,
                                                       &zero_function}},
                                                     constraints,
                                                     bc.component_mask);
          }
        else
          {
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     {{bc.boundary_id,
                                                       bc.function.get()}},
                                                     constraints,
                                                     bc.component_mask);
          }
      }
  }

  template <class VectorType>
  void
  BCHandler::apply_dirichlet(VectorType &vector_owned,
                             VectorType &vector,
                             const bool &reinit)
  {
    Assert(!vector_owned.has_ghost_elements(), ExcGhostsPresent());

    Assert(vector.has_ghost_elements() || (Core::mpi_size == 1),
           ExcParallelNonGhosted());

    if (dirichlet_bcs.size() == 0)
      return;

    if (reinit)
      {
        vec_dirichlet_owned = VectorType();
        vec_dirichlet       = VectorType();

        std::get<VectorType>(vec_dirichlet_owned).reinit(vector_owned);
        std::get<VectorType>(vec_dirichlet).reinit(vector);

        dirichlet_dofs.resize(dirichlet_bcs.size());
      }
    else
      {
        AssertThrow(vector_owned.size() != 0, ExcNotInitialized());
        AssertThrow(vector.size() != 0, ExcNotInitialized());
      }

    // Reverse loop to ensure that the first bc in the list wins.
    // Underflow occurring when decrementing unsigned iterators
    // is prevented by the post-decrement operator.
    for (size_t k = dirichlet_bcs.size(); k-- > 0;)
      {
        if (reinit)
          {
            DoFTools::extract_boundary_dofs(dof_handler,
                                            dirichlet_bcs[k].component_mask,
                                            dirichlet_dofs[k],
                                            {dirichlet_bcs[k].boundary_id});
          }

        VectorTools::interpolate(dof_handler,
                                 *(dirichlet_bcs[k].function),
                                 std::get<VectorType>(vec_dirichlet_owned));

        std::get<VectorType>(vec_dirichlet) =
          std::get<VectorType>(vec_dirichlet_owned);

        for (auto idx : dirichlet_dofs[k])
          {
            vector_owned[idx] = std::get<VectorType>(vec_dirichlet)[idx];
          }
      }

    vector_owned.compress(VectorOperation::insert);
    vector = vector_owned;
  }

  void
  BCHandler::apply_neumann(Vector<double> &cell_rhs,
                           const DoFHandler<dim>::active_cell_iterator &cell)
  {
    Assert(cell->is_locally_owned(), ExcCellNonLocallyOwned());

    if (!cell->at_boundary() || neumann_bcs.size() == 0)
      return;

    const FiniteElement<dim> &fe = cell->get_dof_handler().get_fe();
    const unsigned int        n_face_q_points =
      fe_face_values->get_quadrature().size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Assert(cell_rhs.size() == dofs_per_cell,
           ExcDimensionMismatch(cell_rhs.size(), dofs_per_cell));

    bool on_neumann_boundary;

    BC<FunctionNeumann>
      *neumann_bc; // Pointer to current Neumann BC to be processed.

    for (unsigned int face = 0; face < cell->n_faces(); ++face)
      {
        if (cell->face(face)->at_boundary())
          {
            // Check if current face is on a Neumann boundary.
            on_neumann_boundary = false;

            for (auto &bc : neumann_bcs)
              {
                // The first Neumann boundary detected wins.
                if (cell->face(face)->boundary_id() == bc.boundary_id)
                  {
                    on_neumann_boundary = true;
                    neumann_bc          = &bc;
                    break;
                  }
              }

            // Ignore non-Neumann boundary conditions.
            if (!on_neumann_boundary)
              {
                continue;
              }

            fe_face_values->reinit(cell, face);

            Vector<double> vector_value(neumann_bc->function->n_components);

            // Assemble boundary integral.
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                neumann_bc->function->set_normal_vector(
                  fe_face_values->normal_vector(q));
                neumann_bc->function->vector_value(
                  fe_face_values->quadrature_point(q), vector_value);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;

                    if (neumann_bc->component_mask[component_i])
                      {
                        cell_rhs(i) -= vector_value[component_i] *
                                       fe_face_values->shape_value(i, q) *
                                       fe_face_values->JxW(q);
                      }
                  }
              }
          }
      }
  }

  void
  BCHandler::apply_dirichlet_normal_flux(AffineConstraints<double> &constraints,
                                         const bool &homogeneous_constraints)
  {
    unsigned int first_selected_component;

    for (const auto &bc : dirichlet_normal_flux_bcs)
      {
        // Check that bc.component mask identifies a vector component,
        // i.e. it selects exactly "dim" consecutive components.
        first_selected_component = bc.component_mask.first_selected_component();

        Assert(bc.component_mask.n_selected_components() == dim,
               ExcDimensionMismatch(bc.component_mask.n_selected_components(),
                                    dim));

        for (unsigned int d = 1; d < dim; ++d)
          {
            Assert(bc.component_mask[first_selected_component + d],
                   ExcMessage("Component " +
                              std::to_string(first_selected_component + d) +
                              " must be selected."));
          }

        auto mapping =
          MeshHandler::get_linear_mapping(dof_handler.get_triangulation());

        if (homogeneous_constraints)
          {
            VectorTools::compute_no_normal_flux_constraints(
              dof_handler,
              first_selected_component,
              {bc.boundary_id},
              constraints,
              *mapping);
          }
        else
          {
            VectorTools::compute_nonzero_normal_flux_constraints(
              dof_handler,
              first_selected_component,
              {bc.boundary_id},
              {{bc.boundary_id, bc.function.get()}},
              constraints,
              *mapping);
          }
      }
  }

  void
  BCHandler::apply_dirichlet_tangential_flux(
    AffineConstraints<double> &constraints,
    const bool &               homogeneous_constraints)
  {
    unsigned int first_selected_component;

    for (const auto &bc : dirichlet_tangential_flux_bcs)
      {
        // Check that bc.component mask identifies a vector component,
        // i.e. it selects exactly "dim" consecutive components.
        first_selected_component = bc.component_mask.first_selected_component();

        Assert(bc.component_mask.n_selected_components() == dim,
               ExcDimensionMismatch(bc.component_mask.n_selected_components(),
                                    dim));

        for (unsigned int d = 1; d < dim; ++d)
          {
            Assert(bc.component_mask[first_selected_component + d],
                   ExcMessage("Component " +
                              std::to_string(first_selected_component + d) +
                              " must be selected."));
          }

        auto mapping =
          MeshHandler::get_linear_mapping(dof_handler.get_triangulation());

        if (homogeneous_constraints)
          {
            VectorTools::compute_normal_flux_constraints(
              dof_handler,
              first_selected_component,
              {bc.boundary_id},
              constraints,
              *mapping);
          }
        else
          {
            VectorTools::compute_nonzero_tangential_flux_constraints(
              dof_handler,
              first_selected_component,
              {bc.boundary_id},
              {{bc.boundary_id, bc.function.get()}},
              constraints,
              *mapping);
          }
      }
  }

  template <class VectorType>
  void
  BCHandler::apply_dirichlet_constraints(
    VectorType &                     vector_owned,
    VectorType &                     vector,
    const AffineConstraints<double> &constraints)
  {
    Assert(!vector_owned.has_ghost_elements(), ExcGhostsPresent());
    Assert(vector.has_ghost_elements() || (mpi_size == 1),
           ExcParallelNonGhosted());

    constraints.distribute(vector_owned);
    vector = vector_owned;
  }

  void
  BCHandler::update_bc_geometry_info(BC<FunctionDirichlet> &bc)
  {
    const utils::MeshInfo mesh_info(dof_handler.get_triangulation());

    bc.function->set_normal_vector(
      mesh_info.compute_flat_boundary_normal(bc.boundary_id));

    bc.function->set_surface_area(
      mesh_info.compute_surface_area(bc.boundary_id));

    bc.function->set_barycenter(
      mesh_info.compute_surface_barycenter({bc.boundary_id}));
  }

  /// Explicit instantiation.
  template class BC<FunctionDirichlet>;

  /// Explicit instantiation.
  template class BC<FunctionNeumann>;

  /// Explicit instantiation.
  template void
  BCHandler::apply_dirichlet<LinAlg::MPI::Vector>(LinAlg::MPI::Vector &,
                                                  LinAlg::MPI::Vector &,
                                                  const bool &);

  /// Explicit instantiation.
  template void
  BCHandler::apply_dirichlet<LinAlg::MPI::BlockVector>(
    LinAlg::MPI::BlockVector &,
    LinAlg::MPI::BlockVector &,
    const bool &);

  /// Explicit instantiation.
  template void
  BCHandler::apply_dirichlet_constraints<>(LinAlg::MPI::Vector &,
                                           LinAlg::MPI::Vector &,
                                           const AffineConstraints<double> &);

  /// Explicit instantiation.
  template void
  BCHandler::apply_dirichlet_constraints<>(LinAlg::MPI::BlockVector &,
                                           LinAlg::MPI::BlockVector &,
                                           const AffineConstraints<double> &);

} // namespace lifex::utils
