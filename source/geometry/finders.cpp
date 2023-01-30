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

#include "source/geometry/finders.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/fe/mapping_q1.h>

#include <algorithm>

namespace lifex::utils
{
  std::pair<types::global_dof_index, unsigned int>
  find_closest_vertex(const DoFHandler<dim> &dof_handler,
                      const Point<dim> &     point)
  {
    // Get closest vertex dof.
    double distance;

    double                  distance_min = std::numeric_limits<double>::max();
    types::global_dof_index id_min       = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
            {
              distance = cell->vertex(i).distance(point);

              if (distance <= distance_min)
                {
                  distance_min = distance;
                  id_min       = cell->vertex_dof_index(i, 0);
                }
            }
        }

    // Gather all distance_min across all the processors.
    std::vector<double> distance_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, distance_min);

    std::vector<types::global_dof_index> id_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, id_min);

    // Get rank index of absolute minimum.
    size_t rank =
      std::min_element(distance_min_vec.begin(), distance_min_vec.end()) -
      distance_min_vec.begin();

    return {id_min_vec[rank], rank};
  }

  std::pair<types::global_dof_index, unsigned int>
  find_closest_dof(const Mapping<dim> &   mapping,
                   const DoFHandler<dim> &dof_handler,
                   const Point<dim> &     point)
  {
    // Get support point coordinates.
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    // Get closest support point.
    double distance;

    double                  distance_min = std::numeric_limits<double>::max();
    types::global_dof_index id_min       = 0;

    for (const auto &p : support_points)
      {
        // map_dofs_to_support_points computes support points for all locally
        // relevant DoFs. This function should return the rank of the process
        // owning the DoF. Therefore, if the DoF is (locally relevant, but) not
        // locally owned by this rank, we skip it.
        if (!dof_handler.locally_owned_dofs().is_element(p.first))
          continue;

        distance = point.distance(p.second);

        if (distance <= distance_min)
          {
            distance_min = distance;
            id_min       = p.first;
          }
      }

    // Gather all distance_min across all the processors.
    std::vector<double> distance_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, distance_min);

    std::vector<types::global_dof_index> id_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, id_min);

    // Get rank index of absolute minimum.
    size_t rank =
      std::min_element(distance_min_vec.begin(), distance_min_vec.end()) -
      distance_min_vec.begin();

    return std::make_pair(id_min_vec[rank], rank);
  }

  std::vector<std::vector<std::pair<types::global_dof_index, double>>>
  find_closest_owned_dofs(
    const DoFHandler<dim> &dof_handler,
    const std::vector<std::map<types::global_dof_index, Point<dim>>>
      &                            support_points,
    const std::vector<Point<dim>> &points)
  {
    // Number of components.
    const unsigned int n_components = dof_handler.get_fe().n_components();

    // Result vector.
    std::vector<std::vector<std::pair<types::global_dof_index, double>>> result(
      points.size(),
      std::vector<std::pair<types::global_dof_index, double>>(
        n_components,
        std::make_pair(types::global_dof_index(0),
                       std::numeric_limits<double>::max())));

    double tmp_distance;

    for (unsigned int component = 0; component < n_components; ++component)
      {
        ComponentMask mask(n_components, false);
        mask.set(component, true);

        // Retrieve nearest DoF among locally owned ones.
        for (const auto &p : support_points[component])
          {
            if (!dof_handler.locally_owned_dofs().is_element(p.first))
              continue;

            for (unsigned int i = 0; i < points.size(); ++i)
              {
                tmp_distance = points[i].distance_square(p.second);

                if (tmp_distance < result[i][component].second)
                  {
                    result[i][component].first  = p.first;
                    result[i][component].second = tmp_distance;
                  }
              }
          }
      }

    return result;
  }

  std::tuple<DoFHandler<dim>::active_cell_iterator, unsigned int, unsigned int>
  find_boundary_face(const DoFHandler<dim> &   dof_handler,
                     const Point<dim> &        center,
                     const types::boundary_id &boundary_tag)
  {
    // Get closest center.
    double distance;
    double distance_min = std::numeric_limits<double>::max();
    DoFHandler<dim>::active_cell_iterator cell_min;
    unsigned int                          face_min;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned() || !cell->at_boundary())
          continue;

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
          {
            if (!cell->face(face_number)->at_boundary() ||
                (boundary_tag != numbers::invalid_boundary_id &&
                 cell->face(face_number)->boundary_id() != boundary_tag))
              continue;

            distance = center.distance(cell->face(face_number)->center());

            if (distance <= distance_min)
              {
                distance_min = distance;
                cell_min     = cell;
                face_min     = face_number;
              }
          }
      }

    // Gather all distance_min across all the processors.
    std::vector<double> distance_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, distance_min);

    // Get rank index of absolute minimum.
    size_t rank =
      std::min_element(distance_min_vec.begin(), distance_min_vec.end()) -
      distance_min_vec.begin();

    return {cell_min, face_min, rank};
  }

  std::pair<types::global_dof_index, unsigned int>
  find_boundary_dof(
    const DoFHandler<dim> &                              dof_handler,
    const std::map<types::global_dof_index, Point<dim>> &support_points,
    const Point<dim> &                                   point,
    const unsigned int &                                 component,
    const std::set<types::boundary_id> &                 boundary_tags)
  {
    const FiniteElement<dim> &fe            = dof_handler.get_fe();
    const unsigned int        dofs_per_face = fe.dofs_per_face;

    // Get closest support point.
    double                  distance;
    double                  distance_min = std::numeric_limits<double>::max();
    types::global_dof_index id_min       = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_face);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->at_boundary())
          continue;

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
          {
            if (!cell->face(face_number)->at_boundary() ||
                (!boundary_tags.empty() &&
                 !utils::contains(boundary_tags,
                                  cell->face(face_number)->boundary_id())))
              continue;

            cell->face(face_number)->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < dofs_per_face; ++i)
              {
                // Skip the DoF if it is not owned or it refers to a solution
                // component different from the required one.
                if (!dof_handler.locally_owned_dofs().is_element(
                      local_dof_indices[i]) ||
                    fe.face_system_to_component_index(i).first != component)
                  continue;

                distance =
                  point.distance(support_points.at(local_dof_indices[i]));

                if (distance <= distance_min)
                  {
                    distance_min = distance;
                    id_min       = local_dof_indices[i];
                  }
              }
          }
      }

    // Gather all distance_min across all the processors.
    std::vector<double> distance_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, distance_min);

    std::vector<types::global_dof_index> id_min_vec =
      Utilities::MPI::all_gather(Core::mpi_comm, id_min);

    // Get rank index of absolute minimum.
    size_t rank =
      std::min_element(distance_min_vec.begin(), distance_min_vec.end()) -
      distance_min_vec.begin();

    return std::make_pair(id_min_vec[rank], rank);
  }
} // namespace lifex::utils
