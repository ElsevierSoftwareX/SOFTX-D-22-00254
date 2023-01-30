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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#ifndef LIFEX_UTILS_FINDERS_HPP_
#define LIFEX_UTILS_FINDERS_HPP_

#include "source/lifex.hpp"

#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Find global ID of the closest vertex @b dof to
   * a specified point.
   *
   * @param[in] dof_handler The DoF handler.
   * @param[in] point       The target point.
   * @return                Global ID of the closest vertex to @a point
   *                        and the rank owning it.
   *
   * @note This function, of course, only works if the finite element object
   * used by the DoF handler object actually provides support points,
   * @a i.e. no edge elements or the like. Otherwise, an exception is thrown.
   */
  std::pair<types::global_dof_index, unsigned int>
  find_closest_vertex(const DoFHandler<dim> &dof_handler,
                      const Point<dim> &     point);

  /**
   * @brief Find the global ID of the closest dof to a specified point.
   *
   * @param[in] mapping     Mapping from the reference cell to the real one.
   * @param[in] dof_handler The DoF handler.
   * @param[in] point       The target point.
   * @return                Global ID of the closest vertex to @a point.
   *                        and the rank owning it.
   *
   * If multiple degrees of freedom are associated to the same support point (@a
   * i.e. in case of vector problems), the dof with the highest global index is
   * returned (usually the one in the last block).
   *
   * @note This function, of course, only works if the finite element object
   * used by the DoF handler object actually provides support points,
   * @a i.e. no edge elements or the like. Otherwise, an exception is thrown.
   */
  std::pair<types::global_dof_index, unsigned int>
  find_closest_dof(const Mapping<dim> &   mapping,
                   const DoFHandler<dim> &dof_handler,
                   const Point<dim> &     point);

  /**
   * @brief Find the global ID and distance of the closest locally owned DoFs
   * to specified points.
   *
   * If the DoF handler has multiple components, for each point the function
   * returns the indices of the closest DoFs for each component.
   *
   * @param[in] dof_handler The DoF handler.
   * @param[in] support_points A vector that for each component stores the
   * support points of the associated DoFs. Should be computed by means of the
   * function DoFTools::map_dofs_to_support_points.
   * @param[in] points The vector of target points.
   *
   * @return A vector that, for each element of points, contains a vector with
   * one entry for each component. Each entry of the vector is a pair whose
   * first element is the closest DoF index, and second element is the squared
   * distance between from the point.
   */
  std::vector<std::vector<std::pair<types::global_dof_index, double>>>
  find_closest_owned_dofs(
    const DoFHandler<dim> &dof_handler,
    const std::vector<std::map<types::global_dof_index, Point<dim>>>
      &                            support_points,
    const std::vector<Point<dim>> &points);

  /**
   * @brief Find the face on the boundary whose center is closest to a specified
   * point.
   *
   * @param[in] dof_handler  The DoF handler.
   * @param[in] center       The center to be searched for.
   * @param[in] boundary_tag If different from numbers::invalid_boundary_id, the
   *            search is restricted to faces whose boundary ID is equal to
   *            boundary_tag.
   * @return    a tuple of active_cell_iterator to the cell containing the closest
   *            face, face index of the closest face and rank owning the closest
   *            face.
   *
   * @note For ranks different from that owning the closest face, this returns the
   * cell and face closest to center amongst those owned by the current rank.
   */
  std::tuple<DoFHandler<dim>::active_cell_iterator, unsigned int, unsigned int>
  find_boundary_face(
    const DoFHandler<dim> &   dof_handler,
    const Point<dim> &        center,
    const types::boundary_id &boundary_tag = numbers::invalid_boundary_id);

  /**
   * @brief Find the global index and owner rank of the boundary DoF whose support
   * point is closest to a specified point.
   *
   * @param[in] dof_handler    The DoF handler.
   * @param[in] support_points a map from global DoF index on support points
   *                           for all DoFs relevant to this rank; it should be
   *                           computed with
   *                           <code>DoFTools::map_dofs_to_support_points</code>
   * @param[in] point          The target point.
   * @param[in] component      Select only DoFs that correspond to this
   *                           solution component.
   * @param[in] boundary_tags  If not empty, select only DoFs laying on a boundary with a tag in this set.
   */
  std::pair<types::global_dof_index, unsigned int>
  find_boundary_dof(
    const DoFHandler<dim> &                              dof_handler,
    const std::map<types::global_dof_index, Point<dim>> &support_points,
    const Point<dim> &                                   point,
    const unsigned int &                                 component     = 0,
    const std::set<types::boundary_id> &                 boundary_tags = {});

} // namespace lifex::utils

#endif /* LIFEX_UTILS_FINDERS_HPP_ */
