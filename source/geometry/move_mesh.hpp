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
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#ifndef LIFEX_UTILS_MOVE_MESH_HPP_
#define LIFEX_UTILS_MOVE_MESH_HPP_

#include "source/geometry/mesh_handler.hpp"

namespace lifex::utils
{
  /**
   * @brief Move triangulation according to given displacement.
   *
   * The mesh is moved from the configuration @f$\hat\Omega@f$ to
   * @f$\Omega = (I + \mathbf d)(\hat\Omega)@f$,
   * that is, every vertex @f$\hat{\mathbf v}@f$ of the input mesh is moved to
   * the displaced position @f$\hat{\mathbf v} + \mathbf d@f$.
   *
   * @param[in, out] triangulation Mesh @f$\hat\Omega@f$ to move to @f$\Omega@f$.
   * @param[in] dof_handler DoFHandler used to make up the displacement.
   * @param[in] incremental_displacement Displacement @f$\mathbf d@f$.
   */
  void
  move_mesh(utils::MeshHandler &       triangulation,
            const DoFHandler<dim> &    dof_handler,
            const LinAlg::MPI::Vector &incremental_displacement);

} // namespace lifex::utils

#endif /* LIFEX_UTILS_MOVE_MESH_HPP_ */
