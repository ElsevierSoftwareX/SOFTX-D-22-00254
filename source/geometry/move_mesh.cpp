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

#include "source/geometry/move_mesh.hpp"

#include <deal.II/grid/grid_tools.h>

#include <vector>

namespace lifex::utils
{
  void
  move_mesh(utils::MeshHandler &       triangulation,
            const DoFHandler<dim> &    dof_handler,
            const LinAlg::MPI::Vector &incremental_displacement)
  {
    std::vector<bool> vertex_touched(triangulation.get().n_vertices(), false);

    const std::vector<bool> vertex_owned(
      GridTools::get_locally_owned_vertices(triangulation.get()));

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int v = 0; v < cell->n_vertices(); ++v)
              {
                if (vertex_owned[cell->vertex_index(v)] &&
                    !vertex_touched[cell->vertex_index(v)])
                  {
                    vertex_touched[cell->vertex_index(v)] = true;

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        cell->vertex(v)[d] +=
                          incremental_displacement[cell->vertex_dof_index(v,
                                                                          d)];
                      }
                  }
              }
          }
      }

    triangulation.get().communicate_locally_moved_vertices(vertex_touched);
  }

} // namespace lifex::utils
