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

#ifndef LIFEX_UTILS_SERIALIZATION_HPP_
#define LIFEX_UTILS_SERIALIZATION_HPP_

#include "source/lifex.hpp"

#include "source/geometry/mesh_handler.hpp"

#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Serialize finite element vector and save it to file.
   *
   * @param[in] filename      The output filename.
   * @param[in] vector_in     The vector to be serialized, ghosted.
   * @param[in] triangulation The triangulation.
   * @param[in] dof_handler   The DoFHandler associated with the input vector and the triangulation.
   *
   * Here is an example of usage:
   * @code{.cpp}
   * serialize<LinAlg::MPI::Vector>(filename,
   *                                solution,
   *                                triangulation,
   *                                dof_handler);
   * @endcode
   */
  template <class VectorType>
  void
  serialize(const std::string &       filename,
            const VectorType &        vector_in,
            const utils::MeshHandler &triangulation,
            const DoFHandler<dim> &   dof_handler);

  /**
   * @brief Serialize finite element vectors and save them to file.
   *
   * Same as the function above, for a list of finite element vectors.
   *
   * Here is an example of usage:
   * @code{.cpp}
   * serialize<LinAlg::MPI::Vector>(filename,
   *                                {&solution1, &solution2},
   *                                triangulation,
   *                                dof_handler);
   * @endcode
   */
  template <class VectorType>
  void
  serialize(const std::string &                    filename,
            const std::vector<const VectorType *> &vectors_in,
            const utils::MeshHandler &             triangulation,
            const DoFHandler<dim> &                dof_handler);

  /**
   * @brief Deserialize finite element vector from file.
   *
   * The input file can be processed even if the corresponding @ref serialize was called
   * on a different number of parallel processes.
   *
   * @param[in] filename        The input filename.
   * @param[in, out] vector_out The deserialized vector, unghosted (has to be initialized with the proper size).
   * @param[in] triangulation   The triangulation.
   * @param[in] dof_handler     The DoFHandler associated with the output vector and the triangulation.
   * @param[in] dof_renumbering A callback function to be called upon DoFHandler re-construction
   *                            for dof renumbering or alike.
   * @param[in] autopartition   Repartition the triangulation upon loading if a different number of MPI processes
   *                            is encountered. To be enabled only if it cannot
   *                            be otherwise.
   *
   * Here is an example of usage:
   * @code{.cpp}
   * // Initialize vector that will contain the deserialized vector with
   * // the unghosted structure of a vector sol_owned.
   * sol_owned.reinit(owned_dofs, mpi_comm);
   *
   * // Process input file and fill data_owned.
   * deserialize<LinAlg::MPI::Vector>(filename,
   *                                  sol_owned,
   *                                  triangulation,
   *                                  dof_handler);
   *
   * // If <kbd>sol</kbd> is ghosted, the following assignment
   * // does parallel communication.
   * sol = sol_owned;
   * @endcode
   */
  template <class VectorType>
  void
  deserialize(
    const std::string &                           filename,
    VectorType &                                  vector_out,
    utils::MeshHandler &                          triangulation,
    const DoFHandler<dim> &                       dof_handler,
    const std::function<void(DoFHandler<dim> &)> &dof_renumbering = {},
    const bool &                                  autopartition   = false);

  /**
   * @brief Deserialize finite element vectors from file.
   *
   * Same as the function above, for a list of finite element vectors.
   *
   * Here is an example of usage:
   * @code{.cpp}
   * std::vector<LinAlg::MPI::Vector *> sol_out(
   *   {&sol1_owned, &sol2_owned});
   *
   * deserialize<LinAlg::MPI::Vector>(filename,
   *                                  sol_out,
   *                                  triangulation,
   *                                  dof_handler);
   * @endcode
   */
  template <class VectorType>
  void
  deserialize(
    const std::string &                           filename,
    std::vector<VectorType *> &                   vectors_out,
    utils::MeshHandler &                          triangulation,
    const DoFHandler<dim> &                       dof_handler,
    const std::function<void(DoFHandler<dim> &)> &dof_renumbering = {},
    const bool &                                  autopartition   = false);

  /**
   * @brief Deserialize mesh refinement info from file.
   *
   * If finite element vectors are attached to the input file,
   * they are ignored. The functions
   * <kbd>parallel::DistributedTriangulationBase<dim>::register_data_attach</kbd>
   * and
   * <kbd>parallel::DistributedTriangulationBase<dim>::notify_ready_to_unpack</kbd>
   * are manually called passing empty callbacks, so that de-serialization does
   * not hang and future calls to
   * <kbd>parallel::DistributedTriangulationBase<dim>::load()</kbd> are allowed
   * (otherwise the error <kbd>"Previously loaded data has not been released
   * yet!"</kbd> is thrown).
   */
  void
  deserialize_mesh(const std::string & filename,
                   utils::MeshHandler &triangulation,
                   const bool &        autopartition = false);

} // namespace lifex::utils

#endif /* LIFEX_UTILS_SERIALIZATION_HPP_ */
