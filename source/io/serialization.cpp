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

#include "source/io/serialization.hpp"

#include <deal.II/distributed/solution_transfer.h>

namespace lifex::utils
{
  template <class VectorType>
  void
  serialize(const std::string &       filename,
            const VectorType &        vector_in,
            const utils::MeshHandler &triangulation,
            const DoFHandler<dim> &   dof_handler)
  {
    std::vector<const VectorType *> vectors_in(1, &vector_in);

    serialize<VectorType>(filename, vectors_in, triangulation, dof_handler);
  }

  template <class VectorType>
  void
  serialize(const std::string &                    filename,
            const std::vector<const VectorType *> &vectors_in,
            const utils::MeshHandler &             triangulation,
            const DoFHandler<dim> &                dof_handler)
  {
    parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
      dof_handler);

    solution_transfer.prepare_for_serialization(vectors_in);

    triangulation.get().save(Core::prm_output_directory + filename);
  }

  template <class VectorType>
  void
  deserialize(const std::string &                           filename,
              VectorType &                                  vector_out,
              utils::MeshHandler &                          triangulation,
              const DoFHandler<dim> &                       dof_handler,
              const std::function<void(DoFHandler<dim> &)> &dof_renumbering,
              const bool &                                  autopartition)
  {
    std::vector<VectorType *> vectors_out(1, &vector_out);

    deserialize<VectorType>(filename,
                            vectors_out,
                            triangulation,
                            dof_handler,
                            dof_renumbering,
                            autopartition);
  }

  template <class VectorType>
  void
  deserialize(const std::string &                           filename,
              std::vector<VectorType *> &                   vectors_out,
              utils::MeshHandler &                          triangulation,
              const DoFHandler<dim> &                       dof_handler,
              const std::function<void(DoFHandler<dim> &)> &dof_renumbering,
              const bool &                                  autopartition)
  {
    // After clearing a triangulation, all dof handlers attached to it are
    // cleared. In this function we are sure that the triangulation loaded from
    // file coincides with the input one, so we can safely create a new
    // temporary MeshHandler object to keep the external dof handlers unaltered.
    MeshHandler triangulation_coarse(triangulation);

    // Restore coarse triangulation.
    triangulation_coarse.create_mesh(false);

    AssertThrow(triangulation_coarse.get().n_levels() == 1,
                ExcMessage("Triangulation may only contain coarse cells when "
                           "calling deserialize()."));

    // Create new dof handler associated to triangulation_coarse.
    DoFHandler<dim> dof_handler_new;
    dof_handler_new.reinit(triangulation_coarse.get());

    // Load refinement information and data to deserialize.
    if (triangulation_coarse.is_tet())
      triangulation_coarse.get().clear();

    triangulation_coarse.get().load(filename, autopartition);

    // Re-distribute dofs.
    dof_handler_new.distribute_dofs(dof_handler.get_fe());

    if (dof_renumbering)
      dof_renumbering(dof_handler_new);

    parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
      dof_handler_new);

    solution_transfer.deserialize(vectors_out);
  }

  void
  deserialize_mesh(const std::string & filename,
                   utils::MeshHandler &triangulation,
                   const bool &        autopartition)
  {
    if (triangulation.is_tet())
      triangulation.get().clear();

    // Load refinement information.
    triangulation.get().load(filename, autopartition);

    auto pack_callback = [](const Triangulation<dim>::cell_iterator &,
                            const Triangulation<dim>::CellStatus) {
      return std::vector<char>();
    };

    auto unpack_callback =
      [](const Triangulation<dim>::cell_iterator &,
         const Triangulation<dim>::CellStatus,
         const boost::iterator_range<std::vector<char>::const_iterator> &) {
        return;
      };

    triangulation.get().register_data_attach(
      pack_callback,
      /* returns_variable_size_data = */ false);

    // handle = 1 means that no variable-size data is attached to this file.
    triangulation.get().notify_ready_to_unpack(/* handle = */ 1,
                                               unpack_callback);
  }

  /// Explicit instantiation.
  template void
  serialize<>(const std::string &,
              const LinAlg::MPI::Vector &,
              const utils::MeshHandler &,
              const DoFHandler<dim> &);

  /// Explicit instantiation.
  template void
  serialize<>(const std::string &,
              const LinAlg::MPI::BlockVector &,
              const utils::MeshHandler &,
              const DoFHandler<dim> &);

  /// Explicit instantiation.
  template void
  serialize<>(const std::string &,
              const LinearAlgebra::distributed::Vector<double> &,
              const utils::MeshHandler &,
              const DoFHandler<dim> &);


  /// Explicit instantiation.
  template void
  serialize<>(const std::string &,
              const std::vector<const LinAlg::MPI::Vector *> &,
              const utils::MeshHandler &,
              const DoFHandler<dim> &);

  /// Explicit instantiation.
  template void
  serialize<>(const std::string &,
              const std::vector<const LinAlg::MPI::BlockVector *> &,
              const utils::MeshHandler &,
              const DoFHandler<dim> &);

  /// Explicit instantiation.
  template void
  serialize<>(
    const std::string &,
    const std::vector<const LinearAlgebra::distributed::Vector<double> *> &,
    const utils::MeshHandler &,
    const DoFHandler<dim> &);


  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                LinAlg::MPI::Vector &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);

  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                LinAlg::MPI::BlockVector &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);

  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                LinearAlgebra::distributed::Vector<double> &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);


  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                std::vector<LinAlg::MPI::Vector *> &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);

  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                std::vector<LinAlg::MPI::BlockVector *> &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);

  /// Explicit instantiation.
  template void
  deserialize<>(const std::string &,
                std::vector<LinearAlgebra::distributed::Vector<double> *> &,
                utils::MeshHandler &,
                const DoFHandler<dim> &,
                const std::function<void(DoFHandler<dim> &)> &,
                const bool &);

} // namespace lifex::utils
