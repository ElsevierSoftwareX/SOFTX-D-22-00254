/********************************************************************************
  Copyright (C) 2019 - 2023 by the lifex authors.

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

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/interface_handler.hpp"

#include <deal.II/fe/mapping_q1.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>

namespace lifex::utils
{
  InterfaceDoF::InterfaceDoF(
    const types::global_dof_index &interface_index_,
    const types::global_dof_index &other_subdomain_index_,
    const unsigned int &           component_,
    const unsigned int &           other_component_,
    const unsigned int &           owner_,
    const unsigned int &           other_owner_)
    : interface_index(interface_index_)
    , other_subdomain_index(other_subdomain_index_)
    , component(component_)
    , other_component(other_component_)
    , owner(owner_)
    , other_owner(other_owner_)
  {}

  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  compute_interface_maps(
    const DoFHandler<dim> &                            dof_handler_0,
    const DoFHandler<dim> &                            dof_handler_1,
    const std::array<std::set<types::boundary_id>, 2> &interface_tags,
    const std::map<unsigned int, unsigned int> &       component_map)
  {
    IndexSet owned_interface_dofs;
    IndexSet relevant_interface_dofs;

    const auto &triangulation_0 = dof_handler_0.get_triangulation();
    const auto &triangulation_1 = dof_handler_1.get_triangulation();

    std::array<std::shared_ptr<InterfaceMap>, 2> interface_maps;
    interface_maps[0] = std::make_shared<InterfaceMap>();
    interface_maps[1] = std::make_shared<InterfaceMap>();

    const FiniteElement<dim> &fe_0 = dof_handler_0.get_fe();
    const FiniteElement<dim> &fe_1 = dof_handler_1.get_fe();

    // Retrieve the physical coordinates of DoFs on both subdomains.
    std::array<std::map<types::global_dof_index, Point<dim>>, 2> support_points;
    auto mapping_0 = MeshHandler::get_linear_mapping(triangulation_0);
    auto mapping_1 = MeshHandler::get_linear_mapping(triangulation_1);
    DoFTools::map_dofs_to_support_points(*mapping_0,
                                         dof_handler_0,
                                         support_points[0]);
    DoFTools::map_dofs_to_support_points(*mapping_1,
                                         dof_handler_1,
                                         support_points[1]);

    // Create a copy of the components map. If it is empty, we fill it with
    // pairs of components with the same index.
    std::map<unsigned int, unsigned int> component_map_copy = component_map;
    if (component_map_copy.empty())
      {
        for (unsigned int c = 0;
             c < std::min(fe_0.n_components(), fe_1.n_components());
             ++c)
          component_map_copy[c] = c;
      }

    // Count the total interface DoFs to properly resize index sets. We count
    // them on both domains, even though in practice we only need to on one:
    // this way, we can check that the two interfaces are conforming, at least
    // from the point of view of the total number of interface DoFs.
    unsigned int n_interface_dofs = 0;
    {
      ComponentMask mask_0(fe_0.n_components(), false);
      ComponentMask mask_1(fe_1.n_components(), false);

      for (const auto &component_pair : component_map_copy)
        {
          mask_0.set(component_pair.first, true);
          mask_1.set(component_pair.second, true);
        }

      IndexSet owned_interface_dofs_0;
      DoFTools::extract_boundary_dofs(dof_handler_0,
                                      mask_0,
                                      owned_interface_dofs_0,
                                      interface_tags[0]);
      owned_interface_dofs_0 =
        owned_interface_dofs_0 & dof_handler_0.locally_owned_dofs();

      IndexSet owned_interface_dofs_1;
      DoFTools::extract_boundary_dofs(dof_handler_1,
                                      mask_1,
                                      owned_interface_dofs_1,
                                      interface_tags[1]);
      owned_interface_dofs_1 =
        owned_interface_dofs_1 & dof_handler_1.locally_owned_dofs();

      n_interface_dofs = owned_interface_dofs_0.n_elements();
      n_interface_dofs = Utilities::MPI::sum(n_interface_dofs, Core::mpi_comm);

      unsigned int n_interface_dofs_1 = owned_interface_dofs_1.n_elements();
      n_interface_dofs_1 =
        Utilities::MPI::sum(n_interface_dofs_1, Core::mpi_comm);

      AssertThrow(n_interface_dofs == n_interface_dofs_1,
                  ExcMessage("Subdomain 0 has " +
                             std::to_string(n_interface_dofs) +
                             " interface DoFs, but subdomain 1 has " +
                             std::to_string(n_interface_dofs_1) +
                             ": the two meshes are not conforming. You may "
                             "want to check the interface conformity as well "
                             "as the interface tags."));

      owned_interface_dofs.set_size(n_interface_dofs);
      relevant_interface_dofs.set_size(n_interface_dofs);
    }

    // First available interface DoF.
    unsigned int cur_interface_dof = 0;

    for (const auto &component_pair : component_map_copy)
      {
        // Retrieve the DoFs corresponding to the current component pair at the
        // interface.
        ComponentMask mask_0(fe_0.n_components(), false);
        ComponentMask mask_1(fe_1.n_components(), false);
        IndexSet      relevant_interface_dofs_0;
        IndexSet      relevant_interface_dofs_1;
        IndexSet      mapped_dofs_0(dof_handler_0.n_dofs());
        IndexSet      mapped_dofs_1(dof_handler_1.n_dofs());

        mask_0.set(component_pair.first, true);
        mask_1.set(component_pair.second, true);

        DoFTools::extract_boundary_dofs(dof_handler_0,
                                        mask_0,
                                        relevant_interface_dofs_0,
                                        interface_tags[0]);
        DoFTools::extract_boundary_dofs(dof_handler_1,
                                        mask_1,
                                        relevant_interface_dofs_1,
                                        interface_tags[1]);

        IndexSet owned_interface_dofs_0 =
          relevant_interface_dofs_0 & dof_handler_0.locally_owned_dofs();

        // Collect DoFs and their support points in vectors for communication.
        std::vector<types::global_dof_index> interface_indices;
        std::vector<Point<dim>>              interface_support_points;

        for (const auto dof : owned_interface_dofs_0)
          {
            interface_indices.push_back(dof);
            interface_support_points.push_back(support_points[0][dof]);
          }

        // Communicate DoFs and their support points.
        std::vector<std::vector<types::global_dof_index>>
          all_interface_indices =
            Utilities::MPI::all_gather(Core::mpi_comm, interface_indices);
        std::vector<std::vector<Point<dim>>> all_interface_support_points =
          Utilities::MPI::all_gather(Core::mpi_comm, interface_support_points);

        // Locate the points on the other subdomain.
        for (unsigned int rank = 0; rank < Core::mpi_size; ++rank)
          for (unsigned int i = 0; i < all_interface_indices[rank].size(); ++i)
            {
              types::global_dof_index other_dof;
              unsigned int            other_owner;

              std::tie(other_dof, other_owner) =
                utils::find_boundary_dof(dof_handler_1,
                                         support_points[1],
                                         all_interface_support_points[rank][i],
                                         component_pair.second,
                                         interface_tags[1]);

              const unsigned int cur_dof_index = all_interface_indices[rank][i];

              if (relevant_interface_dofs_0.is_element(cur_dof_index))
                {
                  Assert(interface_maps[0]->find(cur_dof_index) ==
                           interface_maps[0]->end(),
                         ExcMessage("The DoF " + std::to_string(cur_dof_index) +
                                    " is already in the interface map."));

                  interface_maps[0]->emplace(cur_dof_index,
                                             InterfaceDoF(cur_interface_dof,
                                                          other_dof,
                                                          component_pair.first,
                                                          component_pair.second,
                                                          rank,
                                                          other_owner));
                  relevant_interface_dofs.add_index(cur_interface_dof);
                  if (rank == Core::mpi_rank)
                    owned_interface_dofs.add_index(cur_interface_dof);
                  mapped_dofs_0.add_index(cur_dof_index);
                }

              if (relevant_interface_dofs_1.is_element(other_dof))
                {
                  Assert(interface_maps[1]->find(other_dof) ==
                           interface_maps[1]->end(),
                         ExcMessage("The DoF " + std::to_string(other_dof) +
                                    " is already in the interface map."));

                  interface_maps[1]->emplace(other_dof,
                                             InterfaceDoF(cur_interface_dof,
                                                          cur_dof_index,
                                                          component_pair.second,
                                                          component_pair.first,
                                                          other_owner,
                                                          rank));
                  relevant_interface_dofs.add_index(cur_interface_dof);
                  mapped_dofs_1.add_index(other_dof);
                }

              ++cur_interface_dof;
            }

        // Sanity check: mapped DoFs coincide with relevant interface DoFs on
        // both subdomains.
        AssertThrow(mapped_dofs_0 == relevant_interface_dofs_0,
                    ExcMessage("Discrepancy between mapped DoFs and relevant "
                               "interface DoFs on subdomain 0"));
        AssertThrow(mapped_dofs_1 == relevant_interface_dofs_1,
                    ExcMessage("Discrepancy between mapped DoFs and relevant "
                               "interface DoFs on subdomain 1"));
      }

    return {interface_maps, owned_interface_dofs, relevant_interface_dofs};
  }

  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  compute_interface_maps(
    const DoFHandler<dim> &                     dof_handler_0,
    const DoFHandler<dim> &                     dof_handler_1,
    const std::array<types::boundary_id, 2> &   interface_tags,
    const std::map<unsigned int, unsigned int> &component_map)
  {
    std::array<std::set<types::boundary_id>, 2> interface_tags_set;
    interface_tags_set[0].insert(interface_tags[0]);
    interface_tags_set[1].insert(interface_tags[1]);

    return compute_interface_maps(dof_handler_0,
                                  dof_handler_1,
                                  interface_tags_set,
                                  component_map);
  }

  void
  serialize_interface_maps(
    const std::string &                                 filename,
    const std::array<std::shared_ptr<InterfaceMap>, 2> &interface_maps)
  {
    std::ofstream output_file(Core::prm_output_directory + filename);

    AssertThrow(output_file.is_open(), ExcFileNotOpen(filename));

    boost::archive::binary_oarchive oa(output_file);

    // Write number of MPI processes in the archive, to check that serialization
    // and deserialization occur with the same number of cores, to output a
    // meaningful error message if this doesn't happen.
    oa &Utilities::MPI::n_mpi_processes(Core::mpi_comm);

    for (const auto &map : interface_maps)
      {
        // Collect the interface maps from all processes.
        std::vector<InterfaceMap> all_maps =
          Utilities::MPI::gather(Core::mpi_comm, *map, 0);

        if (Core::mpi_rank == 0)
          {
            // Here we construct an interface map containing data from all
            // ranks.
            InterfaceMap serial_interface_map;

            for (const auto &rank_map : all_maps)
              for (const auto &entry : rank_map)
                serial_interface_map.insert(entry);

            // Then we serialize the interface map to the requested file.
            oa << serial_interface_map;
          }
      }
  }

  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  deserialize_interface_maps(const std::string &            filename,
                             const std::array<IndexSet, 2> &owned_dofs,
                             const std::array<IndexSet, 2> &relevant_dofs)
  {
    AssertThrow(boost::filesystem::exists(filename), ExcFileNotOpen(filename));

    unsigned int                serialization_mpi_size = 0;
    std::array<InterfaceMap, 2> serial_interface_maps;

    std::array<std::shared_ptr<InterfaceMap>, 2> interface_maps;
    IndexSet                                     owned_interface_dofs;
    IndexSet                                     relevant_interface_dofs;

    // First, we read data from the serialized file. Each process should read
    // the file. To avoid possible memory bottlenecks and deadlocks due to all
    // processes reading simultaneously, we have processes read the file one at
    // a time.
    for (unsigned int rank = 0; rank < Core::mpi_size; ++rank)
      {
        if (rank == Core::mpi_rank)
          {
            std::ifstream input_file(filename);

            AssertThrow(input_file.is_open(), ExcFileNotOpen(filename));

            boost::archive::binary_iarchive ia(input_file);

            ia >> serialization_mpi_size;
            ia >> serial_interface_maps[0];
            ia >> serial_interface_maps[1];
          }

        MPI_Barrier(Core::mpi_comm);
      }

    AssertThrow(
      serialization_mpi_size == Core::mpi_size,
      ExcMessage(
        "Deserialization of interface maps must be done with the same number "
        "of processes as serialization (serialization processes: " +
        std::to_string(serialization_mpi_size) +
        ", deserialization processes: " + std::to_string(Core::mpi_size) +
        "."));

    AssertThrow(serial_interface_maps[0].size() ==
                  serial_interface_maps[1].size(),
                ExcMessage("Deserialized interface maps have different size."));

    owned_interface_dofs.clear();
    owned_interface_dofs.set_size(serial_interface_maps[0].size());

    relevant_interface_dofs.clear();
    relevant_interface_dofs.set_size(serial_interface_maps[0].size());

    for (unsigned int i = 0; i < 2; ++i)
      {
        interface_maps[i] = std::make_shared<InterfaceMap>();

        // We loop through the serial interface map, that contains all interface
        // DoFs (regardless of their owning processes), and copy the entries
        // that are relevant to the current process into the local interface map
        // that we're going to return.
        for (const auto &entry : serial_interface_maps[i])
          {
            // An interface DoF is owned if it is owned in the first subdomain.
            const bool is_owned =
              (i == 0 && owned_dofs[0].is_element(entry.first)) ||
              (i == 1 && owned_dofs[0].is_element(
                           entry.second.get_other_subdomain_index()));

            // An interface DoF is relevant if it is relevant in the subdomain
            // the map starts from or relevant in the subdomain the map points
            // to.
            const bool is_relevant =
              relevant_dofs[i].is_element(entry.first) ||
              relevant_dofs[1 - i].is_element(
                entry.second.get_other_subdomain_index());

            // If the DoF is owned or relevant, we add it to relevant interface
            // DoF indices, and insert it into the local map.
            if (is_relevant)
              {
                relevant_interface_dofs.add_index(
                  entry.second.get_interface_index());
                interface_maps[i]->insert(entry);
              }

            // If it is owned but not relevant, we also add it to owned
            // interface DoF indices.
            if (is_owned)
              owned_interface_dofs.add_index(
                entry.second.get_interface_index());
          }
      }

    return {interface_maps, owned_interface_dofs, relevant_interface_dofs};
  }

  void
  communicate_interface_constraints(AffineConstraints<double> &constraints,
                                    const InterfaceMap &       interface_map,
                                    const unsigned int &       offset)
  {
    // Request constraints from interface DoFs on the other subdomain.
    std::map<unsigned int, std::vector<unsigned int>> lines_to_request;
    for (const auto &dof : interface_map)
      if (dof.second.get_other_owner() != Core::mpi_rank)
        lines_to_request[dof.second.get_other_owner()].push_back(
          dof.second.get_other_subdomain_index() + offset);

    std::map<unsigned int, std::vector<unsigned int>> lines_requested =
      Utilities::MPI::some_to_some(Core::mpi_comm, lines_to_request);
    std::map<unsigned int,
             std::vector<AffineConstraints<double>::ConstraintLine>>
      data_to_send;

    for (const auto &lines : lines_requested)
      for (const auto &line : lines.second)
        if (constraints.is_constrained(line))
          {
            // Find the line storing the constraint for requested DoF.
            auto line_it = constraints.get_lines().begin();
            for (; line_it != constraints.get_lines().end(); ++line_it)
              if (line_it->index == line)
                break;

            data_to_send[lines.first].push_back(*line_it);
          }

    std::map<unsigned int,
             std::vector<AffineConstraints<double>::ConstraintLine>>
      data_received =
        Utilities::MPI::some_to_some(Core::mpi_comm, data_to_send);

    for (const auto &lines : data_received)
      for (const auto &line : lines.second)
        if (!constraints.is_constrained(line.index))
          {
            constraints.add_line(line.index);
            constraints.set_inhomogeneity(line.index, line.inhomogeneity);

            for (const auto &entry : line.entries)
              constraints.add_entry(line.index, entry.first, entry.second);
          }
  }

  void
  add_interface_constraints(AffineConstraints<double> &constraints,
                            AffineConstraints<double> &interface_constraints,
                            const InterfaceMap &       interface_map,
                            const unsigned int &       subdomain,
                            const unsigned int &       offset)
  {
    for (const auto &dof : interface_map)
      {
        const types::global_dof_index dof_0 =
          (subdomain == 0 ? dof.first : dof.second.get_other_subdomain_index());
        const types::global_dof_index dof_1 =
          (subdomain == 0 ? dof.second.get_other_subdomain_index() :
                            dof.first) +
          offset;

        // If the two DoFs are already identity constrained (most likely due
        // to a previous call to add_interface_constraints), we skip them to
        // avoid introducing cycles in the constraints.
        if (constraints.are_identity_constrained(dof_0, dof_1))
          continue;

        const bool already_constrained_0 = constraints.is_constrained(dof_0);
        const bool already_constrained_1 = constraints.is_constrained(dof_1);

        // If neither DoF is constrained, just add the interface constraint
        // dof_0 = dof_1.
        if (!(already_constrained_0 || already_constrained_1))
          {
            constraints.add_line(dof_0);
            constraints.add_entry(dof_0, dof_1, 1.0);
            interface_constraints.add_line(dof_0);
            interface_constraints.add_entry(dof_0, dof_1, 1.0);
          }

        // If exactly one DoF is constrained, add the constraint for the
        // unconstrained DoF.
        else if (already_constrained_0 && !already_constrained_1)
          {
            constraints.add_line(dof_1);
            constraints.add_entry(dof_1, dof_0, 1.0);
            interface_constraints.add_line(dof_1);
            interface_constraints.add_entry(dof_1, dof_0, 1.0);
          }

        else if (!already_constrained_0 && already_constrained_1)
          {
            constraints.add_line(dof_0);
            constraints.add_entry(dof_0, dof_1, 1.0);
            interface_constraints.add_line(dof_0);
            interface_constraints.add_entry(dof_0, dof_1, 1.0);
          }

        // If both DoFs are constrained, we do nothing.
      }
  }

  std::pair<AffineConstraints<double>, AffineConstraints<double>>
  make_interface_constraints(
    const AffineConstraints<double> & constraints_0,
    const AffineConstraints<double> & constraints_1,
    const InterfaceMap &              interface_map_0,
    const InterfaceMap &              interface_map_1,
    const IndexSet &                  relevant_dofs,
    const unsigned int &              offset,
    const AlreadyConstrainedBehavior &already_constrained_behavior)
  {
    AffineConstraints<double> constraints(relevant_dofs);
    AffineConstraints<double> interface_constraints(relevant_dofs);

    // Copy constraints from subdomain 0.
    for (const auto &line : constraints_0.get_lines())
      {
        const auto mapped_dof = interface_map_0.find(line.index);

        if (mapped_dof == interface_map_0.end() ||
            bitmask_contains(already_constrained_behavior,
                             AlreadyConstrainedBehavior::keep_constraints_0))
          {
            constraints.add_line(line.index);
            constraints.set_inhomogeneity(line.index, line.inhomogeneity);

            for (const auto &entry : line.entries)
              constraints.add_entry(line.index, entry.first, entry.second);
          }
      }

    // Copy constraints from subdomain 1.
    for (const auto &line : constraints_1.get_lines())
      {
        const auto mapped_dof = interface_map_1.find(line.index);

        if (mapped_dof == interface_map_1.end() ||
            bitmask_contains(already_constrained_behavior,
                             AlreadyConstrainedBehavior::keep_constraints_1))
          {
            constraints.add_line(line.index + offset);
            constraints.set_inhomogeneity(line.index + offset,
                                          line.inhomogeneity);

            for (const auto &entry : line.entries)
              constraints.add_entry(line.index + offset,
                                    entry.first + offset,
                                    entry.second);
          }
      }

    communicate_interface_constraints(constraints, interface_map_0, offset);
    communicate_interface_constraints(constraints, interface_map_1);

    // Add constraints due to interface conditions.
    add_interface_constraints(
      constraints, interface_constraints, interface_map_0, 0, offset);
    add_interface_constraints(
      constraints, interface_constraints, interface_map_1, 1, offset);

    return std::make_pair(constraints, interface_constraints);
  }

  void
  make_interface_sparsity_pattern(
    const DoFHandler<dim> &                     dof_handler_0,
    const DoFHandler<dim> &                     dof_handler_1,
    const std::vector<types::global_dof_index> &n_dofs_0,
    const std::vector<types::global_dof_index> &n_dofs_1,
    const AffineConstraints<double> &           constraints,
    BlockDynamicSparsityPattern &               dest)
  {
    // Make the sparsity patterns of the two subdomains.
    std::vector<types::global_dof_index> dofs_per_block_0 = n_dofs_0;
    BlockDynamicSparsityPattern dsp_0(dofs_per_block_0, dofs_per_block_0);
    DoFTools::make_sparsity_pattern(dof_handler_0, dsp_0);

    std::vector<types::global_dof_index> dofs_per_block_1 = n_dofs_1;
    BlockDynamicSparsityPattern dsp_1(dofs_per_block_1, dofs_per_block_1);
    DoFTools::make_sparsity_pattern(dof_handler_1, dsp_1);

    const std::vector<unsigned int> n_blocks       = {dsp_0.n_block_rows(),
                                                dsp_1.n_block_rows()};
    const unsigned int              n_total_blocks = n_blocks[0] + n_blocks[1];

    dest.reinit(n_total_blocks, n_total_blocks);

    // Copy sparsity entries from each block of dsp_0 into the corresponding
    // block of dest.
    for (unsigned int i = 0; i < n_blocks[0]; ++i)
      for (unsigned int j = 0; j < n_blocks[0]; ++j)
        {
          const DynamicSparsityPattern &src_block  = dsp_0.block(i, j);
          DynamicSparsityPattern &      dest_block = dest.block(i, j);

          dest_block.reinit(src_block.n_rows(),
                            src_block.n_cols(),
                            src_block.row_index_set());

          for (auto it = src_block.begin(); it != src_block.end(); ++it)
            dest_block.add(it->row(), it->column());
        }

    // Copy sparsity entries from each block of dsp_1 into the corresponding
    // block of dest.
    for (unsigned int i = 0; i < n_blocks[1]; ++i)
      for (unsigned int j = 0; j < n_blocks[1]; ++j)
        {
          const DynamicSparsityPattern &src_block = dsp_1.block(i, j);
          DynamicSparsityPattern &      dest_block =
            dest.block(i + n_blocks[0], j + n_blocks[0]);

          dest_block.reinit(src_block.n_rows(),
                            src_block.n_cols(),
                            src_block.row_index_set());

          for (auto it = src_block.begin(); it != src_block.end(); ++it)
            dest_block.add(it->row(), it->column());
        }

    // Off-diagonal blocks are initialized empty.
    for (unsigned int i = 0; i < n_blocks[0]; ++i)
      for (unsigned int j = 0; j < n_blocks[1]; ++j)
        dest.block(i, j + n_blocks[0])
          .reinit(dsp_0.block(i, 0).n_rows(),
                  dsp_1.block(0, j).n_cols(),
                  dsp_0.block(i, 0).row_index_set());

    for (unsigned int i = 0; i < n_blocks[1]; ++i)
      for (unsigned int j = 0; j < n_blocks[0]; ++j)
        dest.block(i + n_blocks[0], j)
          .reinit(dsp_1.block(i, 0).n_rows(),
                  dsp_0.block(0, j).n_cols(),
                  dsp_1.block(i, 0).row_index_set());

    // Condense the sparsity pattern.
    dest.collect_sizes();
    constraints.condense(dest);
  }

  void
  compute_interface_mass_matrix(
    const utils::InterfaceMap &             interface_map_0,
    const IndexSet &                        owned_interface_dofs,
    const IndexSet &                        relevant_interface_dofs,
    const DoFHandler<dim> &                 dof_handler_0,
    const std::set<types::boundary_id> &    interface_tags_0,
    LinAlgTrilinos::Wrappers::SparseMatrix &matrix,
    const ComponentMask &                   mask_0)
  {
    const FiniteElement<dim> &fe            = dof_handler_0.get_fe();
    const unsigned int        dofs_per_face = fe.dofs_per_face;

    std::vector<types::global_dof_index> global_indices(dofs_per_face);

    unsigned int                         mapped_dofs_per_face = 0;
    std::vector<types::global_dof_index> local_interface_indices;
    std::vector<unsigned int>            components;

    for (unsigned int i = 0; i < dofs_per_face; ++i)
      {
        const unsigned int component =
          fe.face_system_to_component_index(i).first;

        if (mask_0[component])
          {
            ++mapped_dofs_per_face;
            components.push_back(component);
            local_interface_indices.push_back(i);
          }
      }

    std::vector<types::global_dof_index> interface_indices(
      mapped_dofs_per_face);

    // Construct the sparsity pattern of the matrix.
    {
      DynamicSparsityPattern dsp(relevant_interface_dofs);

      for (const auto &cell : dof_handler_0.active_cell_iterators())
        {
          if (!cell->is_locally_owned() || !cell->at_boundary())
            continue;

          for (unsigned int face = 0; face < cell->n_faces(); ++face)
            {
              if (interface_tags_0.find(cell->face(face)->boundary_id()) ==
                  interface_tags_0.end())
                continue;

              cell->face(face)->get_dof_indices(global_indices);

              // Retrieve interface indices for this face.
              for (unsigned int i = 0; i < mapped_dofs_per_face; ++i)
                interface_indices[i] =
                  interface_map_0.at(global_indices[local_interface_indices[i]])
                    .get_interface_index();

              for (unsigned int i = 0; i < mapped_dofs_per_face; ++i)
                for (unsigned int j = 0; j < mapped_dofs_per_face; ++j)
                  if (components[i] == components[j])
                    dsp.add(interface_indices[i], interface_indices[j]);
            }
        }

      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_interface_dofs,
                                                 Core::mpi_comm,
                                                 relevant_interface_dofs);

      matrix.reinit(owned_interface_dofs, dsp, Core::mpi_comm);
    }

    // Assemble the matrix.
    {
      const auto &triangulation = dof_handler_0.get_triangulation();

      std::unique_ptr<Quadrature<dim - 1>> quadrature =
        MeshHandler::get_quadrature_gauss<dim - 1>(triangulation,
                                                   fe.degree + 1);
      const unsigned int n_q_points = quadrature->size();

      FEFaceValues<dim> fe_face_values(fe,
                                       *quadrature,
                                       update_values | update_JxW_values);

      FullMatrix<double> mass_loc(mapped_dofs_per_face, mapped_dofs_per_face);

      for (const auto &cell : dof_handler_0.active_cell_iterators())
        {
          if (!cell->is_locally_owned() || !cell->at_boundary())
            continue;

          for (unsigned int face = 0; face < cell->n_faces(); ++face)
            {
              if (interface_tags_0.find(cell->face(face)->boundary_id()) ==
                  interface_tags_0.end())
                continue;

              mass_loc = 0.0;
              fe_face_values.reinit(cell, face);

              // Assemble local matrix.
              for (unsigned int i = 0; i < mapped_dofs_per_face; ++i)
                for (unsigned int j = 0; j < mapped_dofs_per_face; ++j)
                  if (components[i] == components[j])
                    {
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        mass_loc(i, j) +=
                          fe_face_values.shape_value(local_interface_indices[i],
                                                     q) *
                          fe_face_values.shape_value(local_interface_indices[j],
                                                     q) *
                          fe_face_values.JxW(q);
                    }

              cell->face(face)->get_dof_indices(global_indices);

              // Retrieve interface indices for this face.
              for (unsigned int i = 0; i < mapped_dofs_per_face; ++i)
                interface_indices[i] =
                  interface_map_0.at(global_indices[local_interface_indices[i]])
                    .get_interface_index();

              // Distribute local matrix to the global one.
              matrix.add(interface_indices, mass_loc);
            }
        }

      matrix.compress(VectorOperation::add);
    }
  } // namespace lifex::utils

  // Explicit instantiation.
  template class InterfaceHandler<LinAlg::MPI::Vector>;

  // Explicit instantiation.
  template class InterfaceHandler<LinAlg::MPI::BlockVector>;
} // namespace lifex::utils
