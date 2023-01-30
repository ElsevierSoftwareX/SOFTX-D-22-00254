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

#include "source/geometry/mesh_handler.hpp"
#include "source/geometry/mesh_info.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/distributed/tria_base.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>

namespace lifex::utils
{
  MeshInfo::MeshInfo(const Triangulation<dim> &triangulation_)
    : _initialized(false)
    , triangulation(triangulation_)
  {}

  void
  MeshInfo::initialize()
  {
    _initialized = true;

    diameter_tot = 0;
    diameter_min = std::numeric_limits<double>::max();
    diameter_max = std::numeric_limits<double>::min();

    // Compute number of locally owned active cell, depending on whether the
    // triangulation is parallel or not. This works for all types of parallel
    // triangulation (shared, distributed, fullydistributed).
    const auto *parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);

    if (parallel_triangulation)
      n_cells = parallel_triangulation->n_locally_owned_active_cells();
    else
      n_cells = triangulation.n_active_cells();

    // Compute diameter vector, as well as
    // its total, minimum, maximum and average value.
    diameters.resize(n_cells);

    unsigned int c = 0;

    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        if (cell->is_locally_owned())
          {
            diameters[c] = cell->diameter();

            diameter_tot += diameters[c];
            diameter_min = std::min(diameter_min, diameters[c]);
            diameter_max = std::max(diameter_max, diameters[c]);

            // Compute volume IDs and face/line boundary IDs.
            ids_volume.insert(cell->material_id());

            for (unsigned int face = 0; face < cell->n_faces(); ++face)
              {
                if (cell->face(face)->at_boundary())
                  {
                    ids_face.insert(cell->face(face)->boundary_id());

                    for (unsigned int line = 0;
                         line < cell->face(face)->n_lines();
                         ++line)
                      {
                        ids_line.insert(
                          cell->face(face)->line(line)->boundary_id());
                      }
                  }
              }

            ++c;
          }
      }

    // Communicate mesh info across different parallel processes.
    n_cells      = Utilities::MPI::sum(n_cells, mpi_comm);
    diameter_tot = Utilities::MPI::sum(diameter_tot, mpi_comm);
    diameter_min = Utilities::MPI::min(diameter_min, mpi_comm);
    diameter_max = Utilities::MPI::max(diameter_max, mpi_comm);

    diameter_avg = diameter_tot / n_cells;

    // Communicate IDs across different parallel processes.
    auto allgather_ids = [this](const auto &ids) -> std::vector<int> {
      int              n_ids = ids.size();
      std::vector<int> n_ids_vec(mpi_size);
      MPI_Allgather(&n_ids, 1, MPI_INT, n_ids_vec.data(), 1, MPI_INT, mpi_comm);

      int offsets = 0;
      MPI_Exscan(&n_ids, &offsets, 1, MPI_INT, MPI_SUM, mpi_comm);
      std::vector<int> offsets_vec(mpi_size);
      MPI_Allgather(
        &offsets, 1, MPI_INT, offsets_vec.data(), 1, MPI_INT, mpi_comm);

      std::vector<int> ids_vec(ids.begin(), ids.end());

      n_ids = Utilities::MPI::sum(n_ids, mpi_comm);
      std::vector<int> ids_vec_global(n_ids);
      MPI_Allgatherv(ids_vec.data(),
                     ids_vec.size(),
                     MPI_INT,
                     ids_vec_global.data(),
                     n_ids_vec.data(),
                     offsets_vec.data(),
                     MPI_INT,
                     mpi_comm);
      return ids_vec_global;
    };

    std::vector<int> ids_volume_global = allgather_ids(ids_volume);
    ids_volume = std::set<types::material_id>(ids_volume_global.begin(),
                                              ids_volume_global.end());

    std::vector<int> ids_face_global = allgather_ids(ids_face);
    ids_face = std::set<types::material_id>(ids_face_global.begin(),
                                            ids_face_global.end());

    std::vector<int> ids_line_global = allgather_ids(ids_line);
    ids_line = std::set<types::material_id>(ids_line_global.begin(),
                                            ids_line_global.end());
  }

  void
  MeshInfo::clear()
  {
    _initialized = false;

    diameters.clear();

    n_cells = 0;

    diameter_tot = 0;
    diameter_min = 0;
    diameter_max = 0;
    diameter_avg = 0;

    ids_volume.clear();
    ids_face.clear();
    ids_line.clear();
  }

  void
  MeshInfo::print(const std::string &label,
                  const std::string &n_dofs_info,
                  const bool &       print_ids) const
  {
    AssertThrow(_initialized, ExcNotInitialized());

    pcout << utils::log::separator_section << std::endl;

    if (!label.empty())
      pcout << label << std::endl << std::endl;


    pcout << "    Element type: ";
    if (MeshHandler::is_hex(triangulation))
      pcout << "Hexahedra";
    else // if (MeshHandler::is_tet(triangulation))
      pcout << "Tetrahedra";

    pcout << std::endl << std::endl;

    pcout << "    Maximum cell diameter: " << diameter_max << std::endl
          << "    Average cell diameter: " << diameter_avg << std::endl
          << "    Minimum cell diameter: " << diameter_min << std::endl
          << std::endl;


    pcout << "    Number of active cells: "
          << triangulation.n_global_active_cells() << std::endl;

    if (!n_dofs_info.empty())
      pcout << "    Number of degrees of freedom: " << n_dofs_info << std::endl;


    if (print_ids)
      {
        if (!ids_volume.empty() || !ids_face.empty() || !ids_line.empty())
          pcout << std::endl;

        if (!ids_volume.empty())
          {
            pcout << "    Volume IDs: ";
            for (auto it = ids_volume.begin(); it != ids_volume.end(); ++it)
              {
                pcout << *it;

                if (it != std::prev(ids_volume.end()))
                  pcout << ", ";
                else
                  pcout << std::endl;
              }
          }

        if (!ids_face.empty())
          {
            pcout << "    Face boundary IDs: ";
            for (auto it = ids_face.begin(); it != ids_face.end(); ++it)
              {
                pcout << *it;

                if (it != std::prev(ids_face.end()))
                  pcout << ", ";
                else
                  pcout << std::endl;
              }
          }

        if (!ids_line.empty())
          {
            pcout << "    Line boundary IDs: ";
            for (auto it = ids_line.begin(); it != ids_line.end(); ++it)
              {
                pcout << *it;

                if (it != std::prev(ids_line.end()))
                  pcout << ", ";
                else
                  pcout << std::endl;
              }
          }
      }

    pcout << utils::log::separator_section << std::endl;
  }

  void
  MeshInfo::print(const std::string & label,
                  const unsigned int &n_dofs,
                  const bool &        print_ids) const
  {
    return print(label, std::to_string(n_dofs), print_ids);
  }

  MeshInfo::MeshQualityInfo
  MeshInfo::print_mesh_quality_info(const bool &verbose) const
  {
    // Create a dummy FE and DoF handler.
    auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // We use a nodal quadrature formula to evaluate jacobians at vertices.
    // Notice that the weights of this quadrature are uninitialized, so it
    // cannot be used for integral computation.
    auto quadrature_formula_jacobians =
      MeshHandler::get_quadrature_nodal(triangulation);

    FEValues<dim> fe_values(*fe,
                            *quadrature_formula_jacobians,
                            update_jacobians);

    double edge_ratio_min = std::numeric_limits<double>::max();
    double edge_ratio_max = std::numeric_limits<double>::min();

    double jac_min = std::numeric_limits<double>::max();
    double jac_max = std::numeric_limits<double>::min();

    // If in verbose mode, this vector is filled by each process with error
    // messages related to individual low-quality elements. They are printed at
    // the end of this function, to avoid mixing messages between different
    // processes.
    std::vector<std::string> error_messages;

    // Helper function used to print material and boundary ID info of an
    // element.
    auto print_element_info =
      [](const DoFHandler<dim>::active_cell_iterator &cell,
         std::stringstream &                          message) {
        message << "Material ID = " << std::setw(4) << cell->material_id();

        if (cell->at_boundary())
          {
            message << " | Boundary IDs = ";
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
              if (cell->face(face)->at_boundary())
                message << std::setw(4) << cell->face(face)->boundary_id()
                        << " ";
          }
      };

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            // Edge ratio.
            {
              double len_min = std::numeric_limits<double>::max();
              double len_max = std::numeric_limits<double>::min();

              for (unsigned int line_idx = 0; line_idx < cell->n_lines();
                   ++line_idx)
                {
                  const double len = cell->line(line_idx)->diameter();

                  len_min = std::min(len_min, len);
                  len_max = std::max(len_max, len);
                }

              const double edge_ratio = len_max / len_min;
              edge_ratio_min          = std::min(edge_ratio_min, edge_ratio);
              edge_ratio_max          = std::max(edge_ratio_max, edge_ratio);

              if (verbose && edge_ratio > 10)
                {
                  std::stringstream message;

                  message << "    [rank " << std::setw(4) << mpi_rank
                          << "] Edge ratio = " << std::scientific
                          << std::setw(13) << edge_ratio << " | ";
                  print_element_info(cell, message);

                  error_messages.push_back(message.str());
                }
            }

            // Jacobian at quadrature points.
            for (unsigned int q = 0; q < quadrature_formula_jacobians->size();
                 ++q)
              {
                const double jac = fe_values.jacobian(q).determinant();
                jac_min          = std::min(jac_min, jac);
                jac_max          = std::max(jac_max, jac);

                if (verbose && !utils::is_positive(jac))
                  {
                    std::stringstream message;

                    message << "    [rank " << std::setw(4) << mpi_rank
                            << "] Jacobian   = " << std::scientific
                            << std::setw(13) << jac << " | ";
                    print_element_info(cell, message);

                    error_messages.push_back(message.str());
                  }
              }
          }
      }

    // Communicate.
    edge_ratio_min = Utilities::MPI::min(edge_ratio_min, mpi_comm);
    edge_ratio_max = Utilities::MPI::max(edge_ratio_max, mpi_comm);
    jac_min        = Utilities::MPI::min(jac_min, mpi_comm);
    jac_max        = Utilities::MPI::max(jac_max, mpi_comm);

    pcout << "\tMinimum edge ratio = " << edge_ratio_min << std::endl;
    pcout << "\tMaximum edge ratio = " << edge_ratio_max << std::endl;
    pcout << "\tMinimum jacobian   = " << jac_min << std::endl;
    pcout << "\tMaximum jacobian   = " << jac_max << std::endl;

    // Print error messages one rank at a time.
    if (verbose)
      {
        pcout << std::endl;

        for (unsigned int rank = 0; rank < mpi_size; ++rank)
          {
            if (rank == Core::mpi_rank)
              {
                for (const auto &error_message : error_messages)
                  std::cout << error_message << std::endl;
              }

            MPI_Barrier(mpi_comm);
          }
      }

    return MeshQualityInfo(edge_ratio_min, edge_ratio_max, jac_min, jac_max);
  }

  void
  MeshInfo::save_diameters(const std::string &filename) const
  {
    AssertThrow(_initialized, ExcNotInitialized());

    std::ofstream output;

    for (unsigned int rank = 0; rank < mpi_size; ++rank)
      {
        if (rank == Core::mpi_rank)
          {
            // Rank 0 opens and writes the file, other ranks append to it.
            output.open(filename,
                        (mpi_rank == 0) ? std::ofstream::out :
                                          std::ofstream::app);

            for (const auto &d : diameters)
              output << d << std::endl;
          }

        MPI_Barrier(mpi_comm);
      }

    output.close();
  }

  double
  MeshInfo::compute_mesh_volume() const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the constant function 1 over each cell,
    // due to Q1 mapping to reference element.
    const auto quadrature_formula =
      MeshHandler::get_quadrature_gauss(triangulation, 2);

    const unsigned int n_q_points = quadrature_formula->size();

    FEValues<dim> fe_values(*fe, *quadrature_formula, update_JxW_values);

    double volume = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                volume += fe_values.JxW(q);
              }
          }
      }

    // Sum over the different parallel processes.
    volume = Utilities::MPI::sum(volume, mpi_comm);

    return volume;
  }

  double
  MeshInfo::compute_submesh_volume(
    const std::set<types::material_id> &material_ids) const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the constant function 1 over each cell,
    // due to Q1 mapping to reference element.
    const auto quadrature_formula =
      MeshHandler::get_quadrature_gauss(triangulation, 2);

    const unsigned int n_q_points = quadrature_formula->size();

    FEValues<dim> fe_values(*fe, *quadrature_formula, update_JxW_values);

    double       volume      = 0;
    unsigned int found_count = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() &&
            material_ids.find(cell->material_id()) != material_ids.end())
          {
            fe_values.reinit(cell);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                volume += fe_values.JxW(q);
              }

            ++found_count;
          }
      }

    found_count = Utilities::MPI::sum(found_count, mpi_comm);

    std::string exc_message =
      "None of the material IDs was found. Check that at least one of the IDs ";
    for (const auto &material_id : material_ids)
      exc_message += std::to_string(material_id) + ", ";
    exc_message += "really exists in the input triangulation.";

    AssertThrow(found_count != 0, ExcMessage(exc_message));

    // Sum over the different parallel processes.
    volume = Utilities::MPI::sum(volume, mpi_comm);

    return volume;
  }

  double
  MeshInfo::compute_submesh_volume(const types::material_id &material_id) const
  {
    return compute_submesh_volume({{material_id}});
  }

  double
  MeshInfo::compute_surface_area(const types::boundary_id &boundary_id) const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the constant function 1 over each face,
    // due to Q1 mapping to reference element.
    const auto face_quadrature_formula =
      MeshHandler::get_quadrature_gauss<dim - 1>(triangulation, 2);

    const unsigned int n_face_q_points = face_quadrature_formula->size();

    FEFaceValues<dim> fe_face_values(*fe,
                                     *face_quadrature_formula,
                                     update_JxW_values);

    double       area        = 0;
    unsigned int found_count = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
              {
                if (cell->face(face)->at_boundary() &&
                    cell->face(face)->boundary_id() == boundary_id)
                  {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        area += fe_face_values.JxW(q);
                      }

                    ++found_count;
                  }
              }
          }
      }

    found_count = Utilities::MPI::sum(found_count, mpi_comm);

    AssertThrow(found_count != 0,
                ExcMessage("Boundary ID " + std::to_string(boundary_id) +
                           " not found."));

    // Sum over the different parallel processes.
    area = Utilities::MPI::sum(area, mpi_comm);

    return area;
  }

  Tensor<1, dim, double>
  MeshInfo::compute_flat_boundary_normal(
    const types::boundary_id &boundary_id) const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // We just compute the normal at the barycenter of
    // the first face found on the considered boundary.
    const auto face_quadrature_formula =
      MeshHandler::get_quadrature_gauss<dim - 1>(triangulation, 1);

    const unsigned int n_face_q_points = face_quadrature_formula->size();

    FEFaceValues<dim> fe_face_values(*fe,
                                     *face_quadrature_formula,
                                     update_normal_vectors);

    Tensor<1, dim, double> normal_vector;
    unsigned int           found_count = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
              {
                if (cell->face(face)->at_boundary() &&
                    cell->face(face)->boundary_id() == boundary_id)
                  {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        normal_vector = fe_face_values.normal_vector(q);
                      }

                    ++found_count;

                    break;
                  }
              }

            if (found_count > 0)
              break;
          }
      }

    // Here found_count is either 0 or 1.
    // Summing found_count returns the number of processes
    // owning a face on the considered boundary.
    found_count = Utilities::MPI::sum(found_count, mpi_comm);

    AssertThrow(found_count != 0,
                ExcMessage("Boundary ID " + std::to_string(boundary_id) +
                           " not found."));

    // Compute average over the different parallel processes.
    // Since we are assuming that the boundary is flat,
    // every process owning a face on that boundary computes the same
    // normal vector as the other processes (up to the numerical precision).
    // Therefore, returning the average is equivalent
    // to return any one of the normal vectors found.
    for (unsigned int d = 0; d < dim; ++d)
      {
        normal_vector[d] =
          Utilities::MPI::sum(normal_vector[d], mpi_comm) / found_count;
      }

    return normal_vector;
  }

  Point<dim>
  MeshInfo::compute_mesh_barycenter() const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the linear functions x, y, z over each cell,
    // due to Q1 mapping to reference element.
    const auto quadrature_formula =
      MeshHandler::get_quadrature_gauss(triangulation, 2);

    const unsigned int n_q_points = quadrature_formula->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_JxW_values | update_quadrature_points);

    Point<dim> barycenter;
    double     volume = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                volume += fe_values.JxW(q);
                barycenter += fe_values.quadrature_point(q) * fe_values.JxW(q);
              }
          }
      }

    // Sum over the different parallel processes.
    volume = Utilities::MPI::sum(volume, mpi_comm);
    for (unsigned int d = 0; d < dim; ++d)
      barycenter[d] = Utilities::MPI::sum(barycenter[d], mpi_comm);

    barycenter /= volume;

    return barycenter;
  }

  Point<dim>
  MeshInfo::compute_surface_barycenter(
    const std::set<types::boundary_id> &boundary_ids) const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the linear functions x, y, z over each face,
    // due to Q1 mapping to reference element.
    const auto face_quadrature_formula =
      MeshHandler::get_quadrature_gauss<dim - 1>(triangulation, 2);

    const unsigned int n_face_q_points = face_quadrature_formula->size();

    FEFaceValues<dim> fe_face_values(*fe,
                                     *face_quadrature_formula,
                                     update_JxW_values |
                                       update_quadrature_points);

    Point<dim>   barycenter;
    double       area        = 0;
    unsigned int found_count = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
              {
                if (cell->face(face)->at_boundary() &&
                    boundary_ids.find(cell->face(face)->boundary_id()) !=
                      boundary_ids.end())
                  {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        barycenter += fe_face_values.quadrature_point(q) *
                                      fe_face_values.JxW(q);

                        area += fe_face_values.JxW(q);
                      }

                    ++found_count;
                  }
              }
          }
      }

    found_count = Utilities::MPI::sum(found_count, mpi_comm);

    std::string exc_message =
      "None of the boundary IDs was found. Check that at least one of the IDs ";
    for (const auto &boundary_id : boundary_ids)
      exc_message += std::to_string(boundary_id) + ", ";
    exc_message += "really exists in the input triangulation.";

    AssertThrow(found_count != 0, ExcMessage(exc_message));

    // Sum over the different parallel processes.
    area = Utilities::MPI::sum(area, mpi_comm);
    for (unsigned int component = 0; component < dim; ++component)
      {
        barycenter[component] =
          Utilities::MPI::sum(barycenter[component], mpi_comm);
      }

    return barycenter / area;
  }

  Point<dim>
  MeshInfo::compute_surface_barycenter(
    const types::boundary_id &boundary_id) const
  {
    return compute_surface_barycenter({{boundary_id}});
  }

  Tensor<1, dim, double>
  MeshInfo::compute_mesh_moment_inertia() const
  {
    // Create a dummy FE and DoF handler.
    const auto fe = MeshHandler::get_fe_lagrange(triangulation, 1);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    // Two points are required to integrate exactly
    // the linear functions x, y, z over each cell,
    // due to Q1 mapping to reference element.
    const auto quadrature_formula =
      MeshHandler::get_quadrature_gauss(triangulation, 2);

    const unsigned int n_q_points = quadrature_formula->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_JxW_values | update_quadrature_points);

    Tensor<1, dim, double> moment;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  // d-th component = integral of x_d^2
                  moment[d] += fe_values.quadrature_point(q)[d] *
                               fe_values.quadrature_point(q)[d] *
                               fe_values.JxW(q);
              }
          }
      }

    // Sum over the different parallel processes.
    for (unsigned int d = 0; d < dim; ++d)
      moment[d] = Utilities::MPI::sum(moment[d], mpi_comm);

    return moment;
  }
} // namespace lifex::utils
