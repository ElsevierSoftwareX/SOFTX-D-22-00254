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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/geometry/mesh_handler.hpp"

#include "source/io/serialization.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <boost/filesystem.hpp>

#include <fstream>

namespace lifex::utils
{
  MeshHandler::MeshHandler(const std::string &           subsection,
                           const MPI_Comm &              mpi_comm_,
                           const std::set<GeometryType> &geometry_type_set_)
    : MeshHandler(subsection,
                  "Mesh and space discretization",
                  mpi_comm_,
                  geometry_type_set_)
  {}

  MeshHandler::MeshHandler(const std::string &           subsection,
                           const std::string &           subsubsection,
                           const MPI_Comm &              mpi_comm_,
                           const std::set<GeometryType> &geometry_type_set_)
    : CoreModel(subsection + " / " + subsubsection)
  {
    data.mpi_comm = mpi_comm_;
    data.mpi_rank = Utilities::MPI::this_mpi_process(data.mpi_comm);
    data.mpi_size = Utilities::MPI::n_mpi_processes(data.mpi_comm);

    data.geometry_type_set = geometry_type_set_;
  }

  MeshHandler::MeshHandler(const MeshHandler &other)
    : CoreModel(other.prm_subsection_path)
  {
    data = other.data;
  }

  MeshHandler &
  MeshHandler::operator=(const MeshHandler &other)
  {
    data = other.data;

    return *this;
  }

  void
  MeshHandler::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    params.set_verbosity(VerbosityParam::Minimal);
    {
      std::string geometry_type_selection;

      for (const auto &geometry_type : data.geometry_type_set)
        {
          geometry_type_selection +=
            data.geometry_type_map.at(geometry_type) + " | ";
        }
      geometry_type_selection =
        geometry_type_selection.substr(0, geometry_type_selection.size() - 3);

      if (data.geometry_type_set.size() > 1)
        {
          params.declare_entry_selection("Mesh type",
                                         "File",
                                         geometry_type_selection);
        }

      if (utils::contains(data.geometry_type_set, GeometryType::File))
        {
          params.enter_subsection("File");
          {
            params.declare_entry("Filename",
                                 "",
                                 Patterns::FileName(
                                   Patterns::FileName::FileType::input),
                                 "Mesh file.");

            params.declare_entry(
              "Scaling factor",
              "1",
              Patterns::Double(0),
              "Mesh scaling factor: 1e-3 => from [mm] to [m].");
          }
          params.leave_subsection();
        }

      if (utils::contains(data.geometry_type_set, GeometryType::Hypercube))
        {
          params.enter_subsection("Hypercube");
          {
            params.declare_entry("Left",
                                 "0",
                                 Patterns::Double(),
                                 "Left endpoint.");

            params.declare_entry("Right",
                                 "1",
                                 Patterns::Double(),
                                 "Right endpoint.");

            params.declare_entry("Colorize",
                                 "true",
                                 Patterns::Bool(),
                                 "Colorize boundary IDs.");
          }
          params.leave_subsection();
        }

      if (utils::contains(data.geometry_type_set, GeometryType::Cylinder))
        {
          params.enter_subsection("Cylinder");
          {
            params.declare_entry("Radius",
                                 "0.01",
                                 Patterns::Double(0),
                                 "Cylinder radius [m].");

            params.declare_entry("Length",
                                 "0.1",
                                 Patterns::Double(0),
                                 "Cylinder length [m].");

            params.declare_entry("Number of slices",
                                 "3",
                                 Patterns::Integer(2),
                                 "Number of slices along the cylinder axis.");
          }
          params.leave_subsection();
        }

      if (utils::contains(data.geometry_type_set,
                          GeometryType::ChannelWithCylinder))
        {
          params.enter_subsection("Channel with cylinder");
          {
            params.declare_entry(
              "Shell region width",
              "0.03",
              Patterns::Double(0),
              "Width of the layer of shells around the cylinder [m].");

            params.declare_entry("Number of shells",
                                 "2",
                                 Patterns::Integer(0),
                                 "Number of shells in the shell layer [-].");

            params.declare_entry("Skewness",
                                 "2.0",
                                 Patterns::Double(0),
                                 "Parameter controlling how close the shells "
                                 "are to the cylinder.");

            params.declare_entry("Colorize",
                                 "true",
                                 Patterns::Bool(),
                                 "Colorize boundary IDs.");
          }
          params.leave_subsection();
        }

      params.declare_entry_selection(
        "Element type",
        "Hex",
        "Hex | Tet",
        "Specify whether the input mesh has hexahedral or tetrahedral "
        "elements. Tetrahedral meshes can only be imported from file.");

      params.declare_entry("Number of refinements",
                           "0",
                           Patterns::Integer(0),
                           "Number of global mesh refinement steps applied to "
                           "the initial grid. Ignored if restart is enabled.");
    }
    params.reset_verbosity();

    params.set_verbosity(VerbosityParam::Full);
    params.declare_entry(
      "Reading group size",
      "1",
      Patterns::Integer(0),
      "This option is only used if the Element type is Tet. In that case, the "
      "mesh is only read by one process for every group of this size, and then "
      "communicated to other processes in that group. A value of 0 indicates "
      "that the group contains all processes (i.e. only one process reads the "
      "mesh). Small groups are faster but may take up a lot of memory during "
      "the mesh creation phase. Conversely, large groups require less memory "
      "but the mesh creation may be slower.");
    params.reset_verbosity();

    params.leave_subsection_path();
  }

  void
  MeshHandler::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      const std::string mesh_type =
        (data.geometry_type_set.size() > 1) ?
          params.get("Mesh type") :
          data.geometry_type_map.at(*(data.geometry_type_set.begin()));

      if (mesh_type == "File")
        {
          params.enter_subsection("File");
          {
            initialize_from_file(params.get("Filename"),
                                 params.get_double("Scaling factor"));
          }
          params.leave_subsection();
        }
      else if (mesh_type == "Hypercube")
        {
          params.enter_subsection("Hypercube");
          {
            initialize_hypercube(params.get_double("Left"),
                                 params.get_double("Right"),
                                 params.get_bool("Colorize"));
          }
          params.leave_subsection();
        }
      else if (mesh_type == "Cylinder")
        {
          params.enter_subsection("Cylinder");
          {
            initialize_cylinder(params.get_double("Radius"),
                                params.get_double("Length"),
                                params.get_integer("Number of slices"));
          }
          params.leave_subsection();
        }
      else // if (mesh_type == "Channel with cylinder")
        {
          params.enter_subsection("Channel with cylinder");
          {
            initialize_channel_with_cylinder(
              params.get_double("Shell region width"),
              params.get_integer("Number of shells"),
              params.get_double("Skewness"),
              params.get_bool("Colorize"));
          }
          params.leave_subsection();
        }

      if (params.get("Element type") == "Hex")
        set_element_type(ElementType::Hex);
      else // if (params.get("Element type") == "Tet")
        set_element_type(ElementType::Tet);

      set_refinement_global(params.get_integer("Number of refinements"));
    }

    data.reading_group_size = params.get_integer("Reading group size");

    params.leave_subsection_path();
  }

  void
  MeshHandler::initialize_from_file(const std::string &filename_,
                                    const double &     scaling_factor_)
  {
    AssertThrow(!filename_.empty(), ExcMessage("Filename cannot be empty."));
    AssertThrow(boost::filesystem::exists(filename_),
                ExcFileNotOpen(filename_));
    AssertThrow(scaling_factor_ > 0,
                ExcMessage("Scaling factor should be greater than zero."));

    data.geometry_type = GeometryType::File;

    data.filename       = filename_;
    data.scaling_factor = scaling_factor_;
  }

  void
  MeshHandler::initialize_hypercube(const double &left,
                                    const double &right,
                                    const bool    colorize)
  {
    data.geometry_type = GeometryType::Hypercube;

    data.hypercube_left     = left;
    data.hypercube_right    = right;
    data.hypercube_colorize = colorize;
  }

  void
  MeshHandler::initialize_cylinder(const double       radius,
                                   const double       length,
                                   const unsigned int n_slices)
  {
    AssertThrow(radius > 0, ExcMessage("Radius should be greater than zero."));
    AssertThrow(length > 0, ExcMessage("Length should be greater than zero."));
    AssertThrow(n_slices >= 2,
                ExcMessage("Number of slices should be at least 2."));

    data.geometry_type = GeometryType::Cylinder;

    data.cylinder_radius   = radius;
    data.cylinder_length   = length;
    data.cylinder_n_slices = n_slices;
  }

  void
  MeshHandler::initialize_channel_with_cylinder(const double shell_region_width,
                                                const unsigned int n_shells,
                                                const double       skewness,
                                                const bool         colorize)
  {
    AssertThrow(shell_region_width > 0,
                ExcMessage("Shell region width should be greater than zero."));
    AssertThrow(skewness > 0,
                ExcMessage("Skewness should be greater than zero."));

    data.geometry_type = GeometryType::ChannelWithCylinder;

    data.channel_shell_region_width = shell_region_width;
    data.channel_n_shells           = n_shells;
    data.channel_skewness           = skewness;
    data.channel_colorize           = colorize;
  }

  void
  MeshHandler::initialize_hyper_shell(const Point<dim>   center,
                                      const double       inner_radius,
                                      const double       outer_radius,
                                      const unsigned int n_cells,
                                      const bool         colorize)
  {
    AssertThrow(outer_radius > 0,
                ExcMessage("Outer radius should be greater than zero."));

    data.geometry_type = GeometryType::HyperShell;

    data.shell_center       = center;
    data.shell_inner_radius = inner_radius;
    data.shell_outer_radius = outer_radius;
    data.shell_n_cells      = n_cells;
    data.shell_colorize     = colorize;
  }

  void
  MeshHandler::set_element_type(const ElementType &element_type)
  {
    if (element_type == ElementType::Tet)
      AssertThrow(data.geometry_type == GeometryType::File,
                  ExcMessage(
                    "Tetrahedral meshes can only be imported from file."));

    data.element_type = element_type;
  }

  void
  MeshHandler::set_refinement_global(const unsigned int &n_refinements_)
  {
    data.refinement_type = RefinementType::Global;

    data.n_refinements = n_refinements_;
  }

  void
  MeshHandler::set_refinement_from_file(
    const std::string &filename_to_deserialize_)
  {
    AssertThrow(!filename_to_deserialize_.empty(),
                ExcMessage("Filename to deserialize cannot be empty."));

    data.refinement_type = RefinementType::Deserialize;

    data.filename_to_deserialize = filename_to_deserialize_;
  }

  void
  MeshHandler::create_mesh(
    const bool &                                      refine,
    const typename Triangulation<dim>::MeshSmoothing &smoothing,
    const std::optional<parallel::distributed::Triangulation<dim>::Settings>
      &settings)
  {
    AssertThrow(data.geometry_type != GeometryType::Other,
                ExcLifexNotImplemented());

    if (data.element_type == ElementType::Hex)
      {
        // Hexahedral meshes are both supported by p::d::T and p::f::T:
        // the former is used here due to its better compatibility and more
        // stable interface.
        triangulation =
          std::make_unique<parallel::distributed::Triangulation<dim>>(
            data.mpi_comm,
            smoothing,
            settings.value_or(
              parallel::distributed::Triangulation<dim>::default_setting));
      }
    else // if (data.element_type == ElementType::Tet)
      {
        // Tetrahedral meshes are only supported by p::f::T.
        triangulation =
          std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
            data.mpi_comm);
      }

    mesh_info = std::make_unique<MeshInfo>(*triangulation);

    // Create coarse mesh.
    auto generate_mesh = [this](::dealii::Triangulation<dim>
                                  &triangulation_current) {
      if (data.geometry_type == GeometryType::File)
        {
          GridIn<dim> gridin;
          gridin.attach_triangulation(triangulation_current);

          std::ifstream file(data.filename);
          gridin.read_msh(file);
        }
      else if (data.geometry_type == GeometryType::Hypercube)
        {
          GridGenerator::hyper_cube(triangulation_current,
                                    data.hypercube_left,
                                    data.hypercube_right,
                                    data.hypercube_colorize);
        }
      else if (data.geometry_type == GeometryType::Cylinder)
        {
          // NB: the following call would generate an analogous geometry
          // (where the cylinder extends from x = -half_length
          // to x = +half_length).
          // GridGenerator::subdivided_cylinder(triangulation_current,
          //                                    cylinder_n_slices,
          //                                    cylinder_radius,
          //                                    cylinder_length / 2);

          // Manually extrude a circle.
          ::dealii::Triangulation<dim - 1> triangulation_circle;
          GridGenerator::hyper_ball(triangulation_circle,
                                    Point<dim - 1>(),
                                    data.cylinder_radius,
                                    false);

          GridGenerator::extrude_triangulation(triangulation_circle,
                                               data.cylinder_n_slices,
                                               data.cylinder_length,
                                               triangulation_current);

          // Set cylindrical manifold along the z-axis.
          GridTools::copy_boundary_to_manifold_id(triangulation_current, false);
          triangulation_current.set_manifold(0, CylindricalManifold<dim>(2));
        }
      else if (data.geometry_type == GeometryType::ChannelWithCylinder)
        {
          GridGenerator::channel_with_cylinder(triangulation_current,
                                               data.channel_shell_region_width,
                                               data.channel_n_shells,
                                               data.channel_skewness,
                                               data.channel_colorize);
        }
      else // if (data.geometry_type == GeometryType::HyperShell)
        {
          GridGenerator::hyper_shell(triangulation_current,
                                     data.shell_center,
                                     data.shell_inner_radius,
                                     data.shell_outer_radius,
                                     data.shell_n_cells,
                                     data.shell_colorize);
        }
    };

    if (data.element_type == ElementType::Tet)
      {
        auto partition_mesh = [](::dealii::Triangulation<dim> &tria,
                                 const MPI_Comm &              mpi_comm,
                                 const unsigned int /*group_size*/) {
          GridTools::partition_triangulation(
            Utilities::MPI::n_mpi_processes(mpi_comm), tria);
        };

        auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation_in_groups<dim, dim>(
            generate_mesh,
            partition_mesh,
            mpi_comm,
            (data.reading_group_size == 0) ? mpi_size :
                                             data.reading_group_size);

        triangulation->create_triangulation(construction_data);
      }
    else // if (data.element_type == ElementType::Hex)
      {
        generate_mesh(*triangulation);
      }

    // Check the element type.
    AssertThrow((data.element_type == ElementType::Hex && is_hex()) ||
                  (data.element_type == ElementType::Tet && is_tet()),
                ExcMessage("Wrong element type specified."));

    if (data.scaling_factor > 0)
      GridTools::scale(data.scaling_factor, *triangulation);

    // Refine.
    if (refine && data.refinement_type != RefinementType::None)
      {
        if (data.refinement_type == RefinementType::Global)
          {
            triangulation->refine_global(data.n_refinements);
          }
        else // if (refinement_type == RefinementType::Deserialize)
          {
            // We deserialize only if the mesh is hexahedral: in the case of
            // tetrahedra, deserialization loads not only the refinement
            // information, but also the mesh itself. Since refinement is
            // currently not implemented for tets, we just skip the
            // deserialization, and do nothing.
            if (is_hex())
              // Enable autopartition so that the current mesh can be
              // deserialized in serial even though it was serialized in
              // parallel.
              utils::deserialize_mesh(
                data.filename_to_deserialize,
                *this,
                /* autopartition = */ triangulation->get_communicator() ==
                  MPI_COMM_SELF);
          }
      }
  }
} // namespace lifex::utils
