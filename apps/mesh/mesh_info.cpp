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

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/geometry/mesh_handler.hpp"
#include "source/geometry/mesh_info.hpp"

#include "source/io/data_writer.hpp"

#include <fstream>
#include <memory>

namespace
{
  using namespace lifex;

  /// This class imports a mesh and prints its geometrical information to
  /// output.
  class MeshInfoApp : public CoreModel
  {
  public:
    /// Constructor.
    MeshInfoApp(const std::string &subsection)
      : CoreModel(subsection)
      , triangulation(std::make_unique<utils::MeshHandler>(
          prm_subsection_path,
          mpi_comm,
          std::initializer_list<utils::MeshHandler::GeometryType>(
            {utils::MeshHandler::GeometryType::File,
             utils::MeshHandler::GeometryType::Hypercube,
             utils::MeshHandler::GeometryType::Cylinder,
             utils::MeshHandler::GeometryType::ChannelWithCylinder})))
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      triangulation->declare_parameters(params);

      params.enter_subsection_path(prm_subsection_path);
      {
        params.declare_entry(
          "Enable output of mesh diameters",
          "false",
          Patterns::Bool(),
          "Export a CSV file containing the diameter of each mesh cell.");

        params.declare_entry("Mesh diameters filename",
                             "mesh_diameters.csv",
                             Patterns::FileName(
                               Patterns::FileName::FileType::output));

        params.declare_entry(
          "Enable output of mesh material IDs",
          "false",
          Patterns::Bool(),
          "Export a HDF5 file containing the material ID of each mesh cell.");

        params.declare_entry("Material IDs basename",
                             "material_ids",
                             Patterns::FileName(
                               Patterns::FileName::FileType::output));
      }
      params.leave_subsection_path();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.parse();

      triangulation->parse_parameters(params);

      params.enter_subsection_path(prm_subsection_path);
      {
        prm_enable_output_diameters =
          params.get_bool("Enable output of mesh diameters");
        prm_output_file_diameters = params.get("Mesh diameters filename");

        if (prm_enable_output_diameters)
          {
            AssertThrow(!prm_output_file_diameters.empty(),
                        ExcMessage("Filename cannot be empty."));
          }


        prm_enable_output_material_ids =
          params.get_bool("Enable output of mesh material IDs");
        prm_output_file_material_ids = params.get("Material IDs basename");

        if (prm_enable_output_material_ids)
          {
            AssertThrow(!prm_output_file_material_ids.empty(),
                        ExcMessage("Filename cannot be empty."));
          }
      }
      params.leave_subsection_path();
    }

    /// Run mesh info printer.
    virtual void
    run() override
    {
      triangulation->create_mesh();

      triangulation->get_info().print(prm_subsection_path, "", true);

      if (prm_enable_output_diameters)
        {
          triangulation->get_info().save_diameters(prm_output_file_diameters);
        }

      if (prm_enable_output_material_ids)
        {
          std::unique_ptr<FiniteElement<dim>> fe =
            triangulation->get_fe_dg(triangulation->is_hex() ? 0 : 1);

          DoFHandler<dim> dof_handler;
          dof_handler.reinit(triangulation->get());
          dof_handler.distribute_dofs(*fe);


          LinAlg::MPI::Vector material_ids(dof_handler.locally_owned_dofs(),
                                           mpi_comm);

          std::vector<types::global_dof_index> dof_indices(fe->dofs_per_cell);

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(dof_indices);

                  // Assign the same material ID to all dofs on current cell
                  // (useful, e.g., when exporting higher-order elements).
                  for (unsigned int i = 0; i < dof_indices.size(); ++i)
                    material_ids[dof_indices[i]] = cell->material_id();
                }
            }

          material_ids.compress(VectorOperation::insert);

          DataOut<dim> data_out;
          data_out.add_data_vector(dof_handler, material_ids, "material_ids");
          data_out.build_patches();

          // Disable filter_duplicate_vertices for proper exporting
          // of cellwise data.
          utils::dataout_write_hdf5(data_out,
                                    prm_output_file_material_ids,
                                    false);

          data_out.clear();
        }
    }

  private:
    /// Triangulation.
    std::unique_ptr<utils::MeshHandler> triangulation;

    /// Enable output of mesh diameters.
    bool prm_enable_output_diameters;
    /// Output filename for mesh diameters.
    std::string prm_output_file_diameters;

    /// Enable output of mesh material IDs.
    bool prm_enable_output_material_ids;
    /// Output filename for mesh material IDs.
    std::string prm_output_file_material_ids;
  };
} // namespace

/// Import a mesh and print its geometrical information to output.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      MeshInfoApp app("Mesh info");

      app.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
