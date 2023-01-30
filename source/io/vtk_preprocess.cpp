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
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 * @author Marco Fedele <marco.fedele@polimi.it>.
 */

#include "source/geometry/move_mesh.hpp"

#include "source/io/serialization.hpp"
#include "source/io/vtk_preprocess.hpp"

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_out.h>

#include <vtkXMLPolyDataWriter.h>

#include <memory>
#include <vector>

namespace lifex::utils
{
  VTKPreprocess::VTKPreprocess(const std::string &subsection)
    : CoreModel(subsection)
    , triangulation(prm_subsection_path, mpi_comm)
  {}

  void
  VTKPreprocess::run()
  {
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path + " / Read mesh");

      pcout << "Create the mesh..." << std::endl;
      triangulation.create_mesh();
    }

    unsigned int dimension = prm_vtk_data_is_vectorial ? dim : 1;

    std::unique_ptr<FiniteElement<dim>> fe_scalar;

    if (prm_fe_degree > 0)
      {
        fe_scalar = triangulation.get_fe_lagrange(prm_fe_degree);
      }
    else // if (prm_fe_degree == 0)
      {
        fe_scalar = triangulation.get_fe_dg(prm_fe_degree);
      }

    std::unique_ptr<FESystem<dim>> fe =
      std::make_unique<FESystem<dim>>(*fe_scalar, dimension);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);

    // Define FE vector.
    LinAlg::MPI::Vector fe_vector;       // FE vector.
    LinAlg::MPI::Vector fe_vector_owned; // FE vector, without ghost entries.
    IndexSet            owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet            relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
    fe_vector.reinit(owned_dofs, relevant_dofs, mpi_comm);
    fe_vector_owned.reinit(owned_dofs, mpi_comm);

    // Interpolate VTK file and store the results.
    std::unique_ptr<VTKFunction> vtk_interpolant;
    {
      // Monitor read and interpolate VTK file.
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Read and interpolate VTK file");

      pcout << "Reading and interpolating VTK file..." << std::endl;

      vtk_interpolant =
        std::make_unique<VTKFunction>(prm_vtk_filename,
                                      prm_vtk_datatype,
                                      prm_vtk_array_scaling_factor,
                                      prm_vtk_data_is_vectorial,
                                      prm_vtk_geometry_scaling_factor);

      if (prm_mode == VTKFunction::Mode::ClosestPointProjection)
        {
          vtk_interpolant->setup_as_closest_point_projection(
            prm_vtk_arrayname, prm_vtk_arraydatatype);
        }
      else if (prm_mode == VTKFunction::Mode::LinearProjection)
        {
          vtk_interpolant->setup_as_linear_projection(prm_vtk_arrayname);
        }
      else if (prm_mode == VTKFunction::Mode::SignedDistance)
        {
          vtk_interpolant->setup_as_signed_distance();
        }

      VectorTools::interpolate(dof_handler, *vtk_interpolant, fe_vector_owned);

      if (prm_mode == VTKFunction::Mode::LinearProjection)
        {
          pcout << "\t percentage of points where linear interpolation "
                   "correction is active: "
                << 100 *
                     static_cast<double>(
                       vtk_interpolant->get_counter_lin_int_correction()) /
                     vtk_interpolant->get_counter_value_calls()
                << "%" << std::endl;
        }

      fe_vector = fe_vector_owned;
    }

    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Store results and serialize");

      pcout << "Saving results..." << std::endl;

      DataOut<dim> data_out;

      std::vector<std::string> vector_names(prm_vtk_data_is_vectorial ? dim : 1,
                                            "processed_vector");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          prm_vtk_data_is_vectorial ? dim : 1,
          prm_vtk_data_is_vectorial ?
            DataComponentInterpretation::component_is_part_of_vector :
            DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector(dof_handler,
                               fe_vector,
                               vector_names,
                               data_component_interpretation);
      data_out.build_patches();
      data_out.write_vtu_in_parallel(prm_output_directory +
                                       prm_output_filename + ".vtu",
                                     mpi_comm);
      data_out.clear();

      utils::serialize(prm_output_filename,
                       fe_vector,
                       triangulation,
                       dof_handler);
    }

    if (prm_VTK_move_mesh)
      {
        AssertThrow(
          prm_vtk_data_is_vectorial &&
            prm_vtk_arraydatatype == VTKArrayDataType::PointData,
          ExcMessage("move_mesh can only be used with a vectorial PointData."));

        // Monitor move mesh on initial mesh according to fe_vector
        // and storage of .msh file containing the moved mesh file.
        // Note: these operations must be serial!
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Move mesh and store .msh file");

        const MPI_Comm mpi_comm_serial = MPI_COMM_SELF;

        // Load the same triangulation as the original mesh.
        utils::MeshHandler triangulation_serial(prm_subsection_path,
                                                mpi_comm_serial);

        triangulation_serial.initialize_from_file(
          triangulation.get_filename(), triangulation.get_scaling_factor());
        triangulation_serial.set_element_type(triangulation.get_element_type());

        triangulation_serial.set_refinement_from_file(prm_output_directory +
                                                      prm_output_filename);
        triangulation_serial.create_mesh(true);

        if (mpi_rank == 0)
          {
            // Create dof_handler.
            DoFHandler<dim> dof_handler_serial(triangulation_serial.get());
            dof_handler_serial.distribute_dofs(*fe);

            LinAlg::MPI::Vector d_serial;

            IndexSet owned_dofs_serial =
              dof_handler_serial.locally_owned_dofs();

            // Deserialize d_serial.
            d_serial.reinit(owned_dofs_serial, mpi_comm_serial);

            utils::deserialize(prm_output_filename,
                               d_serial,
                               triangulation_serial,
                               dof_handler_serial);

            // Move initial mesh with d_serial and save it.
            utils::move_mesh(triangulation_serial,
                             dof_handler_serial,
                             d_serial);

            GridOut           grid_out;
            GridOutFlags::Msh msh_flags(true, true);
            grid_out.set_flags(msh_flags);

            const std::string filename_msh =
              prm_output_directory + prm_output_filename + ".msh";

            std::ofstream output_mesh(filename_msh);
            grid_out.write_msh(triangulation_serial.get(), output_mesh);
          }

        MPI_Barrier(mpi_comm);
      }

    if (prm_VTK_move_input_surface)
      {
        AssertThrow(
          prm_vtk_data_is_vectorial &&
            prm_vtk_datatype == VTKDataType::PolyData &&
            prm_vtk_arraydatatype == VTKArrayDataType::PointData,
          ExcMessage(
            "VTKPreprocess: move input surface can only be used with a "
            "vectorial field defined as PointData on a PolyData."));

        // Monitor move input surface on according to vtk_vector
        // and storage of .vtp file containing the moved surface file.
        // Note: PolyData writing must be serial!
        TimerOutput::Scope timer_section(
          timer_output,
          prm_subsection_path + " / Move input surface and store .vtp file");

        // Move input surface.
        vtk_interpolant->warp_by_input_array();

        if (mpi_rank == 0)
          {
            // Save moved surface.
            vtkSmartPointer<vtkXMLPolyDataWriter> writer =
              vtkSmartPointer<vtkXMLPolyDataWriter>::New();
            std::string vtp_output_filename =
              prm_output_directory + prm_output_filename + ".vtp";
            writer->SetFileName(vtp_output_filename.c_str());
            writer->SetInputData(vtk_interpolant->get_vtk_surface());
            writer->Write();
          }
      }
  }

  void
  VTKPreprocess::declare_parameters(ParamHandler &params) const
  {
    // Declare parameters.
    triangulation.declare_parameters(params);

    params.enter_subsection_path(prm_subsection_path);
    params.enter_subsection("Mesh and space discretization");
    {
      params.declare_entry("FE space degree",
                           "1",
                           Patterns::Integer(0),
                           "Degree of the FE space.");
    }
    params.leave_subsection();

    params.enter_subsection("VTK processing");
    {
      params.declare_entry("VTK filename",
                           "",
                           Patterns::FileName(
                             Patterns::FileName::FileType::input),
                           "VTK filename.");

      params.declare_entry_selection(
        "Mode",
        "Closest point",
        "Closest point | Linear projection | Signed distance");

      params.declare_entry("VTK array name",
                           "",
                           Patterns::FileName(
                             Patterns::FileName::FileType::input));

      params.declare_entry("Geometry scaling factor",
                           "1.0",
                           Patterns::Double());

      params.declare_entry("VTK array scaling factor", "1", Patterns::Double());

      params.declare_entry("VTK data is vectorial", "false", Patterns::Bool());

      params.declare_entry("VTK move mesh",
                           "false",
                           Patterns::Bool(),
                           "VTK move mesh.");

      params.declare_entry("VTK move input surface",
                           "false",
                           Patterns::Bool(),
                           "VTK move input surface.");

      params.declare_entry_selection("VTK data type",
                                     "UnstructuredGrid",
                                     "UnstructuredGrid | PolyData");

      params.declare_entry_selection("VTK array data type",
                                     "PointData",
                                     "PointData | CellData");

      params.declare_entry("Output filename",
                           "",
                           Patterns::FileName(
                             Patterns::FileName::FileType::input));
    }
    params.leave_subsection();
    params.leave_subsection_path();
  }

  void
  VTKPreprocess::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    triangulation.parse_parameters(params);

    params.enter_subsection_path(prm_subsection_path);
    params.enter_subsection("Mesh and space discretization");
    {
      prm_fe_degree = params.get_integer("FE space degree");
    }
    params.leave_subsection();

    params.enter_subsection("VTK processing");
    {
      prm_vtk_filename    = params.get("VTK filename");
      prm_vtk_arrayname   = params.get("VTK array name");
      prm_output_filename = params.get("Output filename");

      prm_vtk_geometry_scaling_factor =
        params.get_double("Geometry scaling factor");
      prm_vtk_array_scaling_factor =
        params.get_double("VTK array scaling factor");

      prm_vtk_data_is_vectorial = params.get_bool("VTK data is vectorial");

      prm_VTK_move_mesh = params.get_bool("VTK move mesh");

      prm_VTK_move_input_surface = params.get_bool("VTK move input surface");

      if (params.get("VTK data type") == "UnstructuredGrid")
        {
          prm_vtk_datatype = VTKDataType::UnstructuredGrid;
        }
      else if (params.get("VTK data type") == "PolyData")
        {
          prm_vtk_datatype = VTKDataType::PolyData;
        }

      if (params.get("VTK array data type") == "PointData")
        {
          prm_vtk_arraydatatype = VTKArrayDataType::PointData;
        }
      else if (params.get("VTK array data type") == "CellData")
        {
          prm_vtk_arraydatatype = VTKArrayDataType::CellData;
        }

      if (params.get("Mode") == "Closest point")
        {
          prm_mode = VTKFunction::Mode::ClosestPointProjection;
        }
      else if (params.get("Mode") == "Linear projection")
        {
          prm_mode = VTKFunction::Mode::LinearProjection;
        }
      else if (params.get("Mode") == "Signed distance")
        {
          prm_mode = VTKFunction::Mode::SignedDistance;
        }
    }
    params.leave_subsection();
    params.leave_subsection_path();
  }
} // namespace lifex::utils
