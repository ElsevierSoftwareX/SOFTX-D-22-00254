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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 *
 * Micro-benchmark for VTKFunction::warp_by_fe_function
 */

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/vtk_function.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::examples
{
  /**
   * @brief Example class for the function @ref lifex::utils::VTKFunction::warp_by_pointwise_vectors.
   *
   * This serves both as a usage example for said function and as a test to
   * assess its efficiency. The function is called several times on dummy data.
   * See also the general documentation for @ref utils::VTKFunction.
   */
  class ExampleVTKWarpByFE : public CoreModel
  {
  public:
    /// Constructor.
    ExampleVTKWarpByFE(const std::string &subsection)
      : CoreModel(subsection)
      , triangulation(subsection,
                      mpi_comm,
                      {utils::MeshHandler::GeometryType::Cylinder})
    {}

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);
      params.declare_entry("Immersed surface path",
                           "../../mesh/surfaces/cylinder_plane_full.vtp",
                           Patterns::FileName(
                             Patterns::FileName::FileType::input));

      params.declare_entry("Number of warp calls",
                           "1000",
                           Patterns::Integer(0));
      params.leave_subsection_path();

      // Dependencies.
      triangulation.declare_parameters(params);
    }

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      // Parse input file.
      params.parse();

      // Read input parameters.
      params.enter_subsection_path(prm_subsection_path);
      prm_surface_path = params.get("Immersed surface path");
      prm_n_warp_calls = params.get_integer("Number of warp calls");
      params.leave_subsection_path();

      // Dependencies.
      triangulation.parse_parameters(params);
    }

    /// Run the example.
    virtual void
    run() override
    {
      setup_system();

      {
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Warp by FE function");

        for (unsigned int i = 0; i < prm_n_warp_calls; ++i)
          {
            surface->warp_by_pointwise_vectors(
              utils::VTKFunction::extract_nearest_neighbor_values(
                surface_closest_dofs, warp_vector),
              1.0);
          }
      }
    }

  private:
    /// Setup system.
    void
    setup_system()
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path + " / Setup system");
      // Create the mesh.
      triangulation.create_mesh();
      mapping = triangulation.get_linear_mapping();

      // Create the finite element field.
      auto fe_scalar = triangulation.get_fe_lagrange(1);
      fe             = std::make_unique<FESystem<dim>>(*fe_scalar, dim);

      // Setup the DoF handler.
      dof_handler.reinit(triangulation.get());
      dof_handler.distribute_dofs(*fe);

      // Print info on the triangulation and DoFs.
      triangulation.get_info().print(prm_subsection_path,
                                     dof_handler.n_dofs(),
                                     true);

      // Load the surface.
      surface =
        std::make_unique<utils::InterpolatedSignedDistance>(prm_surface_path,
                                                            dof_handler);

      // Create a finite element vector to warp the surface by. Since this
      // example has the purpose of showcasing the interface and testing
      // computational efficiency, we just warp by zero.
      IndexSet owned_dofs = dof_handler.locally_owned_dofs();
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

      warp_vector_owned.reinit(owned_dofs, mpi_comm);
      warp_vector.reinit(owned_dofs, relevant_dofs, mpi_comm);

      // Generate the support points maps, to be forwarded to
      // warp_by_fe_function.
      {
        support_points.resize(dim);

        for (unsigned int d = 0; d < dim; ++d)
          {
            ComponentMask mask(dim, false);
            mask.set(d, true);

            DoFTools::map_dofs_to_support_points(*mapping,
                                                 dof_handler,
                                                 support_points[d],
                                                 mask);
          }
      }

      // Generate the nearest-neighbor interpolation pattern by finding the
      // closest points.
      surface_closest_dofs =
        surface->find_closest_owned_dofs(dof_handler, support_points);
    }

    /// Triangulation.
    utils::MeshHandler triangulation;

    /// Mapping.
    std::unique_ptr<Mapping<dim>> mapping;

    /// Finite element space.
    std::unique_ptr<FESystem<dim>> fe;

    /// DoF handler.
    DoFHandler<dim> dof_handler;

    /// Interpolated distance to be warped.
    std::unique_ptr<utils::InterpolatedSignedDistance> surface;

    /// Vector used to warp, owned elements.
    LinAlg::MPI::Vector warp_vector_owned;

    /// Vector used to warp.
    LinAlg::MPI::Vector warp_vector;

    /// Support points.
    std::vector<std::map<types::global_dof_index, Point<dim>>> support_points;

    /// Surface closest DoFs.
    std::vector<std::vector<std::pair<types::global_dof_index, double>>>
      surface_closest_dofs;

    /// @name Parameters read from file.
    /// @{

    /// Filename for the surface.
    std::string prm_surface_path;

    /// Number of calls to warp_by_fe_function.
    unsigned int prm_n_warp_calls;

    /// @}
  };
} // namespace lifex::examples


/// Run the example.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::examples::ExampleVTKWarpByFE model("Example VTK warp by FE");
      model.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
