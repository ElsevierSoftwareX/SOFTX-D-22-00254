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
 *
 * The function @ref lifex::utils::move_mesh is tested against
 * both a ghosted and a non-ghosted displacement vector.
 *
 * A check is performed on the moved mesh, regarding the max of the vertices
 * norms.
 */

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/geometry/move_mesh.hpp"

#include "source/io/data_writer.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/grid/grid_tools.h>

namespace lifex::tests
{
  /// @brief Test class for the function @ref lifex::utils::move_mesh.
  ///
  /// The function is tested against both a ghosted and a non-ghosted
  /// displacement vector. A check is performed on the moved mesh, regarding the
  /// max of the vertices norms.
  class TestMoveMesh : public CoreModel
  {
  public:
    /// Displacement by which the mesh is moved.
    class Displacement : public Function<dim>
    {
    public:
      /// Default constructor (vectorial function).
      Displacement()
        : Function<dim>(dim)
      {}

      /// Value of the displacement function.
      virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override
      {
        for (unsigned int component = 0; component < dim; ++component)
          {
            value[component] = this->get_time() * p[component];
          }
      }
    };

    /// Constructor.
    TestMoveMesh(const std::string &subsection)
      : CoreModel(subsection)
    {}

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);

      params.enter_subsection("Output");
      {
        params.declare_entry("Files basename",
                             "move_mesh",
                             Patterns::FileName(
                               Patterns::FileName::FileType::output),
                             "Output files basename.");
      }
      params.leave_subsection();

      params.leave_subsection_path();
    }

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      // Parse input file.
      params.parse();

      // Read input parameters.
      params.enter_subsection_path(prm_subsection_path);

      params.enter_subsection("Output");
      {
        prm_output_basename = params.get("Files basename");
      }
      params.leave_subsection();

      params.leave_subsection_path();
    }

    /// Run the test. Throw an exception if the test has not passed.
    virtual void
    run() override
    {
      const double tol = 1e-12;
      const double actual_edge_length =
        100 * 1e-3; // edge length * scaling factor.

      // Mesh to move.
      utils::MeshHandler triangulation(prm_subsection_path, mpi_comm);

      triangulation.initialize_hypercube(0, actual_edge_length, true);
      triangulation.set_refinement_global(3);
      triangulation.create_mesh();

      // FE space and DoF handler, for displacement field.
      const auto    fe_scalar = triangulation.get_fe_lagrange(1);
      FESystem<dim> fe(*fe_scalar, dim);

      DoFHandler<dim> dof_handler;
      dof_handler.reinit(triangulation.get());
      dof_handler.distribute_dofs(fe);

      triangulation.get_info().print(prm_subsection_path,
                                     dof_handler.n_dofs(),
                                     true);

      const auto    quadrature_formula = triangulation.get_quadrature_gauss(2);
      FEValues<dim> fe_values(fe,
                              *quadrature_formula,
                              update_values | update_JxW_values);

      // Displacement FE vector setup.
      IndexSet owned_dofs = dof_handler.locally_owned_dofs();
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
      LinAlg::MPI::Vector displacement, displacement_owned;
      displacement.reinit(owned_dofs, relevant_dofs, mpi_comm);
      displacement_owned.reinit(owned_dofs, mpi_comm);

      // Fictitious time, to save all outputs in the same file.
      double       time_step       = 1;
      double       time            = 0.0;
      unsigned int timestep_number = 0;

      // Initialization of the displacement.
      Displacement displacement_function;
      {
        displacement_function.set_time(time);
        VectorTools::interpolate(fe_values.get_mapping(),
                                 dof_handler,
                                 displacement_function,
                                 displacement_owned);
        displacement = displacement_owned;

        output_results(
          dof_handler, "displacement", displacement, time, timestep_number);

        ++timestep_number;
        time += time_step;
      }

      // Update of the displacement and mesh motion (by ghosted vector).
      {
        displacement_function.set_time(time);
        VectorTools::interpolate(fe_values.get_mapping(),
                                 dof_handler,
                                 displacement_function,
                                 displacement_owned);
        displacement = displacement_owned;

        utils::move_mesh(triangulation, dof_handler, displacement);

        output_results(
          dof_handler, "displacement", displacement, time, timestep_number);

        ++timestep_number;
        time += time_step;
      }

      AssertThrow(check_moved_mesh(triangulation,
                                   std::sqrt(3 * 2 * actual_edge_length * 2 *
                                             actual_edge_length),
                                   tol),
                  ExcTestFailed());

      // Update of the displacement and mesh motion (by ghosted vector).
      {
        displacement_function.set_time(time);
        VectorTools::interpolate(fe_values.get_mapping(),
                                 dof_handler,
                                 displacement_function,
                                 displacement_owned);
        displacement = displacement_owned;

        utils::move_mesh(triangulation, dof_handler, displacement_owned);

        output_results(
          dof_handler, "displacement", displacement, time, timestep_number);
      }

      AssertThrow(check_moved_mesh(triangulation,
                                   std::sqrt(3 * 6 * actual_edge_length * 6 *
                                             actual_edge_length),
                                   tol),
                  ExcTestFailed());
    }

    /// @brief Save displacement with correspondingly moved mesh.

    /// @param[in] dof_handler     DoFHandler.
    /// @param[in] var_name        Output variable name.
    /// @param[in] var             Output vector.
    /// @param[in] time            Fictitious time at which var is stored.
    /// @param[in] timestep_number Fictitious-time index used for output.
    ///
    void
    output_results(const DoFHandler<dim> &    dof_handler,
                   const std::string &        var_name,
                   const LinAlg::MPI::Vector &var,
                   const double &             time,
                   const unsigned int &       timestep_number)
    {
      DataOut<dim> data_out;

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);

      std::vector<std::string> var_names(dim, var_name);
      data_out.add_data_vector(dof_handler,
                               var,
                               var_names,
                               data_component_interpretation);

      data_out.build_patches();

      utils::dataout_write_hdf5(
        data_out, prm_output_basename, timestep_number, timestep_number, time);

      data_out.clear();
    }

    /// @brief Auxiliary function to check moved mesh.
    ///
    /// @param[in] triangulation_in    The mesh to check.
    /// @param[in] max_point_magnitude The max of the norms of the mesh points.
    /// @param[in] tol                 Tolerance used to check the moved mesh.
    bool
    check_moved_mesh(const utils::MeshHandler &triangulation_in,
                     const double &            max_point_magnitude,
                     const double &            tol)
    {
      double local_max_point_magnitude = 0;

      const std::vector<bool> vertex_owned(
        GridTools::get_locally_owned_vertices(triangulation_in.get()));

      // Compute max local to processor.
      for (Triangulation<dim>::active_vertex_iterator v =
             triangulation_in.get().begin_active_vertex();
           v != triangulation_in.get().end_vertex();
           ++v)
        {
          if (vertex_owned[v->vertex_index(0)])
            {
              local_max_point_magnitude =
                std::max(local_max_point_magnitude, v->vertex(0).norm());
            }
        }

      // Max over processors.
      return utils::is_zero(Utilities::MPI::max(local_max_point_magnitude,
                                                mpi_comm) -
                              max_point_magnitude,
                            tol);
    }

  private:
    std::string prm_output_basename; ///< Output basename.
  };
} // namespace lifex::tests


/// Test mesh motion.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tests::TestMoveMesh test("Test move mesh");

      test.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
