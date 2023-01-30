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
 */

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/io/data_writer.hpp"

#include "source/numerics/laplace.hpp"

namespace lifex::examples
{
  /// @brief Example class for the Laplace solver.
  class ExampleLaplace : public CoreModel
  {
  public:
    /// Constructor.
    ExampleLaplace(const std::string &subsection)
      : CoreModel(subsection)
      , laplace(prm_subsection_path + " / Laplace")
    {}

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);
      params.enter_subsection("Output");
      {
        params.declare_entry("Output filename",
                             "solution",
                             Patterns::FileName(
                               Patterns::FileName::FileType::output),
                             "Output files basename.");
      }
      params.leave_subsection();
      params.leave_subsection_path();

      laplace.declare_parameters(params);
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
        prm_output_basename = params.get("Output filename");
      }
      params.leave_subsection();
      params.leave_subsection_path();

      laplace.parse_parameters(params);
    }

    /// Run the example.
    virtual void
    run() override
    {
      utils::MeshHandler triangulation(prm_subsection_path, mpi_comm);

      // Create the mesh.
      {
        triangulation.initialize_hypercube(0, 1, true);
        triangulation.set_refinement_global(3);
        triangulation.create_mesh();
      }

      // Setup and solve the Laplace problem.
      {
        laplace.setup_system(triangulation, 1);

        laplace.clear_bcs();
        laplace.apply_dirichlet_boundary(0, 0.0);
        laplace.apply_dirichlet_boundary(1, 1.0);
        laplace.apply_dirichlet_boundary(2, 2.0);

        laplace.solve();
      }

      // Output results.
      {
        DataOut<dim> data_out;
        laplace.attach_output(data_out, "solution");
        data_out.build_patches();
        utils::dataout_write_hdf5(data_out, prm_output_basename);
        data_out.clear();
      }
    }

  protected:
    /// Laplace problem.
    Laplace laplace;

    /// Output filename.
    std::string prm_output_basename;
  };
} // namespace lifex::examples


/// Example mesh motion.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::examples::ExampleLaplace example("Example Laplace");
      example.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
