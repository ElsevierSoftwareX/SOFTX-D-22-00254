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

#include "source/geometry/mesh_handler.hpp"
#include "source/geometry/mesh_optimization.hpp"

#include "source/io/data_writer.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/fe/mapping_fe_field.h>

#include <random>

namespace lifex::tests
{
  /// Mesh optimization example.
  class TestMeshOptimization : public CoreModel
  {
  public:
    /// Constructor.
    TestMeshOptimization(const std::string &subsection)
      : CoreModel(subsection)
      , triangulation(subsection,
                      mpi_comm,
                      {utils::MeshHandler::GeometryType::File})
      , mesh_optimization(subsection + " / Mesh optimization")
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);
      {
        params.declare_entry("Noise amplitude", "0.01", Patterns::Double(0));
      }
      params.leave_subsection_path();

      // Dependencies.
      {
        triangulation.declare_parameters(params);
        mesh_optimization.declare_parameters(params);
      }
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.parse();

      params.enter_subsection_path(prm_subsection_path);
      {
        prm_noise_amplitude = params.get_double("Noise amplitude");
      }
      params.leave_subsection_path();

      // Dependencies.
      {
        triangulation.parse_parameters(params);
        mesh_optimization.parse_parameters(params);
      }
    }

    /// Run.
    virtual void
    run() override
    {
      setup_system();
      output_results(0);

      pcout << "\nQuality before noise:       "
            << mesh_optimization.compute_loss(x) << std::endl;

// This test is only run in release mode. The test starts from a mesh,
// introduces random perturbations of its nodes, inverting some of the elements,
// then runs a mesh optimization algorithm to repair them. In debug mode,
// deal.II throws an exception if an inverted element is encountered, preventing
// this test from being completed succesfully.
#ifndef BUILD_TYPE_DEBUG
      // Apply noise to the interior nodes of the mesh, to introduce distortion
      // and inverted elements.
      {
        IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(*dof_handler);

        // We use the position vector to create a random number generation seed,
        // to ensure reproducibility independently of DoF numbering and number
        // of processes used.
        std::hash<double> hash;
        for (const auto &idx : dof_handler->locally_owned_dofs())
          {
            std::default_random_engine             engine(hash(x_owned[idx]));
            std::uniform_real_distribution<double> rand(-1.0, 1.0);

            if (!boundary_dofs.is_element(idx))
              x_owned[idx] += prm_noise_amplitude * rand(engine);
          }

        x_owned.compress(VectorOperation::add);
        x = x_owned;
      }
#endif

      output_results(1);

      pcout << "Quality after noise:        "
            << mesh_optimization.compute_loss(x) << std::endl;

      pcout << "\nRunning NLCG for mesh untangling and optimization: "
            << std::endl;

      mesh_optimization.run(x_owned);
      x_owned += mesh_optimization.get_displacement_owned();
      x = x_owned;

      output_results(2);

      pcout << "\nQuality after NLCG:         "
            << mesh_optimization.compute_loss(x) << std::endl;

      // Retrieve mesh quality information.
      const utils::MeshInfo::MeshQualityInfo mesh_quality_info =
        triangulation.get_info().print_mesh_quality_info();

      AssertThrow(utils::is_positive(mesh_quality_info.jacobian_min),
                  ExcTestFailed());
    }

  protected:
    /// Setup system.
    void
    setup_system()
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path + " / Setup");

      triangulation.create_mesh();

      auto fe_scalar = triangulation.get_fe_lagrange(1);
      fe             = std::make_unique<FESystem<dim>>(*fe_scalar, dim);

      dof_handler = std::make_shared<DoFHandler<dim>>();
      dof_handler->reinit(triangulation.get());
      dof_handler->distribute_dofs(*fe);

      triangulation.get_info().print(prm_subsection_path,
                                     dof_handler->n_dofs(),
                                     true);

      // Initialize the position vectors.
      {
        IndexSet owned_dofs = dof_handler->locally_owned_dofs();
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler, relevant_dofs);

        x_owned.reinit(owned_dofs, mpi_comm);
        x.reinit(owned_dofs, relevant_dofs, mpi_comm);

        VectorTools::get_position_vector(*dof_handler, x_owned);
        x_owned.compress(VectorOperation::insert);
        x = x_owned;
      }

      mesh_optimization.setup_system(dof_handler);
    }

    /// Output results.
    void
    output_results(const unsigned int &n)
    {
      // Construct the mapping associated to the position vector.
      MappingFEField<dim, dim, LinAlg::MPI::Vector> mapping(*dof_handler, x);

      DataOut<dim> data_out;

      data_out.attach_triangulation(triangulation.get());
      data_out.build_patches(mapping);

      utils::dataout_write_hdf5(data_out, "meshopt", n, n, n);
    }

    /// Triangulation.
    utils::MeshHandler triangulation;

    /// Finite element space.
    std::unique_ptr<FESystem<dim>> fe;

    /// DoF handler.
    std::shared_ptr<DoFHandler<dim>> dof_handler;

    /// Mesh optimizer.
    utils::MeshOptimization mesh_optimization;

    /// Vector that stores the position vector, owned DoFs.
    LinAlg::MPI::Vector x_owned;

    /// Vector that stores the position vector.
    LinAlg::MPI::Vector x;

    /// @name Parameters read from file.
    /// @{

    /// Amplitude of the noise initially applied to the mesh.
    double prm_noise_amplitude;

    /// @}
  };
} // namespace lifex::tests

/// Run mesh optimization example.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tests::TestMeshOptimization test("Mesh optimization test");

      test.main_run_generate_from_json({"nondefault"});
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
