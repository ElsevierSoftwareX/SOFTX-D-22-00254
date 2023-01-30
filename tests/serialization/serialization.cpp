/********************************************************************************
  Copyright (C) 2021 - 2022 by the lifex authors.

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

#include "source/init.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/serialization.hpp"

#include <memory>

/// Namespace for all the tests in @lifex.
namespace lifex::tests
{
  /**
   * @brief Auxiliary class to test serialization.
   */
  class TestSerialization : public Function<dim>, public Core
  {
  public:
    /// Filename for serialization.
    static inline constexpr auto filename = "test_serialization";

    /// Filename for vector serialization.
    static inline constexpr auto filename_vector = "test_serialization_vector";

    /// Class constructor.
    TestSerialization(const std::string &subsection);

    /// Point evaluation of a test function.
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    /// Run test.
    void
    run();

  protected:
    /// Setup triangulation.
    void
    setup_mesh();

    /// Serialize vectors.
    void
    serialize();

    /// Deserialize vectors and compute errors.
    void
    deserialize();

    utils::MeshHandler triangulation; ///< Triangulation.

    std::unique_ptr<FiniteElement<dim>> fe; ///< FE space.

    LinAlg::MPI::Vector vec1_in; ///< First vector to serialize.
    LinAlg::MPI::Vector
      vec1_in_owned; ///< First vector to serialize, without ghost entries.

    LinAlg::MPI::Vector vec2_in; ///< Second vector to serialize.
    LinAlg::MPI::Vector
      vec2_in_owned; ///< Second vector to serialize, without ghost entries.

    LinAlg::MPI::Vector
      vec1_out_owned; ///< First vector to deserialize, without ghost entries.

    LinAlg::MPI::Vector
      vec2_out_owned; ///< Second vector to deserialize, without ghost entries.
  };

  TestSerialization::TestSerialization(const std::string &subsection)
    : triangulation(subsection, mpi_comm)
  {}

  double
  TestSerialization::value(const Point<dim> &p,
                           const unsigned int /*component*/) const
  {
    return p.norm();
  }

  void
  TestSerialization::setup_mesh()
  {
    triangulation.initialize_hypercube(-1, 1, true);
    triangulation.set_refinement_global(3);
  }

  void
  TestSerialization::serialize()
  {
    // Create triangulation.
    triangulation.create_mesh();

    // Create FE space and DoFHandler.
    fe = triangulation.get_fe_lagrange(2);

    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);

    triangulation.get_info().print("", dof_handler.n_dofs(), true);

    IndexSet owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

    // Initialize vectors.
    vec1_in.reinit(owned_dofs, relevant_dofs, mpi_comm);
    vec1_in_owned.reinit(owned_dofs, mpi_comm);

    vec2_in.reinit(owned_dofs, relevant_dofs, mpi_comm);
    vec2_in_owned.reinit(owned_dofs, mpi_comm);

    // Fill vectors.
    VectorTools::interpolate(dof_handler, *this, vec1_in_owned);
    vec1_in = vec1_in_owned;

    vec2_in_owned = 1e100;
    vec2_in       = vec2_in_owned;

    // Serialize vector.
    utils::serialize<LinAlg::MPI::Vector>(filename,
                                          vec1_in,
                                          triangulation,
                                          dof_handler);

    // Serialize vectors.
    utils::serialize<LinAlg::MPI::Vector>(filename_vector,
                                          {&vec1_in, &vec2_in},
                                          triangulation,
                                          dof_handler);
  }

  void
  TestSerialization::deserialize()
  {
    // Create coarse triangulation and load the de-serialized triangulation in
    // order to properly initialize the DoFHandler.
    triangulation.set_refinement_from_file(filename);
    triangulation.create_mesh();

    // Create FE space and DoFHandler.
    DoFHandler<dim> dof_handler;
    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);

    IndexSet owned_dofs = dof_handler.locally_owned_dofs();

    // Initialize vectors.
    vec1_out_owned.reinit(owned_dofs, mpi_comm);
    vec2_out_owned.reinit(owned_dofs, mpi_comm);

    {
      // Deserialize vector.
      utils::deserialize<LinAlg::MPI::Vector>(filename,
                                              vec1_out_owned,
                                              triangulation,
                                              dof_handler);

      // Compute error. The two vectors cannot be subtracted
      // because their parallel partitioning may be different
      // (triangulation.load() may change the mesh partitioning).
      const double error =
        vec1_in_owned.linfty_norm() - vec1_out_owned.linfty_norm();

      pcout << "Error on vec1: " << error << std::endl;

      AssertThrow(vec1_out_owned.linfty_norm() != 0 && error == 0,
                  ExcTestFailed());
    }

    {
      // Deserialize vectors.
      std::vector<LinAlg::MPI::Vector *> vec_out(
        {&vec1_out_owned, &vec2_out_owned});

      utils::deserialize<LinAlg::MPI::Vector>(filename_vector,
                                              vec_out,
                                              triangulation,
                                              dof_handler);

      // Compute errors. The two vectors cannot be subtracted
      // because their parallel partitioning may be different
      // (triangulation.load() may change the mesh partitioning).
      const double error1 =
        vec1_in_owned.linfty_norm() - vec1_out_owned.linfty_norm();
      const double error2 =
        vec2_in_owned.linfty_norm() - vec2_out_owned.linfty_norm();

      pcout << "Error on vec1 (vector): " << error1 << std::endl;
      pcout << "Error on vec2 (vector): " << error2 << std::endl;

      AssertThrow(vec1_out_owned.linfty_norm() != 0 && error1 == 0,
                  ExcTestFailed());
      AssertThrow(vec2_out_owned.linfty_norm() != 0 && error2 == 0,
                  ExcTestFailed());
    }
  }

  void
  TestSerialization::run()
  {
    setup_mesh();
    serialize();
    deserialize();
  }
} // namespace lifex::tests

/// Test for serialization.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tests::TestSerialization test("Test serialization");

      test.run();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
