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

#include "source/init.hpp"
#include "source/quadrature_evaluation.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/data_writer.hpp"

#include "source/numerics/projection.hpp"

#include <memory>

namespace lifex::tests
{
  /**
   * @brief Auxiliary class to test @ref utils::ProjectionL2.
   *
   * The equation solved is:
   * @f[ \left\{
   * \begin{alignedat}{2}
   * -\Delta u + u & = \left\|\textbf{x}\right\|, & \quad & \text{in } \Omega,
   * \\ \nabla u \cdot \mathbf{n} & = 0, & \quad & \text{on } \partial\Omega,
   * \end{alignedat}
   * \right.
   * @f]
   * where @f$\Omega = [0, 1]^3@f$.
   */
  class TestProjection : public Core
  {
  public:
    /// Class constructor.
    TestProjection(const std::string & subsection,
                   const unsigned int &fe_degree);

    /// Run test.
    void
    run();

  protected:
    /// Create mesh.
    void
    create_mesh();

    /// Setup system.
    void
    setup_system();

    /// Assemble matrix and rhs.
    void
    assemble_system();

    /// Solve problem.
    void
    solve();

    /// Compute solution projection and error.
    /// Since the original function belongs to the space of projection,
    /// the error should be zero.
    ///
    /// Also, project the gradient to the same solution space.
    void
    project();

    /// Output results.
    void
    output_results() const;

    unsigned int fe_degree; ///< FE degree.

    /// Triangulation.
    utils::MeshHandler triangulation;

    std::unique_ptr<FiniteElement<dim>> fe;          ///< FE space.
    DoFHandler<dim>                     dof_handler; ///< Dof handler.

    std::unique_ptr<FESystem<dim>>
                    fe_grad;          ///< FE space for gradient projection.
    DoFHandler<dim> dof_handler_grad; ///< Dof handler for gradient projection.

    std::unique_ptr<Quadrature<dim>>
      quadrature_formula; ///< Quadrature formula.

    LinAlg::MPI::SparseMatrix mat; ///< Problem matrix
    LinAlg::MPI::Vector       rhs; ///< Problem right hand side.

    LinAlg::MPI::Vector sol;       ///< Solution vector.
    LinAlg::MPI::Vector sol_owned; ///< Solution vector, without ghost entries.

    LinAlg::MPI::Vector sol_projected;       ///< Projected solution vector.
    LinAlg::MPI::Vector sol_projected_owned; ///< Projected solution vector,
                                             ///< without ghost entries.

    LinAlg::MPI::Vector grad_projected;       ///< Projected solution gradient.
    LinAlg::MPI::Vector grad_projected_owned; ///< Projected solution gradient,
                                              ///< without ghost entries.
  };

  TestProjection::TestProjection(const std::string & subsection,
                                 const unsigned int &fe_degree_)
    : fe_degree(fe_degree_)
    , triangulation(subsection, mpi_comm)
  {}

  void
  TestProjection::create_mesh()
  {
    triangulation.initialize_hypercube(-1, 1);
    triangulation.set_refinement_global(3);
    triangulation.create_mesh();
  }

  void
  TestProjection::setup_system()
  {
    // Create FE spaces and DoFHandler.
    fe = triangulation.get_fe_lagrange(fe_degree);

    fe_grad = std::make_unique<FESystem<dim>>(*fe, dim);

    quadrature_formula = triangulation.get_quadrature_gauss(fe_degree + 1);

    dof_handler.reinit(triangulation.get());
    dof_handler.distribute_dofs(*fe);

    triangulation.get_info().print("", dof_handler.n_dofs(), true);

    IndexSet owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

    DynamicSparsityPattern dsp(relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    AffineConstraints<double>(),
                                    false);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs,
                                               mpi_comm,
                                               relevant_dofs);

    mat.reinit(owned_dofs, owned_dofs, dsp, mpi_comm);
    rhs.reinit(owned_dofs, mpi_comm);

    sol.reinit(owned_dofs, relevant_dofs, mpi_comm);
    sol_owned.reinit(owned_dofs, mpi_comm);

    sol_projected.reinit(owned_dofs, relevant_dofs, mpi_comm);
    sol_projected_owned.reinit(owned_dofs, mpi_comm);

    // Gradient projection.
    dof_handler_grad.reinit(triangulation.get());
    dof_handler_grad.distribute_dofs(*fe_grad);

    IndexSet owned_dofs_grad = dof_handler_grad.locally_owned_dofs();
    IndexSet relevant_dofs_grad;
    DoFTools::extract_locally_relevant_dofs(dof_handler_grad,
                                            relevant_dofs_grad);

    grad_projected.reinit(owned_dofs_grad, relevant_dofs_grad, mpi_comm);
    grad_projected_owned.reinit(owned_dofs_grad, mpi_comm);
  }

  void
  TestProjection::assemble_system()
  {
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);

          cell_matrix = 0;
          cell_rhs    = 0;

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              Point<dim> point_q = fe_values.quadrature_point(q);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += (fe_values.shape_grad(i, q) *
                                              fe_values.shape_grad(j, q) +
                                            fe_values.shape_value(i, q) *
                                              fe_values.shape_value(j, q)) *
                                           fe_values.JxW(q);
                    }

                  cell_rhs(i) += point_q.norm() * fe_values.shape_value(i, q) *
                                 fe_values.JxW(q);
                }
            }

          mat.add(dof_indices, cell_matrix);
          rhs.add(dof_indices, cell_rhs);
        }

    mat.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  void
  TestProjection::solve()
  {
    SolverControl solver_control(sol.size(),
                                 1e-8 * rhs.l2_norm(),
                                 false,
                                 false);

    SolverCG<LinAlg::MPI::Vector> solver(solver_control);

    LinAlg::MPI::PreconditionJacobi preconditioner;
    preconditioner.initialize(mat);

    solver.solve(mat, sol_owned, rhs, preconditioner);
    sol = sol_owned;
  }

  void
  TestProjection::project()
  {
    // Project solution.
    utils::ProjectionL2 project_l2(dof_handler, *quadrature_formula, false);

    QuadratureFEMSolution quadrature_sol(sol, dof_handler, *quadrature_formula);

    project_l2.project<QuadratureEvaluationScalar>(quadrature_sol,
                                                   sol_projected_owned);
    sol_projected = sol_projected_owned;

    // Compute error.
    sol_owned -= sol_projected_owned;
    const double error = sol_owned.linfty_norm();

    pcout << "FE degree: " << fe_degree << ", error: " << error << std::endl;

    AssertThrow(error <= 1e-10, ExcTestFailed());

    // Project gradient. Also test regularization.
    utils::ProjectionL2 project_l2_grad(
      dof_handler_grad, *quadrature_formula, true, 0.5, 1);

    QuadratureFEMGradient quadrature_grad(sol,
                                          dof_handler,
                                          *quadrature_formula);

    project_l2_grad.project<QuadratureEvaluationVector>(quadrature_grad,
                                                        grad_projected_owned);

    grad_projected = grad_projected_owned;
  }

  void
  TestProjection::output_results() const
  {
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, sol, "u");
    data_out.add_data_vector(dof_handler, sol_projected, "u_projected");
    data_out.build_patches();

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(dof_handler_grad,
                             grad_projected,
                             std::vector<std::string>(dim, "grad_u_projected"),
                             data_component_interpretation);
    data_out.build_patches();

    utils::dataout_write_hdf5(data_out,
                              "solution_Q" + std::to_string(fe_degree));

    data_out.clear();
  }

  void
  TestProjection::run()
  {
    create_mesh();
    setup_system();
    assemble_system();
    solve();
    project();
    output_results();
  }
} // namespace lifex::tests

/// Test for @ref lifex::utils::ProjectionL2.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      for (unsigned int fe_degree = 1; fe_degree <= 2; ++fe_degree)
        {
          lifex::tests::TestProjection test_projection("Test projection",
                                                       fe_degree);

          test_projection.run();
        }
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
