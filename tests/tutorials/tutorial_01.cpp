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

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/data_writer.hpp"

#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

#include <memory>
#include <vector>

/// @lifex tutorials.
namespace lifex::tutorials
{
  namespace
  {
    class ExactSolution : public utils::FunctionDirichlet
    {
    public:
      ExactSolution()
        : utils::FunctionDirichlet()
      {}

      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        return std::exp(p[0] + p[1] + p[2]);
      }
    };


    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>()
      {}

      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        return -3 * std::exp(p[0] + p[1] + p[2]);
      }
    };
  } // namespace

  /**
   * @brief Prototype solver class for a stationary linear PDE.
   *
   * The equation solved is:
   * @f[
   * \begin{aligned}
   * -\Delta u &= f, & \quad & \text{in } \Omega = (-1, 1)^3, \\
   * u &= u_\mathrm{ex}, & \quad & \text{on } \partial\Omega,
   * \end{aligned}
   * @f]
   * where @f$f@f$ is chosen such that the exact solution is
   * @f$u_\mathrm{ex} = e^{x+y+z}@f$.
   *
   * @note To see all the parameters, generate parameter file
   * with the <kbd>-g full</kbd> flag, as explained in @ref run.
   */
  class Tutorial01 : public CoreModel
  {
  public:
    /// Constructor.
    Tutorial01()
      : CoreModel("Tutorial 01")
      , triangulation(prm_subsection_path, mpi_comm)
      , linear_solver(prm_subsection_path + " / Linear solver",
                      {"CG", "GMRES", "BiCGStab"},
                      "GMRES")
      , preconditioner(prm_subsection_path + " / Preconditioner", true)
      , bc_handler(dof_handler)
      , u_ex(std::make_shared<ExactSolution>())
    {}

    /// Declare input parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection("Mesh and space discretization");
      {
        params.declare_entry(
          "Number of refinements",
          "4",
          Patterns::Integer(0),
          "Number of global mesh refinement steps applied to initial grid.");

        params.set_verbosity(VerbosityParam::Standard);
        params.declare_entry("FE space degree",
                             "1",
                             Patterns::Integer(1),
                             "Degree of the FE space.");
        params.reset_verbosity();
      }
      params.leave_subsection();

      linear_solver.declare_parameters(params);
      preconditioner.declare_parameters(params);
    }

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      // Parse input file.
      params.parse();

      // Read input parameters.
      params.enter_subsection("Mesh and space discretization");
      {
        prm_n_refinements = params.get_integer("Number of refinements");
        prm_fe_degree     = params.get_integer("FE space degree");
      }
      params.leave_subsection();

      linear_solver.parse_parameters(params);
      preconditioner.parse_parameters(params);
    }

    /// Run simulation.
    virtual void
    run() override
    {
      create_mesh();
      setup_system();

      VectorTools::interpolate(dof_handler, *u_ex, solution_ex_owned);
      solution_ex = solution_ex_owned;

      // Initial guess.
      solution = solution_owned = 0;

      assemble_system();
      solve_system();

      output_results();

      solution_owned -= solution_ex_owned;
      pcout << "L-inf error norm: " << solution_owned.linfty_norm()
            << std::endl;
    }

  private:
    /// Create mesh.
    void
    create_mesh()
    {
      triangulation.initialize_hypercube(-1, 1, true);
      triangulation.set_refinement_global(prm_n_refinements);
      triangulation.create_mesh();
    }

    /// Setup system, @a i.e. setup DoFHandler, allocate matrices and vectors
    /// and initialize handlers.
    void
    setup_system()
    {
      fe                 = triangulation.get_fe_lagrange(prm_fe_degree);
      quadrature_formula = triangulation.get_quadrature_gauss(fe->degree + 1);

      dof_handler.reinit(triangulation.get());
      dof_handler.distribute_dofs(*fe);

      triangulation.get_info().print(prm_subsection_path,
                                     dof_handler.n_dofs(),
                                     true);

      IndexSet owned_dofs = dof_handler.locally_owned_dofs();
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

      // Setup BCs.
      std::vector<utils::BC<utils::FunctionDirichlet>> bcs_dirichlet;
      for (size_t i = 0; i < triangulation.n_faces_per_cell(); ++i)
        bcs_dirichlet.emplace_back(i, u_ex);

      bc_handler.initialize(bcs_dirichlet);

      bc_handler.apply_dirichlet(constraints_dirichlet, false);
      constraints_dirichlet.close();

      DynamicSparsityPattern dsp(relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_dirichlet);

      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 mpi_comm,
                                                 relevant_dofs);

      utils::initialize_matrix(matrix, owned_dofs, dsp);

      rhs.reinit(owned_dofs, mpi_comm);

      solution_owned.reinit(owned_dofs, mpi_comm);
      solution.reinit(owned_dofs, relevant_dofs, mpi_comm);

      solution_ex_owned.reinit(owned_dofs, mpi_comm);
      solution_ex.reinit(owned_dofs, relevant_dofs, mpi_comm);
    }

    /// Assemble linear system.
    void
    assemble_system()
    {
      matrix = 0;
      rhs    = 0;

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
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              fe_values.reinit(cell);

              cell_matrix = 0;
              cell_rhs    = 0;

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                               fe_values.shape_grad(j, q) *
                                               fe_values.JxW(q);
                        }

                      cell_rhs(i) += f_ex.value(fe_values.quadrature_point(q)) *
                                     fe_values.shape_value(i, q) *
                                     fe_values.JxW(q);
                    }
                }

              constraints_dirichlet.distribute_local_to_global(
                cell_matrix, cell_rhs, dof_indices, matrix, rhs);
            }
        }

      matrix.compress(VectorOperation::add);
      rhs.compress(VectorOperation::add);
    }

    /// Solve linear system.
    void
    solve_system()
    {
      preconditioner.initialize(matrix);

      linear_solver.solve(matrix, solution_owned, rhs, preconditioner);
      constraints_dirichlet.distribute(solution_owned);

      solution = solution_owned;
    }

    /// Output results.
    void
    output_results()
    {
      DataOut<dim> data_out;

      // Solutions.
      data_out.add_data_vector(dof_handler, solution, "u");
      data_out.add_data_vector(dof_handler, solution_ex, "u_ex");
      data_out.build_patches();

      utils::dataout_write_hdf5(data_out, "solution");

      data_out.clear();
    }

    /// Number of mesh refinements.
    unsigned int prm_n_refinements;

    /// FE space degree.
    unsigned int prm_fe_degree;

    /// Triangulation.
    utils::MeshHandler triangulation;

    /// FE space.
    std::unique_ptr<FiniteElement<dim>> fe;

    /// Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature_formula;

    /// DoFHandler.
    DoFHandler<dim> dof_handler;

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::Vector> linear_solver;

    /// Preconditioner handler.
    utils::PreconditionerHandler preconditioner;

    /// Boundary condition handler.
    utils::BCHandler bc_handler;
    /// Dirichlet BCs constraints.
    AffineConstraints<double> constraints_dirichlet;

    /// Distributed system matrix.
    LinAlg::MPI::SparseMatrix matrix;
    /// Distributed system rhs vector.
    LinAlg::MPI::Vector rhs;
    /// Distributed solution vector, without ghost entries.
    LinAlg::MPI::Vector solution_owned;
    /// Distributed solution vector, with ghost entries.
    LinAlg::MPI::Vector solution;

    /// Distributed exact solution vector, without ghost entries.
    LinAlg::MPI::Vector solution_ex_owned;
    /// Distributed exact solution vector, without ghost entries.
    LinAlg::MPI::Vector solution_ex;

    /// Pointer to exact solution function.
    std::shared_ptr<ExactSolution> u_ex;

    /// Right hand side.
    RightHandSide f_ex;
  };
} // namespace lifex::tutorials

/// Run tutorial 01.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tutorials::Tutorial01 tutorial;

      tutorial.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
