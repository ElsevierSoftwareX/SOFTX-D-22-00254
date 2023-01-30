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
#include "source/numerics/non_linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/time_handler.hpp"
#include "source/numerics/tools.hpp"

#include <memory>
#include <vector>

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
        return this->get_time() * std::exp(p[0] + p[1] + p[2]);
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
        return (1 - 3 * this->get_time()) * std::exp(p[0] + p[1] + p[2]) +
               this->get_time() * this->get_time() *
                 std::exp(2 * (p[0] + p[1] + p[2]));
      }
    };
  } // namespace

  /**
   * @brief Prototype solver class for a time-dependent non-linear PDE, linearized using
   * automatic differentiation.
   *
   * This class solves the same problem as @ref Tutorial04,
   * but here the Jacobian matrix is assembled via automatic differentiation.
   *
   * @note To see all the parameters, generate parameter file
   * with the <kbd>-g full</kbd> flag, as explained in @ref run.
   */
  class Tutorial04_AD : public CoreModel
  {
  public:
    /// Constructor.
    Tutorial04_AD()
      : CoreModel("Tutorial 04")
      , timestep_number(0)
      , triangulation(prm_subsection_path, mpi_comm)
      , non_linear_solver(prm_subsection_path + " / Non-linear solver")
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

      params.enter_subsection("Time solver");
      {
        params.declare_entry("Initial time",
                             "0",
                             Patterns::Double(),
                             "Initial time.");

        params.declare_entry("Final time",
                             "1",
                             Patterns::Double(),
                             "Final time.");

        params.declare_entry("Time step",
                             "1e-1",
                             Patterns::Double(0),
                             "Time step.");

        params.set_verbosity(VerbosityParam::Standard);
        params.declare_entry("BDF order",
                             "3",
                             Patterns::Integer(1, 3),
                             "BDF order: 1, 2, 3.");
        params.reset_verbosity();
      }
      params.leave_subsection();

      non_linear_solver.declare_parameters(params);
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

      params.enter_subsection("Time solver");
      {
        prm_time_init  = params.get_double("Initial time");
        prm_time_final = params.get_double("Final time");
        prm_time_step  = params.get_double("Time step");
        prm_bdf_order  = params.get_integer("BDF order");
      }
      params.leave_subsection();

      non_linear_solver.parse_parameters(params);
      linear_solver.parse_parameters(params);
      preconditioner.parse_parameters(params);
    }

    /// Run simulation.
    virtual void
    run() override
    {
      create_mesh();
      setup_system();
      output_results();

      auto assemble_fun = [this](const bool &assemble_jac) {
        assemble_system(assemble_jac);

        return res.l2_norm();
      };

      auto solve_fun = [this](const bool &         assemble_prec,
                              LinAlg::MPI::Vector &incr) {
        const double norm_sol = solution_owned.l2_norm();

        solve_system(assemble_prec, incr);

        const double norm_incr = incr.l2_norm();

        const unsigned int n_iterations_linear =
          linear_solver.get_n_iterations();

        return std::make_tuple(norm_sol, norm_incr, n_iterations_linear);
      };

      LinAlg::MPI::Vector error_owned;
      error_owned.reinit(solution_owned);

      while (time < prm_time_final)
        {
          time += prm_time_step;
          ++timestep_number;

          pcout << "Time step " << std::setw(6) << timestep_number
                << " at t = " << std::setw(8) << std::fixed
                << std::setprecision(6) << time;

          bdf_handler.time_advance(solution_owned, true);

          u_ex->set_time(time);
          f_ex.set_time(time);
          bc_handler.set_time(time, false);

          VectorTools::interpolate(dof_handler, *u_ex, solution_ex_owned);
          solution_ex = solution_ex_owned;

          // Copy into ghosted vectors.
          solution_bdf = bdf_handler.get_sol_bdf();
          solution_ext = bdf_handler.get_sol_extrapolation();

          // Initial guess.
          solution = solution_owned = solution_ext;
          bc_handler.apply_dirichlet(solution_owned,
                                     solution,
                                     timestep_number == 1);

          const bool converged =
            non_linear_solver.solve(assemble_fun, solve_fun);
          AssertThrow(converged, ExcNonlinearNotConverged());

          output_results();

          error_owned = solution_owned;
          error_owned -= solution_ex_owned;
          pcout << "\tL-inf error norm: " << error_owned.linfty_norm()
                << std::endl
                << std::endl;
        }
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

      // Setup BCs: we start from an initial guess having the correct Dirichlet
      // values and impose homogeneous BCs on the Newton increment.
      std::vector<utils::BC<utils::FunctionDirichlet>> bcs_dirichlet;
      for (size_t i = 0; i < triangulation.n_faces_per_cell(); ++i)
        bcs_dirichlet.emplace_back(i, u_ex);

      bc_handler.initialize(bcs_dirichlet);

      bc_handler.apply_dirichlet(constraints_dirichlet, true);
      constraints_dirichlet.close();

      DynamicSparsityPattern dsp(relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_dirichlet);

      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 mpi_comm,
                                                 relevant_dofs);

      utils::initialize_matrix(jac, owned_dofs, dsp);

      res.reinit(owned_dofs, mpi_comm);

      solution_owned.reinit(owned_dofs, mpi_comm);
      solution.reinit(owned_dofs, relevant_dofs, mpi_comm);

      solution_ex_owned.reinit(owned_dofs, mpi_comm);
      solution_ex.reinit(owned_dofs, relevant_dofs, mpi_comm);

      solution_bdf.reinit(owned_dofs, relevant_dofs, mpi_comm);
      solution_ext.reinit(owned_dofs, relevant_dofs, mpi_comm);

      non_linear_solver.initialize(&solution_owned, &solution);

      // Initialize BDF handler.
      time = prm_time_init;
      u_ex->set_time(time);
      VectorTools::interpolate(dof_handler, *u_ex, solution_ex_owned);
      solution_ex = solution_ex_owned;
      solution = solution_owned = solution_ex_owned;

      const std::vector<LinAlg::MPI::Vector> sol_init(prm_bdf_order,
                                                      solution_owned);
      bdf_handler.initialize(prm_bdf_order, sol_init);
    }

    /// Assemble linear system.
    void
    assemble_system(const bool &assemble_jac)
    {
      if (assemble_jac)
        {
          jac = 0;
        }
      res = 0;

      FEValues<dim> fe_values(*fe,
                              *quadrature_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

      const unsigned int dofs_per_cell = fe->dofs_per_cell;
      const unsigned int n_q_points    = quadrature_formula->size();

      FullMatrix<double> cell_jac(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_res(dofs_per_cell);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      // Automatic differentiation.
      const unsigned int n_independent_variables = dof_indices.size();
      const unsigned int n_dependent_variables   = dof_indices.size();

      std::vector<double_AD> u_dof_AD(dofs_per_cell);
      std::vector<double_AD> cell_res_AD(dofs_per_cell);

      // BDF quantities.
      const double &alpha_bdf = bdf_handler.get_alpha();

      std::vector<double_AD> u_loc(n_q_points);
      std::vector<double>    u_bdf_loc(n_q_points);

      std::vector<Tensor<1, dim, double_AD>> grad_u_loc(n_q_points);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              // Helper for automatic differentiation.
              ADHelper ad_helper(n_independent_variables,
                                 n_dependent_variables);

              cell->get_dof_indices(dof_indices);
              fe_values.reinit(cell);

              // Prepare for automatic differentiation.
              ad_helper.register_dof_values(solution, dof_indices);
              u_dof_AD = ad_helper.get_sensitive_dof_values();

              if (assemble_jac)
                {
                  cell_jac = 0;
                }
              cell_res = 0;

              std::fill(cell_res_AD.begin(), cell_res_AD.end(), double_AD(0));
              std::fill(u_loc.begin(), u_loc.end(), double_AD(0));
              std::fill(u_bdf_loc.begin(), u_bdf_loc.end(), 0);

              std::fill(grad_u_loc.begin(),
                        grad_u_loc.end(),
                        Tensor<1, dim, double_AD>());

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      u_loc[q] += u_dof_AD[i] * fe_values.shape_value(i, q);

                      u_bdf_loc[q] += solution_bdf[dof_indices[i]] *
                                      fe_values.shape_value(i, q);

                      for (unsigned int d = 0; d < dim; ++d)
                        grad_u_loc[q][d] +=
                          u_dof_AD[i] * fe_values.shape_grad(i, q)[d];
                    }
                }

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      cell_res_AD[i] +=
                        (alpha_bdf * u_loc[q] - u_bdf_loc[q]) / prm_time_step *
                        fe_values.shape_value(i, q) * fe_values.JxW(q);

                      cell_res_AD[i] += grad_u_loc[q] *
                                        fe_values.shape_grad(i, q) *
                                        fe_values.JxW(q);

                      cell_res_AD[i] += u_loc[q] * u_loc[q] *
                                        fe_values.shape_value(i, q) *
                                        fe_values.JxW(q);

                      cell_res_AD[i] -=
                        f_ex.value(fe_values.quadrature_point(q)) *
                        fe_values.shape_value(i, q) * fe_values.JxW(q);
                    }
                }

              ad_helper.register_residual_vector(cell_res_AD);
              ad_helper.compute_residual(cell_res);

              if (assemble_jac)
                {
                  ad_helper.compute_linearization(cell_jac);

                  constraints_dirichlet.distribute_local_to_global(
                    cell_jac, cell_res, dof_indices, jac, res);
                }
              else
                {
                  constraints_dirichlet.distribute_local_to_global(cell_res,
                                                                   dof_indices,
                                                                   res);
                }
            }
        }

      jac.compress(VectorOperation::add);
      res.compress(VectorOperation::add);
    }

    /// Solve linear system.
    void
    solve_system(const bool &assemble_prec, LinAlg::MPI::Vector &incr)
    {
      if (assemble_prec)
        preconditioner.initialize(jac);

      linear_solver.solve(jac, incr, res, preconditioner);
      constraints_dirichlet.distribute(incr);
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

      utils::dataout_write_hdf5(data_out, "solution", timestep_number, 0, time);

      data_out.clear();
    }

    /// Number of mesh refinements.
    unsigned int prm_n_refinements;

    /// FE space degree.
    unsigned int prm_fe_degree;

    /// Initial time.
    double prm_time_init;

    /// Final time.
    double prm_time_final;

    /// Time step.
    double prm_time_step;

    /// BDF order.
    unsigned int prm_bdf_order;

    /// Current time.
    double time;

    /// Timestep number.
    unsigned int timestep_number;

    /// Triangulation.
    utils::MeshHandler triangulation;

    /// FE space.
    std::unique_ptr<FiniteElement<dim>> fe;

    /// Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature_formula;

    /// DoFHandler.
    DoFHandler<dim> dof_handler;

    /// BDF time advancing handler.
    utils::BDFHandler<LinAlg::MPI::Vector> bdf_handler;

    /// Non-linear solver handler.
    utils::NonLinearSolverHandler<LinAlg::MPI::Vector> non_linear_solver;

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::Vector> linear_solver;

    /// Preconditioner handler.
    utils::PreconditionerHandler preconditioner;

    /// Boundary condition handler.
    utils::BCHandler bc_handler;
    /// Dirichlet BCs constraints.
    AffineConstraints<double> constraints_dirichlet;

    /// Distributed system jacobian matrix.
    LinAlg::MPI::SparseMatrix jac;
    /// Distributed system residual vector.
    LinAlg::MPI::Vector res;
    /// Distributed solution vector, without ghost entries.
    LinAlg::MPI::Vector solution_owned;
    /// Distributed solution vector, with ghost entries.
    LinAlg::MPI::Vector solution;

    /// BDF solution, with ghost entries.
    LinAlg::MPI::Vector solution_bdf;
    /// BDF extrapolated solution, with ghost entries.
    LinAlg::MPI::Vector solution_ext;

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

/// Run tutorial 04 with automatic differentiation.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tutorials::Tutorial04_AD tutorial;

      tutorial.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
