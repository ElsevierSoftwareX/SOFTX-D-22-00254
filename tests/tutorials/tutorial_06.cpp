/********************************************************************************
  Copyright (C) 2022 by the lifex authors.

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

#include <cmath>
#include <memory>
#include <vector>

namespace lifex::tutorials
{
  namespace
  {
    class SolverU : public CoreModel
    {
    public:
      class ExactSolution : public utils::FunctionDirichlet
      {
      public:
        ExactSolution()
          : utils::FunctionDirichlet()
        {}

        virtual double
        value(const Point<dim> &p,
              const unsigned int /* component */ = 0) const override
        {
          return (this->get_time() * std::cos(M_PI * p[0]) *
                  std::cos(M_PI * p[1]) * std::cos(M_PI * p[2]));
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
              const unsigned int /* component */ = 0) const override
        {
          const double t = this->get_time();

          const double u = t * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                           std::cos(M_PI * p[2]);

          return (u / t + 3 * M_PI * M_PI * u + u * u);
        }
      };

      /// Constructor.
      SolverU(const std::string &subsection)
        : CoreModel(subsection)
        , timestep_number(0)
        , non_linear_solver(prm_subsection_path + " / Non-linear solver")
        , linear_solver(prm_subsection_path + " / Linear solver",
                        {"CG", "GMRES", "BiCGStab"},
                        "GMRES")
        , preconditioner(prm_subsection_path + " / Preconditioner", true)
        , bc_handler(dof_handler)
        , u_ex(std::make_shared<ExactSolution>())
      {}

      const LinAlg::MPI::Vector &
      get_solution() const
      {
        return solution;
      }

      const DoFHandler<dim> &
      get_dof_handler() const
      {
        return dof_handler;
      }

      /// Declare input parameters.
      virtual void
      declare_parameters(ParamHandler &params) const override
      {
        params.enter_subsection_path(prm_subsection_path);
        params.enter_subsection("Mesh and space discretization");
        {
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
          params.set_verbosity(VerbosityParam::Standard);
          params.declare_entry("BDF order",
                               "1",
                               Patterns::Integer(1, 3),
                               "BDF order: 1, 2, 3.");
          params.reset_verbosity();
        }
        params.leave_subsection();
        params.leave_subsection_path();

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
        params.enter_subsection_path(prm_subsection_path);
        params.enter_subsection("Mesh and space discretization");
        {
          prm_fe_degree = params.get_integer("FE space degree");
        }
        params.leave_subsection();

        params.enter_subsection("Time solver");
        {
          prm_bdf_order = params.get_integer("BDF order");
        }
        params.leave_subsection();
        params.leave_subsection_path();

        non_linear_solver.parse_parameters(params);
        linear_solver.parse_parameters(params);
        preconditioner.parse_parameters(params);
      }

      void
      set_mesh(const std::shared_ptr<utils::MeshHandler> &triangulation_)
      {
        triangulation = triangulation_;
      };

      /// Set time discretization parameters.
      void
      set_time_discretization(const double &      time_initial,
                              const double &      time_final,
                              const double &      time_step,
                              const double &      time_,
                              const unsigned int &timestep_number_)
      {
        prm_time_init  = time_initial;
        prm_time_final = time_final;
        prm_time_step  = time_step;

        time            = time_;
        timestep_number = timestep_number_;
      }

      void
      time_advance()
      {
        time += prm_time_step;
        ++timestep_number;

        bdf_handler.time_advance(solution_owned, true);
      }

      void
      solve_time_step()
      {
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

        const bool converged = non_linear_solver.solve(assemble_fun, solve_fun);
        AssertThrow(converged, ExcNonlinearNotConverged());

        error_owned = solution_owned;
        error_owned -= solution_ex_owned;
        pcout << "\tL-inf error norm: " << error_owned.linfty_norm()
              << std::endl
              << std::endl;
      }

      /// Run simulation.
      /// This method is not supposed to be called explicitly.
      virtual void
      run() override
      {
        AssertThrow(false, ExcLifexInternalError());
      }

      /// Attach output.
      void
      attach_output(DataOut<dim> &data_out) const
      {
        // Solutions.
        data_out.add_data_vector(dof_handler, solution, "u");
        data_out.add_data_vector(dof_handler, solution_ex, "u_ex");
      }

      /// Setup system, @a i.e. setup DoFHandler, allocate matrices and vectors
      /// and initialize handlers.
      void
      setup_system()
      {
        fe = triangulation->get_fe_lagrange(prm_fe_degree);

        // Quadrature formula has to integrate exactly the mass matrix
        // so (fe degree + 1) points are needed.
        quadrature_formula =
          triangulation->get_quadrature_gauss(prm_fe_degree + 1);

        dof_handler.reinit(triangulation->get());
        dof_handler.distribute_dofs(*fe);

        triangulation->get_info().print(prm_subsection_path,
                                        dof_handler.n_dofs(),
                                        true);

        IndexSet owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

        // Setup BCs: we start from an initial guess having the correct
        // Dirichlet values and impose homogeneous BCs on the Newton increment.
        std::vector<utils::BC<utils::FunctionDirichlet>> bcs_dirichlet;
        for (size_t i = 0; i < triangulation->n_faces_per_cell(); ++i)
          bcs_dirichlet.emplace_back(i, u_ex);

        bc_handler.initialize(bcs_dirichlet);

        bc_handler.apply_dirichlet(constraints_dirichlet, true);
        constraints_dirichlet.close();

        DynamicSparsityPattern dsp(relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        constraints_dirichlet);

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

        error_owned.reinit(owned_dofs, mpi_comm);

        non_linear_solver.initialize(&solution_owned, &solution);
      }

      void
      setup_initial_conditions()
      {
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

    private:
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

        // BDF quantities.
        const double &alpha_bdf = bdf_handler.get_alpha();

        std::vector<double> u_loc(n_q_points);
        std::vector<double> u_bdf_loc(n_q_points);
        std::vector<double> u_ext_loc(n_q_points);

        std::vector<Tensor<1, dim, double>> grad_u_loc(n_q_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(dof_indices);
                fe_values.reinit(cell);

                if (assemble_jac)
                  {
                    cell_jac = 0;
                  }
                cell_res = 0;

                std::fill(u_loc.begin(), u_loc.end(), 0);
                std::fill(u_bdf_loc.begin(), u_bdf_loc.end(), 0);
                std::fill(u_ext_loc.begin(), u_ext_loc.end(), 0);

                std::fill(grad_u_loc.begin(),
                          grad_u_loc.end(),
                          Tensor<1, dim, double>());

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        u_loc[q] += solution[dof_indices[i]] *
                                    fe_values.shape_value(i, q);

                        u_bdf_loc[q] += solution_bdf[dof_indices[i]] *
                                        fe_values.shape_value(i, q);

                        u_ext_loc[q] += solution_ext[dof_indices[i]] *
                                        fe_values.shape_value(i, q);

                        for (unsigned int d = 0; d < dim; ++d)
                          grad_u_loc[q][d] += solution[dof_indices[i]] *
                                              fe_values.shape_grad(i, q)[d];
                      }
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        if (assemble_jac)
                          {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                cell_jac(i, j) += alpha_bdf / prm_time_step *
                                                  fe_values.shape_value(i, q) *
                                                  fe_values.shape_value(j, q) *
                                                  fe_values.JxW(q);

                                cell_jac(i, j) += fe_values.shape_grad(i, q) *
                                                  fe_values.shape_grad(j, q) *
                                                  fe_values.JxW(q);

                                cell_jac(i, j) += 2 * u_ext_loc[q] *
                                                  fe_values.shape_value(i, q) *
                                                  fe_values.shape_value(j, q) *
                                                  fe_values.JxW(q);
                              }
                          }

                        cell_res(i) += (alpha_bdf * u_loc[q] - u_bdf_loc[q]) /
                                       prm_time_step *
                                       fe_values.shape_value(i, q) *
                                       fe_values.JxW(q);

                        cell_res(i) += grad_u_loc[q] *
                                       fe_values.shape_grad(i, q) *
                                       fe_values.JxW(q);

                        cell_res(i) += u_loc[q] * u_loc[q] *
                                       fe_values.shape_value(i, q) *
                                       fe_values.JxW(q);

                        cell_res(i) -=
                          f_ex.value(fe_values.quadrature_point(q)) *
                          fe_values.shape_value(i, q) * fe_values.JxW(q);
                      }
                  }

                if (assemble_jac)
                  {
                    constraints_dirichlet.distribute_local_to_global(
                      cell_jac, cell_res, dof_indices, jac, res);
                  }
                else
                  {
                    constraints_dirichlet.distribute_local_to_global(
                      cell_res, dof_indices, res);
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
      std::shared_ptr<utils::MeshHandler> triangulation;

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

      /// Error between numerical and exact solution, without ghost entries.
      LinAlg::MPI::Vector error_owned;

      /// Pointer to exact solution function.
      std::shared_ptr<ExactSolution> u_ex;

      /// Right hand side.
      RightHandSide f_ex;
    }; // namespace lifex::tutorials

    class SolverV : public CoreModel
    {
    public:
      class ExactSolution : public utils::FunctionDirichlet
      {
      public:
        ExactSolution()
          : utils::FunctionDirichlet()
        {}

        virtual double
        value(const Point<dim> &p,
              const unsigned int /* component */ = 0) const override
        {
          return (std::exp(this->get_time()) * p.norm_square());
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
              const unsigned int /* component */ = 0) const override
        {
          const double t = this->get_time();

          const double u = t * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                           std::cos(M_PI * p[2]);

          const double v = std::exp(t) * p.norm_square();

          return (-5 * std::exp(t) + u * v);
        }
      };

      /// Constructor.
      SolverV(const std::string &subsection)
        : CoreModel(subsection)
        , timestep_number(0)
        , linear_solver(prm_subsection_path + " / Linear solver",
                        {"CG", "GMRES", "BiCGStab"},
                        "GMRES")
        , preconditioner(prm_subsection_path + " / Preconditioner", true)
        , bc_handler(dof_handler)
        , v_ex(std::make_shared<ExactSolution>())
      {}

      const Quadrature<dim> &
      get_quadrature() const
      {
        AssertThrow(quadrature_formula != nullptr, ExcNotInitialized());

        return *quadrature_formula;
      }

      /// Declare input parameters.
      virtual void
      declare_parameters(ParamHandler &params) const override
      {
        params.enter_subsection_path(prm_subsection_path);
        params.enter_subsection("Mesh and space discretization");
        {
          params.set_verbosity(VerbosityParam::Standard);
          params.declare_entry("FE space degree",
                               "2",
                               Patterns::Integer(1),
                               "Degree of the FE space.");
          params.reset_verbosity();
        }
        params.leave_subsection();

        params.enter_subsection("Time solver");
        {
          params.set_verbosity(VerbosityParam::Standard);
          params.declare_entry("BDF order",
                               "3",
                               Patterns::Integer(1, 3),
                               "BDF order: 1, 2, 3.");
          params.reset_verbosity();
        }
        params.leave_subsection();
        params.leave_subsection_path();

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
        params.enter_subsection_path(prm_subsection_path);
        params.enter_subsection("Mesh and space discretization");
        {
          prm_fe_degree = params.get_integer("FE space degree");
        }
        params.leave_subsection();

        params.enter_subsection("Time solver");
        {
          prm_bdf_order = params.get_integer("BDF order");
        }
        params.leave_subsection();
        params.leave_subsection_path();

        linear_solver.parse_parameters(params);
        preconditioner.parse_parameters(params);
      }

      void
      set_mesh(const std::shared_ptr<utils::MeshHandler> &triangulation_)
      {
        triangulation = triangulation_;
      };

      /// Set time discretization parameters.
      void
      set_time_discretization(const double &      time_initial,
                              const double &      time_final,
                              const double &      time_step,
                              const double &      time_,
                              const unsigned int &timestep_number_)
      {
        prm_time_init  = time_initial;
        prm_time_final = time_final;
        prm_time_step  = time_step;

        time            = time_;
        timestep_number = timestep_number_;
      }

      void
      time_advance()
      {
        time += prm_time_step;
        ++timestep_number;

        bdf_handler.time_advance(solution_owned, true);
      }

      void
      solve_time_step(QuadratureEvaluationScalar &u_fun)
      {
        v_ex->set_time(time);
        f_ex.set_time(time);
        bc_handler.set_time(time, false);

        VectorTools::interpolate(dof_handler, *v_ex, solution_ex_owned);
        solution_ex = solution_ex_owned;

        // Copy into ghosted vectors.
        solution_bdf = bdf_handler.get_sol_bdf();
        solution_ext = bdf_handler.get_sol_extrapolation();

        // Initial guess.
        solution = solution_owned = solution_ext;

        constraints_dirichlet.clear();
        bc_handler.apply_dirichlet(constraints_dirichlet, false);
        constraints_dirichlet.close();

        assemble_system(u_fun);
        solve_system();

        error_owned = solution_owned;
        error_owned -= solution_ex_owned;
        pcout << "\tL-inf error norm: " << error_owned.linfty_norm()
              << std::endl
              << std::endl;
      }

      /// Run simulation.
      /// This method is not supposed to be called explicitly.
      virtual void
      run() override
      {
        AssertThrow(false, ExcLifexInternalError());
      }

      /// Attach output.
      void
      attach_output(DataOut<dim> &data_out) const
      {
        // Solutions.
        data_out.add_data_vector(dof_handler, solution, "v");
        data_out.add_data_vector(dof_handler, solution_ex, "v_ex");
      }

      /// Setup system, @a i.e. setup DoFHandler, allocate matrices and vectors
      /// and initialize handlers.
      void
      setup_system()
      {
        fe = triangulation->get_fe_lagrange(prm_fe_degree);
        quadrature_formula =
          triangulation->get_quadrature_gauss(fe->degree + 1);

        dof_handler.reinit(triangulation->get());
        dof_handler.distribute_dofs(*fe);

        triangulation->get_info().print(prm_subsection_path,
                                        dof_handler.n_dofs(),
                                        true);

        IndexSet owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

        // Setup BCs: we start from an initial guess having the correct
        // Dirichlet values and impose homogeneous BCs on the Newton increment.
        std::vector<utils::BC<utils::FunctionDirichlet>> bcs_dirichlet;
        for (size_t i = 0; i < triangulation->n_faces_per_cell(); ++i)
          bcs_dirichlet.emplace_back(i, v_ex);

        bc_handler.initialize(bcs_dirichlet);

        bc_handler.apply_dirichlet(constraints_dirichlet, false);
        constraints_dirichlet.close();

        DynamicSparsityPattern dsp(relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        constraints_dirichlet);

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

        solution_bdf.reinit(owned_dofs, relevant_dofs, mpi_comm);
        solution_ext.reinit(owned_dofs, relevant_dofs, mpi_comm);

        error_owned.reinit(owned_dofs, mpi_comm);
      }

      void
      setup_initial_conditions()
      {
        // Initialize BDF handler.
        time = prm_time_init;
        v_ex->set_time(time);
        VectorTools::interpolate(dof_handler, *v_ex, solution_ex_owned);
        solution_ex = solution_ex_owned;
        solution = solution_owned = solution_ex_owned;

        const std::vector<LinAlg::MPI::Vector> sol_init(prm_bdf_order,
                                                        solution_owned);
        bdf_handler.initialize(prm_bdf_order, sol_init);
      }

    private:
      /// Assemble linear system.
      void
      assemble_system(QuadratureEvaluationScalar &u_fun)
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

        // BDF quantities.
        const double &alpha_bdf = bdf_handler.get_alpha();

        std::vector<double> v_bdf_loc(n_q_points);
        std::vector<double> v_ext_loc(n_q_points);

        std::vector<Tensor<1, dim, double>> grad_v_ext_loc(n_q_points);

        // Initialize quadrature evaluation.
        u_fun.init();

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(dof_indices);
                fe_values.reinit(cell);

                // Reinitialize quadrature evaluation on current cell.
                u_fun.reinit(cell);

                cell_matrix = 0;
                cell_rhs    = 0;

                std::fill(v_bdf_loc.begin(), v_bdf_loc.end(), 0);
                std::fill(v_ext_loc.begin(), v_ext_loc.end(), 0);

                std::fill(grad_v_ext_loc.begin(),
                          grad_v_ext_loc.end(),
                          Tensor<1, dim, double>());

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        v_bdf_loc[q] += solution_bdf[dof_indices[i]] *
                                        fe_values.shape_value(i, q);

                        v_ext_loc[q] += solution_ext[dof_indices[i]] *
                                        fe_values.shape_value(i, q);

                        for (unsigned int d = 0; d < dim; ++d)
                          grad_v_ext_loc[q][d] += solution_ext[dof_indices[i]] *
                                                  fe_values.shape_grad(i, q)[d];
                      }
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            cell_matrix(i, j) += alpha_bdf / prm_time_step *
                                                 fe_values.shape_value(i, q) *
                                                 fe_values.shape_value(j, q) *
                                                 fe_values.JxW(q);

                            cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                                 fe_values.shape_grad(j, q) *
                                                 fe_values.JxW(q);

                            // The QuadratureEvaluation object is a functor:
                            // u_fun(q) simply returns the evaluation of the
                            // solution u at the q-th quadrature node.
                            cell_matrix(i, j) +=
                              u_fun(q) * fe_values.shape_value(i, q) *
                              fe_values.shape_value(j, q) * fe_values.JxW(q);
                          }

                        cell_rhs(i) += v_bdf_loc[q] / prm_time_step *
                                       fe_values.shape_value(i, q) *
                                       fe_values.JxW(q);

                        cell_rhs(i) +=
                          f_ex.value(fe_values.quadrature_point(q)) *
                          fe_values.shape_value(i, q) * fe_values.JxW(q);
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
      std::shared_ptr<utils::MeshHandler> triangulation;

      /// FE space.
      std::unique_ptr<FiniteElement<dim>> fe;

      /// Quadrature formula.
      std::unique_ptr<Quadrature<dim>> quadrature_formula;

      /// DoFHandler.
      DoFHandler<dim> dof_handler;

      /// BDF time advancing handler.
      utils::BDFHandler<LinAlg::MPI::Vector> bdf_handler;

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

      /// BDF solution, with ghost entries.
      LinAlg::MPI::Vector solution_bdf;
      /// BDF extrapolated solution, with ghost entries.
      LinAlg::MPI::Vector solution_ext;

      /// Distributed exact solution vector, without ghost entries.
      LinAlg::MPI::Vector solution_ex_owned;
      /// Distributed exact solution vector, without ghost entries.
      LinAlg::MPI::Vector solution_ex;

      /// Error between numerical and exact solution, without ghost entries.
      LinAlg::MPI::Vector error_owned;

      /// Pointer to exact solution function.
      std::shared_ptr<ExactSolution> v_ex;

      /// Right hand side.
      RightHandSide f_ex;
    };
  }; // namespace


  /**
   * @brief Prototype solver class for a time-dependent non-linear system of PDEs.
   *
   * This class solves the same problem as @ref Tutorial05,
   * but here the two equations are solved in a segregated way, rather than
   * monolithically.
   *
   * This tutorial also shows the use of the @ref QuadratureEvaluation functionalities,
   * here demonstrated through the @ref QuadratureFEMSolution class.
   *
   * The problem is time-discretized using the implicit finite difference scheme
   * @f$\mathrm{BDF}\sigma@f$ (where @f$\sigma = 1,2,...@f$
   * is the order of the BDF formula, see @ref utils::BDFHandler) as follows:
   * @f[
   * \left\{
   * \begin{aligned}
   * \frac{\alpha_{\mathrm{BDF}\sigma} u^{n+1} -
   * u_{\mathrm{BDF}\sigma}^n}{\Delta t} - \Delta u^{n+1} +
   * \left(u^{n+1}\right)^2 &= f^{n+1}, \\
   * \frac{\alpha_{\mathrm{BDF}\sigma} v^{n+1} -
   * v_{\mathrm{BDF}\sigma}^n}{\Delta t} - \Delta v^{n+1} +
   * u^{n+1}v^{n+1} &= g^{n+1},
   * \end{aligned}
   * \right.
   * @f]
   * where @f$\Delta t = t^{n+1}-t^{n}@f$ is the time step.
   *
   * The solution at each time (@f$t^{n} \rightarrow t^{n+1}@f$) consists of two
   * steps.
   *
   * 1. Solve for @f$u^{n+1}@f$: we first solve the equation for @f$u@f$
   * by linearizing it using Newton's method, @a i.e., given an initial guess
   * @f$u_0^{n+1}@f$ and for @f$k=1, \dots, n_\mathrm{max}@f$ until convergence,
   * the following iterative scheme is solved:
   * @f[
   * \left\{
   * \begin{aligned}
   * \frac{\alpha_{\mathrm{BDF}\sigma} \delta u}{\Delta t} -
   * \Delta \delta u + 2 u_{\mathrm{EXT}\sigma}^n \delta u &=
   * \frac{\alpha_{\mathrm{BDF}\sigma} u_k^{n+1} - u_{\mathrm{BDF}\sigma}^n}
   * {\Delta t} - \Delta u_k^{n+1} + \left(u_k^{n+1}\right)^2 - f^{n+1}, \\
   * u_{k+1}^{n+1} &= u_k^{n+1} - \delta u,
   * \end{aligned}
   * \right.
   * @f]
   * where @f$u^n_{\mathrm{EXT}\sigma}@f$ is an extrapolation of @f$u^{n+1}@f$,
   * computed as a linear combination of the previous time steps(see @ref utils::BDFHandler).
   *
   * 2. Solve for @f$v^{n+1}@f$: given @f$u^{n+1}@f$ from step 1, solve
   * @f[
   * \left\{
   * \frac{\alpha_{\mathrm{BDF}\sigma} v^{n+1} -
   * v_{\mathrm{BDF}\sigma}^n}{\Delta t} - \Delta v^{n+1} +
   * u^{n+1}v^{n+1} = g^{n+1}.
   * \right.
   * @f]
   * .
   * The two equations are space-discretized using @f$\mathbb{Q}^1@f$ and
   * @f$\mathbb{Q}^2@f$ finite elements, respectively.
   *
   * @note To see all the parameters, generate parameter file
   * with the <kbd>-g full</kbd> flag, as explained in @ref run.
   */
  class Tutorial06 : public CoreModel
  {
  public:
    /// Constructor.
    Tutorial06()
      : CoreModel("Tutorial 06")
      , timestep_number(0)
      , triangulation(
          std::make_shared<utils::MeshHandler>(prm_subsection_path, mpi_comm))
      , solver_u(prm_subsection_path + " / Solver for u")
      , solver_v(prm_subsection_path + " / Solver for v")
    {}

    /// Declare input parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);
      params.enter_subsection("Mesh and space discretization");
      {
        params.declare_entry(
          "Number of refinements",
          "3",
          Patterns::Integer(0),
          "Number of global mesh refinement steps applied to initial grid.");
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
      }
      params.leave_subsection();
      params.leave_subsection_path();

      solver_u.declare_parameters(params);
      solver_v.declare_parameters(params);
    }

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      // Parse input file.
      params.parse();

      // Read input parameters.
      params.enter_subsection_path(prm_subsection_path);
      params.enter_subsection("Mesh and space discretization");
      {
        prm_n_refinements = params.get_integer("Number of refinements");
      }
      params.leave_subsection();

      params.enter_subsection("Time solver");
      {
        prm_time_init  = params.get_double("Initial time");
        prm_time_final = params.get_double("Final time");
        prm_time_step  = params.get_double("Time step");
      }
      params.leave_subsection();
      params.leave_subsection_path();

      solver_u.parse_parameters(params);
      solver_v.parse_parameters(params);
    }

    /// Run simulation.
    virtual void
    run() override
    {
      create_mesh();
      setup_system();
      output_results();

      QuadratureFEMSolution u_fun(solver_u.get_solution(),
                                  solver_u.get_dof_handler(),
                                  solver_v.get_quadrature());

      while (time < prm_time_final)
        {
          time += prm_time_step;
          ++timestep_number;

          pcout << "Time step " << std::setw(6) << timestep_number
                << " at t = " << std::setw(8) << std::fixed
                << std::setprecision(6) << time;

          {
            pcout << "\nSolver for u: " << std::endl;

            solver_u.time_advance();
            solver_u.solve_time_step();
          }

          {
            pcout << "\nSolver for v: " << std::endl;

            solver_v.time_advance();
            solver_v.solve_time_step(u_fun);
          }

          output_results();
        }
    }

  private:
    /// Create mesh.
    void
    create_mesh()
    {
      triangulation->initialize_hypercube(-1, 1, true);
      triangulation->set_refinement_global(prm_n_refinements);
      triangulation->create_mesh();
    }

    /// Setup system, @a i.e. setup DoFHandler, allocate matrices and vectors
    /// and initialize handlers.
    void
    setup_system()
    {
      time = prm_time_init;

      {
        solver_u.set_mesh(triangulation);

        solver_u.set_time_discretization(
          prm_time_init, prm_time_final, prm_time_step, time, timestep_number);

        solver_u.setup_system();
        solver_u.setup_initial_conditions();
      }

      {
        solver_v.set_mesh(triangulation);

        solver_v.set_time_discretization(
          prm_time_init, prm_time_final, prm_time_step, time, timestep_number);

        solver_v.setup_system();
        solver_v.setup_initial_conditions();
      }
    }

    /// Output results.
    void
    output_results()
    {
      DataOut<dim> data_out;

      solver_u.attach_output(data_out);
      solver_v.attach_output(data_out);

      data_out.build_patches();

      utils::dataout_write_hdf5(data_out, "solution", timestep_number, 0, time);

      data_out.clear();
    }

    /// Number of mesh refinements.
    unsigned int prm_n_refinements;

    /// Initial time.
    double prm_time_init;

    /// Final time.
    double prm_time_final;

    /// Time step.
    double prm_time_step;

    /// Current time.
    double time;

    /// Timestep number.
    unsigned int timestep_number;

    /// Triangulation.
    std::shared_ptr<utils::MeshHandler> triangulation;

    /// Solver for the equation in @f$u@f$.
    SolverU solver_u;

    /// Solver for the equation in @f$v@f$.
    SolverV solver_v;
  };
} // namespace lifex::tutorials

/// Run tutorial 06.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tutorials::Tutorial06 tutorial;

      tutorial.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
