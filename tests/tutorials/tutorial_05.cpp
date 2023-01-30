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

#include <cmath>
#include <memory>
#include <vector>

namespace lifex::tutorials
{
  namespace
  {
    /// Number of components.
    static inline constexpr unsigned int n_blocks = 2;

    class ExactSolution : public utils::FunctionDirichlet
    {
    public:
      ExactSolution()
        : utils::FunctionDirichlet(n_blocks)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        const double t = this->get_time();

        if (component == 0)
          return (t * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                  std::cos(M_PI * p[2]));
        else // if (component == 1)
          return (std::exp(t) * p.norm_square());
      }
    };


    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>(n_blocks)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        const double t = this->get_time();

        const double u = t * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) *
                         std::cos(M_PI * p[2]);

        const double v = std::exp(t) * p.norm_square();

        if (component == 0)
          return (u / t + 3 * M_PI * M_PI * u + u * u);
        else // if (component == 1)
          return (-5 * std::exp(t) + u * v);
      }
    };
  } // namespace

  /**
   * @brief Prototype solver class for a time-dependent non-linear system of PDEs.
   *
   * The equations solved are:
   * @f[
   * \left\{
   * \begin{aligned}
   * \frac{\partial u}{\partial t} - \Delta u + u^2 &= f, & \quad & \text{in }
   * \Omega \times (0, T] = (-1, 1)^3 \times (0, T], \\
   * \frac{\partial v}{\partial t} - \Delta v + uv &= g, & \quad & \text{in }
   * \Omega \times (0, T] = (-1, 1)^3 \times (0, T], \\
   * u &= u_\mathrm{ex}, & \quad & \text{on } \partial\Omega \times (0, T], \\
   * v &= v_\mathrm{ex}, & \quad & \text{on } \partial\Omega \times (0, T], \\
   * u &= u^0, & \quad & \text{in } \Omega \times \{0\}, \\
   * v &= v^0, & \quad & \text{in } \Omega \times \{0\},
   * \end{aligned}
   * \right.
   * @f]
   * where @f$f, g, u^0, v^0@f$ are chosen such that the exact solution is
   * @f[
   * \left\{
   * \begin{aligned}
   * u_\mathrm{ex}(\mathbf{x}, t) &= t \cos(\pi x_0) \cos(\pi x_1) \cos(\pi
   * x_2), \\
   * v_\mathrm{ex}(\mathbf{x}, t) &= e^t \left\|\mathbf{x}\right\|^2.
   * \end{aligned}
   * \right.
   * @f]
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
   * At each time (@f$t^{n} \rightarrow t^{n+1}@f$) the problem above is
   * linearized and solved @b monolithically using Newton's method, @a i.e.,
   * given an initial guess @f$u_0^{n+1}, v_0^{n+1}@f$ and for @f$k=1, \dots,
   * n_\mathrm{max}@f$ until convergence, the following iterative scheme is
   * solved:
   * @f[
   * \left\{
   * \begin{aligned}
   * \frac{\alpha_{\mathrm{BDF}\sigma} \delta u}{\Delta t} -
   * \Delta \delta u + 2 u_{\mathrm{EXT}\sigma}^n \delta u &=
   * \frac{\alpha_{\mathrm{BDF}\sigma} u_k^{n+1} - u_{\mathrm{BDF}\sigma}^n}
   * {\Delta t} - \Delta u_k^{n+1} + \left(u_k^{n+1}\right)^2 - f^{n+1}, \\
   * \frac{\alpha_{\mathrm{BDF}\sigma} \delta v}{\Delta t} -
   * \Delta \delta v + v_{\mathrm{EXT}\sigma}^n \delta u +
   * u_{\mathrm{EXT}\sigma}^n \delta v &= \frac{\alpha_{\mathrm{BDF}\sigma}
   * v_k^{n+1} - v_{\mathrm{BDF}\sigma}^n}
   * {\Delta t} - \Delta v_k^{n+1} + u_k^{n+1}v_k^{n+1} - g^{n+1}, \\
   * u_{k+1}^{n+1} &= u_k^{n+1} - \delta u, \\
   * v_{k+1}^{n+1} &= v_k^{n+1} - \delta v, \\
   * \end{aligned}
   * \right.
   * @f]
   * where @f$u^n_{\mathrm{EXT}\sigma}, v^n_{\mathrm{EXT}\sigma}@f$ are
   * extrapolations of @f$u^{n+1}, v^{n+1}@f$, respectively, computed as a
   * linear combination of the previous time steps (see @ref utils::BDFHandler).
   *
   * The two equations are space-discretized using @f$\mathbb{Q}^1@f$ and
   * @f$\mathbb{Q}^2@f$ finite elements, respectively.
   *
   * @note To see all the parameters, generate parameter file
   * with the <kbd>-g full</kbd> flag, as explained in @ref run.
   */
  class Tutorial05 : public CoreModel
  {
  public:
    /// Constructor.
    Tutorial05()
      : CoreModel("Tutorial 05")
      , timestep_number(0)
      , triangulation(prm_subsection_path, mpi_comm)
      , non_linear_solver(prm_subsection_path + " / Non-linear solver")
      , linear_solver(prm_subsection_path + " / Linear solver",
                      {"GMRES", "BiCGStab"},
                      "GMRES")
      , preconditioner(prm_subsection_path + " / Preconditioner", jac, true)
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
          "3",
          Patterns::Integer(0),
          "Number of global mesh refinement steps applied to initial grid.");

        params.set_verbosity(VerbosityParam::Standard);
        params.declare_entry("FE space degree (u)",
                             "1",
                             Patterns::Integer(1),
                             "Degree of the FE space for u.");

        params.declare_entry("FE space degree (v)",
                             "2",
                             Patterns::Integer(1),
                             "Degree of the FE space for v.");
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
        prm_fe_degree_u   = params.get_integer("FE space degree (u)");
        prm_fe_degree_v   = params.get_integer("FE space degree (v)");
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

      auto solve_fun = [this](const bool &              assemble_prec,
                              LinAlg::MPI::BlockVector &incr) {
        const double norm_sol = solution_owned.l2_norm();

        solve_system(assemble_prec, incr);

        const double norm_incr = incr.l2_norm();

        const unsigned int n_iterations_linear =
          linear_solver.get_n_iterations();

        return std::make_tuple(norm_sol, norm_incr, n_iterations_linear);
      };

      LinAlg::MPI::BlockVector error_owned;
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
      const auto fe_scalar_1 = triangulation.get_fe_lagrange(prm_fe_degree_u);
      const auto fe_scalar_2 = triangulation.get_fe_lagrange(prm_fe_degree_v);

      fe = std::make_unique<FESystem<dim>>(*fe_scalar_1, 1, *fe_scalar_2, 1);
      quadrature_formula = triangulation.get_quadrature_gauss(fe->degree + 1);

      dof_handler.reinit(triangulation.get());
      dof_handler.distribute_dofs(*fe);

      const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler);

      triangulation.get_info().print(
        prm_subsection_path,
        std::to_string(dof_handler.n_dofs()) +
          " (u_0: " + std::to_string(dofs_per_block[0]) +
          ", u_1: " + std::to_string(dofs_per_block[1]) + ")",
        true);

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      std::vector<IndexSet> owned_dofs(n_blocks);
      std::vector<IndexSet> relevant_dofs(n_blocks);

      const unsigned int n0 = dofs_per_block[0];
      const unsigned int n1 = dofs_per_block[1];

      owned_dofs[0] = dof_handler.locally_owned_dofs().get_view(0, n0);
      owned_dofs[1] = dof_handler.locally_owned_dofs().get_view(n0, n0 + n1);

      relevant_dofs[0] = locally_relevant_dofs.get_view(0, n0);
      relevant_dofs[1] = locally_relevant_dofs.get_view(n0, n0 + n1);

      // Setup BCs: we start from an initial guess having the correct Dirichlet
      // values and impose homogeneous BCs on the Newton increment.
      std::vector<utils::BC<utils::FunctionDirichlet>> bcs_dirichlet;
      for (size_t i = 0; i < triangulation.n_faces_per_cell(); ++i)
        bcs_dirichlet.emplace_back(i, u_ex);

      bc_handler.initialize(bcs_dirichlet);

      bc_handler.apply_dirichlet(constraints_dirichlet, true);
      constraints_dirichlet.close();

      Table<2, DoFTools::Coupling> coupling(n_blocks, n_blocks);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          for (unsigned int j = 0; j < n_blocks; ++j)
            {
              // No coupling on block (0, 1).
              coupling[i][j] =
                (i == 0 && j == 1) ? DoFTools::none : DoFTools::always;
            }
        }

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints_dirichlet, false);

      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_comm, dof_handler.locally_owned_dofs()),
        mpi_comm,
        locally_relevant_dofs);

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

      const std::vector<LinAlg::MPI::BlockVector> sol_init(prm_bdf_order,
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

      std::vector<unsigned int> component(dofs_per_cell);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        component[i] = fe->system_to_component_index(i).first;

      FullMatrix<double> cell_jac(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_res(dofs_per_cell);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      // BDF quantities.
      const double &alpha_bdf = bdf_handler.get_alpha();

      std::vector<double> u_loc(n_q_points);
      std::vector<double> u_bdf_loc(n_q_points);
      std::vector<double> u_ext_loc(n_q_points);

      std::vector<Tensor<1, dim, double>> grad_u_loc(n_q_points);

      std::vector<double> v_loc(n_q_points);
      std::vector<double> v_bdf_loc(n_q_points);
      std::vector<double> v_ext_loc(n_q_points);

      std::vector<Tensor<1, dim, double>> grad_v_loc(n_q_points);

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

              std::fill(v_loc.begin(), v_loc.end(), 0);
              std::fill(v_bdf_loc.begin(), v_bdf_loc.end(), 0);
              std::fill(v_ext_loc.begin(), v_ext_loc.end(), 0);

              std::fill(grad_v_loc.begin(),
                        grad_v_loc.end(),
                        Tensor<1, dim, double>());

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      if (component[i] == 0)
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
                      else // if (component[i] == 1)
                        {
                          v_loc[q] += solution[dof_indices[i]] *
                                      fe_values.shape_value(i, q);

                          v_bdf_loc[q] += solution_bdf[dof_indices[i]] *
                                          fe_values.shape_value(i, q);

                          v_ext_loc[q] += solution_ext[dof_indices[i]] *
                                          fe_values.shape_value(i, q);

                          for (unsigned int d = 0; d < dim; ++d)
                            grad_v_loc[q][d] += solution[dof_indices[i]] *
                                                fe_values.shape_grad(i, q)[d];
                        }
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
                              if (component[i] == 0 && component[j] == 0)
                                {
                                  cell_jac(i, j) +=
                                    alpha_bdf / prm_time_step *
                                    fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    fe_values.JxW(q);

                                  cell_jac(i, j) += fe_values.shape_grad(i, q) *
                                                    fe_values.shape_grad(j, q) *
                                                    fe_values.JxW(q);

                                  cell_jac(i, j) +=
                                    2 * u_ext_loc[q] *
                                    fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    fe_values.JxW(q);
                                }
                              else if (component[i] == 0 && component[j] == 1)
                                {
                                  // Nothing to do here.
                                }
                              else if (component[i] == 1 && component[j] == 0)
                                {
                                  cell_jac(i, j) +=
                                    v_ext_loc[q] * fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    fe_values.JxW(q);
                                }
                              else // if (component[i] == 1 && component[j] ==
                                   // 1)
                                {
                                  cell_jac(i, j) +=
                                    alpha_bdf / prm_time_step *
                                    fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    fe_values.JxW(q);

                                  cell_jac(i, j) += fe_values.shape_grad(i, q) *
                                                    fe_values.shape_grad(j, q) *
                                                    fe_values.JxW(q);

                                  cell_jac(i, j) +=
                                    u_ext_loc[q] * fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    fe_values.JxW(q);
                                }
                            }
                        }

                      if (component[i] == 0)
                        {
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
                            f_ex.value(fe_values.quadrature_point(q),
                                       component[i]) *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);
                        }
                      else // if (component[i] == 1)
                        {
                          cell_res(i) += (alpha_bdf * v_loc[q] - v_bdf_loc[q]) /
                                         prm_time_step *
                                         fe_values.shape_value(i, q) *
                                         fe_values.JxW(q);

                          cell_res(i) += grad_v_loc[q] *
                                         fe_values.shape_grad(i, q) *
                                         fe_values.JxW(q);

                          cell_res(i) += u_loc[q] * v_loc[q] *
                                         fe_values.shape_value(i, q) *
                                         fe_values.JxW(q);

                          cell_res(i) -=
                            f_ex.value(fe_values.quadrature_point(q),
                                       component[i]) *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);
                        }
                    }
                }

              if (assemble_jac)
                {
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
    solve_system(const bool &assemble_prec, LinAlg::MPI::BlockVector &incr)
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
      const std::vector<std::string> solution_names{"u", "v"};
      const std::vector<std::string> solution_names_ex{"u_ex", "v_ex"};

      const std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          n_blocks, DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector(dof_handler,
                               solution,
                               solution_names,
                               data_component_interpretation);
      data_out.add_data_vector(dof_handler,
                               solution_ex,
                               solution_names_ex,
                               data_component_interpretation);
      data_out.build_patches(2);

      utils::dataout_write_hdf5(data_out, "solution", timestep_number, 0, time);

      data_out.clear();
    }

    /// Number of mesh refinements.
    unsigned int prm_n_refinements;

    /// FE space degree for @f$u@f$.
    unsigned int prm_fe_degree_u;

    /// FE space degree for @f$v@f$.
    unsigned int prm_fe_degree_v;

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
    std::unique_ptr<FESystem<dim>> fe;

    /// Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature_formula;

    /// DoFHandler.
    DoFHandler<dim> dof_handler;

    /// BDF time advancing handler.
    utils::BDFHandler<LinAlg::MPI::BlockVector> bdf_handler;

    /// Non-linear solver handler.
    utils::NonLinearSolverHandler<LinAlg::MPI::BlockVector> non_linear_solver;

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::BlockVector> linear_solver;

    /// Preconditioner handler.
    utils::BlockPreconditionerHandler preconditioner;

    /// Boundary condition handler.
    utils::BCHandler bc_handler;
    /// Dirichlet BCs constraints.
    AffineConstraints<double> constraints_dirichlet;

    /// Distributed system jacobian matrix.
    LinAlg::MPI::BlockSparseMatrix jac;
    /// Distributed system residual vector.
    LinAlg::MPI::BlockVector res;
    /// Distributed solution vector, without ghost entries.
    LinAlg::MPI::BlockVector solution_owned;
    /// Distributed solution vector, with ghost entries.
    LinAlg::MPI::BlockVector solution;

    /// BDF solution, with ghost entries.
    LinAlg::MPI::BlockVector solution_bdf;
    /// BDF extrapolated solution, with ghost entries.
    LinAlg::MPI::BlockVector solution_ext;

    /// Distributed exact solution vector, without ghost entries.
    LinAlg::MPI::BlockVector solution_ex_owned;
    /// Distributed exact solution vector, without ghost entries.
    LinAlg::MPI::BlockVector solution_ex;

    /// Pointer to exact solution function.
    std::shared_ptr<ExactSolution> u_ex;

    /// Right hand side.
    RightHandSide f_ex;
  };
} // namespace lifex::tutorials

/// Run tutorial 05.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tutorials::Tutorial05 tutorial;

      tutorial.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
