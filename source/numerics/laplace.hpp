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

#ifndef LIFEX_PHYSICS_LAPLACE_HPP_
#define LIFEX_PHYSICS_LAPLACE_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>

namespace lifex
{
  /**
   * @brief Laplace problem solver.
   *
   * This class allows to solve a Laplace problem with arbitrary Dirichlet
   * boundary conditions (imposed on boundary surfaces or material IDs).
   *
   * An example of usage is the following:
   * @code{cpp}
   * Laplace laplace;
   *
   * // ... declare and parse parameters ...
   *
   * // Set boundary conditions.
   * laplace.apply_dirichlet_boundary(0, 1.23);
   * laplace.apply_dirichlet_boundary(1, 4.56);
   *
   * // Solve.
   * laplace.solve();
   *
   * // Get solution and do stuff...
   * laplace.get_solution();
   * @endcode
   */
  class Laplace : public CoreModel
  {
  public:
    /**
     * @brief Constructor.
     */
    Laplace(const std::string &subsection_path);

    /**
     * @brief Override of @ref CoreModel::declare_parameters.
     */
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /**
     * @brief Override of @ref CoreModel::declare_parameters.
     */
    virtual void
    parse_parameters(ParamHandler &params) override;

    /**
     * @brief Setup the system.
     *
     * This method also assembles the system matrix.
     *
     * @param[in] triangulation The mesh on which the Laplace problem must be
     * solved.
     * @param[in] fe_degree The polynomial degree used to discretize the Laplace
     * problem.
     */
    void
    setup_system(const utils::MeshHandler &triangulation,
                 const unsigned int &      fe_degree = 1);

    /**
     * @brief Remove any previously set boundary condition.
     */
    void
    clear_bcs();

    /**
     * @brief Apply a Dirichlet condition on a set of boundary tags.
     */
    void
    apply_dirichlet_boundary(const std::set<types::boundary_id> &tags,
                             const double &                      value);

    /**
     * @brief Apply a Dirichlet condition on a boundary tag.
     */
    void
    apply_dirichlet_boundary(const types::boundary_id &tag, const double &value)
    {
      const std::set<types::boundary_id> tags{tag};
      apply_dirichlet_boundary(tags, value);
    }

    /**
     * @brief Apply a Dirichlet-type condition on a set of volume tags.
     *
     * In practice, this means that all DoFs that belong to a cell with one of
     * the provided tags will be constrained to the given value. Notice that
     * this is not technically a boundary condition, because we are constraining
     * DoFs that do not lie on the boundary.
     */
    void
    apply_dirichlet_volume(const std::set<types::material_id> &tags,
                           const double &                      value);

    /**
     * @brief Apply a Dirichlet-type condition on a volume tag.
     *
     * In practice, this means that all DoFs that belong to a cell with one of
     * the provided tags will be constrained to the given value. Notice that
     * this is not technically a boundary condition, because we are constraining
     * DoFs that do not lie on the boundary.
     */
    void
    apply_dirichlet_volume(const types::material_id &tag, const double &value)
    {
      const std::set<types::material_id> tags{tag};
      apply_dirichlet_volume(tags, value);
    }

    /**
     * @brief Compute the solution.
     */
    void
    solve();

    /**
     * @brief Get the solution vector, without ghost elements.
     */
    const LinAlg::MPI::Vector &
    get_solution_owned() const
    {
      return sol_owned;
    }

    /**
     * @brief Get the solution vector, with ghost elements.
     */
    const LinAlg::MPI::Vector &
    get_solution() const
    {
      return sol;
    }

    /**
     * @brief Get the DoF handler.
     */
    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return dof_handler;
    }

    /**
     * @brief Attach the solution vector to a DataOut object.
     */
    void
    attach_output(DataOut<dim> &data_out, const std::string &name) const;

  protected:
    /// Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    /// DoF handler.
    DoFHandler<dim> dof_handler;

    /// Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    /// System matrix, without boundary conditions.
    LinAlg::MPI::SparseMatrix matrix_no_bcs;

    /// System matrix, with boundary conditions.
    LinAlg::MPI::SparseMatrix matrix;

    /// System right-hand-side.
    LinAlg::MPI::Vector rhs;

    /// Problem solution, without ghost elements.
    LinAlg::MPI::Vector sol_owned;

    /// Problem solution, with ghost elements.
    LinAlg::MPI::Vector sol;

    /// Linear solver.
    utils::LinearSolverHandler<LinAlg::MPI::Vector> linear_solver;

    /// Preconditioner handler.
    utils::PreconditionerHandler preconditioner;

    /// Boundary values map.
    std::map<types::global_dof_index, double> boundary_values;
  };
} // namespace lifex

#endif
