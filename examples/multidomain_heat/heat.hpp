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

#ifndef LIFEX_EXAMPLE_HEAT_HPP_
#define LIFEX_EXAMPLE_HEAT_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/bc_handler.hpp"
#include "source/numerics/interface_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/time_handler.hpp"
#include "source/numerics/tools.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace lifex::examples
{
  /**
   * @brief Heat equation on a subdomain.
   *
   * Solves the equation @f[ \frac{\partial u}{\partial t} - \Delta u +
   * \mathbf\beta\cdot\nabla u = 0 @f] with homogeneous initial conditions.
   *
   * See @ref ExampleMultidomainHeat .
   */
  class ExampleHeat : public CoreModel
  {
    /// This class needs access to the dof_handler, fe and owned_dofs members to
    /// build interface maps between subdomains.
    friend class ExampleMultidomainHeat;

  public:
    /// Constructor.
    ExampleHeat(
      const unsigned int &  subdomain_id_,
      const Tensor<1, dim> &transport_coefficient_,
      const std::vector<utils::BC<utils::FunctionDirichlet>> &dirirchlet_bcs_,
      const std::string &                                     subsection);

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Assemble, solve and output the results. This does not advance time.
    void
    step();

    /// Compute the interface residual.
    void
    compute_interface_residual();

    /// Setup interface handler and interface conditions.
    void
    setup_interface(
      const DoFHandler<dim> &other_dof_handler,
      const std::vector<std::shared_ptr<
        utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataDirichlet>>
        interface_data_dirichlet,
      const std::vector<std::shared_ptr<
        utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataNeumann>>
        interface_data_neumann,
      const std::vector<std::shared_ptr<
        utils::InterfaceHandler<LinAlg::MPI::Vector>::InterfaceDataRobinLinear>>
        interface_data_robin);

    /// Advance time.
    void
    time_advance();

    /// Set the solution vector.
    void
    set_solution(const LinAlg::MPI::Vector &src)
    {
      solution_owned = src;
      solution       = solution_owned;
    }

  protected:
    /// Create the mesh.
    void
    create_mesh(const unsigned int &n_refinements);

    /// Setup the system.
    void
    setup_system();

    /// Assemble the system.
    void
    assemble_system();

    /// Assemble a single cell.
    void
    assemble_cell(const DoFHandler<dim>::active_cell_iterator &cell,
                  const unsigned int &                         c,
                  const std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> &                        cell_matrix,
                  Vector<double> &                            cell_rhs);

    /// Solve the linear system.
    void
    solve();

    /// Output the results.
    void
    output_results() const;

    unsigned int subdomain_id; ///< Subdomain identifier

    utils::MeshHandler                  triangulation; ///< Triangulation.
    DoFHandler<dim>                     dof_handler;   ///< DoF handler.
    std::unique_ptr<FiniteElement<dim>> fe; ///< Finite element space.
    std::unique_ptr<Quadrature<dim>>
      quadrature_formula; ///< Quadrature formula.
    std::unique_ptr<Quadrature<dim - 1>>
      face_quadrature_formula; ///< Face quadrature formula.

    LinAlg::MPI::SparseMatrix system_matrix; ///< Linear system matrix.
    LinAlg::MPI::Vector       solution;      ///< Linear system solution.
    LinAlg::MPI::Vector
                        solution_owned; ///< Linear system solution, without ghosts.
    LinAlg::MPI::Vector system_rhs;     ///< Linear system right hand side.

    LinAlg::MPI::Vector u_bdf; ///< BDF solution, with ghost entries.

    LinAlg::MPI::SparseMatrix
      system_matrix_no_interface; ///< System matrix, without interface
                                  ///< constraints.
    LinAlg::MPI::Vector
      system_rhs_no_interface; ///< System rhs, without interface constraints.

    /// Interface residual, i.e. residual of the linear system without the
    /// interface terms.
    LinAlg::MPI::Vector residual_no_interface;

    utils::BDFHandler<LinAlg::MPI::Vector>
      bdf_handler; ///< Handler for time advancing.

    std::vector<utils::BC<utils::FunctionDirichlet>>
                     dirichlet_bcs; ///< Dirichlet boundary conditions.
    utils::BCHandler bc_handler;    ///< Handler for boundary conditions.
    std::unique_ptr<utils::InterfaceHandler<LinAlg::MPI::Vector>>
      interface_handler; ///< Handler for interface conditions.

    AffineConstraints<double>
      constraints; ///< Affine constraints for the linear system.

    /// Affine constraints for the linear system, without constraints for
    /// interface conditions.
    AffineConstraints<double> constraints_no_interface;

    IndexSet owned_dofs;    ///< Locally owned DoFs.
    IndexSet relevant_dofs; ///< Locally relevant DoFs.

    Tensor<1, dim> transport_coefficient; ///< Transport coefficient.

    double       dt              = 1e-2; ///< Timestep.
    double       time            = 0.0;  ///< Current time.
    unsigned int timestep_number = 0;    ///< Current timestep.

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::Vector> linear_solver;

    /// Preconditioner handler.
    utils::PreconditionerHandler preconditioner;

    /// Constant diffusion coefficient.
    double diffusion;

    /// Coefficient for heterogeneous conductivity in the torso.
    std::vector<std::vector<double>> diffusion_coeff;

    /// Linear solver handler for Neumann interface conditions computation.
    /// Using domain decomposition technique, in particular in presence of
    /// non-conforming interface grids between the two subdomains, Neumann
    /// boundary conditions can not be obtained computing the simple residual
    /// $r$. Instead, a mass interface matrix @f$\mathbb{M}@f$ must be assembled
    /// and an intermediate vector @f$z@f$ is to be considered, which is the
    /// results of the linear system @f$\mathbb{M}z = r@f$. This
    /// linear_solver_bc is used to solve such problem.
    utils::LinearSolverHandler<Vector<double>> linear_solver_neumann;
  };
} // namespace lifex::examples

#endif /* LIFEX_EXAMPLE_HEAT_HPP_ */
