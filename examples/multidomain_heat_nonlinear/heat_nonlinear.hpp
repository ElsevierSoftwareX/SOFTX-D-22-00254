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

#ifndef LIFEX_EXAMPLE_HEAT_NONLINEAR_HPP_
#define LIFEX_EXAMPLE_HEAT_NONLINEAR_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/bc_handler.hpp"
#include "source/numerics/interface_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/non_linear_solver_handler.hpp"
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
   * @brief Non-linear heat equation on a subdomain.
   *
   * Solves on a subdomain the following non-linear advection-diffusion-reaction
   * equation: @f[ \frac{\partial u}{\partial t} - D \Delta u +
   * \mathbf\beta\cdot\nabla u - r u^2 = f\;. @f]
   *
   * See @ref ExampleMultidomainHeatNonLinear .
   */
  class ExampleHeatNonLinear : public CoreModel
  {
    /// This class needs access to the dof_handler, fe and owned_dofs members to
    /// build interface maps between subdomains.
    friend class ExampleMultidomainHeatNonLinear;

  public:
    /// Constructor.
    ExampleHeatNonLinear(
      const unsigned int &                                    subdomain_id_,
      const unsigned int &                                    bdf_order_,
      const double &                                          initial_time_,
      const double &                                          time_step_,
      const std::vector<utils::BC<utils::FunctionDirichlet>> &dirichlet_bcs_,
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
      const std::vector<std::shared_ptr<utils::InterfaceHandler<
        LinAlg::MPI::Vector>::InterfaceDataRobinNonLinear>>
        interface_data_robin);

    /// Advance time.
    void
    time_advance();

    /// Set solution.
    void
    set_solution(const LinAlg::MPI::Vector &src);

  protected:
    /// Create the mesh.
    void
    create_mesh(const unsigned int &n_refinements);

    /// Setup the system.
    void
    setup_system();

    /// Assemble the jacobian matrix and residual vector.
    void
    assemble_system(const bool &assemble_jacobian = true);

    /// Assemble a single cell.
    void
    assemble_cell(const DoFHandler<dim>::active_cell_iterator &cell,
                  const unsigned int &                         c,
                  const std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> &                        cell_matrix,
                  Vector<double> &                            cell_rhs,
                  const bool &assemble_jacobian = true);

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

    LinAlg::MPI::SparseMatrix jacobian; ///< Linear system matrix.
    LinAlg::MPI::Vector       solution; ///< Linear system solution.
    LinAlg::MPI::Vector
                        solution_owned; ///< Linear system solution, without ghosts.
    LinAlg::MPI::Vector solution_old; ///< Solution at previous time step.
    LinAlg::MPI::Vector residual;     ///< Linear system right hand side.
    LinAlg::MPI::Vector
      residual_no_interface; ///< Residual without interface contributions.

    LinAlg::MPI::Vector u_bdf; ///< BDF solution, with ghost entries.

    utils::NonLinearSolverHandler<LinAlg::MPI::Vector>
      non_linear_solver; ///< Non-linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::Vector>
                                 linear_solver;  ///< Linear solver handler.
    utils::PreconditionerHandler preconditioner; ///< Preconditioner handler.

    utils::BDFHandler<LinAlg::MPI::Vector>
      bdf_handler; ///< Handler for time advancing.

    AffineConstraints<double>
                              constraints; ///< Affine constraints for Dirichlet boundary conditions.
    AffineConstraints<double> zero_constraints; ///< Zero constraints.

    /// Affine constraints for the linear system, without constraints for
    /// interface conditions.
    AffineConstraints<double> constraints_no_interface;

    IndexSet owned_dofs;    ///< Locally owned DoFs.
    IndexSet relevant_dofs; ///< Locally relevant DoFs.

    unsigned int bdf_order;           ///< BDF order.
    double       time;                ///< Current time.
    double       time_step;           ///< Timestep.
    unsigned int timestep_number = 0; ///< Number of timesteps.

    std::vector<utils::BC<utils::FunctionDirichlet>>
                     dirichlet_bcs; ///< Dirichlet boundary conditions.
    utils::BCHandler bc_handler;    ///< Handler for boundary conditions.
    std::unique_ptr<utils::InterfaceHandler<LinAlg::MPI::Vector>>
      interface_handler; ///< Handler for interface conditions.

    /// @name Parameters read from file.
    /// @{

    double         prm_diffusion_coefficient; ///< Diffusion coefficient.
    Tensor<1, dim> prm_transport_coefficient; ///< Transport coefficient.
    double         prm_reaction_coefficient;  ///< Reaction coefficient.

    /// @}
  };
} // namespace lifex::examples

#endif /* LIFEX_EXAMPLE_HEAT_NONLINEAR_HPP_ */
