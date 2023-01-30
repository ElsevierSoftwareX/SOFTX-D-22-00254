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

#ifndef LIFEX_EXAMPLE_MULTIDOMAIN_HEAT_HPP_
#define LIFEX_EXAMPLE_MULTIDOMAIN_HEAT_HPP_

#include "source/numerics/fixed_point_acceleration.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

#include "examples/multidomain_heat/heat.hpp"

#include <memory>
#include <string>
#include <vector>

namespace lifex::examples
{
  /**
   * @brief Class for multidomain solution of heat equation.
   *
   * Solves the following boundary value problem:
   * @f[ \left\{\begin{alignedat}{2}
   *  \frac{\partial u}{\partial t} -\Delta u + \mathbf\beta\cdot\nabla u &= f &
   * \quad & \text{in}\;\Omega\times(0,T) \\
   *  \nabla u \cdot \mathbf n & = 0 & \quad & \text{on}\;\partial\Omega
   * \times(0,T) \\
   * u(t = 0) & = 0 & \quad & \text{in}\;\Omega \end{alignedat}\right. @f]
   * where @f[ \Omega = (-0.5, 1.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,. @f]
   *
   * @f$\Omega@f$ is split into the two subdomains @f$\Omega_0@f$ and
   * @f$\Omega_1@f$ along the interface @f$\Sigma@f$, defined as
   * @f[ \begin{gather}
   *  \Omega_0 = (-0.5, 0.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Omega_1 = (0.5, 1.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Sigma = \left\{x = 0.5\right\}
   * \end{gather} @f]
   * Denoting with @f$u_i@f$ the solution on the subdomain @f$\Omega_i@f$, the
   * multidomain formulation of the problem is the following:
   * @f[ \left\{ \begin{alignedat}{2}
   *  \frac{\partial u_0}{\partial t} -\Delta u_0 + \mathbf\beta\cdot\nabla u_0
   * & = f|_{\Omega_0} & \quad & \text{in}\;\Omega_0\times(0,T) \\
   *  \nabla u_0 \cdot \mathbf n_0 & = 0 & \quad &
   * \text{on}\;(\partial\Omega_0\backslash\Sigma)\times(0,T) \\
   *  u_0 & = u_1 & \quad & \text{on}\;\Sigma\times(0,T) \\
   *  \nabla u_0 \cdot \mathbf n_0 & = -\nabla u_1 \cdot \mathbf n_1 & \quad &
   * \text{on}\;\Sigma\times(0,T) \\ \frac{\partial u_1}{\partial t} - \Delta
   * u_1 + \mathbf\beta\cdot\nabla u_1 & = f|_{\Omega_1} & \quad &
   * \text{in}\;\Omega_1\times(0,T) \\
   *  \nabla u_1 \cdot \mathbf n_1 & = 0 & \quad &
   * \text{on}\;(\partial\Omega_1\backslash\Sigma)\times(0,T) \\
   *  u_0(t=0) & = 0 & \quad & \text{in}\;\Omega_0 \\
   *  u_1(t=0) & = 0 & \quad & \text{in}\;\Omega_1
   * \\ \end{alignedat}\right. @f]
   *
   * At each timestep, two coupling schemes are available for the solution: the
   * fixed point scheme, and a monolithic scheme. The scheme can be chosen
   * setting the parameter <kbd>Scheme</kbd>.
   *
   * * **fixed point scheme**: at each timestep, the problem is solved iterating
   * on the two subdomains
   * with a staggered algorithm as described in @ref utils::InterfaceHandler. At
   * each iteration, each of the interface conditions is assigned to one of the
   * two subdomains. Which interface condition is assigned to which subdomain
   * can be specified in the parameter file. On top of the Dirichlet and Neumann
   * conditions shown above, it is possible to apply a Robin condition in the
   * form:
   * @f[ \alpha u_0 + \nabla u_0 \cdot \mathbf n_0 = \alpha u_1 - \nabla u_1
   * \cdot \mathbf n_1\;. @f]
   * If Robin interface conditions are applied to both subdomains, the
   * @f$\alpha@f$ coefficients must be different for the two subdomains.
   * <br> At each iteration, the interface data obtained evaluating the solution
   * on @f$\Omega_1@f$ is relaxed to guarantee convergence. Denoting with
   * @f$\lambda^{(k)}@f$ the interface datum (Neumann, Dirichlet or Robin)
   * applied to subdomain @f$\Omega_0@f$ at the k-th iteration,
   * @f[ \lambda^{(k+1)} = \omega u_1|_\Sigma + (1 - \omega)\lambda^{(k)}\;, @f]
   * where @f$\omega \in (0,1]@f$ is the relaxation parameter, specified in the
   * parameter file. At each timestep, the initial guess @f$\lambda^{(0)}@f$ is
   * its last value from previous timestep. Iterations end when the norm of
   * @f$\lambda^{(k+1)} - \lambda^{(k)}@f$ falls below a given tolerance.
   * * **monolithic scheme**: a single linear system is assembled containing the
   * unknown for both subdomains; interface DoFs from the two subdomains are
   * constrained to being equal, thus enforcing the continuity of the solution
   * across the interface. The weak continuity of the derivative follows
   * naturally, and no extra terms are needed.
   */
  class ExampleMultidomainHeat : public CoreModel
  {
  public:
    /// Interface condition type enumeration.
    enum class InterfaceType
    {
      Dirichlet, ///< Dirichlet interface conditions.
      Neumann,   ///< Neumann interface conditions.
      Robin      ///< Robin interface conditions.
    };

    /// Constructor.
    ExampleMultidomainHeat(const std::string &subsection,
                           const std::string &subsection_subproblem_0_,
                           const std::string &subsection_subproblem_1_);

    /// Setup and solve the problem.
    virtual void
    run() override;

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

  protected:
    /// Create the meshes.
    void
    create_mesh(const unsigned int &n_refinements);

    /// Setup the system.
    void
    setup_system();

    /// Build the interface maps.
    void
    setup_interface_maps();

    /// Setup the monolithic system.
    void
    setup_system_monolithic();

    /// Assemble the monolithic system.
    void
    assemble_monolithic();

    /// Solve the monolithic system.
    void
    solve_monolithic();

    /// Output results.
    void
    output_results();

    /// Problems defined on the two subdomains.
    std::array<std::unique_ptr<ExampleHeat>, 2> subproblems;

    /// Interface maps for DoFs.
    std::array<std::shared_ptr<utils::InterfaceMap>, 2> interface_maps;

    /// Index set for owned interface DoFs.
    IndexSet owned_interface_dofs;

    /// Index set for relevant interface DoFs.
    IndexSet relevant_interface_dofs;

    std::array<types::boundary_id, 2> interface_tag; ///< Interface tags.

    /// File and subsection for parameters of subproblem 0.
    std::string subsection_subproblem_0;

    /// File and subsection for parameters of subproblem 1.
    std::string subsection_subproblem_1;

    /// @name Fixed point solver
    /// @{

    /// Fixed point acceleration scheme.
    std::unique_ptr<utils::FixedPointAcceleration<LinAlg::MPI::Vector>>
      fixed_point_acceleration;

    double res_norm = 0.0;   ///< Norm of the interface residual.
    double weight_dirichlet; ///< Weight of Dirichlet term in the residual.
    double weight_neumann;   ///< Weight of Neumann term in the residual.
    unsigned int n_iter = 0; ///< Number of iterations.

    /// Interface mass matrix.
    std::shared_ptr<LinAlgTrilinos::Wrappers::SparseMatrix>
      interface_mass_matrix;

    /// @}

    /// @name Monolithic solver
    /// @{

    /// Matrix of the monolithic system.
    LinAlg::MPI::BlockSparseMatrix system_matrix;

    /// RHS of the monolithic system.
    LinAlg::MPI::BlockVector system_rhs;

    /// Solution of the monolithic system.
    LinAlg::MPI::BlockVector solution;

    /// Solution of the monolithic system, owned DoFs.
    LinAlg::MPI::BlockVector solution_owned;

    /// Constraints for the monolithic system.
    AffineConstraints<double> constraints;

    std::vector<IndexSet> block_owned_dofs;    ///< Owned DoFs for each block.
    std::vector<IndexSet> block_relevant_dofs; ///< Relevant DoFs for each block
    IndexSet              owned_dofs_global; ///< Owned DoFs of the whole system
    IndexSet relevant_dofs_global; ///< Relevant DoFs of the whole system.

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::BlockVector> linear_solver;

    /// Linear solver preconditioner.
    utils::BlockPreconditionerHandler preconditioner;

    /// Number of DoFs in each subdomain.
    std::vector<unsigned int> n_dofs;

    /// @}

    /// @name Parameters read from file.
    /// @{

    std::string prm_scheme; ///< Scheme (fixed point or monolithic).

    std::array<InterfaceType, 2> prm_fixed_point_ICs; ///< Interface types.
    double       prm_tolerance;  ///< Tolerance on res_norm for convergence.
    unsigned int prm_n_max_iter; ///< Maximum number of iterations.

    std::array<double, 2>
      prm_robin_coefficient; ///< Coefficients for Robin interface conditions.

    Tensor<1, dim> prm_transport_coefficient; ///< Transport coefficient.

    unsigned int prm_n_refinements; ///< Number of refinements on the mesh.

    /// Fixed point acceleration scheme.
    std::string prm_fixed_point_acceleration_type;

    /// @}
  };
} // namespace lifex::examples

#endif /* LIFEX_EXAMPLE_MULTIDOMAIN_HEAT_HPP_ */
