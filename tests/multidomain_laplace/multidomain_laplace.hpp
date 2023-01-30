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

#ifndef LIFEX_TEST_MULTIDOMAIN_LAPLACE_HPP_
#define LIFEX_TEST_MULTIDOMAIN_LAPLACE_HPP_

#include "source/numerics/fixed_point_acceleration.hpp"
#include "source/numerics/interface_handler.hpp"

#include "tests/multidomain_laplace/laplace.hpp"

#include <memory>
#include <string>
#include <vector>

namespace lifex::tests
{
  /**
   * @brief Class for multidomain solution of a Laplace problem.
   *
   * Solves the following boundary value problem:
   * @f[ \left\{ \begin{alignedat}{2}
   *  -\Delta u + \mathbf\beta\cdot\nabla u & = 0 & \quad & \text{in}\;\Omega \\
   *  u & = u_{\mathrm{D}} & \quad & \text{on}\;\Gamma_{\mathrm{D}} \\
   *  \nabla u \cdot \mathbf n & = 0 & \quad & \text{on}\;\Gamma_{\mathrm{N}}
   * \end{alignedat} \right. @f]
   * where @f[ \begin{gather}
   *  \Omega = (-0.5, 1.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Gamma_{\mathrm{D},\mathrm{left}} = \left\{ x = -0.5 \right\}\,,\\
   *  \Gamma_{\mathrm{D},\mathrm{right}} = \left\{ x = 1.5 \right\}\,,\\
   *  \Gamma_{\mathrm{D}} =
   * \Gamma_{\mathrm{D},\mathrm{left}}\cup\Gamma_{\mathrm{D},\mathrm{right}}\,,\\
   *  \Gamma_{\mathrm{N}} =
   * (\partial\Omega\backslash\Gamma_{\mathrm{D}})^{\mathrm{o}}\,. \end{gather}
   * @f]
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
   *  -\Delta u_0 + \mathbf\beta\cdot\nabla u_0 & = 0 & \quad &
   * \text{in}\;\Omega_0 \\
   *  u_0 & = u_{\mathrm{D},\mathrm{left}} & \quad &
   * \text{on}\;\Gamma_{\mathrm{D},\mathrm{left}} \\ \nabla u_0 \cdot \mathbf
   * n_0 & = 0 & \quad & \text{on}\;\Gamma_{\mathrm{N}}\cap\partial\Omega_0
   * \\
   *  u_0 & = u_1 & \quad & \text{on}\;\Sigma \\
   *  \nabla u_0 \cdot \mathbf n_0 & = -\nabla u_1 \cdot \mathbf n_1 & \quad &
   * \text{on}\;\Sigma
   * \\
   *  -\Delta u_1 + \mathbf\beta\cdot\nabla u_1 & = 0 & \quad &
   * \text{in}\;\Omega_1 \\
   *  u_1 & = u_{\mathrm{D},\mathrm{right}} & \quad &
   * \text{on}\;\Gamma_{\mathrm{D},\mathrm{right}} \\ \nabla u_1 \cdot \mathbf
   * n_1 & = 0 & \quad & \text{on}\;\Gamma_{\mathrm{N}}\cap\partial\Omega_1
   * \\ \end{alignedat}\right . @f]
   *
   * Two coupling schemes are available for the solution: the fixed point
   * scheme, and a monolithic scheme. The scheme can be chosen setting the
   * parameter <kbd>Scheme</kbd>.
   *
   * * **fixed point scheme**: the problem is solved iterating on the two
   * subdomains with a staggered algorithm as described in @ref utils::InterfaceHandler.
   * At each iteration, each of the interface conditions is assigned to one of
   * the two subdomains. Which interface condition is assigned to which
   * subdomain can be specified in the parameter file. On top of the Dirichlet
   * and Neumann conditions shown above, it is possible to apply equivalently a
   * Robin condition in the form:
   * @f[ \alpha u_0 + \nabla u_0 \cdot \mathbf n_0 = \alpha u_1 - \nabla u_1
   * \cdot \mathbf n_1\;. @f]
   * If Robin interface conditions are applied to both subdomains, the
   * @f$\alpha@f$ coefficients must be different for the two subdomains.
   * * **monolithic scheme**: a single linear system is assembled containing the
   * unknown for both subdomains; interface DoFs from the two subdomains are
   * constrained to being equal, thus enforcing the continuity of the solution
   * across the interface. The weak continuity of the derivative follows
   * naturally, and no extra terms are needed.
   *
   * If @f$\mathbf\beta = 0@f$, the exact solution to the problem can be
   * computed. The numerical error will be computed, and the test will pass if
   * such error is below the specified tolerance. Error checking for
   * @f$\mathbf\beta\neq 0@f$ is not implemented, and the test will fail in that
   * case.
   */
  class TestMultidomainLaplace : public CoreModel
  {
  public:
    /// Constant Dirichlet data class.
    class DirichletData : public utils::FunctionDirichlet
    {
    public:
      /// Constructor.
      DirichletData(const std::vector<double> &dirichlet_data_);

      /// Evaluation.
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /// Vector evaluation.
      virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    protected:
      /// Vector of Dirichlet values, one for each component.
      std::vector<double> dirichlet_data;
    };

    /// Exact solution class.
    class ExactSolution : public Function<dim>
    {
    protected:
      std::vector<double>
        dirichlet_data_left; ///< Dirichlet value for all components on the
                             ///< left face (x = -0.5).
      std::vector<double>
        dirichlet_data_right; ///< Dirichlet value for all components on the
                              ///< right face (x = 1.5).
    public:
      /// Constructor.
      ExactSolution(const unsigned int &n_components,
                    std::vector<double> dirichlet_data_left_,
                    std::vector<double> dirichlet_data_right_);

      /// Evaluation of the function.
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /// Vector evaluation of the function.
      virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;
    };

    /// Interface condition type enumeration.
    enum class InterfaceType
    {
      Dirichlet, ///< Dirichlet interface conditions.
      Neumann,   ///< Neumann interface conditions.
      Robin      ///< Robin interface conditions.
    };

    /// Constructor.
    TestMultidomainLaplace(const std::string &subsection,
                           const std::string &subsection_subproblem_0_,
                           const std::string &subsection_subproblem_1_);

    /// Run the test. Throw an exception if the L2 error against the exact
    /// solution is above the tolerance.
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

    /// Build the interface maps and the interface handlers if needed.
    void
    setup_interface();

    /// Setup the monolithic system.
    void
    setup_system_monolithic();

    /// Assemble the monolithic system.
    void
    assemble_monolithic();

    /// Solve the monolithic system.
    void
    solve_monolithic();

    /// Compute the error between computed and exact solution.
    void
    compute_error();

    /// Problems defined on the two subdomains.
    std::array<std::unique_ptr<TestLaplace>, 2> subproblems;

    /// Interface maps for DoFs.
    std::array<std::shared_ptr<utils::InterfaceMap>, 2> interface_maps;

    /// Index set for owned interface DoFs.
    IndexSet owned_interface_dofs;

    /// Index set for relevant interface DoFs.
    IndexSet relevant_interface_dofs;

    std::array<types::boundary_id, 2> interface_tag; ///< Interface tags.

    std::unique_ptr<ExactSolution> exact_solution; ///< Exact solution.
    double                         error = 0.0;    ///< Error.

    /// File and subsection for parameters of subproblem 0.
    std::string subsection_subproblem_0;

    /// File and subsection for parameters of subproblem 1.
    std::string subsection_subproblem_1;

    /// @name Fixed point scheme
    /// @{

    /// Fixed point acceleration scheme.
    std::unique_ptr<utils::FixedPointAcceleration<LinAlg::MPI::Vector>>
      fixed_point_acceleration;

    double res_norm = 0.0;   ///< Increment on interface data.
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

    /// Number of DoFs in each subdomain.
    std::vector<unsigned int> n_dofs;

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::BlockVector> linear_solver;

    /// Preconditioner handler.
    utils::BlockPreconditionerHandler preconditioner;


    /// @}

    /// @name Parameters read from file.
    /// @{

    std::string prm_scheme; ///< Scheme for the multidomain problem.

    std::array<InterfaceType, 2>
      prm_fixed_point_ICs;       ///< Type of interface conditions for the fixed
                                 ///< point scheme.
    double       prm_tolerance;  ///< Tolerance on res_norm for convergence.
    unsigned int prm_n_max_iter; ///< Maximum number of iterations.

    std::array<double, 2>
      prm_robin_coefficient; ///< Coefficients for Robin interface conditions.

    unsigned int
      prm_n_components; ///< Number of solution components of the subproblems.

    std::vector<double>
      prm_dirichlet_data_left; ///< Dirichlet value for all components on the
                               ///< left face (x = -0.5).
    std::vector<double>
      prm_dirichlet_data_right; ///< Dirichlet value for all components on the
                                ///< right face (x = 1.5).

    Tensor<1, dim> prm_transport_coefficient; ///< Transport coefficient.

    unsigned int prm_n_refinements; ///< Number of refinements on the mesh.

    double
      prm_error_tolerance; ///< Tolerance on the final error, to pass the test.

    /// Fixed point acceleration scheme.
    std::string prm_fixed_point_acceleration_type;

    /// @}
  };
} // namespace lifex::tests

#endif /* LIFEX_TEST_MULTIDOMAIN_LAPLACE_HPP_ */
