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

#ifndef LIFEX_EXAMPLE_MULTIDOMAIN_STOKES_HPP_
#define LIFEX_EXAMPLE_MULTIDOMAIN_STOKES_HPP_

#include "source/numerics/fixed_point_acceleration.hpp"

#include "examples/multidomain_stokes/stokes.hpp"

#include <memory>
#include <string>
#include <vector>

namespace lifex::examples
{
  /**
   * @brief Solution of Stokes problem on multiple subdomains.
   *
   * The problem being solved is:
   * @f[ \left\{\begin{alignedat}{2}
   * -\mu \Delta \mathbf{u} + \nabla p & = 0 & \quad & \text{in}\;\Omega \\
   * \nabla\cdot\mathbf{u} & = 0 & \quad & \text{in}\;\Omega \\
   * \mathbf{u} & = (1, 0, 0)^T & \quad & \text{on}\;\Gamma_{\mathrm{in}} \\
   * \mathbf{u}  & = (0, 0, 0)^T & \quad & \text{on}\;\Gamma_{\mathrm{sides}} \\
   * \nabla \mu \mathbf{u} \mathbf{n} - p\mathbf{n} & = 0 & \quad &
   * \text{on}\;\Gamma_{\mathrm{out}} \end{alignedat}\right. @f]
   * where @f[ \begin{gather}
   *  \Omega = (-0.5, 1.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Gamma_{\mathrm{in}} = \left\{ x = -0.5 \right\}\,,\\
   *  \Gamma_{\mathrm{out}} = \left\{ x = 1.5 \right\}\,,\\
   *  \Gamma_{\mathrm{sides}} =
   * (\partial\Omega\backslash(\Gamma_{\mathrm{in}}\cup\Gamma_{\mathrm{out}}))^{\mathrm{o}}\,.
   * \end{gather} @f]
   *
   * @f$\Omega@f$ is split into the two subdomains @f$\Omega_0@f$ and
   * @f$\Omega_1@f$ along the interface @f$\Sigma@f$, defined as
   * @f[ \begin{gather}
   *  \Omega_0 = (-0.5, 0.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Omega_1 = (0.5, 1.5)\times(-0.5, 0.5)\times(-0.5, 0.5)\,,\\
   *  \Sigma = \left\{x = 0.5\right\}
   * \end{gather} @f]
   *
   * Denoting with @f$\mathbf{u}_i,\, p_i@f$ the solution on the subdomain
   * @f$\Omega_i@f$, the multidomain formulation of the problem is the
   * following:
   * @f[ \left\{ \begin{alignedat}{2}
   * -\mu \Delta \mathbf{u}_i + \nabla p_i & = 0 & \quad & \text{in}\;\Omega_i
   * \\
   * \nabla\cdot\mathbf{u}_i & = 0 & \quad & \text{in}\;\Omega_i \\
   * \mathbf{u}_0 & = (1, 0, 0)^T & \quad & \text{on}\;\Gamma_{\mathrm{in}} \\
   * \mathbf{u}_i & = (0, 0, 0)^T & \quad &
   * \text{on}\;\Gamma_{\mathrm{sides}}\cap\partial\Omega_i
   * \\ \nabla \mu \mathbf{u}_1 \mathbf{n} - p\mathbf{n} & = 0 & \quad &
   * \text{on}\;\Gamma_{\mathrm{out}} \\
   * \mathbf{u}_0 & = \mathbf{u}_1 & \quad & \text{on}\Sigma \\
   * -\mu \nabla \mathbf{u}_0 \mathbf{n}_0 - p_0 \mathbf{n}_0 & = \mu \nabla
   * \mathbf{u}_1 \mathbf{n}_1 + p_1 \mathbf{n}_1 & \quad & \text{on}\Sigma
   * \end{alignedat} \right.
   * @f]
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
   * @f[ \alpha\mathbf u_0 + \nabla \mathbf u_0 \mathbf n_0 - p_0 \mathbf n_0 =
   * \alpha \mathbf u_1 - \nabla\mathbf u_1 \mathbf n_1 + p_1 \mathbf n_1\;. @f]
   * If Robin interface conditions are applied to both subdomains, the
   * @f$\alpha@f$ coefficients must be different for the two subdomains.
   * * **monolithic scheme**: a single linear system is assembled containing the
   * unknown for both subdomains; interface DoFs from the two subdomains are
   * constrained to being equal, thus enforcing the continuity of the solution
   * across the interface. The weak continuity of the derivative follows
   * naturally, and no extra terms are needed.
   */
  class ExampleMultidomainStokes : public CoreModel
  {
  public:
    /// @brief Preconditioner class for the monolithic system. Block triangular
    /// preconditioner on each subdomain's block.
    template <class PreconditionerA, class PreconditionerS, class BTranspose>
    class Preconditioner
    {
    public:
      /// Constructor.
      Preconditioner(const PreconditionerA &prec_A_0_,
                     const PreconditionerS &prec_S_0_,
                     const BTranspose &     BT_0_,
                     const PreconditionerA &prec_A_1_,
                     const PreconditionerS &prec_S_1_,
                     const BTranspose &     BT_1_)
        : prec_A_0(prec_A_0_)
        , prec_S_0(prec_S_0_)
        , BT_0(BT_0_)
        , prec_A_1(prec_A_1_)
        , prec_S_1(prec_S_1_)
        , BT_1(BT_1_)
      {}

      /// Application of the preconditioner.
      void
      vmult(LinAlg::MPI::BlockVector &      dst,
            const LinAlg::MPI::BlockVector &src) const
      {
        LinAlg::MPI::BlockVector tmp(src);
        tmp = 0;

        /// Subdomain 0.
        {
          prec_S_0.vmult(tmp.block(1), src.block(1));
          BT_0.vmult(tmp.block(0), tmp.block(1));
          tmp.block(0).sadd(-1.0, src.block(0));
          prec_A_0.vmult(dst.block(0), tmp.block(0));

          prec_S_0.vmult(dst.block(1), src.block(1));
        }

        /// Subdomain 1.
        {
          prec_S_1.vmult(tmp.block(3), src.block(3));
          BT_0.vmult(tmp.block(2), tmp.block(3));
          tmp.block(2).sadd(-1.0, src.block(2));
          prec_A_1.vmult(dst.block(2), tmp.block(2));

          prec_S_1.vmult(dst.block(3), src.block(3));
        }
      }

    protected:
      const PreconditionerA &prec_A_0; ///< Preconditioner for A on subdomain 0.
      const PreconditionerS &prec_S_0; ///< Preconditioner for S on subdomain 0.
      const BTranspose &     BT_0;     ///< B^T on subdomain 0.

      const PreconditionerA &prec_A_1; ///< Preconditioner for A on subdomain 1.
      const PreconditionerS &prec_S_1; ///< Preconditioner for S on subdomain 1.
      const BTranspose &     BT_1;     ///< B^T on subdomain 1.
    };

    /// Interface condition type enumeration.
    enum class InterfaceType
    {
      Dirichlet, ///< Dirichlet interface conditions.
      Neumann,   ///< Neumann interface conditions.
      Robin      ///< Robin interface conditions.
    };

    /// Constructor.
    ExampleMultidomainStokes(const std::string &subsection,
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

    /// Build the interface maps and initialize interface handlers if needed.
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

    /// Problems defined on the two subdomains.
    std::array<std::unique_ptr<ExampleStokes>, 2> subproblems;

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
    std::unique_ptr<utils::FixedPointAcceleration<LinAlg::MPI::BlockVector>>
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

    /// @}

    /// @name Parameters read from file.
    /// @{

    double prm_viscosity; ///< Viscosity.

    std::string prm_scheme; ///< Scheme (fixed point or monolithic).

    std::array<InterfaceType, 2> prm_fixed_point_ICs; ///< Interface types.
    double       prm_tolerance;  ///< Tolerance on res_norm for convergence.
    unsigned int prm_n_max_iter; ///< Maximum number of iterations.

    Tensor<1, dim> prm_transport_coefficient; ///< Transport coefficient.

    unsigned int prm_n_refinements; ///< Number of refinements on the mesh.

    std::array<double, 2>
      prm_robin_coefficient; ///< Coefficients for Robin interface conditions.

    /// Type of fixed point acceleration scheme.
    std::string prm_fixed_point_acceleration_type;

    /// @}
  };
} // namespace lifex::examples

#endif /* LIFEX_EXAMPLE_MULTIDOMAIN_STOKES_HPP_ */
