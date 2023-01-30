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

#ifndef LIFEX_EXAMPLE_STOKES_HPP_
#define LIFEX_EXAMPLE_STOKES_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/bc_handler.hpp"
#include "source/numerics/interface_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace lifex::examples
{
  /// @brief Class for the solution of Stokes problem on a subdomain.
  ///
  /// See @ref ExampleMultidomainStokes.
  class ExampleStokes : public CoreModel
  {
  public:
    friend class ExampleMultidomainStokes;

    /**
     * @brief Preconditioner class.
     *
     * The discretized Stokes problem has saddle-point form
     * @f[
     * \begin{bmatrix} A & B \\ B^T & 0 \end{bmatrix}
     * \begin{bmatrix} \mathbf{U} \\ \mathbf{P} \end{bmatrix} =
     * \begin{bmatrix} \mathbf{f}_\mathbf{u} \\ \mathbf{f}_p \end{bmatrix}\;.
     * @f]
     * This class represents the following preconditioner:
     * @f[
     * P = \begin{bmatrix} A & B \\ 0 & \frac{1}{\mu} M_p \end{bmatrix}\;,
     * @f]
     * where @f$M_p@f$ is the pressure mass matrix. The inverses of @f$A@f$ and
     * @f$M_p@f$ are computed with an inner iterative solver by means of
     * @ref utils::InverseMatrix.
     */
    class Preconditioner
    {
    public:
      /// Constructor.
      Preconditioner(
        const utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                   LinAlg::Wrappers::PreconditionILU>
          &inverse_A_,
        const utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                   LinAlg::Wrappers::PreconditionILU>
          &                              inverse_Mp_,
        const LinAlg::MPI::SparseMatrix &B_)
        : inverse_A(inverse_A_)
        , inverse_Mp(inverse_Mp_)
        , B(B_)
      {}

      /// Multiplication by a vector.
      void
      vmult(LinAlg::MPI::BlockVector &      dst,
            const LinAlg::MPI::BlockVector &src) const
      {
        LinAlg::MPI::Vector tmp(B.locally_owned_range_indices(), mpi_comm);

        inverse_Mp.vmult(dst.block(1), src.block(1));
        B.vmult(tmp, dst.block(1));
        tmp.sadd(-1.0, src.block(0));
        inverse_A.vmult(dst.block(0), tmp);
      }

    protected:
      /// Object to apply the inverse of the velocity-velocity block.
      const utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                 LinAlg::Wrappers::PreconditionILU> &inverse_A;

      /// Object to apply the inverse of the pressure mass matrix.
      const utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                 LinAlg::Wrappers::PreconditionILU> &inverse_Mp;

      /// Velocity-pressure block.
      const LinAlg::MPI::SparseMatrix &B;
    };

    /// Constructor.
    ExampleStokes(
      const unsigned int &                                    subdomain_id_,
      const double &                                          viscosity,
      const std::vector<utils::BC<utils::FunctionDirichlet>> &dirichlet_bcs_,
      const std::string &                                     subsection);

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Setup interface handler and interface conditions.
    void
    setup_interface(const DoFHandler<dim> &other_dof_handler,
                    const std::vector<std::shared_ptr<utils::InterfaceHandler<
                      LinAlg::MPI::BlockVector>::InterfaceDataDirichlet>>
                      interface_data_dirichlet,
                    const std::vector<std::shared_ptr<utils::InterfaceHandler<
                      LinAlg::MPI::BlockVector>::InterfaceDataNeumann>>
                      interface_data_neumann,
                    const std::vector<std::shared_ptr<utils::InterfaceHandler<
                      LinAlg::MPI::BlockVector>::InterfaceDataRobinLinear>>
                      interface_data_robin);

    /// Assemble, solve and output the results.
    void
    step(const unsigned int &n_iter);

    /// Compute the interface residual.
    void
    compute_interface_residual();

    /// Set solution vector.
    void
    set_solution(const LinAlg::MPI::BlockVector &src)
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

    /// Assemble local contributions of a single cell.
    void
    assemble_cell(const DoFHandler<dim>::active_cell_iterator &cell,
                  FullMatrix<double> &                         cell_matrix,
                  Vector<double> &                             cell_rhs);

    /// Assemble the pressure mass matrix.
    void
    assemble_pressure_mass_matrix();

    /// Solve the linear system.
    void
    solve();

    /// Output the results.
    void
    output_results(const unsigned int &n_iter) const;

    unsigned int subdomain_id; ///< Subdomain identifier

    utils::MeshHandler             triangulation; ///< Triangulation.
    DoFHandler<dim>                dof_handler;   ///< DoF handler.
    std::unique_ptr<FESystem<dim>> fe;            ///< Finite element space.
    std::unique_ptr<Quadrature<dim>>
      quadrature_formula; ///< Quadrature formula.
    std::unique_ptr<Quadrature<dim - 1>>
      face_quadrature_formula; ///< Face quadrature formula.

    LinAlg::MPI::BlockSparseMatrix system_matrix; ///< Linear system matrix.
    LinAlg::MPI::BlockVector       solution;      ///< Linear system solution.
    LinAlg::MPI::BlockVector
                             solution_owned; ///< Linear system solution, owned elements.
    LinAlg::MPI::BlockVector system_rhs; ///< Linear system right hand side.
    LinAlg::MPI::BlockSparseMatrix
      pressure_mass_matrix; ///< Pressure mass matrix Mp.

    /// Matrix of the linear system, without the interface contributions.
    LinAlg::MPI::BlockSparseMatrix system_matrix_no_interface;

    /// RHS of the linear system without the interface contribution.
    LinAlg::MPI::BlockVector system_rhs_no_interface;

    /// Residual of the linear system without the interface contributions.
    LinAlg::MPI::BlockVector residual_no_interface;

    LinAlg::Wrappers::PreconditionILU precondition_A; ///< Preconditioner for A.
    LinAlg::Wrappers::PreconditionILU
      precondition_Mp; ///< Preconditioner for Mp.
    std::unique_ptr<utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                         LinAlg::Wrappers::PreconditionILU>>
      inverse_A; ///< Inverse of A.
    std::unique_ptr<utils::InverseMatrix<LinAlg::MPI::SparseMatrix,
                                         LinAlg::Wrappers::PreconditionILU>>
      inverse_Mp; ///< Inverse of Mp.

    AffineConstraints<double>
      constraints; ///< Affine constraints for the linear system.

    /// Constraints without the interface contributions (i.e. without the
    /// Dirichlet interface conditions).
    AffineConstraints<double> constraints_no_interface;

    IndexSet              owned_dofs;          ///< Locally owned DoFs.
    std::vector<IndexSet> block_owned_dofs;    ///< Owned DoFs for each block.
    IndexSet              relevant_dofs;       ///< Locally relevant DoFs.
    std::vector<IndexSet> block_relevant_dofs; ///< Relevant DoFs for each block

    std::vector<utils::BC<utils::FunctionDirichlet>>
                     dirichlet_bcs; ///< Dirichlet boundary conditions.
    utils::BCHandler bc_handler;    ///< Handler for boundary conditions.
    std::unique_ptr<utils::InterfaceHandler<LinAlg::MPI::BlockVector>>
      interface_handler; ///< Handler for interface conditions.

    double viscosity; ///< Viscosity.

    /// Linear solver handler.
    utils::LinearSolverHandler<LinAlg::MPI::BlockVector> linear_solver;
  };
} // namespace lifex::examples

#endif /* LIFEX_EXAMPLE_STOKES_HPP_ */
