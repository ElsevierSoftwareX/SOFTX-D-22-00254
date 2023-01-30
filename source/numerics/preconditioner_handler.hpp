/********************************************************************************
  Copyright (C) 2020 - 2022 by the lifex authors.

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
 * @author Nicolas Alejandro Barnafi <nicolas.barnafi@unipv.it>.
 */

#ifndef LIFEX_UTILS_PRECONDITIONER_HANDLER_HPP_
#define LIFEX_UTILS_PRECONDITIONER_HANDLER_HPP_

#include "source/core_model.hpp"

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <memory>
#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Class representing the operation of multiplying a vector by the
   * inverse of a matrix.
   *
   * For a given matrix @f$M@f$, this allows to compute @f$\mathbf y =
   * M^{-1}\mathbf x@f$ by solving the system @f$M\mathbf y = \mathbf x@f$ using
   * GMRES.
   */
  template <class Matrix, class Preconditioner>
  class InverseMatrix : public Subscriptor
  {
  public:
    /// Constructor.
    InverseMatrix(const Matrix &matrix_, const Preconditioner &preconditioner)
      : matrix(matrix_)
      , preconditioner(preconditioner)
    {}

    /// Method to solve the linear system to invert the matrix.
    ///
    /// @param[out] dst @f$M^{-1}\mathbf y@f$.
    /// @param[in] src @f$\mathbf y@f$.
    template <class VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
      SolverGMRES<LinAlg::MPI::Vector> solver(solver_control);
      dst = 0;

      solver.solve(matrix, dst, src, preconditioner);
    }

  private:
    const Matrix &        matrix;         ///< The matrix to be inverted.
    const Preconditioner &preconditioner; ///< The preconditioner.
  };

  /// @brief Helper class to handle preconditioning abstractly.
  ///
  /// This class wraps the following types of preconditioner:
  /// - <kbd>None</kbd>: identity preconditioner;
  /// - Algebraic Multi-Grid (AMG) (from both Trilinos and PETSc);
  /// - Block Jacobi;
  /// - (Trilinos only) Additive Schwarz preconditioners:
  /// <kbd>PreconditionSOR</kbd>, <kbd>PreconditionSSOR</kbd>,
  /// <kbd>PreconditionBlockSOR</kbd>, <kbd>PreconditionBlockSSOR</kbd>,
  /// <kbd>PreconditionILU</kbd>, <kbd>PreconditionILUT</kbd>; all these
  /// preconditioners are actually Additive Schwarz preconditioners when run in
  /// parallel, using different inner solver (SOR, SSOR etc.) on the local
  /// matrices.
  ///
  /// For each possible option, the class provides an interface to declare and
  /// parse the corresponding parameters from file.
  ///
  /// @note An object of this class has to be destroyed after the matrix used
  /// to initialize it: consider declaring the preconditioner handler @b after
  /// the matrix used in the @ref initialize method.
  class PreconditionerHandler : public CoreModel, public Subscriptor
  {
  public:
    friend class BlockPreconditionerHandler;

    /// Alias for base preconditioner class.
    using PreconditionerBaseType =
#if defined(LIN_ALG_TRILINOS)
      LinAlg::Wrappers::PreconditionBase
#elif defined(LIN_ALG_PETSC)
      LinAlg::Wrappers::PreconditionerBase
#endif
      ;

    /// Constructor
    ///
    /// @param[in] subsection Parameter subsection path.
    /// @param[in] elliptic_  If compiling with Trilinos backend,
    ///                       enable optimizations for elliptic problems by
    ///                       default.
    PreconditionerHandler(const std::string &subsection,
                          const bool &       elliptic_ = false);

    /// Copy constructor.
    ///
    /// The parameters of the preconditioner passed as argument are copied,
    /// whereas the underlying preconditioner object is not. In particular, the
    /// constructed preconditioner will not be initialized, regardless of the
    /// state of the input one.
    PreconditionerHandler(const PreconditionerHandler &other);

    /// Assignment operator.
    /// The parameters of the preconditioner passed as argument are copied,
    /// whereas the underlying preconditioner object is not. In particular, the
    /// constructed preconditioner will not be initialized, regardless of the
    /// state of the input one.
    PreconditionerHandler &
    operator=(const PreconditionerHandler &other);

    /// Destructor.
    ~PreconditionerHandler() = default;

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;


    /// Initialize preconditioner data on <kbd>matrix</kbd>. Additional
    /// information required for constant modes in Trilinos AMG.
    void
    initialize(const LinAlg::MPI::SparseMatrix &matrix,
               const DoFHandler<dim> *          dof_handler = nullptr,
               const ComponentMask &component_mask          = ComponentMask());

    /// Method used to apply the chosen preconditioner.
    void
    vmult(LinAlg::MPI::Vector &dst, const LinAlg::MPI::Vector &src) const;

    /// Obtain the underlying preconditioner object. This is used for the
    /// PETSc::GMRES linear solver, which supports only PETSc preconditioners.
    std::shared_ptr<PreconditionerBaseType>
    get_preconditioner_base() const
    {
      return preconditioner;
    }

  private:
    /// Templated method for preconditioner construction and initialization.
    template <class PreconditionerType>
    void
    create_preconditioner(
      const LinAlg::MPI::SparseMatrix &                  matrix,
      const typename PreconditionerType::AdditionalData &additional_data)
    {
      if (!initialized)
        {
          preconditioner = std::make_shared<PreconditionerType>();
        }
      else
        {
          // If the preconditioner is already initialized, we clear it before
          // doing anything else. Otherwise, segfaults might happen during
          // preconditioner initialization.
          preconditioner->clear();
        }

      auto preconditioner_ptr =
        dynamic_cast<PreconditionerType *>(preconditioner.get());

      AssertThrow(preconditioner_ptr != nullptr, ExcLifexInternalError());

      preconditioner_ptr->initialize(matrix, additional_data);
    }

    /// Declare parameters of the AMG preconditioner.
    void
    declare_parameters_amg(ParamHandler &params) const;

    /// Parse parameters of the AMG preconditioner and store them in the
    /// prm_additional_data_amg variable.
    void
    parse_parameters_amg(ParamHandler &params);

// Additive Schwarz preconditioners are available only from TrilinosWrappers.
#if defined(LIN_ALG_TRILINOS)
    /// Declare parameters of the Additive Schwarz preconditioner.
    void
    declare_parameters_additive_schwarz(ParamHandler &params) const;

    /// Parse parameters of the Additive Schwarz preconditioner.
    void
    parse_parameters_additive_schwarz(ParamHandler &params);
#endif

    /// Boolean value to specify whether @ref initialize has been called at least once.
    bool initialized;

    /// @brief Structure to bind parameters read from file and additional data.
    ///
    /// This is used to make all such members trivially copyable at once.
    class Data
    {
    public:
      /// Boolean value to enable optimizations for elliptic problems by
      /// default, if compiling with Trilinos backend.
      bool elliptic;

      /// Type of preconditioner used.
      std::string prm_preconditioner_type;

#if defined(LIN_ALG_TRILINOS)
      /// @name Trilinos parameters.
      /// @{

      /// Smoother type.
      std::string prm_smoother_type;

      /// Coarse type.
      std::string prm_coarse_type;

      /// Boolean to enable using the constant modes of the matrix.
      bool prm_constant_modes;

      /// @}
#endif

      /// @name Preconditioners additional data.
      /// @{

      /// Additional data for identity preconditioner.
      LinAlg::Wrappers::
#if defined(LIN_ALG_TRILINOS)
        PreconditionIdentity
#elif defined(LIN_ALG_PETSC)
        PreconditionNone
#endif
        ::AdditionalData prm_additional_data_identity;

      /// AMG additional data.
      LinAlg::MPI::PreconditionAMG::AdditionalData prm_additional_data_amg;

      /// BlockJacobi additional data.
      LinAlg::Wrappers::PreconditionBlockJacobi::AdditionalData
        prm_additional_data_block_jacobi;

// Additive Schwarz preconditioners are available only from TrilinosWrappers.
#if defined(LIN_ALG_TRILINOS)
      /// Type of inner solver for Additive Schwarz.
      std::string prm_additive_schwarz_inner_solver;

      /// AdditiveSchwarz/SOR additional data.
      LinAlg::Wrappers::PreconditionSOR::AdditionalData prm_additional_data_sor;

      /// AdditiveSchwarz/SSOR additional data.
      LinAlg::Wrappers::PreconditionSSOR::AdditionalData
        prm_additional_data_ssor;

      /// AdditiveSchwarz/BlockSOR additional data.
      LinAlg::Wrappers::PreconditionBlockSOR::AdditionalData
        prm_additional_data_block_sor;

      /// AdditiveSchwarz/BlockSSOR additional data.
      LinAlg::Wrappers::PreconditionBlockSSOR::AdditionalData
        prm_additional_data_block_ssor;

      /// AdditiveSchwarz/ILU additional data.
      LinAlg::Wrappers::PreconditionILU::AdditionalData prm_additional_data_ilu;

      /// AdditiveSchwarz/ILUT additional data.
      LinAlg::Wrappers::PreconditionILUT::AdditionalData
        prm_additional_data_ilut;
#endif

      /// @}
    };

    /// Parameters and additional data.
    Data data;

    /// Pointer to preconditioner.
    /// @todo: PETSc base class name will be consistent with Trilinos
    /// since deal.II v9.3.0.
    std::shared_ptr<PreconditionerBaseType> preconditioner;
  };


  /**
   * @brief Block preconditioner.
   */
  class BlockPreconditionerHandler : public CoreModel, public Subscriptor
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param[in] subsection Corresponding subsection in parameter file.
     * @param[in] matrix_    Problem matrix.
     * @param[in] elliptic   If compiling with Trilinos backend,
     *                       enable optimizations for elliptic problems by
     *                       default.
     */
    BlockPreconditionerHandler(const std::string &                   subsection,
                               const LinAlg::MPI::BlockSparseMatrix &matrix_,
                               const bool &elliptic = false);

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Declare parameters for a single block.
    void
    declare_parameters_single_block(ParamHandler &params) const;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// parse parameters for a single block.
    void
    parse_parameters_single_block(ParamHandler &params);

    /// Initialize the preconditioner.
    void
    initialize(const LinAlg::MPI::BlockSparseMatrix &matrix_);

    /// Action of preconditioner according to deal.II interface.
    void
    vmult(LinAlg::MPI::BlockVector &      dst,
          const LinAlg::MPI::BlockVector &src) const;

  private:
    /// Vector containing the preconditioners for each diagonal block.
    std::vector<PreconditionerHandler> diagonal_blocks;

    /// Subsection of single block parameters.
    std::string subsection_diagonal_block;

    /// Boolean value to specify whether @ref initialize has been called at least once.
    bool initialized;

    /// Dummy preconditioner, used only to declare and parse parameters.
    PreconditionerHandler preconditioner_dummy;

    /// Reference to problem matrix. Used during Gauss-Seidel iterations.
    const LinAlg::MPI::BlockSparseMatrix &matrix;

    /// Temporary vector used in Gauss-Seidel iterations.
    mutable LinAlg::MPI::Vector tmp;

    /// Type of block solver.
    std::string prm_block_solver_type;

    /// Number of blocks.
    unsigned int n_blocks;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_PRECONDITIONER_HANDLER_HPP_ */
