/********************************************************************************
  Copyright (C) 2019 - 2023 by the lifex authors.

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

#ifndef LIFEX_UTILS_TOOLS_HPP_
#define LIFEX_UTILS_TOOLS_HPP_

#include "source/core.hpp"

namespace lifex::utils
{
  /// Integrate a scalar function using the trapezoidal rule.
  template <class VectorType1, class VectorType2>
  static double
  integrate_trap(const VectorType1 &t,
                 const VectorType2 &y,
                 const double &     offset = 0.0)
  {
    AssertThrow(t.size() == y.size(), ExcDimensionMismatch(t.size(), y.size()));

    double result = 0.0;
    for (unsigned int i = 1; i < t.size(); ++i)
      {
        result +=
          0.5 * (t[i] - t[i - 1]) * ((y[i] - offset) + (y[i - 1] - offset));
      }

    return result;
  }

  /// Integrate the absolute value of a scalar function using the trapezoidal
  /// rule.
  template <class VectorType1, class VectorType2>
  static double
  integrate_trap_abs(const VectorType1 &t,
                     const VectorType2 &y,
                     const double &     offset = 0.0)
  {
    AssertThrow(t.size() == y.size(), ExcDimensionMismatch(t.size(), y.size()));

    double result = 0.0;
    for (unsigned int i = 1; i < t.size(); ++i)
      {
        result += 0.5 * (t[i] - t[i - 1]) *
                  (std::abs(y[i] - offset) + std::abs(y[i - 1] - offset));
      }

    return result;
  }

  /**
   * @brief Diagonalize a row of a linear system <kbd>A * x = b</kbd>.
   *
   * After this process, the row <kbd>A(row_idx) * x = b(row_idx)</kbd> becomes
   * <kbd>diag_value * x = diag_value * rhs_value</kbd>.
   * By default, <kbd>diag_value</kbd> is equal to 1.
   * @param[in, out] A          Matrix to be diagonalized.
   * @param[in, out] b          Right-hand side vector to be updated accordingly.
   * @param[in]      row_idx    Index of the row to be diagonalized.
   * @param[in]      rhs_value  Value to insert in <kbd>b(row_idx)</kbd>.
   * @param[in]      diag_value Value to set in <kbd>A(row_idx, row_idx)</kbd>
   *                            and to multiply <kbd>b(row_idx)</kbd> by.
   */
  template <class MatrixType, class VectorType>
  void
  diagonalize(MatrixType &                           A,
              VectorType &                           b,
              const types::global_dof_index &        row_idx,
              const typename VectorType::value_type &rhs_value,
              const typename VectorType::value_type &diag_value = 1)
  {
    // Clear diagonal entry before calling clear_row().
    A.set(row_idx, row_idx, 0);
    A.compress(VectorOperation::insert);

    A.clear_row(row_idx, diag_value);

    b[row_idx] = diag_value * rhs_value;
    b.compress(VectorOperation::insert);
  }


  /// @brief Allocate matrix memory using @dealii reinit methods.
  ///
  /// @param[in,out] matrix  Matrix to be allocated.
  /// @param[in] owned_dofs  <kbd>IndexSet</kbd> (or <kbd>std::vector<IndexSet></kbd> for block matrices) with locally owned dofs.
  /// @param[in] dsp         (Block)DynamicSparsityPattern with problem structure.
  template <class MatrixType, class SparsityType, class IndexType>
  void
  initialize_matrix(MatrixType &        matrix,
                    const IndexType &   owned_dofs,
                    const SparsityType &dsp)
  {
    matrix.clear();

#if defined(LIN_ALG_PETSC)
    matrix.reinit(owned_dofs, owned_dofs, dsp, Core::mpi_comm);
#else
    matrix.reinit(owned_dofs, dsp, Core::mpi_comm);
#endif
  }

  /**
   * @brief Wrapper for iterating through a vector.
   *
   * Depending on the implementation, going through a distributed vector by
   * means of iterators may be significantly faster than iterating over its
   * indices and repeatedly calling operator[]. That is the case, for instance,
   * for TrilinosWrappers::MPI::Vector. However, other classes, such as
   * PETScWrappers::MPI::Vector, do not expose iterators. Writing loops that are
   * both generic and efficient may become cumbersome if different vector
   * implementations must be supported.
   *
   * The purpose of this class is to provide a generic iterator-like interface
   * to go through the elements of a vector, regardless of the underlying
   * implementation.
   */
  template <class VectorType>
  class VectorConstIterator
  {
  public:
    /// Default constructor.
    VectorConstIterator()
      : vector(nullptr)
      , iterator(nullptr)
    {}

    /// Constructor.
    VectorConstIterator(const VectorType &vector)
    {
      initialize(vector);
    }

    /// Initialize.
    void
    initialize(const VectorType &vector_)
    {
      vector   = &vector_;
      iterator = vector->begin();
    }

    /// Increment operator.
    VectorConstIterator &
    operator++()
    {
      ++iterator;
      return *this;
    }

    /// Dereference operator.
    typename VectorType::value_type
    operator*() const
    {
      return *iterator;
    }

    /// Check if the end was reached.
    bool
    at_end() const
    {
      return iterator == vector->end();
    }

  protected:
    /// Pointer to the vector.
    const VectorType *vector;

    /// Internal iterator.
    typename VectorType::const_iterator iterator;
  };

  /// Specialization of VectorConstIterator for PETSc vectors.
  template <>
  class VectorConstIterator<PETScWrappers::MPI::Vector>
  {
  public:
    /// Default constructor.
    VectorConstIterator()
      : vector(nullptr)
      , iterator(0)
    {}

    /// Constructor.
    VectorConstIterator(const PETScWrappers::MPI::Vector &vector)
    {
      initialize(vector);
    }

    /// Initialize.
    void
    initialize(const PETScWrappers::MPI::Vector &vector_)
    {
      vector   = &vector_;
      iterator = vector->local_range().first;
    }

    /// Increment operator.
    VectorConstIterator &
    operator++()
    {
      ++iterator;
      return *this;
    }

    /// Dereference operator.
    PETScWrappers::MPI::Vector::value_type
    operator*() const
    {
      return (*vector)[iterator];
    }

    /// Check if the end was reached.
    bool
    at_end() const
    {
      return iterator == vector->local_range().second;
    }

  protected:
    /// Pointer to the vector.
    const PETScWrappers::MPI::Vector *vector;

    /// Internal iterator: the current index within the vector.
    unsigned int iterator;
  };

  /// Explicit instantiation.
  template class VectorConstIterator<TrilinosWrappers::MPI::Vector>;
  template class VectorConstIterator<PETScWrappers::MPI::Vector>;
  template class VectorConstIterator<
    LinearAlgebra::distributed::Vector<double>>;

  /**
   * @brief Set one entry of a vector.
   *
   * Depending on the implementation, the method Vector::set may be
   * significantly faster than accessing elements through operator[]. Different
   * vector implementations offer different interfaces for the set method. This
   * function wraps them in a generic way.
   */
  template <class VectorType>
  void
  set_vector_entry(VectorType &                           vector,
                   const unsigned int &                   index,
                   const typename VectorType::value_type &value)
  {
    if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
      vector.set(1, &index, &value);
    else if constexpr (std::is_same_v<VectorType, PETScWrappers::MPI::Vector>)
      vector.set({index}, {value});
    else
      vector[index] = value;
  }

} // namespace lifex::utils

#endif /* LIFEX_UTILS_TOOLS_HPP_ */
