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
 * @author Nicolas Alejandro Barnafi <nicolas.barnafi@unipv.it>.
 */

#ifndef LIFEX_PETSC_TOOLS_HPP_
#define LIFEX_PETSC_TOOLS_HPP_

#include "source/lifex.hpp"

#include <vector>

#ifdef LIN_ALG_PETSC
namespace lifex::utils
{
  /// @brief Internally defines the block size of the given matrix.
  /// This is fundamental for the AMG and BDDC preconditioners.
  void
  set_block_size(Mat &mat, const PetscInt &block_size);

  /// @brief Allocate memory for @p dst matrix and copy data from @p src.
  void
  copy_matrix(const Mat &src, Mat &dst);

  /// @brief Copy vectors from @dealii format to <kbd>PETSc</kbd>.
  void
  copy_vectors(const std::vector<LinAlg::MPI::Vector> &src,
               std::vector<Vec> &                      dst);

  /// @brief Initialize the nullspace object.
  void
  create_nullspace(MatNullSpace &nullspace, const std::vector<Vec> &vectors);

  /// @brief Set a nullspace as the near-nullspace of a matrix.
  void
  set_near_nullspace(Mat &mat, const MatNullSpace &near_nullspace);
} // namespace lifex::utils
#endif /* LIN_ALG_PETSC */

#endif /* LIFEX_PETSC_TOOLS_HPP_ */
