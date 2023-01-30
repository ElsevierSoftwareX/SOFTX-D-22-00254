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

#include "source/core.hpp"

#include "source/numerics/petsc_tools.hpp"

#ifdef LIN_ALG_PETSC
namespace lifex::utils
{
  void
  set_block_size(Mat &mat, const PetscInt &block_size)
  {
    MatSetBlockSize(mat, block_size);
  }

  void
  copy_matrix(const Mat &src, Mat &dst)
  {
    MatConvert(src, MATSAME, MAT_INITIAL_MATRIX, &dst);
  }

  void
  copy_vectors(const std::vector<LinAlg::MPI::Vector> &src,
               std::vector<Vec> &                      dst)
  {
    dst.clear();
    dst.resize(src.size());

    for (unsigned int i = 0; i < src.size(); ++i)
      {
        VecDuplicate(src[i], &dst[i]);
        VecCopy(src[i], dst[i]);
      }
  }

  void
  create_nullspace(MatNullSpace &nullspace, const std::vector<Vec> &vectors)
  {
    MatNullSpaceCreate(
      Core::mpi_comm, PETSC_FALSE, vectors.size(), vectors.data(), &nullspace);
  }

  void
  set_near_nullspace(Mat &mat, const MatNullSpace &near_nullspace)
  {
    MatSetNearNullSpace(mat, near_nullspace);
  }
} // namespace lifex::utils

#endif /* LIN_ALG_PETSC */
