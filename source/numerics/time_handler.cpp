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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 * @author Marco Fedele <marco.fedele@polimi.it>.
 */

#include "source/numerics/time_handler.hpp"

namespace lifex::utils
{
  template <class VectorType>
  BDFHandler<VectorType>::BDFHandler()
    : initialized(false)
  {
    static_assert(is_any_v<VectorType,
                           double,
                           LinAlg::MPI::Vector,
                           LinAlg::MPI::BlockVector,
                           LinearAlgebra::distributed::Vector<double>,
                           std::vector<std::vector<double>>,
                           std::vector<std::vector<Tensor<1, dim, double>>>,
                           std::vector<std::vector<Tensor<2, dim, double>>>>,
                  "BDFHandler: template parameter not allowed.");

    initialized = false;
  }

  template <class VectorType>
  BDFHandler<VectorType>::BDFHandler(
    const unsigned int &           order_,
    const std::vector<VectorType> &initial_solutions)
    : initialized(false)
    , order(order_)
  {
    initialize(order, initial_solutions);
  }

  template <class VectorType>
  void
  BDFHandler<VectorType>::copy_from(const BDFHandler<VectorType> &other)
  {
    initialized = other.initialized;

    order = other.order;

    solutions = other.solutions;

    alpha = other.alpha;

    *sol_bdf           = *(other.sol_bdf);
    *sol_extrapolation = *(other.sol_extrapolation);
  }

  template <class VectorType>
  void
  BDFHandler<VectorType>::initialize(
    const unsigned int &           order_,
    const std::vector<VectorType> &initial_solutions)
  {
    initialized = true;

    order = order_;

    Assert(order > 0 && order <= 3, ExcIndexRange(order, 1, 4));
    Assert(initial_solutions.size() == order,
           ExcDimensionMismatch(initial_solutions.size(), order));

    // Initialize alpha.
    switch (order)
      {
        case 1:
          alpha = 1.0;
          break;

        case 2:
          alpha = 1.5;
          break;

        case 3:
          alpha = 11.0 / 6;
          break;

        default:
          break;
      }

    // Initialize solutions, sol_bdf and sol_extrapolation.
    solutions.clear();

    for (size_t i = 0; i < initial_solutions.size() - 1; ++i)
      {
        if constexpr (is_any_v<VectorType,
                               LinAlg::MPI::Vector,
                               LinAlg::MPI::BlockVector,
                               LinearAlgebra::distributed::Vector<double>>)
          {
            Assert(!initial_solutions[i].has_ghost_elements(),
                   ExcGhostsPresent());
          }

        solutions.push_back(initial_solutions[i]);
      }

    sol_bdf           = std::make_shared<VectorType>(initial_solutions[0]);
    sol_extrapolation = std::make_shared<VectorType>(initial_solutions[0]);

    // Update sol_bdf and sol_extrapolation using the initial guess.
    {
      // Backup initial solutions to prevent overwriting the oldest one.
      const auto solutions_backup = solutions;
      time_advance(initial_solutions.back(), true);
      solutions = solutions_backup;
    }
  }

  template <class VectorType>
  void
  BDFHandler<VectorType>::time_advance(const VectorType &sol_new,
                                       const bool &      update_extrapolation)
  {
    AssertThrow(initialized, ExcNotInitialized());

    Assert(order > 0 && order <= 3, ExcIndexRange(order, 1, 4));

    if constexpr (is_any_v<VectorType,
                           LinAlg::MPI::Vector,
                           LinAlg::MPI::BlockVector,
                           LinearAlgebra::distributed::Vector<double>>)
      {
        Assert(sol_new.has_ghost_elements() == false, ExcGhostsPresent());
      }

    solutions.push_back(sol_new);

    // Initialize vectors.
    *sol_bdf = solutions[order - 1];

    if (update_extrapolation)
      {
        *sol_extrapolation = solutions[order - 1];
      }

    switch (order)
      {
        case 1:
          // Already initialized right before the switch statement.
          break;

        case 2:
          // Specialization for scalars.
          if constexpr (is_any_v<VectorType, double>)
            {
              *sol_bdf = 2 * solutions[order - 1] - 0.5 * solutions[order - 2];

              if (update_extrapolation)
                {
                  *sol_extrapolation =
                    2 * solutions[order - 1] - solutions[order - 2];
                }
            }
          // Specialization for MPI vectors.
          else if constexpr (is_any_v<
                               VectorType,
                               LinAlg::MPI::Vector,
                               LinAlg::MPI::BlockVector,
                               LinearAlgebra::distributed::Vector<double>>)
            {
              // sol_bdf = 2 * solutions[order - 1] -
              //           0.5 * solutions[order - 2];
              sol_bdf->sadd(2, -0.5, solutions[order - 2]);

              if (update_extrapolation)
                {
                  // sol_extrapolation = 2 * solutions[order - 1] -
                  //                     solutions[order - 2];
                  sol_extrapolation->sadd(2, -1, solutions[order - 2]);
                }
            }
          // Specialization for std::vector<std::vector<*>>.
          else
            {
              // sol_bdf = 2 * solutions[order - 1] -
              //           0.5 * solutions[order - 2];
              for (size_t c = 0; c < solutions[0].size(); ++c)
                {
                  for (size_t q = 0; q < solutions[0][c].size(); ++q)
                    {
                      (*sol_bdf)[c][q] = 2 * solutions[order - 1][c][q] -
                                         0.5 * solutions[order - 2][c][q];
                    }
                }

              if (update_extrapolation)
                {
                  // sol_extrapolation = 2 * solutions[order - 1] -
                  //                     solutions[order - 2];
                  for (size_t c = 0; c < solutions[0].size(); ++c)
                    {
                      for (size_t q = 0; q < solutions[0][c].size(); ++q)
                        {
                          (*sol_extrapolation)[c][q] =
                            2 * solutions[order - 1][c][q] -
                            solutions[order - 2][c][q];
                        }
                    }
                }
            }
          break;

        case 3:
          // Specialization for scalars.
          if constexpr (is_any_v<VectorType, double>)
            {
              *sol_bdf = 3 * solutions[order - 1] - 1.5 * solutions[order - 2] +
                         1.0 / 3 * solutions[order - 3];

              if (update_extrapolation)
                {
                  *sol_extrapolation = 3 * solutions[order - 1] -
                                       3 * solutions[order - 2] +
                                       solutions[order - 3];
                }
            }
          // Specialization for MPI vectors.
          else if constexpr (is_any_v<
                               VectorType,
                               LinAlg::MPI::Vector,
                               LinAlg::MPI::BlockVector,
                               LinearAlgebra::distributed::Vector<double>>)
            {
              // sol_bdf = 3 * solutions[order - 1] -
              //           1.5 * solutions[order - 2] +
              //           1.0 / 3 * solutions[order - 3];
              sol_bdf->sadd(3, -1.5, solutions[order - 2]);
              sol_bdf->add(1.0 / 3, solutions[order - 3]);

              if (update_extrapolation)
                {
                  // sol_extrapolation = 3 * solutions[order - 1] -
                  //                     3 * solutions[order - 2] +
                  //                     solutions[order - 3];
                  sol_extrapolation->sadd(3, -3, solutions[order - 2]);
                  sol_extrapolation->add(1, solutions[order - 3]);
                }
            }
          // Specialization for std::vector<std::vector<*>>.
          else
            {
              // sol_bdf = 3 * solutions[order - 1] -
              //           1.5 * solutions[order - 2] +
              //           1.0 / 3 * solutions[order - 3];
              for (size_t c = 0; c < solutions[0].size(); ++c)
                {
                  for (size_t q = 0; q < solutions[0][c].size(); ++q)
                    {
                      (*sol_bdf)[c][q] = 3 * solutions[order - 1][c][q] -
                                         1.5 * solutions[order - 2][c][q] +
                                         1.0 / 3 * solutions[order - 3][c][q];
                    }
                }

              if (update_extrapolation)
                {
                  // sol_extrapolation = 3 * solutions[order - 1] -
                  //                     3 * solutions[order - 2] +
                  //                     solutions[order - 3];
                  for (size_t c = 0; c < solutions[0].size(); ++c)
                    {
                      for (size_t q = 0; q < solutions[0][c].size(); ++q)
                        {
                          (*sol_extrapolation)[c][q] =
                            3 * solutions[order - 1][c][q] -
                            3 * solutions[order - 2][c][q] +
                            solutions[order - 3][c][q];
                        }
                    }
                }
            }
          break;

        default:
          break;
      }

    // Prepare for next time step.
    solutions.pop_front();
  }


  /// Explicit instantiation.
  template class BDFHandler<double>;

  /// Explicit instantiation.
  template class BDFHandler<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class BDFHandler<LinAlg::MPI::BlockVector>;

  /// Explicit instantiation.
  template class BDFHandler<LinearAlgebra::distributed::Vector<double>>;

  /// Explicit instantiation.
  template class BDFHandler<std::vector<std::vector<double>>>;

  /// Explicit instantiation.
  template class BDFHandler<std::vector<std::vector<Tensor<1, dim, double>>>>;

  /// Explicit instantiation.
  template class BDFHandler<std::vector<std::vector<Tensor<2, dim, double>>>>;

} // namespace lifex::utils
