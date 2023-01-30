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
 * @author Roberto Piersanti <roberto.piersanti@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#ifndef LIFEX_UTILS_QUATERNIONS_HPP_
#define LIFEX_UTILS_QUATERNIONS_HPP_

#include "source/lifex.hpp"

namespace lifex::utils
{
  /// Class to represent a quaternion.
  class Quaternion
  {
  public:
    /// Constructor
    Quaternion(const double &a_ = 0,
               const double &b_ = 0,
               const double &c_ = 0,
               const double &d_ = 0)
      : a(a_)
      , b(b_)
      , c(c_)
      , d(d_)
    {}

    double a; ///< Quaternion @f$a@f$ component.
    double b; ///< Quaternion @f$b@f$ component.
    double c; ///< Quaternion @f$c@f$ component.
    double d; ///< Quaternion @f$d@f$ component.
  };

  /// Quaternion sum.
  Quaternion
  operator+(const Quaternion &q1, const Quaternion &q2);

  /// Quaternion difference.
  Quaternion
  operator-(const Quaternion &q1, const Quaternion &q2);

  /// Quaternion (Hamilton) multiplication.
  Quaternion
  operator*(const Quaternion &q1, const Quaternion &q2);

  /// Quaternion scalar product.
  double
  quaternion_dot(const Quaternion &q1, const Quaternion &q2);

  /// Quaternion norm.
  double
  quaternion_norm(const Quaternion &q);

  /// Quaternion normalization.
  Quaternion
  quaternion_normalize(const Quaternion &q);

  /// Trasform a quaternion into a matrix rotation.
  Tensor<2, 3, double>
  quaternion_to_rotation(const Quaternion &q);

  /// Trasform a matrix rotation into a quaternion.
  Quaternion
  rotation_to_quaternion(const Tensor<2, 3, double> &Q);

  /// Rotate an axis system, specified by the rotation matrix Q. Reference:
  /// Bayer et al., 2012.
  Tensor<2, 3, double>
  orient(const Tensor<2, 3, double> &Q,
         const double &              alpha,
         const double &              beta);

  /// Quaternions slerp interpolation: evaluate the spherical interpolation
  /// between q0 (for t=0) and q1(for t=1). Reference: Shoemake, 1985.
  Quaternion
  slerp(const Quaternion &q0, const Quaternion &q1, const double &t);

  /// Quaternions bidirectional-slerp interpolation: evaluates the spherical
  /// interpolation between the quaternions equivalent to Qa and Qb, ignoring
  /// changes in orientation. Reference: Bayer et al., 2012.
  Tensor<2, 3, double>
  bislerp(const Tensor<2, 3, double> &Qa,
          const Tensor<2, 3, double> &Qb,
          const double                t);

  /// Computes an axis system from the solution of suitable laplacian problems.
  /// Reference: Bayer et al., 2012.
  Quaternion
  axis_system(const Tensor<1, 3, double> &grad_psi_apex,
              const Tensor<1, 3, double> &grad_phi_trans);

  /// Fibers generation in BT algo for BiV geo. Reference: Bayer et al., 2012.
  Tensor<2, 3, double>
  compute_fibers_BT(const double &              phi_epi,
                    const Tensor<1, 3, double> &grad_phi_epi,
                    const double &              phi_lv,
                    const Tensor<1, 3, double> &grad_phi_lv,
                    const double &              phi_rv,
                    const Tensor<1, 3, double> &grad_phi_rv,
                    const Tensor<1, 3, double> &grad_psi_ab,
                    const double &              alpha_epi,
                    const double &              alpha_endo,
                    const double &              beta_epi,
                    const double &              beta_endo);

} // namespace lifex::utils

#endif /* LIFEX_UTILS_QUATERNIONS_HPP_ */
