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
 * @author Roberto Piersanti <roberto.piersanti@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/numerics/numbers.hpp"
#include "source/numerics/quaternions.hpp"

#include <vector>

namespace lifex::utils
{
  Quaternion
  operator+(const Quaternion &q1, const Quaternion &q2)
  {
    return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d + q2.d);
  }

  Quaternion
  operator-(const Quaternion &q1, const Quaternion &q2)
  {
    return Quaternion(q1.a - q2.a, q1.b - q2.b, q1.c - q2.c, q1.d - q2.d);
  }

  Quaternion
  operator*(const Quaternion &q1, const Quaternion &q2)
  {
    return Quaternion(q1.a * q2.a - q1.b * q2.b - q1.c * q2.c - q1.d * q2.d,
                      q1.a * q2.b + q1.b * q2.a + q1.c * q2.d - q1.d * q2.c,
                      q1.a * q2.c - q1.b * q2.d + q1.c * q2.a + q1.d * q2.b,
                      q1.a * q2.d + q1.b * q2.c - q1.c * q2.b + q1.d * q2.a);
  }

  double
  quaternion_dot(const Quaternion &q1, const Quaternion &q2)
  {
    return q1.a * q2.a + q1.b * q2.b + q1.c * q2.c + q1.d * q2.d;
  }

  double
  quaternion_norm(const Quaternion &q)
  {
    return std::sqrt(quaternion_dot(q, q));
  }

  Quaternion
  quaternion_normalize(const Quaternion &q)
  {
    const double norm = quaternion_norm(q);

    Quaternion result;

    if (!utils::is_zero(norm))
      {
        result.a = q.a / norm;
        result.b = q.b / norm;
        result.c = q.c / norm;
        result.d = q.d / norm;
      }
    else
      {
        result.a = 0.0;
        result.b = 0.0;
        result.c = 0.0;
        result.d = 0.0;
      }

    return result;
  }

  Tensor<2, 3, double>
  quaternion_to_rotation(const Quaternion &q)
  {
    Tensor<2, 3, double> Q;

    Q[0][0] = 1 - 2 * (q.c * q.c + q.d * q.d);
    Q[0][1] = 2 * (q.b * q.c - q.d * q.a);
    Q[0][2] = 2 * (q.b * q.d + q.c * q.a);
    Q[1][0] = 2 * (q.b * q.c + q.d * q.a);
    Q[1][1] = 1 - 2 * (q.b * q.b + q.d * q.d);
    Q[1][2] = 2 * (q.c * q.d - q.b * q.a);
    Q[2][0] = 2 * (q.b * q.d - q.c * q.a);
    Q[2][1] = 2 * (q.c * q.d + q.b * q.a);
    Q[2][2] = 1 - 2 * (q.b * q.b + q.c * q.c);

    return Q;
  }

  Quaternion
  rotation_to_quaternion(const Tensor<2, 3, double> &Q)
  {
    Quaternion   q;
    const double t = trace(Q);

    if (t > 0)
      {
        const double r = std::sqrt(1.0 + t);
        const double s = 0.5 / r;

        q.a = 0.5 * r;
        q.b = (Q[2][1] - Q[1][2]) * s;
        q.c = (Q[0][2] - Q[2][0]) * s;
        q.d = (Q[1][0] - Q[0][1]) * s;
      }
    else if (Q[0][0] > Q[1][1] && Q[0][0] > Q[2][2])
      {
        const double s = 2 * std::sqrt(1.0 + Q[0][0] - Q[1][1] - Q[2][2]);

        q.a = (Q[2][1] - Q[1][2]) / s;
        q.b = 0.25 * s;
        q.c = (Q[0][1] + Q[1][0]) / s;
        q.d = (Q[0][2] + Q[2][0]) / s;
      }
    else if (Q[1][1] > Q[0][0] && Q[1][1] > Q[2][2])
      {
        const double s = 2 * std::sqrt(1.0 + Q[1][1] - Q[0][0] - Q[2][2]);

        q.a = (Q[0][2] - Q[2][0]) / s;
        q.b = (Q[0][1] + Q[1][0]) / s;
        q.c = 0.25 * s;
        q.d = (Q[1][2] + Q[2][1]) / s;
      }
    else
      {
        const double s = 2 * std::sqrt(1.0 + Q[2][2] - Q[0][0] - Q[1][1]);

        q.a = (Q[1][0] - Q[0][1]) / s;
        q.b = (Q[0][2] + Q[2][0]) / s;
        q.c = (Q[1][2] + Q[2][1]) / s;
        q.d = 0.25 * s;
      }

    return quaternion_normalize(q);
  }

  Quaternion
  slerp(const Quaternion &q0_, const Quaternion &q1_, const double &t)
  {
    const Quaternion q0 = quaternion_normalize(q0_);
    Quaternion       q1 = quaternion_normalize(q1_);

    double dot = quaternion_dot(q0, q1);

    if (utils::is_negative(dot))
      {
        q1 = q1 * (-1);
        dot *= (-1);
      }

    const double tol = 1e-8;

    if (dot > 1 - tol)
      {
        return quaternion_normalize(q0 + t * (q1 - q0));
      }

    const double theta_0 = std::acos(dot); // angle between input vectors.

    const double theta = theta_0 * t; // angle between v0 and result.

    const double sin_theta   = std::sin(theta);
    const double sin_theta_0 = std::sin(theta_0);

    const double s1 = sin_theta / sin_theta_0;
    const double s0 =
      std::cos(theta) - dot * s1; // = sin(theta_0 - theta) / sin(theta_0)

    return quaternion_normalize(s0 * q0 + s1 * q1);
  }

  Tensor<2, 3, double>
  bislerp(const Tensor<2, 3, double> &Qa,
          const Tensor<2, 3, double> &Qb,
          const double                t)
  {
    const Quaternion i = Quaternion(0, 1, 0, 0);
    const Quaternion j = Quaternion(0, 0, 1, 0);
    const Quaternion k = Quaternion(0, 0, 0, 1);

    const Quaternion qa = rotation_to_quaternion(Qa);
    const Quaternion qb = rotation_to_quaternion(Qb);

    // We check qa * i, qa * j, qa * k instead of i * qa, j * qa, k * qa as in
    // the original paper. That is because we want to express rotations of the
    // axis system represented by qa around its principal axes, not around the
    // Cartesian axes x, y, z.
    const std::vector<Quaternion> candidates = {qa, qa * i, qa * j, qa * k};

    size_t max_idx = 0;

    double max_norm = std::abs(quaternion_dot(qa, qb));

    for (size_t n = 0; n < candidates.size(); ++n)
      {
        double tmp = std::abs(quaternion_dot(candidates[n], qb));
        if (tmp > max_norm)
          {
            max_norm = tmp;
            max_idx  = n;
          }
      }

    return quaternion_to_rotation(slerp(candidates[max_idx], qb, t));
  }

  Quaternion
  axis_system(const Tensor<1, 3, double> &grad_psi_apex,
              const Tensor<1, 3, double> &grad_phi_trans)
  {
    Tensor<1, 3, double> e_t;
    Tensor<1, 3, double> e_l;
    Tensor<1, 3, double> e_c;

    Tensor<2, 3, double> Q;

    e_t = grad_phi_trans;
    e_t /= e_t.norm();

    e_l = grad_psi_apex - scalar_product(e_t, grad_psi_apex) * e_t;
    e_l /= e_l.norm();

    e_c = cross_product_3d(e_l, e_t);
    e_c /= e_c.norm();

    Q = Tensor<2, 3, double>({{e_c[0], e_l[0], e_t[0]},
                              {e_c[1], e_l[1], e_t[1]},
                              {e_c[2], e_l[2], e_t[2]}});

    return quaternion_normalize(rotation_to_quaternion(Q));
  }

  Tensor<2, 3, double>
  orient(const Tensor<2, 3, double> &Q, const double &alpha, const double &beta)
  {
    const double sina = std::sin(alpha);
    const double cosa = std::cos(alpha);

    const double sinb = std::sin(beta);
    const double cosb = std::cos(beta);

    Tensor<2, 3, double> Q1;

    Q1[0][0] = Q[0][0] * cosa + Q[0][1] * sina;
    Q1[0][1] = -Q[0][0] * sina * cosb + Q[0][1] * cosa * cosb - Q[0][2] * sinb;
    Q1[0][2] = -Q[0][0] * sina * sinb + Q[0][1] * cosa * sinb + Q[0][2] * cosb;
    Q1[1][0] = Q[1][0] * cosa + Q[1][1] * sina;
    Q1[1][1] = -Q[1][0] * sina * cosb + Q[1][1] * cosa * cosb - Q[1][2] * sinb;
    Q1[1][2] = -Q[1][0] * sina * sinb + Q[1][1] * cosa * sinb + Q[1][2] * cosb;
    Q1[2][0] = Q[2][0] * cosa + Q[2][1] * sina;
    Q1[2][1] = -Q[2][0] * sina * cosb + Q[2][1] * cosa * cosb - Q[2][2] * sinb;
    Q1[2][2] = -Q[2][0] * sina * sinb + Q[2][1] * cosa * sinb + Q[2][2] * cosb;

    return Q1;
  }

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
                    const double &              beta_endo)
  {
    auto alpha_wall = [&](const double &tt) {
      return tt * alpha_epi + (1 - tt) * alpha_endo;
    };
    auto alpha_septum = [&](const double &tt) {
      return alpha_endo * (1 - tt) - alpha_endo * tt;
    };
    auto beta_wall = [&](const double &tt) {
      return tt * beta_epi + (1 - tt) * beta_endo;
    };
    auto beta_septum = [&](const double &tt) {
      return beta_endo * (1 - tt) - beta_endo * tt;
    };

    Tensor<2, 3, double> Qtmp, Qlv, Qrv, Qepi, Qendo, Qall;

    Quaternion lv_axis, rv_axis, epi_axis;

    double t = (phi_rv > 0 || phi_lv > 0) ? phi_rv / (phi_lv + phi_rv) : 1;

    lv_axis = axis_system(grad_psi_ab, grad_phi_lv);
    Qtmp    = quaternion_to_rotation(lv_axis);
    Qlv     = orient(Qtmp, alpha_septum(t), beta_septum(t));

    rv_axis = axis_system(grad_psi_ab, grad_phi_rv);
    Qtmp    = quaternion_to_rotation(rv_axis);
    Qrv     = orient(Qtmp, alpha_septum(t), beta_septum(t));

    Qendo = bislerp(Qlv, Qrv, t);

    epi_axis = axis_system(grad_psi_ab, grad_phi_epi);
    Qtmp     = quaternion_to_rotation(epi_axis);
    Qepi     = orient(Qtmp, alpha_wall(phi_epi), beta_wall(phi_epi));

    Qall = bislerp(Qendo, Qepi, phi_epi);

    return Qall;
  }

} // namespace lifex::utils
