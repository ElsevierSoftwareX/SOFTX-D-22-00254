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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#ifndef LIFEX_UTILS_NUMBERS_HPP_
#define LIFEX_UTILS_NUMBERS_HPP_

#include "source/lifex.hpp"

#include <deal.II/physics/elasticity/kinematics.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /// Helper template variable: this is true whenever
  /// type @p T represents a floating point type
  /// (since C++17).
  template <class T>
  inline constexpr bool is_floating_v = (std::is_floating_point_v<T>);

  /// Check if two floating point numbers <kbd>a</kbd> and <kbd>b</kbd>
  /// are equal, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_equal(const T a,
           const T b,
           const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(is_floating_v<T>,
                  "is_equal: template parameter does not represent a floating "
                  "point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return std::abs(a - b) <= (std::max(std::abs(a), std::abs(b)) * tolerance);
  }

  /// Check if a floating point number <kbd>a</kbd>
  /// is zero, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_zero(const T a, const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(is_floating_v<T>,
                  "is_zero: template parameter does not represent a floating "
                  "point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return (std::abs(a) <= tolerance);
  }

  /// Check if a floating point number <kbd>a</kbd>
  /// is positive, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_positive(const T a, const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(is_floating_v<T>,
                  "is_positive: template parameter does not represent a "
                  "floating point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return (a > tolerance);
  }

  /// Check if a floating point number <kbd>a</kbd>
  /// is negative, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_negative(const T a, const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(is_floating_v<T>,
                  "is_negative: template parameter does not represent a "
                  "floating point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return (a < -tolerance);
  }

  /// Check if a floating point number <kbd>a</kbd> is @a definitely greater
  /// than a floating point number <kbd>b</kbd>, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_definitely_greater_than(
    const T a,
    const T b,
    const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(
      is_floating_v<T>,
      "is_definitely_greather_than: template parameter does not represent a "
      "floating point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return (a - b) > (std::max(std::abs(a), std::abs(b)) * tolerance);
  }

  /// Check if a floating point number <kbd>a</kbd> is @a definitely smaller
  /// than a floating point number <kbd>b</kbd>, up to a given tolerance.
  template <class T>
  inline constexpr bool
  is_definitely_less_than(const T a,
                          const T b,
                          const T tolerance = std::numeric_limits<T>::epsilon())
  {
    static_assert(
      is_floating_v<T>,
      "is_definitely_less_than: template parameter does not represent a "
      "floating point number.");

    Assert(tolerance >= 0, ExcMessage("Tolerance must be non-negative."));

    return (b - a) > (std::max(std::abs(a), std::abs(b)) * tolerance);
  }

  /**
   * Dummy type used for type traits below.
   */
  template <typename... T>
  using void_t = void;

  /**
   * Type trait to tell if a container has a <kbd>.find()</kbd> member function.
   */
  template <class ContainerType, class ValueType, class = void_t<>>
  struct has_member_find : std::false_type
  {};

  /**
   * Partial specialization: true if container has a <kbd>.find()</kbd> member
   * function.
   */
  template <class ContainerType, class ValueType>
  struct has_member_find<ContainerType,
                         ValueType,
                         void_t<decltype(std::declval<ContainerType>().find(
                           std::declval<ValueType>()))>> : std::true_type
  {};

  /**
   * Check if an arbitary container contains a given value.
   */
  template <class ContainerType, class ValueType>
  inline constexpr bool
  contains(const ContainerType &container, const ValueType &value)
  {
    if constexpr (has_member_find<ContainerType, ValueType>())
      return container.find(value) != std::end(container);
    else
      return std::find(std::begin(container), std::end(container), value) !=
             std::end(container);
  }

  /**
   * Evaluate a scalar function (of scalar variable) and its derivative at a point @p x, using
   * @ref ADHelperScalarFunction.
   */
  std::pair<double, double>
  compute_value_and_derivative(const std::function<double_AD(double_AD)> &f,
                               const double &                             x);

  /**
   * Compute the sharp Heaviside function
   * @f[
   * H(x) =
   * \begin{cases}
   * 1, & x > x0, \\
   * 0, & x \leq x0.
   * \end{cases}
   * @f]
   */
  double
  heaviside_sharp(const double &x, const double &x0 = 0);

  /**
   * Compute the smoothed Heaviside function
   * @f[
   * H(x) = \frac{1}{2}\left(1 + \tanh\left(k\left(x -
   * x_0\right)\right)\right).
   * @f]
   */
  double
  heaviside(const double &x, const double &x0 = 0, const double &k = 200);

  /**
   * Compute the smoothed trigonometric Heaviside function
   * @f[
   * H(x) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{k\pi}{2}(x -
   * x_0)\right).
   * @f]
   */
  double
  heaviside_trig(const double &x, const double &x0 = 0, const double &k = 200);

  /**
   * Cosinusoidal ramp.
   *@f[
   * f(t) = \left\{
   * \begin{alignedat}{3}
   * & v_0, & \quad & \text{if } t < t_0, \\
   * & v_0 + (v_1 - v_0)\left[\frac{1}{2} - \frac{1}{2}\cos\left(\frac{t -
   *t_0}{t_1 - t_0}\pi\right)\right], & \quad & \text{if } t_0 \leq t < t_1, \\
   * & v_1, & \quad & \text{if } t \geq t_1.
   * \end{alignedat}
   * \right.
   * @f]
   */
  double
  cosine_ramp(const double &t,
              const double &t0 = 0.0,
              const double &t1 = 1.0,
              const double &v0 = 0.0,
              const double &v1 = 1.0);

  /**
   * Compute minimum value of a parallel vector over all MPI processes,
   * optionally limiting this operation to a subset of indices.
   */
  template <class VectorType>
  typename VectorType::real_type
  vec_min(const VectorType &             vec,
          const std::optional<IndexSet> &active_indices = {});

  /**
   * Compute average value of a parallel vector over all MPI processes,
   * optionally limiting this operation to a subset of indices.
   */
  template <class VectorType>
  typename VectorType::real_type
  vec_avg(const VectorType &             vec,
          const std::optional<IndexSet> &active_indices = {});

  /**
   * Compute maximum value of a parallel vector over all MPI processes,
   * optionally limiting this operation to a subset of indices.
   */
  template <class VectorType>
  typename VectorType::real_type
  vec_max(const VectorType &             vec,
          const std::optional<IndexSet> &active_indices = {});

  /**
   * @brief Compute the [Bessel functions of the first kind]
   * (https://en.wikipedia.org/wiki/Bessel_function#Bessel_functions_of_the_first_kind:_J%CE%B1).
   *
   * The function computes
   * @f[
   * J_n(z) = \sum_{m = 0}^{\infty} \frac{(-1)^m}{m! (m + n)!}
   * \left(\frac{z}{2}\right)^{2m + n}\;,
   * @f]
   * for @f$n \in \mathbb{N}@f$ and @f$z \in \mathbb{C}@f$.
   *
   * #### Implementation details
   * The function is computed by separately computing its real and imaginary
   * parts. Denoting by @f$r@f$ the modulus and by @f$\theta@f$ the argument of
   * @f$z@f$,
   * @f[
   * \begin{aligned}
   * \text{Re} J_n(z) &= \sum_{m = 0}^{\infty} \frac{(-1)^m}{m! (m +
   * n)!}\left(\frac{r}{2}\right)^{2m + n} \cos((2m + n)\theta) = \sum_{m =
   * 0}^{\infty} J_{n,m}^\text{Re} \;, \\
   * \text{Im} J_n(z) &= \sum_{m = 0}^{\infty} \frac{(-1)^m}{m! (m
   * + n)!}\left(\frac{r}{2}\right)^{2m + n} \sin((2m + n)\theta) = \sum_{m =
   * 0}^{\infty} J_{n,m}^\text{Im} \;.
   * \end{aligned}
   * @f]
   *
   * In practice, we compute the series as
   * @f[
   * \sum_{k = 0}^{K_{\max}}\left(J^*_{2k} + J^*_{2k + 1}\right)\;,
   * @f]
   * where @f$*@f$ is either @f$\text{Re}@f$ or @f$\text{Im}@f$. Since
   * subsequent terms of the series cancel out, this aids numerical stability
   * reduces the risk of getting <kbd>Inf</kbd> or <kbd>NaN</kbd> values
   * appearing in the computation.
   * @f$K_{\max}@f$ is chosen so that
   * @f[
   * \left|\sum_{k = 0}^{K_{\max}}\left(J^*_{2k} + J^*_{2k + 1}\right) - \sum_{k
   * = 0}^{K_{\max} - 1}\left(J^*_{2k} + J^*_{2k + 1}\right)\right| \leq
   * \varepsilon\;,
   * @f]
   * with @f$\varepsilon@f$ a prescribed tolerance.
   *
   * @param[in] n Order @f$n@f$ of the Bessel function to be computed.
   * @param[in] z Argument @f$z@f$ for the Bessel function.
   * @param[in] tolerance Tolerance @f$\varepsilon@f$ used in the computation of
   * the Bessel function.
   *
   * @note This extends the standard-library function cyl_bessel_j, that does
   * not support complex arguments.
   */
  std::complex<double>
  bessel_j(const unsigned int &       n,
           const std::complex<double> z,
           const double &             tolerance = 1e-6);

  namespace convert
  {
    /// Conversion factor from @f$[mmHg]@f$ to @f$[Pa]@f$.
    static inline constexpr double mmHg_to_Pa = 133.322;

    /// Conversion factor from @f$[Pa]@f$ to @f$[mmHg]@f$.
    static inline constexpr double Pa_to_mmHg = 1.0 / mmHg_to_Pa;

    /// Conversion factor from @f$[ml]@f$ to @f$[m^3]@f$.
    static inline constexpr double mL_to_m3 = 1e-6;

    /// Conversion factor from @f$[m^3]@f$ to @f$[ml]@f$.
    static inline constexpr double m3_to_mL = 1e6;
  } // namespace convert

  /**
   * Namespace containing functions for kinematics problems.
   *
   * @note The functions included here are surrogates of the ones provided by
   * @dealii in <kbd>Physics::Elasticity::Kinematics</kbd>, as those functions
   * lead to issues on some compilers. More specifically, on some versions of
   * Intel compilers they result in either zero or undefined/incorrect behavior
   * depending on the level of compiler optimization.
   */
  namespace kinematics
  {
    /**
     * Compute deformation gradient tensor @f$F = I + \nabla\mathbf d@f$.
     */
    template <class NumberType>
    Tensor<2, dim, NumberType>
    F(const Tensor<2, dim, NumberType> &grad_d)
    {
      return unit_symmetric_tensor<dim, NumberType>() + grad_d;
    }

    /**
     * Compute the symmetric Green-Lagrange strain tensor @f$E =
     * \frac{1}{2}\left(F^T F - I\right)@f$.
     */
    template <class NumberType>
    SymmetricTensor<2, dim, NumberType>
    E(const Tensor<2, dim, NumberType> &F)
    {
      return dealii::internal::NumberType<NumberType>::value(0.5) *
             (symmetrize(transpose(F) * F) -
              unit_symmetric_tensor<dim, NumberType>());
    }

  } // namespace kinematics

  namespace MPI
  {
    /// Structure to store the data of MPI_Allreduce calls with MPI_MINLOC
    /// operation.
    struct MinLocDoubleData
    {
      double value; ///< Stored value.
      int    rank;  ///< Rank owning the minimum.
    };

    /// Return the minimum value across different processes, as well as the
    /// rank owning the minimum.
    MinLocDoubleData
    minloc(const double &in, const MPI_Comm &mpi_comm);

    /// Return the union of maps across MPI processes. If duplicate keys exist
    /// among different processes, only the value with the lowest rank is kept.
    template <class T1, class T2>
    std::map<T1, T2>
    compute_map_union(const std::map<T1, T2> &map, const MPI_Comm &mpi_comm)
    {
      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(mpi_comm);
      const unsigned int mpi_size = Utilities::MPI::n_mpi_processes(mpi_comm);

      const std::vector<std::map<T1, T2>> all_maps =
        Utilities::MPI::all_gather(mpi_comm, map);

      std::map<T1, T2> result = map;

      for (unsigned int rank = 0; rank < mpi_size; ++rank)
        if (mpi_rank != rank)
          result.insert(all_maps[rank].begin(), all_maps[rank].end());

      return result;
    }
  } // namespace MPI
} // namespace lifex::utils

#endif /* LIFEX_UTILS_NUMBERS_HPP_ */
