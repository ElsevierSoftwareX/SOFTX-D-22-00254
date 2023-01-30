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
 */

#include "source/core.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values_extractors.h>

namespace lifex::utils
{
  std::pair<double, double>
  compute_value_and_derivative(const std::function<double_AD(double_AD)> &f,
                               const double &                             x)
  {
    ADHelperScalarFunction     ad_helper(1);
    FEValuesExtractors::Scalar idx(0);

    ad_helper.register_independent_variable(x, idx);

    double_AD x_AD = ad_helper.get_sensitive_variables(idx);
    double_AD y_AD = f(x_AD);

    ad_helper.register_dependent_variable(y_AD);
    double y = ad_helper.compute_value();

    Vector<double> dy_dx_AD(1);
    ad_helper.compute_gradient(dy_dx_AD);
    double dy_dx = dy_dx_AD(0);

    return std::make_pair(y, dy_dx);
  }

  double
  heaviside_sharp(const double &x, const double &x0)
  {
    return x > x0 ? 1.0 : 0.0;
  }

  double
  heaviside(const double &x, const double &x0, const double &k)
  {
    return (0.5 * (1 + std::tanh(k * (x - x0))));
  }

  double
  heaviside_trig(const double &x, const double &x0, const double &k)
  {
    return 0.5 + std::atan(0.5 * M_PI * k * (x - x0)) / M_PI;
  }

  double
  cosine_ramp(const double &t,
              const double &t0,
              const double &t1,
              const double &v0,
              const double &v1)
  {
    if (t < t0)
      {
        return v0;
      }
    else if (t < t1)
      {
        return v0 +
               (v1 - v0) * (0.5 - 0.5 * std::cos((t - t0) / (t1 - t0) * M_PI));
      }
    else // if (t >= t_1)
      {
        return v1;
      }
  }

  template <class VectorType>
  typename VectorType::real_type
  vec_min(const VectorType &vec, const std::optional<IndexSet> &active_indices)
  {
    typename VectorType::real_type min_value =
      std::numeric_limits<typename VectorType::real_type>::max();

    for (const auto &idx : vec.locally_owned_elements())
      if (!active_indices.has_value() || active_indices.value().is_element(idx))
        min_value = std::min(min_value, vec[idx]);

    return Utilities::MPI::min(min_value, Core::mpi_comm);
  }

  template <class VectorType>
  typename VectorType::real_type
  vec_avg(const VectorType &vec, const std::optional<IndexSet> &active_indices)
  {
    if (active_indices.has_value())
      {
        typename VectorType::real_type num_loc = 0.0;
        unsigned int                   den_loc = 0;

        for (const auto &idx : vec.locally_owned_elements())
          if (active_indices.value().is_element(idx))
            {
              num_loc += vec[idx];
              ++den_loc;
            }

        const unsigned int den = Utilities::MPI::sum(den_loc, Core::mpi_comm);

        AssertThrow(den > 0,
                    ExcMessage("The provided active indices are not present in "
                               "the input vector."));

        return Utilities::MPI::sum(num_loc, Core::mpi_comm) / den;
      }
    else
      {
        return vec.mean_value();
      }
  }

  template <class VectorType>
  typename VectorType::real_type
  vec_max(const VectorType &vec, const std::optional<IndexSet> &active_indices)
  {
    typename VectorType::real_type max_value =
      std::numeric_limits<typename VectorType::real_type>::lowest();

    for (const auto &idx : vec.locally_owned_elements())
      if (!active_indices.has_value() || active_indices.value().is_element(idx))
        max_value = std::max(max_value, vec[idx]);

    return Utilities::MPI::max(max_value, Core::mpi_comm);
  }


  /// Explicit instantiation.
  template LinAlg::MPI::Vector::real_type
  vec_min(const LinAlg::MPI::Vector &, const std::optional<IndexSet> &);

  /// Explicit instantiation.
  template LinAlg::MPI::Vector::real_type
  vec_avg(const LinAlg::MPI::Vector &, const std::optional<IndexSet> &);

  /// Explicit instantiation.
  template LinAlg::MPI::Vector::real_type
  vec_max(const LinAlg::MPI::Vector &, const std::optional<IndexSet> &);


  /// Explicit instantiation.
  template LinearAlgebra::distributed::Vector<double>::real_type
  vec_min(const LinearAlgebra::distributed::Vector<double> &,
          const std::optional<IndexSet> &);

  /// Explicit instantiation.
  template LinearAlgebra::distributed::Vector<double>::real_type
  vec_avg(const LinearAlgebra::distributed::Vector<double> &,
          const std::optional<IndexSet> &);

  /// Explicit instantiation.
  template LinearAlgebra::distributed::Vector<double>::real_type
  vec_max(const LinearAlgebra::distributed::Vector<double> &,
          const std::optional<IndexSet> &);

  namespace MPI
  {
    MinLocDoubleData
    minloc(const double &in, const MPI_Comm &mpi_comm)
    {
      MinLocDoubleData in_data{
        in, static_cast<int>(Utilities::MPI::this_mpi_process(mpi_comm))};
      MinLocDoubleData out_data;

      MPI_Allreduce(
        &in_data, &out_data, 1, MPI_DOUBLE_INT, MPI_MINLOC, mpi_comm);

      return out_data;
    }
  } // namespace MPI

  std::complex<double>
  bessel_j(const unsigned int &       n,
           const std::complex<double> z,
           const double &             tolerance)
  {
    double real_result = 0.0;
    double imag_result = 0.0;

    const double abs_z_half   = 0.5 * std::abs(z);
    const double abs_z_half_2 = abs_z_half * abs_z_half;
    const double abs_z_half_4 = abs_z_half_2 * abs_z_half_2;

    const double arg_z     = std::arg(z);
    const double cos_2_arg = std::cos(2 * arg_z);
    const double sin_2_arg = std::sin(2 * arg_z);

    unsigned int k = 0;

    double real_old = 0.0;
    double imag_old = 0.0;

    // Within the do ... while loop below we compute the individual terms of the
    // series and sum them to the result. Each term is of the form:
    //   (abs_z / 2)^(4k + n) / ((2k)!(2k + n)!) [ cos((4k + n) arg_z) -
    //     (abs_z / 2)^2 / ((2k + 1)(2k + n + 1) cos((4k + n + 2) arg_z) ] =
    //   c_k ( cos(theta_0) - b_k cos(theta_1) )
    // (and similar with a sine for the imaginary part). To avoid computing the
    // factorials, we exploit the fact that
    //   c_{k+1} = c_k * (arg_z / 2)^4
    //     / ((2k + 1)(2k + 2)(2k + n + 1)(2k + n + 2))
    // and compute the c_k terms recursively into the variable c. In the for
    // loop below we compute c_0.

    double c = 1.0;
    double b = 0.0;

    for (unsigned int k = 1; k <= n; ++k)
      c *= abs_z_half / k;

    do
      {
        real_old = real_result;
        imag_old = imag_result;

        const double arg_0 = (4.0 * k + n) * arg_z;
        const double cos_0 = std::cos(arg_0);
        const double sin_0 = std::sin(arg_0);

        // Since theta_1 = theta_0 + 2 arg_z, we use the formulas for the cosine
        // and sine of a sum of angles here. This reduces the number of calls to
        // std::cos and std::sin, which were found to be computationally
        // demanding.
        const double cos_1 = cos_0 * cos_2_arg - sin_0 * sin_2_arg;
        const double sin_1 = sin_0 * cos_2_arg + cos_0 * sin_2_arg;

        b = abs_z_half_2 / ((2.0 * k + 1.0) * (2.0 * k + n + 1.0));

        real_result += c * (cos_0 - b * cos_1);
        imag_result += c * (sin_0 - b * sin_1);

        c *= abs_z_half_4 / ((2.0 * k + 1.0) * (2.0 * k + 2.0) *
                             (2.0 * k + n + 1.0) * (2.0 * k + n + 2.0));

        ++k;
      }
    while (std::abs(real_old - real_result) > tolerance ||
           std::abs(imag_old - imag_result) > tolerance);

    return std::complex<double>(real_result, imag_result);
  }

} // namespace lifex::utils
