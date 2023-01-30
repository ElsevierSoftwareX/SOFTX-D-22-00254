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
 * @author Elena Zappon <elena.zappon@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 * @author Alberto Zingaro <alberto.zingaro@polimi.it>.
 */

#include "source/io/vtk_function.hpp"

#include "source/numerics/time_interpolation.hpp"

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <algorithm>
#include <complex>

namespace lifex::utils
{
  TimeInterpolation::TimeInterpolation()
    : Core()
    , mode(Mode::NotInitialized)
  {}

  void
  TimeInterpolation::setup_as_linear_interpolation(
    const std::vector<double> &times,
    const std::vector<double> &data)
  {
    mode = Mode::LinearInterpolation; // Define the type of interpolation used.

    a = times.front();
    b = times.back();

    Assert(b >= a, ExcMessage("b cannot be smaller than a."));

    input_data  = data;
    input_times = times;
  }

  void
  TimeInterpolation::setup_as_cubic_spline(const std::vector<double> &  data,
                                           const double &               a_,
                                           const double &               b_,
                                           const std::optional<double> &a_prime,
                                           const std::optional<double> &b_prime)
  {
    Assert(b_ >= a_, ExcMessage("b cannot be smaller than a."));

    mode = Mode::CubicSpline; // Define the type of interpolation used.

    a = a_;
    b = b_;

    input_data = data;

    size_t size_y = data.size(); // Number of data to interpolate.

    // Stepsize to define equispaced points to be interpolated.
    double step = (b - a) / (size_y - 1.0);

    input_times.resize(size_y);
    for (size_t i = 0; i < size_y; ++i)
      {
        input_times[i] = a + i * step;
      }

    // Compute the spline coefficients.
    AssertThrow((!a_prime.has_value() && !b_prime.has_value()) ||
                  (a_prime.has_value() && b_prime.has_value()),
                ExcMessage("Both a_prime and "
                           "b_prime (or none of them) need to be set."));

    spline.resize(1);
    if (!a_prime.has_value() && !b_prime.has_value())
      {
        spline[0] = boost::math::interpolators::cardinal_cubic_b_spline<double>(
          data.begin(), data.end(), a, step);
      }
    else // if (a_prime.has_value() && b_prime.has_value())
      {
        spline[0] = boost::math::interpolators::cardinal_cubic_b_spline<double>(
          data.begin(), data.end(), a, step, a_prime.value(), b_prime.value());
      }
  }

  void
  TimeInterpolation::setup_as_smoothing_spline(const std::vector<double> &data,
                                               const double &             a_,
                                               const double &             b_,
                                               const double &lambda)
  {
    Assert(b_ >= a_, ExcMessage("b cannot be smaller than a."));

    mode = Mode::SmoothingCubicSpline;

    a                       = a_;
    b                       = b_;
    smoothing_spline_lambda = lambda;

    input_data = data;

    // Class used to represent the matrix of the system we need to solve.
    // @todo Better documentation.
    class SystemMatrix
    {
    public:
      // Constructor.
      SystemMatrix(const unsigned int &n_,
                   const double &      h_,
                   const double &      lambda_)
        : n(n_)
        , h(h_)
        , lambda(lambda_)
        , L_diag_lower(n - 2)
        , U_diag(n - 2)
        , tmp(n - 2)
      {
        // Compute the LU factorization of the tridiagonal matrix W using
        // Thomas' algorithm.
        const double main_diagonal = 2.0 * h / 3.0;
        const double off_diagonal  = h / 6.0;

        U_diag[0] = main_diagonal;
        for (unsigned int i = 1; i < n - 2; ++i)
          {
            L_diag_lower[i - 1] = off_diagonal / U_diag[i - 1];
            U_diag[i] = main_diagonal - L_diag_lower[i - 1] * off_diagonal;
          }
      }

      // Vector multiplication.
      void
      vmult(Vector<double> &dst, const Vector<double> &src) const
      {
        // tmp = Delta * src.
        for (unsigned int i = 0; i < n - 2; ++i)
          {
            tmp[i] = (src[i] - 2 * src[i + 1] + src[i + 2]) / h;
          }

        // tmp = L^{-1} * tmp.
        for (unsigned int i = 1; i < n - 2; ++i)
          {
            tmp[i] = tmp[i] - L_diag_lower[i - 1] * tmp[i - 1];
          }

        // tmp = U^{-1} * tmp.
        tmp[n - 3] /= U_diag[n - 3];
        for (int i = n - 4; i >= 0; --i)
          {
            tmp[i] = (tmp[i] - (h / 6.0) * tmp[i + 1]) / U_diag[i];
          }

        // dst = Delta^t * tmp.
        for (unsigned int i = 0; i < n; ++i)
          {
            dst[i] = 0.0;

            if (i >= 2)
              dst[i] += tmp[i - 2] / h;

            if (1 <= i && i < n - 1)
              dst[i] -= 2 * tmp[i - 1] / h;

            if (i < n - 2)
              dst[i] += tmp[i] / h;
          }

        // dst = src + h * h * h * lambda * dst.
        dst.sadd(h * h * h * lambda, src);
      }

    protected:
      // Number of rows.
      const unsigned int n;

      // Spacing between interpolation nodes.
      const double h;

      // Regularization weight.
      const double lambda;

      // Diagonal of order -1 of the lower-triangular factor of W.
      Vector<double> L_diag_lower;

      // Main diagonal of the upper-triangular factor of U.
      Vector<double> U_diag;

      // Temporary vector.
      mutable Vector<double> tmp;
    };

    // Compute the value of the approximant at the input points.
    const unsigned int n = data.size();
    const double       h = (b - a) / (n - 1);
    const SystemMatrix matrix(n, h, lambda);

    Vector<double> m(n);
    Vector<double> y(data.begin(), data.end());

    ReductionControl         control(1000, 0, 1e-12, false, false);
    SolverCG<Vector<double>> solver(control);
    solver.solve(matrix, m, y, PreconditionIdentity());

    // Construct the interpolating spline that passes through the points m.
    // @todo Why is spline a vector?
    spline.resize(1);
    spline[0] = boost::math::interpolators::cardinal_cubic_b_spline<double>(
      m.begin(), m.end(), a, h);
  }

  void
  TimeInterpolation::setup_as_fourier(const std::vector<double> &times,
                                      const std::vector<double> &data)
  {
    mode = Mode::FourierSeries; // Define the type of interpolation used.

    a = times.front();
    b = times.back();

    Assert(b >= a, ExcMessage("b cannot be smaller than a."));

    input_times = times;
    input_data  = data;

    size_t size_x = times.size(); // Number of data to interpolate.
    size_t n      = size_x - 2;

    // Scale the data_x in the [0, 2*M_PI] interval.
    std::vector<double> x_scaled(size_x);

    const double period = 2.0 * M_PI;

    for (size_t i = 0; i < size_x; ++i)
      {
        x_scaled[i] = (times[i] - a) / (b - a) * period;
      }

    // mu = 0 if the number of points to interpolate is odd, 1 otherwise.
    mu = (n % 2);
    M  = (n - mu) / 2;

    size_t n_columns = 2 * M + 1 + mu;

    FullMatrix<double> A(size_x - 1, n_columns);
    FullMatrix<double> B(size_x - 1, n_columns);
    FullMatrix<double> B_minus(size_x - 1, n_columns);

    const std::complex<double> i(0, 1);

    for (size_t k = 0; k < size_x - 1; ++k)
      {
        for (size_t j = 0; j <= 2 * M; ++j)
          {
            A[k][j] = std::real(
              std::exp(i * (static_cast<double>(j) - static_cast<double>(M)) *
                       x_scaled[k]));
            B[k][j] = std::imag(
              std::exp(i * (static_cast<double>(j) - static_cast<double>(M)) *
                       x_scaled[k]));
            B_minus[k][j] = -std::imag(
              std::exp(i * (static_cast<double>(j) - static_cast<double>(M)) *
                       x_scaled[k]));
          }
      }

    if (mu == 1)
      {
        for (size_t k = 0; k < size_x - 1; ++k)
          {
            A[k][2 * M + 1]       = 2 * std::cos((M + 1) * x_scaled[k]);
            B[k][2 * M + 1]       = 0.0;
            B_minus[k][2 * M + 1] = -0.0;
          }
      }

    FullMatrix<double> matrix(2 * (size_x - 1), 2 * n_columns);
    matrix.fill(A, 0, 0, 0, 0);
    matrix.fill(B, size_x - 1, 0, 0, 0);
    matrix.fill(B_minus, 0, n_columns, 0, 0);
    matrix.fill(A, size_x - 1, n_columns, 0, 0);

    Vector<double> rhs;
    rhs.reinit(2 * (size_x - 1));
    for (size_t j = 0; j < size_x - 1; ++j)
      {
        rhs[j]              = data[j];
        rhs[j + size_x - 1] = 0.0;
      }

    Vector<double> sol;
    sol.reinit(2 * n_columns);

    SolverControl solver_control(std::max<size_t>(1000, rhs.size() / 10),
                                 1e-10 * rhs.l2_norm());

    SolverGMRES<Vector<double>> solver(solver_control);
    solver.solve(matrix, sol, rhs, PreconditionIdentity());

    re_ck.resize(n_columns);
    im_ck.resize(n_columns);
    for (size_t j = 0; j < n_columns; ++j)
      {
        re_ck[j] = sol[j];
        im_ck[j] = sol[n_columns + j];
      }
  }


  void
  TimeInterpolation::setup_as_derivative_linear_interpolation(
    const std::vector<double> &times,
    const std::vector<double> &data)
  {
    mode = Mode::DerivativeLinearInterpolation; // Define the type of
                                                // interpolation used.

    a = times.front();
    b = times.back();

    Assert(b >= a, ExcMessage("b cannot be smaller than a."));

    input_times = times;
    input_data  = data;

    // Compute derivative of input data via backward finite difference.
    data_derivative.resize(times.size());

    data_derivative[0] = 0.0;

    for (size_t k = 1; k < times.size(); k++)
      data_derivative[k] = (data[k] - data[k - 1]) / (times[k] - times[k - 1]);
  }

  void
  TimeInterpolation::setup_as_derivative_spline_interpolation(
    const std::vector<double> &data,
    const double &             a_,
    const double &             b_)
  {
    mode = Mode::DerivativeSplineInterpolation;

    a = a_;
    b = b_;

    Assert(b >= a, ExcMessage("b cannot be smaller than a."));

    const double step = (b - a) / (data.size() - 1.0);

    input_times.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i)
      input_times[i] = a + i * step;

    input_data = data;

    // Compute derivative of input data via backward finite differences.
    data_derivative.resize(data.size());

    data_derivative[0] = 0.0;

    for (size_t k = 1; k < data.size(); k++)
      data_derivative[k] =
        (data[k] - data[k - 1]) / (input_times[k] - input_times[k - 1]);

    // Build a spline that interpolates the derivative.
    spline.resize(1);
    spline[0] = boost::math::interpolators::cardinal_cubic_b_spline<double>(
      data_derivative.begin(), data_derivative.end(), a, step);
  }

  double
  TimeInterpolation::evaluate(const double &t) const
  {
    double interpolated_value = 0.0;

    if (mode == Mode::LinearInterpolation)
      {
        size_t size_data = input_times.size();

        if (t >= b)
          {
            interpolated_value = input_data.back();
          }
        // Do a binary search.
        else
          {
            size_t low  = 0;
            size_t high = size_data;
            size_t mid  = static_cast<size_t>((low + high) / 2.0);
            while (t < input_times[mid] || t >= input_times[mid + 1])
              {
                if (t < input_times[mid])
                  {
                    high = mid;
                  }
                else
                  {
                    low = mid;
                  }
                mid = static_cast<size_t>((low + high) / 2.0);
              }
            interpolated_value =
              input_data[mid] + (input_data[mid + 1] - input_data[mid]) /
                                  (input_times[mid + 1] - input_times[mid]) *
                                  (t - input_times[mid]);
          }
      }
    else if (mode == Mode::CubicSpline || mode == Mode::SmoothingCubicSpline)
      {
        interpolated_value = spline[0](t);
      }
    else if (mode == Mode::FourierSeries)
      {
        const std::complex<double> i(0, 1);

        double t_mod = (t - a) / (b - a) * 2 * M_PI;

        std::complex<double> sol = 0;

        for (size_t j = 0; j <= 2 * M; ++j)
          {
            sol +=
              (re_ck[j] + i * im_ck[j]) *
              std::exp(i * (static_cast<double>(j) - static_cast<double>(M)) *
                       t_mod);
          }

        if (mu == 1)
          {
            sol += (re_ck[2 * M + 1] + i * im_ck[2 * M + 1]) * 2.0 *
                   std::cos((M + 1) * t_mod);
          }
        interpolated_value = sol.real();
      }
    else if (mode == Mode::DerivativeLinearInterpolation)
      {
        // The integral of a piecewise linear polynomial can be computed exactly
        // with the midpoint quadrature rule.

        interpolated_value = input_data[0];

        unsigned int i = 0; // Current integration interval is [t_i, t_{i+1}].

        while (i + 1 < input_times.size() && input_times[i] < t)
          {
            const double left  = input_times[i];
            const double right = std::min(input_times[i + 1], t);

            const double derivative_left = data_derivative[i];
            const double derivative_right =
              input_times[i + 1] < t ?
                data_derivative[i + 1] :
                data_derivative[i] +
                  (t - input_times[i]) / (input_times[i + 1] - input_times[i]) *
                    (data_derivative[i + 1] - data_derivative[i]);

            interpolated_value +=
              0.5 * (right - left) * (derivative_left + derivative_right);

            ++i;
          }
      }
    else if (mode == Mode::DerivativeSplineInterpolation)
      {
        // Splines are piecewise cubic over each knot span. Therefore, we can
        // integrate them exactly by using piecewise Cavalieri-Simpson
        // integration.

        interpolated_value = input_data[0];

        unsigned int i = 0; // Current integration interval is [t_i, t_{i+1}].

        while (i + 1 < input_times.size() && input_times[i] < t)
          {
            const double left  = input_times[i];
            const double right = std::min(input_times[i + 1], t);
            const double mid   = 0.5 * (left + right);

            interpolated_value +=
              (right - left) / 6.0 *
              (spline[0](left) + 4.0 * spline[0](mid) + spline[0](right));

            ++i;
          }
      }

    return interpolated_value;
  }


  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_linear_interpolation(
    const std::vector<double> &    times,
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_linear_interpolation(times, data_i);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_cubic_spline(
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs,
    const double &                 a_,
    const double &                 b_,
    const std::optional<double> &  a_prime,
    const std::optional<double> &  b_prime)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_cubic_spline(data_i, a_, b_, a_prime, b_prime);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_smoothing_spline(
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs,
    const double &                 a_,
    const double &                 b_,
    const double &                 lambda)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_smoothing_spline(data_i, a_, b_, lambda);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_fourier(
    const std::vector<double> &    times,
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_fourier(times, data_i);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_derivative_linear_interpolation(
    const std::vector<double> &    times,
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_derivative_linear_interpolation(times, data_i);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::setup_as_derivative_spline_interpolation(
    const std::vector<VectorType> &data,
    const IndexSet &               data_owned_dofs,
    const double &                 a_,
    const double &                 b_)
  {
    // Possibly resize.
    if (this->size() != data_owned_dofs.n_elements())
      {
        this->resize(data_owned_dofs.n_elements());
      }

    size_t              el(0);
    std::vector<double> data_i(data.size());
    for (IndexSet::ElementIterator i = data_owned_dofs.begin();
         i != data_owned_dofs.end();
         ++i, ++el)
      {
        for (size_t l = 0; l < data.size(); ++l)
          {
            data_i[l] = data[l][*i];
          }
        (*this)[el].setup_as_derivative_spline_interpolation(data_i, a_, b_);
      }
  }

  template <class VectorType>
  void
  TimeInterpolationFEM<VectorType>::evaluate(const double &  t,
                                             const IndexSet &owned_dofs,
                                             VectorType &    values) const
  {
    // Check size set by setup_as_*
    AssertThrow(this->size() == owned_dofs.n_elements(),
                ExcDimensionMismatch(this->size(), owned_dofs.n_elements()));

    size_t el(0);
    for (IndexSet::ElementIterator i = owned_dofs.begin();
         i != owned_dofs.end();
         ++i, ++el)
      {
        values[*i] = (*this)[el].evaluate(t);
      }

    values.compress(VectorOperation::insert);
  }

  /// Explicit instantiation.
  template class TimeInterpolationFEM<LinAlg::MPI::Vector>;

  /// Explicit instantiation.
  template class TimeInterpolationFEM<LinAlg::MPI::BlockVector>;

  /// Explicit instantiation.
  template class TimeInterpolationFEM<InterpolatedSignedDistance>;

} // namespace lifex::utils
