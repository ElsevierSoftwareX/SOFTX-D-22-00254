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
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 * @author Alberto Zingaro <alberto.zingaro@polimi.it>.
 */

#ifndef LIFEX_UTILS_TIME_INTERPOLATION_HPP_
#define LIFEX_UTILS_TIME_INTERPOLATION_HPP_

#include "source/core.hpp"

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

#include <optional>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Class to interpolate a function in a time interval.
   *
   * The class implements the linear interpolation, the cubic spline
   * interpolation and the Fourier interpolation, and evaluates the interpolated
   * function in an input time.
   *
   * The formulas implemented are taken from "Calcolo Scientifico. Esercizi e
   * problemi risolti con MATLAB e Octave" (Quarteroni, Saleri, & Gervasio,
   * 2017, 6th ed.).
   *
   * Reference: https://www.springer.com/gp/book/9788847039520
   *
   * @note The cubic spline interpolation needs at least the version 1.72 of the Boost library
   * (see
   * https://www.boost.org/doc/libs/1_72_0/libs/math/doc/html/math_toolkit/cardinal_cubic_b.html).
   */
  class TimeInterpolation : public Core
  {
  public:
    /// Constructor
    TimeInterpolation();

    /// Interpolation type enumeration.
    enum class Mode
    {
      LinearInterpolation,
      CubicSpline,                   ///< Interpolation with cubic spline.
      SmoothingCubicSpline,          ///< Smoothing cubic spline.
      FourierSeries,                 ///< Interpolation with Fourier series.
      DerivativeLinearInterpolation, ///< Linear interpolation of the derivative
                                     ///< of the data.
      DerivativeSplineInterpolation, ///< Spline interpolation of the derivative
                                     ///< of the data.
      NotInitialized                 ///< Not initialized.
    };

    /** Method to setup the function as linear interpolation.
     *
     * Save the data and the interval endpoints in private variables.
     * @param[in] times the vector with the times that correspond to the data to interpolate
     * @param[in] data the vector with the data to interpolate.
     */
    void
    setup_as_linear_interpolation(const std::vector<double> &times,
                                  const std::vector<double> &data);

    /** Method to setup the function as cubic spline interpolation.
     *
     * Compute the cubic spline coefficients to interpolate a set of data
     * points with equispace distribution in an interval.
     * @param[in] data the vector with the data to interpolate.
     * @param[in] a_ the first endpoint of the interval in which to interpolate the data.
     * @param[in] b_ the second enpoint of the interval in which to interpolate the data.
     * @param[in] a_prime the first derivative of the function to be interpolated
     *                    on the first endpoint of the interpolation interval.
     * @param[in] b_prime the first derivative of the function to be interpolated
     *                    on the second endpoint of the interpolation interval.
     * @note The first derivatives are optional data (since C++17), therefore the user may avoid imposing them.
     */
    void
    setup_as_cubic_spline(const std::vector<double> &  data,
                          const double &               a_,
                          const double &               b_,
                          const std::optional<double> &a_prime = {},
                          const std::optional<double> &b_prime = {});

    /**
     * @brief Setup as smoothing spline.
     *
     * Given input points @f$(t_i, y_i)@f$, with @f$i = 0, \dots, n@f$
     * and @f$t_i = a + h i@f$, finds the function @f$\hat{f}:[a,
     * b]\to\mathbb{R}@f$ in the set of cubic splines that minimizes the
     * functional
     * @f[
     * J(f) = \sum_{i = 0}^n \left(f(t_i) - y_i\right)^2 + \lambda\int_a^b
     * f^"(t)dt\;,
     * @f]
     * with @f$\lambda > 0@f$. For @f$\lambda = 0@f$, the interpolant cubic
     * spline is computed. For @f$\lambda\to\infty@f$, the approximant tends to
     * the least-squares regression line of the input points.
     *
     * **Reference**:
     * [Wikipedia](https://en.wikipedia.org/wiki/Smoothing_spline)
     */
    void
    setup_as_smoothing_spline(const std::vector<double> &data,
                              const double &             a_,
                              const double &             b_,
                              const double &             lambda);

    /// Method to return the real part of Fourier coefficients.
    std::vector<double>
    get_re_ck() const
    {
      AssertThrow(mode == Mode::FourierSeries,
                  ExcMessage(
                    "get_re_ck() cannot be called before setup_as_fourier()."));

      return re_ck;
    };

    /// Method to return the imaginary part of Fourier coefficients.
    std::vector<double>
    get_im_ck() const
    {
      AssertThrow(mode == Mode::FourierSeries,
                  ExcMessage(
                    "get_im_ck() cannot be called before setup_as_fourier()."));

      return im_ck;
    };

    /// Method to return the interpolation mode.
    Mode
    get_mode() const
    {
      return mode;
    }

    /** Method to setup the function through a Fourier series.
     *
     * Compute the coefficients of the Fourier serie that interpolates the
     data
     * points.
     *
     * @param[in] times the vector with the times that correspond to the data to interpolate
     * @param[in] data the vector with the data to interpolate.
     *
     * @todo Add possibility to set number of terms.
     */
    void
    setup_as_fourier(const std::vector<double> &times,
                     const std::vector<double> &data);

    /** Method to setup the function as linear interpolation of the derivative.
     *
     * The interpolant of input data @f$(t_i, y_i)@f$ is computed as follows:
     * 1. compute the finite difference derivative of the data, i.e.
     * @f[
     * u_i = \begin{cases}
     * 0 & i = 0\;, \\
     * \frac{y_{i} - y_{i-1}}{t_{i} - t_{i-1}} & i > 0\;;
     * \end{cases}
     * @f]
     * 2. compute the piecewise linear interpolant @f$\tilde{u}@f$ of @f$(t_i,
     * u_i)@f$;
     * 3. define the interpolant of the data through
     * @f[
     * \tilde{y}(t) = y_0 + \int_{t_0}^t \tilde{u}(\tau)\mathrm{d}\tau\;.
     * @f]
     *
     * @param[in] times the vector with the times that correspond to the data to interpolate
     * @param[in] data the vector with the data to interpolate.
     *
     */
    void
    setup_as_derivative_linear_interpolation(const std::vector<double> &times,
                                             const std::vector<double> &data);

    /**
     * Method to setup the function as spline interpolation of the derivative.
     */
    void
    setup_as_derivative_spline_interpolation(const std::vector<double> &data,
                                             const double &             a_,
                                             const double &             b_);

    /** Method to evaluate the interpolated functions in a point in time t.
     *
     * @param[in] t the value in time in which to evaluate the interpolated function.
     * The method returns the value of the interpolated function in t.
     *
     * @note If @ref mode = Mode::LinearInterpolation and @f$t \geq b@f$,
     * returns evaluation at @ref b.
     */
    double
    evaluate(const double &t) const;

  protected:
    /// The times corresponding to the input data to interpolate.
    std::vector<double> input_times;
    /// The data to interpolate.
    std::vector<double> input_data;
    /// The first endpoint of the interval in which to interpolate the data.
    double a;
    /// The second enpoint of the interval in which to interpolate the data.
    double b;
    /// The cubic spline object obtained through the setup_as_cubic_spline.
    /// method.
    std::vector<boost::math::interpolators::cardinal_cubic_b_spline<double>>
      spline;
    /// The real part of the coefficients obtained through the
    /// setup_as_fourier method.
    std::vector<double> re_ck;
    /// The imaginary part of the coefficients obtained through
    /// the setup_as_fourier method.
    std::vector<double> im_ck;
    /// The index to compute the series of the Fourier interpolation.
    size_t M;
    /// 0 if the number of points to interpolate is odd, 1 otherwise
    unsigned int mu;
    /// Method used to interpolate the data points, i.e.
    /// LinearInterpolation or CubicSpline or FourierSeries or
    /// DerivativeLinearInterpolation.
    Mode mode;
    /// Derivate of the data to interpolate.
    std::vector<double> data_derivative;

    /// Penalty coefficient for second derivative for smoothing cubic splines.
    double smoothing_spline_lambda;
  };


  /// @brief Extension of @ref TimeInterpolation to FEM vectors.
  ///
  /// This class can be seen as a <kbd>std::vector<TimeInterpolation></kbd>
  /// in which the vector index scans the <b>owned</b> dofs of a
  /// <kbd>LinAlg::MPI::Vector</kbd> or a <kbd> LinAlg::MPI::BlockVector</kbd>.
  ///
  /// The size of the corresponding FEM vector is set by <kbd>setup_as_*()</kbd>
  /// methods, as the size <kbd>IndexSet::n_elements()</kbd> of the input vector
  /// dofs, and it is assumed to be fixed.
  ///
  /// @note The class contains no member objects,
  /// thus inheriting from STL is fine.
  template <class VectorType = LinAlg::MPI::Vector>
  class TimeInterpolationFEM : public std::vector<TimeInterpolation>
  {
  public:
    using std::vector<TimeInterpolation>::vector; // use the constructors from
                                                  // std::vector

    /// Extension  of @ref TimeInterpolation::setup_as_linear_interpolation().
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] times the vector with the times that correspond to the data
    /// to interpolate.
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    void
    setup_as_linear_interpolation(const std::vector<double> &    times,
                                  const std::vector<VectorType> &data,
                                  const IndexSet &data_owned_dofs);

    /// Extension of @ref TimeInterpolation::setup_as_cubic_spline().
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    /// @param[in] a_ the first endpoint of the interval in which to interpolate the data.
    /// @param[in] b_ the second enpoint of the interval in which to interpolate the data.
    /// @param[in] a_prime the first derivative of the function to be
    /// interpolated on the first endpoint of the interval
    /// in which to interpolate the data.
    /// @param[in] b_prime the first derivative of the function to be
    /// interpolated on the second endpoint of the interval
    /// in which to interpolate the data.
    /// @note The first derivatives are optional data (since C++17),
    /// therefore the user may avoid imposing them.
    void
    setup_as_cubic_spline(const std::vector<VectorType> &data,
                          const IndexSet &               data_owned_dofs,
                          const double &                 a_,
                          const double &                 b_,
                          const std::optional<double> &  a_prime = {},
                          const std::optional<double> &  b_prime = {});

    /// Extension of @ref TimeInterpolation::setup_as_cubic_spline().
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    /// @param[in] a_ the first endpoint of the interval in which to interpolate the data.
    /// @param[in] b_ the second enpoint of the interval in which to interpolate the data.
    /// @param[in] lambda Regularization weight for the smoothing.
    void
    setup_as_smoothing_spline(const std::vector<VectorType> &data,
                              const IndexSet &               data_owned_dofs,
                              const double &                 a_,
                              const double &                 b_,
                              const double &                 lambda);

    /// Extension of @ref TimeInterpolation::setup_as_fourier().
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] times the vector with the times that correspond to the data
    /// to interpolate.
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    void
    setup_as_fourier(const std::vector<double> &    times,
                     const std::vector<VectorType> &data,
                     const IndexSet &               data_owned_dofs);

    /// Extension  of @ref TimeInterpolation::setup_as_derivative_linear_interpolation().
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] times the vector with the times that correspond to the data
    /// to interpolate.
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    void
    setup_as_derivative_linear_interpolation(
      const std::vector<double> &    times,
      const std::vector<VectorType> &data,
      const IndexSet &               data_owned_dofs);

    /// Extension of @ref TimeInterpolation::setup_as_derivative_spline_interpolation.
    ///
    /// Resizing is possibly performed to ensure that <kbd>this->size() ==
    /// </kbd> @p data_owned_dofs <kbd>.n_elements()</kbd>.
    ///
    /// @param[in] data the vector with the FEM vector data to interpolate
    /// (can be either ghosted or not).
    /// @param[in] data_owned_dofs the non-ghost dofs of data.
    /// @param[in] a_ the first endpoint of the interval in which to interpolate the data.
    /// @param[in] b_ the second enpoint of the interval in which to interpolate the data.
    void
    setup_as_derivative_spline_interpolation(
      const std::vector<VectorType> &data,
      const IndexSet &               data_owned_dofs,
      const double &                 a_,
      const double &                 b_);

    /// Extension of @ref TimeInterpolation::evaluate().
    ///
    /// @param[in] t the value in time in which to evaluate the
    /// interpolated FEM vector.
    /// @param[in] owned_dofs the non-ghost dofs of the FEM vector.
    /// @param[out] values FEM vector containing the interpolated values.
    void
    evaluate(const double &  t,
             const IndexSet &owned_dofs,
             VectorType &    values) const;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_TIME_INTERPOLATION_HPP_ */
