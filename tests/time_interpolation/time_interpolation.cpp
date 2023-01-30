/********************************************************************************
  Copyright (C) 2021 - 2022 by the lifex authors.

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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/init.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/csv_test.hpp"
#include "source/io/csv_writer.hpp"

#include "source/numerics/time_interpolation.hpp"

#include <memory>
#include <random>

/// Namespace for all the tests in @lifex.
namespace lifex::tests
{
  /**
   * @brief Auxiliary class to test time interpolation.
   */
  class TestTimeInterpolation : public CoreModel
  {
  public:
    /// Constructor.
    TestTimeInterpolation(const std::string &subsection)
      : CoreModel(subsection)
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection_path(prm_subsection_path);

      params.declare_entry("Number of input points",
                           "10",
                           Patterns::Integer(2),
                           "Number of (t_i, y_i) pairs to be interpolated. "
                           "Coordinates t_i are evenly spaced in [0, 1], while "
                           "y_i are randomly generated in (-0.1, 0.1).");

      params.declare_entry("Number of output points",
                           "1000",
                           Patterns::Integer(0),
                           "Number of evenly spaced points in [0, 1] on which "
                           "the interpolation is computed.");

      params.declare_entry_selection(
        "Interpolation method",
        "Linear",
        "Linear|Cubic spline|Smoothing spline|Fourier|Derivative linear "
        "interpolation|Derivative spline interpolation",
        "Method used for the interpolation of input data.");

      params.enter_subsection("Cubic spline");
      params.declare_entry(
        "Zero endpoint derivatives",
        "true",
        Patterns::Bool(),
        "Toggle whether endpoint derivatives should be constrained to zero. If "
        "false, not-a-knot spline interpolation is used.");
      params.leave_subsection();

      params.enter_subsection("Smoothing spline");
      params.declare_entry(
        "Regularization coefficient",
        "1.0",
        Patterns::Double(0),
        "Weight of the regularization term in the least-squares fit. If set to "
        "0, an interpolant spline is obtained.");
      params.leave_subsection();

      params.leave_subsection_path();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.parse();

      params.enter_subsection_path(prm_subsection_path);

      prm_n_input_points  = params.get_integer("Number of input points");
      prm_n_output_points = params.get_integer("Number of output points");

      const std::string interpolation_method =
        params.get("Interpolation method");
      if (interpolation_method == "Linear")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::LinearInterpolation;
        }
      else if (interpolation_method == "Cubic spline")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::CubicSpline;

          params.enter_subsection("Cubic spline");
          prm_spline_zero_end_derivatives =
            params.get_bool("Zero endpoint derivatives");
          params.leave_subsection();
        }
      else if (interpolation_method == "Smoothing spline")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::SmoothingCubicSpline;

          params.enter_subsection("Smoothing spline");
          prm_smoothing_spline_lambda =
            params.get_double("Regularization coefficient");
          params.leave_subsection();
        }
      else if (interpolation_method == "Fourier")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::FourierSeries;
        }
      else if (interpolation_method == "Derivative linear interpolation")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::DerivativeLinearInterpolation;
        }
      else // if (interpolation_method == "Derivative spline interpolation")
        {
          prm_interpolation_method =
            utils::TimeInterpolation::Mode::DerivativeSplineInterpolation;
        }

      params.leave_subsection_path();
    }

    /// Run the test.
    void
    run()
    {
      setup_system();
      compute_interpolation();
      output_results();
    }

    /// Return the output CSV.
    std::string
    get_output_csv_filename() const
    {
      return "output.csv";
    }

  protected:
    /// Setup.
    void
    setup_system()
    {
      // Generate artificial data to be interpolated.
      {
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Generate random data");

        // We initialize the random engine with a fixed seed, to ensure
        // reproducibility.
        std::default_random_engine             engine(1);
        std::uniform_real_distribution<double> rand(-0.1, 0.1);

        const double h = 1.0 / (prm_n_input_points - 1.0);

        for (unsigned int i = 0; i < prm_n_input_points; ++i)
          {
            input_times.push_back(h * i);
            input_data.push_back(rand(engine));
          }

        // To avoid any possible floating point inaccuracy:
        input_times.back() = 1.0;
      }

      // Setup the time interpolation object.
      {
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Setup interpolation");

        if (prm_interpolation_method ==
            utils::TimeInterpolation::Mode::LinearInterpolation)
          {
            time_interpolation.setup_as_linear_interpolation(input_times,
                                                             input_data);
          }
        else if (prm_interpolation_method ==
                 utils::TimeInterpolation::Mode::CubicSpline)
          {
            if (prm_spline_zero_end_derivatives)
              time_interpolation.setup_as_cubic_spline(input_data,
                                                       input_times.front(),
                                                       input_times.back());
            else
              time_interpolation.setup_as_cubic_spline(
                input_data, input_times.front(), input_times.back(), 0.0, 0.0);
          }
        else if (prm_interpolation_method ==
                 utils::TimeInterpolation::Mode::SmoothingCubicSpline)
          {
            time_interpolation.setup_as_smoothing_spline(
              input_data,
              input_times.front(),
              input_times.back(),
              prm_smoothing_spline_lambda);
          }
        else if (prm_interpolation_method ==
                 utils::TimeInterpolation::Mode::FourierSeries)
          {
            time_interpolation.setup_as_fourier(input_times, input_data);
          }
        else if (prm_interpolation_method ==
                 utils::TimeInterpolation::Mode::DerivativeLinearInterpolation)
          {
            time_interpolation.setup_as_derivative_linear_interpolation(
              input_times, input_data);
          }
        else
          { // if (prm_interpolation_method ==
            // utils::TimeInterpolation::Mode::DerivativeSplineInterpolation)
            time_interpolation.setup_as_derivative_spline_interpolation(
              input_data, input_times.front(), input_times.back());
          }
      }
    }

    /// Compute the interpolation.
    void
    compute_interpolation()
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Compute interpolation");

      const double h = 1.0 / (prm_n_output_points - 1.0);

      for (unsigned int i = 0; i < prm_n_output_points; ++i)
        {
          output_times.push_back(h * i);
          output_data.push_back(
            time_interpolation.evaluate(output_times.back()));
        }
    }

    /// Output interpolation data to a CSV file.
    void
    output_results()
    {
      // Write the input data to a CSV file.
      {
        utils::CSVWriter input_data_csv(mpi_rank == 0);
        input_data_csv.declare_entries({"t", "y"});
        input_data_csv.open("input.csv");

        for (unsigned int i = 0; i < input_times.size(); ++i)
          {
            input_data_csv.set_entries(
              {{"t", input_times[i]}, {"y", input_data[i]}});
            input_data_csv.write_line();
          }
      }

      // Write the output data to a CSV file.
      {
        utils::CSVWriter output_data_csv(mpi_rank == 0);
        output_data_csv.declare_entries({"t", "y"});
        output_data_csv.open(get_output_csv_filename());

        for (unsigned int i = 0; i < output_times.size(); ++i)
          {
            output_data_csv.set_entries(
              {{"t", output_times[i]}, {"y", output_data[i]}});
            output_data_csv.write_line();
          }
      }
    }

    /// Time interpolation instance.
    utils::TimeInterpolation time_interpolation;

    /// Input times.
    std::vector<double> input_times;

    /// Input data.
    std::vector<double> input_data;

    /// Output times.
    std::vector<double> output_times;

    /// Output data.
    std::vector<double> output_data;

    /// @name Parameters read from file.
    /// @{

    /// Number of input points.
    unsigned int prm_n_input_points;

    /// Number of output points.
    unsigned int prm_n_output_points;

    /// Interpolation method.
    utils::TimeInterpolation::Mode prm_interpolation_method;

    /// Toggle zero endpoint derivatives for cubic spline interpolation.
    bool prm_spline_zero_end_derivatives;

    /// Regularization coefficient for smoothing spline interpolation.
    double prm_smoothing_spline_lambda;

    /// @}
  };
} // namespace lifex::tests

/// Test for time interpolation.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      using namespace lifex::utils;

      CSVTest<lifex::tests::TestTimeInterpolation, WithLinAlg::No> test(
        "Test time interpolation");

      test.main_run_generate([&test]() {
        test.generate_parameters_from_json({"linear"}, "linear");
        test.generate_parameters_from_json({"spline"}, "spline");
        test.generate_parameters_from_json({"smoothing_spline"},
                                           "smoothing_spline");
        test.generate_parameters_from_json({"fourier"}, "fourier");
        test.generate_parameters_from_json({"derivative_linear"},
                                           "derivative_linear");
        test.generate_parameters_from_json({"derivative_spline"},
                                           "derivative_spline");
      });
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
