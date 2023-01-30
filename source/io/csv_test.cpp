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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/io/csv_reader.hpp"
#include "source/io/csv_test.hpp"

#include "source/numerics/numbers.hpp"
#include "source/numerics/time_interpolation.hpp"

#include <deal.II/base/table_handler.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace lifex::utils
{
  CSVDiff::CSVDiff(const std::string &subsection,
                   const std::string &path_reference_solutions_)
    : CoreModel(subsection)
    , path_reference_solutions(path_reference_solutions_)
    , prm_tolerance(1e-8)
  {}

  void
  CSVDiff::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);
    {
      params.declare_entry(
        "Reference solution filename",
        "model/filename.csv",
        Patterns::FileName(Patterns::FileName::input),
        "Name of the CSV file the reference solution is read from. "
        "Assumed to be relative to " +
          std::string(path_reference_solutions));

      params.declare_entry(
        "Variable names",
        "",
        Patterns::List(Patterns::Anything()),
        "Names of the variables to be compared. If empty, all variables in the "
        "simulation file are considered.");
    }
    params.leave_subsection_path();
  }

  void
  CSVDiff::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      prm_filename_solution_reference =
        path_reference_solutions + params.get("Reference solution filename");

      prm_variable_names =
        Utilities::split_string_list(params.get("Variable names"));
    }
    params.leave_subsection_path();
  }

  void
  CSVDiff::compare_to_reference_solution(
    const std::string &filename_solution_simulation) const
  {
    AssertThrow(filename_solution_simulation != prm_filename_solution_reference,
                ExcMessage("The two files to be compared cannot be the same."));

    TimerOutput::Scope timer_section(
      timer_output,
      prm_subsection_path + " / Compare simulation and reference solutions");

    if (mpi_rank != 0)
      return;

    bool files_are_equal = true;

    // Load the headers of the two files and construct the vectors
    // col_variables_reference and col_variables_simulation.
    std::vector<size_t> col_variables_reference;
    std::vector<size_t> col_variables_simulation;

    const std::vector<std::string> header_reference =
      csv_read_header(prm_filename_solution_reference);
    const std::vector<std::string> header_simulation =
      csv_read_header(filename_solution_simulation);

    if (prm_variable_names.empty())
      {
        // Use the simulation CSV header. The first column (time) is ignored.
        prm_variable_names =
          std::vector<std::string>(header_simulation.begin() + 1,
                                   header_simulation.end());
      }

    // For each reference variable, locate it in both headers and store the
    // corresponding index. If a variable is not found in either header, an
    // exception is thrown.
    for (const auto &var : prm_variable_names)
      {
        const auto it_reference =
          std::find(header_reference.begin(), header_reference.end(), var);
        const auto it_simulation =
          std::find(header_simulation.begin(), header_simulation.end(), var);

        AssertThrow(it_reference != header_reference.end(),
                    ExcMessage(
                      "Variable " + var +
                      " is not present in the reference solution CSV file."));
        AssertThrow(it_simulation != header_simulation.end(),
                    ExcMessage(
                      "Variable " + var +
                      " is not present in the simulation solution CSV file."));

        col_variables_reference.push_back(it_reference -
                                          header_reference.begin());
        col_variables_simulation.push_back(it_simulation -
                                           header_simulation.begin());
      }

    // Load the CSV files of the simulation and of the reference solutions.
    const std::vector<std::vector<double>> csv_simulation =
      csv_read(filename_solution_simulation);

    AssertThrow(!csv_simulation.empty(),
                ExcMessage(filename_solution_simulation + " cannot be empty."));

    const std::vector<std::vector<double>> csv_reference =
      csv_read(prm_filename_solution_reference);

    AssertThrow(!csv_reference.empty(),
                ExcMessage(prm_filename_solution_reference +
                           " cannot be empty."));

    // Extract simulation times and variables (to be compared as vectors).
    std::vector<double>              time_simulation;
    std::vector<std::vector<double>> variables_simulation(
      col_variables_simulation.size());

    for (const auto &time_row : csv_simulation)
      {
        time_simulation.push_back(time_row[0]);

        for (size_t j = 0; j < col_variables_simulation.size(); ++j)
          variables_simulation[j].push_back(
            time_row[col_variables_simulation[j]]);
      }

    // Extract reference times and variables (to be compared as vectors).
    std::vector<double>              time_reference;
    std::vector<std::vector<double>> variables_reference(
      col_variables_reference.size());

    for (const auto &time_row : csv_reference)
      {
        time_reference.push_back(time_row[0]);

        for (size_t j = 0; j < col_variables_reference.size(); ++j)
          variables_reference[j].push_back(
            time_row[col_variables_reference[j]]);
      }

    // Construct a common time vector, to interpolate the two solutions on
    // the same grid. The error is evaluated only where both solutions are
    // defined
    AssertThrow(
      time_reference.front() <= time_simulation.front(),
      ExcMessage(
        "The reference CSV file should contain the simulation time interval."));
    AssertThrow(
      time_reference.back() >= time_simulation.back(),
      ExcMessage(
        "The reference CSV file should contain the simulation time interval."));

    const double t0 = time_simulation.front();
    const double t1 = time_simulation.back();

    const double dt = (t1 - t0) / (time_simulation.size() - 1);

    std::vector<double> times(time_simulation.size());
    for (size_t i = 0; i < times.size(); ++i)
      times[i] = t0 + i * dt;

    // Compare the variables, computing the L^inf norm of the difference,
    // normalized with respect to the L^inf norm of the absolute deviation from
    // the mean of the reference solution.
    TableHandler error_table; // Error table for output

    error_table.declare_column("Variable");

    for (const auto &column : {"Absolute L^inf error",
                               "Reference solution norm",
                               "Relative L^inf error"})
      {
        error_table.declare_column(column);
        error_table.set_precision(column, 12);
        error_table.set_scientific(column, true);
      }

    error_table.declare_column(" ");

    double err_max = 0.0;

    for (size_t i = 0; i < col_variables_simulation.size(); ++i)
      {
        // Interpolate simulation and reference solution on the same time grid.
        TimeInterpolation interpolation_simulation;
        interpolation_simulation.setup_as_linear_interpolation(
          time_simulation, variables_simulation[i]);

        TimeInterpolation interpolation_reference;
        interpolation_reference.setup_as_linear_interpolation(
          time_reference, variables_reference[i]);

        // Compute errors.
        double err_absolute = 0.0;
        double err_relative = 0.0;

        double norm_reference = 0.0;

        for (const double t : times)
          {
            const double v   = interpolation_simulation.evaluate(t);
            const double r_v = interpolation_reference.evaluate(t);

            err_absolute = std::max(err_absolute, std::abs(v - r_v));

            norm_reference = std::max(norm_reference, std::abs(r_v));
          }

        err_relative = err_absolute / norm_reference;

        // An arbitrarily small number.
        const double tolerance_zero =
          100 * std::numeric_limits<double>::epsilon();

        const bool normalization_is_zero =
          utils::is_zero(norm_reference, tolerance_zero);

        err_max = std::max(err_max,
                           normalization_is_zero ? err_absolute : err_relative);

        // If the normalization factor is zero, it is checked that
        // the numerator is zero, rather than checking against the
        // user-specified tolerance.
        const bool solutions_are_equal = normalization_is_zero ?
                                           (err_absolute <= tolerance_zero) :
                                           (err_relative <= prm_tolerance);

        error_table.add_value("Variable", prm_variable_names[i]);
        error_table.add_value("Absolute L^inf error", err_absolute);
        error_table.add_value("Reference solution norm", norm_reference);

        if (normalization_is_zero)
          error_table.add_value("Relative L^inf error", "Ignored");
        else
          error_table.add_value("Relative L^inf error", err_relative);

        error_table.add_value(" ", solutions_are_equal ? " " : "*");

        files_are_equal &= solutions_are_equal;
      }

    // Write the error table.
    if (pcout.is_active())
      {
        pcout << std::endl;
        error_table.write_text(pcout.get_stream(),
                               TableHandler::org_mode_table);

        if (!files_are_equal)
          pcout << std::endl
                << "Maximum error: " << std::scientific << err_max << " > "
                << prm_tolerance << std::endl;
      }

    AssertThrow(files_are_equal, ExcTestFailed());
  }

} // namespace lifex::utils
