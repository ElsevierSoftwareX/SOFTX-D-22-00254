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

#ifndef LIFEX_UTILS_CSV_TEST_HPP_
#define LIFEX_UTILS_CSV_TEST_HPP_

#include "source/core_model.hpp"

#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Helper class use to compare a CSV (Comma-Separated Values) file containing
   * a (time-dependent) simulation solution to a reference one.
   *
   * This class is used for automatic testing (see, @a e.g., @ref CSVTest).
   *
   * @note The CSV files are assumed to start with a header line,
   * and the first column should represent the time variable.
   */
  class CSVDiff : public CoreModel
  {
  public:
    /// Constructor.
    CSVDiff(const std::string &subsection,
            const std::string &path_reference_solution_);

    /// Declare input parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /**
     * Compare the given CSV file to the reference solution specified
     * in the parameter file.
     *
     * Let @f$v^i(t)@f$ be the piecewise linear time interpolant of the
     * @f$i@f$-th computed variable of interest, and @f$v_\mathrm{ex}^i(t)@f$
     * the interpolant of the corresponding reference solution.
     *
     * Then for all @f$i@f$, the following error is computed:
     * @f[
     * \varepsilon^i = \frac{\|v^i(t) -
     * v_\mathrm{ex}^i(t)\|_{L^\infty}}{\|v_\mathrm{ex}^i(t)\|_{L^\infty}}.
     * @f]
     *
     * Throw an exception if for some @f$i@f$ @f$\varepsilon^i@f$ is above
     * the tolerance @ref prm_tolerance.
     */
    void
    compare_to_reference_solution(
      const std::string &filename_solution_simulation) const;

  private:
    /// Path to reference solutions folder.
    std::string path_reference_solutions;

    /// @name Parameters read from file.
    /// @{

    /// CSV file containing the reference solution.
    std::string prm_filename_solution_reference;

    /// Names of reference variables.
    mutable std::vector<std::string> prm_variable_names;

    /// Tolerance on the relative @f$L^\infty@f$ error.
    double prm_tolerance;

    /// @}
  };

  /// Enumeration to enable/disable Trilinos/PETSc linear algebra in @ref CSVTest.
  ///
  /// This flag has impact on the folder used to locate reference solution
  /// files.
  enum class WithLinAlg : bool
  {
    Yes = true,
    No  = false
  };

  /// @brief Class to perform a test on a core model.
  ///
  /// The test runs a simulation and compares a CSV solution to a reference
  /// one.
  ///
  /// The class specified as template argument should expose:
  /// - a <kbd>Model::declare_parameters()</kbd> method, that sets the
  /// parameters structure;
  /// - a <kbd>Model::parse_parameters()</kbd> method, that parses the
  /// parameter file;
  /// - a <kbd>Model::run()</kbd> method, that runs the simulation;
  /// - a <kbd>Model::get_output_csv_filename()</kbd> method, that returns the
  ///   name of the (simulated) output CSV file being generated.
  template <class Model, WithLinAlg LinAlg = WithLinAlg::Yes>
  class CSVTest : public CoreModel
  {
  public:
    /// Constructor.
    template <class... Args>
    CSVTest(const std::string &subsection, Args... args)
      : CoreModel(subsection)
      , model(subsection, args...)
      , csv_diff(subsection + " / CSV test",
                 static_cast<bool>(LinAlg) ?
                   lifex_path_reference_solutions_linalg :
                   lifex_path_reference_solutions)
    {}

    /// Declare input parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      model.declare_parameters(params);
      csv_diff.declare_parameters(params);
    }

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      model.parse_parameters(params);
      csv_diff.parse_parameters(params);
    }

    /// Run the test. Throw an exception if the test has not passed.
    virtual void
    run() override
    {
      pcout << "Running test..." << std::endl;

      // Run the model simulation.
      model.run();

      csv_diff.compare_to_reference_solution(prm_output_directory +
                                             model.get_output_csv_filename());
    }

  protected:
    /// Model to be tested.
    Model model;

    /// CSV comparison.
    CSVDiff csv_diff;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_CSV_TEST_HPP_ */
