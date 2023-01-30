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
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 */

#ifndef __LIFEX_COREMODEL_HPP_
#define __LIFEX_COREMODEL_HPP_

#include "source/core.hpp"
#include "source/param_handler.hpp"
#include "source/quadrature_evaluation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lifex
{
  /// Namespace for utilities with the purpose of logging.
  namespace utils::log
  {
    /// Section separator.
    static inline constexpr auto separator_section =
      "======================================================================";

    /// Subsection separator.
    static inline constexpr auto separator_subsection =
      "----------------------------------------------------------------------";
  } // namespace utils::log

  /**
   * @brief Helper class implementing the core of a @lifex physical model.
   *
   * This class is a decorator of @ref Core, adding a common interface
   * for declaring and parsing input parameters.
   *
   * @note The methods @ref declare_parameters and @ref parse_parameters
   * must be overridden in derived classes. If no parameter is needed,
   * the implementation can be left empty.
   */
  class CoreModel : public Core
  {
  public:
    /// Constructor.
    /// @param[in] subsection The name of the subsection where to declare this class parameters.
    CoreModel(const std::string &subsection);

    /// Virtual destructor.
    virtual ~CoreModel() = default;

    /// Declare input parameters.
    virtual void
    declare_parameters(ParamHandler &params) const = 0;

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) = 0;

    /// Run model simulation. Empty by default.
    virtual void
    run()
    {}

    /// Depending on flags from @ref CommandLineParser, run
    /// current app to generate parameter file(s) or to start the actual
    /// simulation.
    ///
    /// @param[in] generate_custom A optional lambda function used to customize
    ///                            the parameter file(s) generation.
    ///                            If unspecified, defaults to @ref generate_parameters.
    virtual void
    main_run_generate(const std::function<void()> &generate_custom = {}) final;

    /// Same as the function above, using @ref generate_parameters_from_json
    /// as the default generator.
    virtual void
    main_run_generate_from_json(const std::vector<std::string> &filenames_input,
                                const std::string &suffix = "") final
    {
      main_run_generate([this, &filenames_input, &suffix]() {
        generate_parameters_from_json(filenames_input, suffix);
      });
    }

    /// Print parameters of the current class and of its dependencies to a
    /// <kbd>.prm</kbd>, <kbd>.json</kbd> or <kbd>.xml</kbd> file.
    ///
    /// The extension used is the same as that of @ref Core::prm_filename.
    ///
    /// @param[in] nondefault_params Map containing parameter names and their
    /// modified values. This can be used to modify the printed parameter values
    /// with respect to their default. The inner key is given in the format
    /// "Path / To / Subsection / Param name". Trailing whitespaces around
    /// separators are ignored.
    /// @param[in] suffix If non-empty, the parameter file will be named
    /// <kbd>executable_name_suffix.{prm,json,xml}</kbd>.
    ///
    /// The input map is forwarded to @ref ParamHandler, so that non-default
    /// parameters are always declared and printed, disregarding their verbosity
    /// value.
    ///
    /// Here is an example of usage:
    /// @code{.cpp}
    /// std::map<std::string, std::string> default_params(
    ///   {{"Electrophysiology / Linear solver / Solver type",
    ///     "GMRES"},
    ///    ...});
    ///
    /// // Re-use a previously defined map.
    /// std::map<std::string, std::string> lv_params(default_params);
    /// lv_params[
    ///   "Electrophysiology / Fiber generation / Mesh and space "
    ///   "discretization / Geometry type"] = "Left ventricle";
    ///
    /// // Or overwrite everything.
    /// std::map<std::string, std::string> ra_params(
    ///   {{"Electrophysiology / Linear solver / Solver type",
    ///     "CG"},
    ///    {"Electrophysiology / Fiber generation / Mesh and space "
    ///     "discretization / Geometry type", "Right atrium"},
    ///    ...});
    ///
    /// model.generate_parameters(default_params);
    /// model.generate_parameters(lv_params, "left_ventricle");
    /// model.generate_parameters(ra_params, "right_atrium");
    /// @endcode
    void
    generate_parameters(
      const std::map<std::string, std::string> &nondefault_params = {},
      const std::string &                       suffix            = "") const;

    /// Same as @ref generate_parameters, but reading non-default
    /// parameters from a <kbd>JSON</kbd> files stored in the
    /// <kbd>config</kbd> folder.
    ///
    /// @param[in] filenames_input The input vector of <kbd>JSON</kbd> files
    ///   (without extensions).
    ///   The files are read and merged together in the order they appear in
    ///   this vector, potentially overwriting parameters already defined in
    ///   previous files.
    ///
    /// @param[in] suffix If non-empty, the parameter file will be named
    ///                   <kbd>executable_name_suffix.{prm,json,xml}</kbd>.
    ///
    /// Here are some examples of usage:
    /// @code{.cpp}
    /// // Read non-default parameters from
    /// // "config/nondefault.json"
    /// // and generate the ready-to-run parameter file
    /// // "executable_name.{prm,json,xml}".
    /// model.generate_parameters_from_json({"nondefault"});
    ///
    /// // Initialize non-default parameters with the content of
    /// // "config/nondefault.json",
    /// // merge them with the content of
    /// // "config/left_ventricle.json" and "config/restart.json"
    /// // and generate the ready-to-run parameter file
    /// // "executable_name_foo.{prm,json,xml}".
    /// model.generate_parameters_from_json(
    ///   {"nondefault", "left_ventricle", "restart"},
    ///   "foo");
    /// @endcode
    void
    generate_parameters_from_json(
      const std::vector<std::string> &filenames_input,
      const std::string &             suffix = "") const;

    /**
     * Convert a string-to-string map into a <kbd>Boost</kbd> property tree
     * and write it to a <kbd>config/executable_name_suffix.json</kbd> file.
     */
    void
    write_map_as_json(const std::map<std::string, std::string> &input_map,
                      const std::string &suffix = "") const;

    /**
     * Overwrite
     * <kbd>config/executable_name_suffix2.json</kbd>
     * with the difference between the two files
     * <kbd>config/executable_name_suffix1.json</kbd>,
     * <kbd>config/executable_name_suffix2.json</kbd>.
     */
    void
    write_json_diff(const std::string &suffix1 = "",
                    const std::string &suffix2 = "") const;

  protected:
    /// Utility function to check whether the @ref generate_parameters
    /// run successfully, @a i.e. all the entries have been found.
    /// If this is not the case, an exception is thrown.
    static void
    check_nondefault_parameters(
      const std::map<std::string, std::string> &nondefault_params);

    std::string prm_subsection_path; ///< Parameter subsection.
  };

} // namespace lifex

#endif /* __LIFEX_CORE_COREMODEL_HPP_ */
