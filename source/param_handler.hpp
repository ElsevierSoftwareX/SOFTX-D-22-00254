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
 */

#ifndef __LIFEX_PARAM_HANDLER_HPP_
#define __LIFEX_PARAM_HANDLER_HPP_

#include "source/core.hpp"
#include "source/param_handler_helpers.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <set>
#include <string>
#include <vector>

namespace lifex
{
  /// @brief Wrapper class to @dealii <kbd>ParameterHandler</kbd>, with verbosity
  /// control.
  class ParamHandler : public ::dealii::ParameterHandler
  {
  public:
    friend class CoreModel;

    /// Default constructor.
    ParamHandler();

    /// Set new verbosity value.
    ///
    /// The verbosity is automatically set to @ref VerbosityParam::Minimal
    /// for non-default parameters, disregarding the value set here, so they are
    /// always declared and printed.
    void
    set_verbosity(const VerbosityParam &verbosity_new)
    {
      AssertThrow(
        verbosity_new >= verbosity,
        ExcMessage(
          "Please call reset_verbosity() before setting a lower verbosity."));

      verbosity = verbosity_new;
    }

    /// Reset verbosity value to @ref VerbosityParam::Minimal.
    void
    reset_verbosity()
    {
      verbosity = VerbosityParam::Minimal;
    }

    /// Get @ref verbosity.
    const VerbosityParam &
    get_verbosity() const
    {
      return verbosity;
    }

    /// Wraps @dealii <kbd>declare_entry</kbd>.
    ///
    /// If the currently set verbosity is higher than @ref Core::prm_verbosity_param,
    /// the entry is not declared.
    void
    declare_entry(const std::string &          entry,
                  const std::string &          default_value,
                  const Patterns::PatternBase &pattern = Patterns::Anything(),
                  const std::string &          documentation = "",
                  const bool &                 has_to_be_set = false);

    /// Wraps @dealii <kbd>declare_entry</kbd> for
    /// <kbd>Patterns::Selection</kbd>, so that the available options are
    /// automatically appended to the generated documentation.
    ///
    /// If the currently set verbosity is higher than @ref Core::prm_verbosity_param,
    /// the entry is not declared.
    void
    declare_entry_selection(const std::string &entry,
                            const std::string &default_value,
                            const std::string &options,
                            const std::string &documentation_pre  = "",
                            const std::string &documentation_post = "",
                            const bool &       has_to_be_set      = false);

    /// Enter a subsection. If it does not yet exist, create it.
    ///
    /// If the currently set verbosity is higher than @ref Core::prm_verbosity_param,
    /// the subsection is not entered.
    void
    enter_subsection(const std::string &subsection);

    /// Leave present subsection.
    ///
    /// If the currently set verbosity is higher than @ref Core::prm_verbosity_param,
    /// the subsection is not left.
    void
    leave_subsection();

    /// Enter a subsection path, given a string where the subsections are
    /// separated by a '/' character.
    void
    enter_subsection_path(const std::string &subsection_path);

    /// Leave current subsection path, since last call to
    /// @ref enter_subsection_path.
    void
    leave_subsection_path();

    /// Parse @ref Core::prm_filename.
    /// The filename extension is used to automatically determine whether
    /// to parse from a <kbd>.prm</kbd>, <kbd>.json</kbd> or <kbd>.xml</kbd>
    /// file format.
    void
    parse();

    /// Save parameters to a <kbd>.prm</kbd>, <kbd>.json</kbd> or
    /// <kbd>.xml</kbd> file.
    ///
    /// @param[in] filename Output filename. If provided without extension,
    ///                     that of @ref Core::prm_filename will be used.
    void
    save(const std::string &filename) const;

    /// Template method to get a value of types <kbd>bool</kbd>,
    /// <kbd>int</kbd>, <kbd>unsigned int</kbd>, <kbd>double</kbd>,
    /// <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
    ///
    /// The behavior of this method is the same as the helper function
    /// @ref utils::string_to, but taking as input the name of the
    /// parameter entry corresponding to the value.
    template <class ValueType>
    ValueType
    get_value(const std::string &              entry_string,
              const std::optional<std::string> point_delimiter = {})
    {
      return utils::string_to<ValueType>(
        this->::dealii::ParameterHandler::get(entry_string), point_delimiter);
    }

    /// Template method to get a vector of values of types <kbd>bool</kbd>,
    /// <kbd>int</kbd>, <kbd>unsigned int</kbd>, <kbd>double</kbd>,
    /// <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
    ///
    /// The behavior of this methods is the same of the helper function
    /// @ref utils::string_to_vector, but taking as input the name of the
    /// parameter entry corresponding to the values.
    template <class ValueType>
    std::vector<ValueType>
    get_vector(const std::string &              entry_string,
               const std::optional<size_t>      expected_size          = {},
               const bool                       duplicate_single_entry = true,
               const std::string &              delimiter              = ",",
               const std::optional<std::string> point_delimiter        = {})
    {
      try
        {
          return utils::string_to_vector<ValueType>(
            this->::dealii::ParameterHandler::get(entry_string),
            expected_size,
            duplicate_single_entry,
            delimiter,
            point_delimiter);
        }
      // If wrong dimensions are detected by string_to_vector, we throw an
      // exception with an error message that helps identifying the
      // parameter.
      catch (const ExcDimensionMismatch & /*exc*/)
        {
          throw ExcMessage(
            "Parameter \"" + entry_string + "\" expected to have " +
            std::to_string(expected_size.value()) +
            " entries, but its value is \"" +
            this->::dealii::ParameterHandler::get(entry_string) + "\".");
        }
      catch (const ExcDimensionMismatch2 & /*exc*/)
        {
          throw ExcMessage(
            "Parameter \"" + entry_string + "\" expected to have either 1 or " +
            std::to_string(expected_size.value()) +
            " entries, but its value is \"" +
            this->::dealii::ParameterHandler::get(entry_string) + "\".");
        }
    }

    /// Template method to get a set of values of types <kbd>bool</kbd>,
    /// <kbd>int</kbd>, <kbd>unsigned int</kbd>, <kbd>double</kbd>,
    /// <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
    ///
    /// The behavior of this methods is the same of the helper function
    /// @ref utils::string_to_set, but taking as input the name of the
    /// entry string corresponding to the values.
    template <class ValueType>
    std::set<ValueType>
    get_set(const std::string &  entry_string,
            std::set<ValueType> *already_assigned = nullptr,
            const std::string &  delimiter        = ",")
    {
      return utils::string_to_set<ValueType>(
        this->::dealii::ParameterHandler::get(entry_string),
        already_assigned,
        delimiter);
    }

    /// Get @f$i@f$-th item of a list, optionally converted to a given type.
    ///
    /// @param[in] entry_string    The parameter name.
    /// @param[in] index           The index of the element to be retrieved.
    /// @param[in] expected_size   Optional input for error checking
    ///   purposes, an exception is thrown if the number of elements in
    ///   the vector is different from this value.
    /// @param[in] delimiter       The delimiter used in the list.
    /// @param[in] point_delimiter Delimiter used to split point coordinates
    ///   in case of values of type <kbd>Point<dim></kbd>. If not provided,
    ///   the default point delimiter <kbd>" "</kbd> is used.
    template <class EntryType = std::string>
    EntryType
    get_vector_item(const std::string &              entry_string,
                    const size_t &                   index,
                    const std::optional<size_t> &    expected_size   = {},
                    const std::string &              delimiter       = ",",
                    const std::optional<std::string> point_delimiter = {})
    {
      const std::vector<EntryType> &output_vec = get_vector<EntryType>(
        entry_string, expected_size, false, delimiter, point_delimiter);

      AssertThrow(index < output_vec.size(),
                  ExcMessage(
                    "Not enough entries in vector \"" + entry_string +
                    "\" (vector has " + std::to_string(output_vec.size()) +
                    " items, requested item " + std::to_string(index) + ")."));

      return output_vec.at(index);
    }

  private:
    /// Set @ref nondefault_paths.
    void
    set_nondefault_paths(const std::vector<std::string> &nondefault_paths_)
    {
      nondefault_paths = nondefault_paths_;
    };

    /// Return <kbd>true</kbd> is @p entry is the name of a non-default entry
    /// (as stored in @ref nondefault_paths), <kbd>false</kbd> otherwise.
    bool
    is_default_entry(const std::string &entry) const;

    /// Return <kbd>true</kbd> is @p subsection is the name of a section containing
    /// non-default entries (as stored in @ref nondefault_paths),
    /// <kbd>false</kbd> otherwise.
    bool
    is_default_subsection(const std::string &subsection) const;

    /// Stack for subsection paths entered so far.
    std::vector<std::vector<std::string>> current_subsection_paths;

    /// Current verbosity, @ref VerbosityParam::Minimal by default.
    VerbosityParam verbosity;

    /// List of parameters having non-default values, as provided by
    /// @ref CoreModel::generate_parameters.
    std::vector<std::string> nondefault_paths;

    /// Stack for subsection names entered so far.
    std::vector<std::string> subsection_names;

    /// Stack for subsections to be skipped for verbosity reasons, that don't
    /// containing only default parameters.
    std::vector<bool> subsections_to_skip;
  };
} // namespace lifex

#endif /* __LIFEX_PARAM_HANDLER_HPP_ */
