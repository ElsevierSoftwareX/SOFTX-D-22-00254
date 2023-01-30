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

#ifndef __LIFEX_PARAM_HANDLER_HELPERS_HPP_
#define __LIFEX_PARAM_HANDLER_HELPERS_HPP_

#include "source/core.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>

#include <boost/property_tree/ptree.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * Mangle string. Convert every letter to lowercase and every special
   * character to underscore.
   */
  std::string
  mangle(const std::string &input);

  /**
   * Helper template function to convert an input string to
   * a value of type <kbd>bool</kbd>, <kbd>int</kbd>, <kbd>unsigned int</kbd>,
   * <kbd>double</kbd>, <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
   *
   * @param[in] input             The input string.
   * @param[in] point_delimiter   Optional delimiter to split the point
   *  coordinates, settable only in case the template parameter is
   *  <kbd>Point<dim></kbd>. If not provided, the default point delimiter
   *  <kbd>" "</kbd> is used.
   *
   * If the template parameter is <kbd>std::string</kbd>, the input string
   * itself will be returned, for compatibility with the functions defined
   * below.
   */
  template <class ValueType>
  ValueType
  string_to(const std::string &              input,
            const std::optional<std::string> point_delimiter = {})
  {
    static_assert(is_any_v<ValueType,
                           bool,
                           int,
                           unsigned int,
                           double,
                           std::string,
                           Point<dim>>);

    if constexpr (!std::is_same_v<ValueType, Point<dim>>)
      {
        AssertThrow(!point_delimiter.has_value(),
                    ExcMessage("Optional input \"point_delimiter\" can only be "
                               "specified for the Point<dim> type."));
      }

    if constexpr (std::is_same_v<ValueType, bool>)
      {
        AssertThrow(mangle(input) == "true" || mangle(input) == "false",
                    ExcMessage("Input string " + input +
                               " must be either \"true\" or \"false\"."));

        return (mangle(input) == "true");
      }
    else if constexpr (is_any_v<ValueType, int, unsigned int>)
      {
        return static_cast<ValueType>(Utilities::string_to_int(input));
      }
    else if constexpr (std::is_same_v<ValueType, double>)
      {
        return static_cast<ValueType>(Utilities::string_to_double(input));
      }
    else if constexpr (std::is_same_v<ValueType, Point<dim>>)
      {
        const std::vector<double> point_coords = Utilities::string_to_double(
          Utilities::split_string_list(input, point_delimiter.value_or(" ")));

        AssertThrow(point_coords.size() == dim,
                    ExcDimensionMismatch(point_coords.size(), dim));

        Point<dim> output;

        for (unsigned int d = 0; d < dim; ++d)
          {
            output[d] = point_coords[d];
          }

        return output;
      }
    else // if constexpr (std::is_same_v<ValueType, std::string>)
      {
        return input;
      }

    // Silence warnings from Intel compilers.
    // Actually never reached.
    AssertThrow(false, ExcLifexInternalError());
    return ValueType();
  }

  /**
   * Helper template function to convert an input string to a vector of values
   * of type <kbd>bool</kbd>, <kbd>int</kbd>, <kbd>unsigned int</kbd>,
   * <kbd>double</kbd>, <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
   *
   * Optionally, for error checking purposes, the expected size of the vector
   * can be provided and an exception is thrown if the number of elements is
   * different from this expected size. Moreover, if the expected size is
   * provided, in case the input string has only one entry the third optional
   * parameter enables to duplicate such entry in all positions of the output
   * vector.
   *
   * @param[in] input               The input string.
   * @param[in] expected_size       Optional expected size of the output vector.
   * @param[in] duplicate_single_entry In case of a one-entry input vector
   *   and a provided expected size, toggle copying the element value in
   *   all the expected-size entries of the output vector.
   * @param[in] delimiter           Delimiter to split the input string.
   * @param[in] point_delimiter     Delimiter used to split point coordinates
   *   in case of values of type <kbd>Point<dim></kbd>. If not provided, the
   *   default point delimiter <kbd>" "</kbd> is used.
   *
   * @return The output vector.
   */
  template <class ValueType>
  std::vector<ValueType>
  string_to_vector(const std::string &         input,
                   const std::optional<size_t> expected_size          = {},
                   const bool                  duplicate_single_entry = true,
                   const std::string &         delimiter              = ",",
                   const std::optional<std::string> point_delimiter   = {})
  {
    const std::vector<std::string> input_vec =
      Utilities::split_string_list(input, delimiter);

    std::vector<ValueType> output_vec(input_vec.size());

    if (expected_size.has_value())
      {
        if (duplicate_single_entry && output_vec.size() == 1)
          {
            output_vec.resize(expected_size.value());
            std::fill(output_vec.begin(),
                      output_vec.end(),
                      string_to<ValueType>(input_vec.front()));

            return output_vec;
          }
        else
          {
            if (duplicate_single_entry)
              {
                AssertThrow(output_vec.size() == expected_size.value(),
                            ExcDimensionMismatch2(input_vec.size(),
                                                  1,
                                                  expected_size.value()));
              }
            else
              {
                AssertThrow(output_vec.size() == expected_size.value(),
                            ExcDimensionMismatch(input_vec.size(),
                                                 expected_size.value()));
              }
          }
      }

    for (size_t i = 0; i < output_vec.size(); ++i)
      {
        output_vec[i] = string_to<ValueType>(input_vec[i], point_delimiter);
      }

    return output_vec;
  }

  /**
   * Helper template function to convert an input string to a set of values of
   * type <kbd>bool</kbd>, <kbd>int</kbd>, <kbd>unsigned int</kbd>,
   * <kbd>double</kbd>, <kbd>std::string</kbd> or <kbd>Point<dim></kbd>.
   *
   * An optional input set may be provided: the conversion succeeds only if it
   * does not contain any of the elements in the input string. If the
   * conversion succeeds, elements are also inserted into the additional set.
   * A possible use case of this functionality is to check that the same
   * boundary tag does not get assigned to multiple different boundary
   * conditions.
   *
   * @param[in]      input            The input string.
   * @param[in, out] already_assigned An exception is thrown if at least one
   *    entry in the input string is already contained in this optional set,
   *    otherwise insert entries also into this set.
   * @param[in]      delimiter        Delimiter to split the input string.
   * @param[in]      point_delimiter  Delimiter used to split point coordinates
   *   in case of values of type <kbd>Point<dim></kbd>. If not provided, the
   *   default point delimiter <kbd>" "</kbd> is used.
   *
   * @return The output set.
   */
  template <class ValueType>
  std::set<ValueType>
  string_to_set(const std::string &              input,
                std::set<ValueType> *            already_assigned = nullptr,
                const std::string &              delimiter        = ",",
                const std::optional<std::string> point_delimiter  = {})
  {
    const std::vector<std::string> input_vec =
      Utilities::split_string_list(input, delimiter);

    std::set<ValueType> output_set;

    for (const std::string &s : input_vec)
      {
        const ValueType value = string_to<ValueType>(s, point_delimiter);

        // std::insert.second returns false if value was already in the set.
        if (already_assigned != nullptr)
          AssertThrow(already_assigned->insert(value).second,
                      ExcMessage("The value " + s + " is already assigned."));

        output_set.insert(value);
      }

    return output_set;
  }

  /**
   * Extract all the initial letters of all words in a string, given a word
   * delimiter, and export them in uppercase. For example,
   * <kbd>"This is a   string"</kbd> will be converted to <kbd>"TIAS"</kbd>.
   */
  std::string
  string_to_initials(const std::string &input,
                     const std::string &delimiter = " ");

  /**
   * Convert a <kbd>Boost</kbd> property tree (@a e.g. read from
   * a <kbd>JSON</kbd> file) into a string-to-string map,
   * for use within @ref CoreModel::generate_parameters.
   *
   * @param[in]      nondefault_params_tree The input property tree.
   * @param[in, out] nondefault_params_map  The map to fill or to append new values to.
   * @param[in]      current_path           The path of the current tree node
   *                                        being processed.
   */
  void
  convert_ptree_to_map(
    const boost::property_tree::ptree & nondefault_params_tree,
    std::map<std::string, std::string> &nondefault_params_map,
    std::string                         current_path = "");

  /**
   * Convert a string-to-string map into a <kbd>Boost</kbd> property tree
   * (@a e.g. for writing to a <kbd>JSON</kbd> file).
   *
   * @param[in]      nondefault_params_map  The input map.
   * @param[in, out] nondefault_params_tree The property tree to fill or
   *                                        to append new values to.
   */
  void
  convert_map_to_ptree(
    const std::map<std::string, std::string> &nondefault_params_map,
    boost::property_tree::ptree &             nondefault_params_tree);

} // namespace lifex::utils

#endif /* __LIFEX_PARAM_HANDLER_HELPERS_HPP_ */
