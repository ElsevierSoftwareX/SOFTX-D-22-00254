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

#include "source/param_handler_helpers.hpp"

#include <algorithm>

namespace lifex::utils
{
  std::string
  mangle(const std::string &input)
  {
    std::string output;

    static const std::string allowed_characters(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

    for (const char &c : input)
      {
        if (allowed_characters.find(c) != std::string::npos)
          output.push_back(std::tolower(c));
        else
          output.push_back('_');
      }

    return output;
  }

  std::string
  string_to_initials(const std::string &input, const std::string &delimiter)
  {
    const std::vector<std::string> input_vec =
      Utilities::split_string_list(input, delimiter);

    std::string output;

    for (const std::string &word : input_vec)
      {
        output.push_back(std::toupper(word.at(0)));
      }

    return output;
  }

  void
  convert_ptree_to_map(
    const boost::property_tree::ptree & nondefault_params_tree,
    std::map<std::string, std::string> &nondefault_params_map,
    std::string                         current_path)
  {
    for (boost::property_tree::ptree::const_iterator it =
           nondefault_params_tree.begin();
         it != nondefault_params_tree.end();
         ++it)
      {
        if (!it->second.empty())
          {
            // Backup current path before appending the new subsection.
            const std::string old_path = current_path;

            current_path += it->first + " / ";

            // Recurse on child.
            convert_ptree_to_map(it->second,
                                 nondefault_params_map,
                                 current_path);

            // Restore old path.
            current_path = old_path;
          }
        else
          {
            // Leaf: add current entry to the map.
            nondefault_params_map[current_path + it->first] =
              it->second.get_value<std::string>();
          }
      }
  }

  void
  convert_map_to_ptree(
    const std::map<std::string, std::string> &nondefault_params_map,
    boost::property_tree::ptree &             nondefault_params_tree)
  {
    for (const auto &entry : nondefault_params_map)
      {
        // Remove trailing whitespaces around separators.
        std::string key_clean;
        for (const auto &key : Utilities::split_string_list(entry.first, "/"))
          key_clean += key + "/";

        key_clean.pop_back();

        // Specify '/' as a custom separator (Boost default is '.').
        nondefault_params_tree.put(
          boost::property_tree::ptree::path_type(key_clean, '/'), entry.second);
      }
  }

} // namespace lifex::utils
