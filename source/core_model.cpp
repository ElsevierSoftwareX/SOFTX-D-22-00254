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

#include "source/core_model.hpp"

#include "source/numerics/numbers.hpp"

#include <boost/property_tree/json_parser.hpp>

namespace lifex
{
  CoreModel::CoreModel(const std::string &subsection)
  {
    // Convert paths like "  A  /B/ C/ " into "A / B / C".
    const std::vector<std::string> subsection_split =
      Utilities::split_string_list(subsection, "/");

    for (size_t i = 0; i < subsection_split.size(); ++i)
      {
        // Trim leading and trailing whitespaces.
        const size_t first = subsection_split[i].find_first_not_of(' ');
        const size_t last  = subsection_split[i].find_last_not_of(' ');

        AssertThrow(first != std::string::npos,
                    ExcMessage("Subsection path passed to CoreModel contains "
                               "empty subsections: \"" +
                               subsection + "\"."));

        prm_subsection_path +=
          subsection_split[i].substr(first, (last - first + 1));

        if (i < subsection_split.size() - 1)
          prm_subsection_path += " / ";
      }
  }

  void
  CoreModel::main_run_generate(const std::function<void()> &generate_custom)
  {
    if (lifex::Core::prm_generate_mode)
      {
        if (generate_custom)
          generate_custom();
        else
          generate_parameters();
      }
    else
      {
        ParamHandler params;
        declare_parameters(params);
        parse_parameters(params);

        params.save(Core::prm_log_filename);

        if (!Core::prm_dry_run)
          run();
      }
  }

  void
  CoreModel::generate_parameters(
    const std::map<std::string, std::string> &nondefault_params,
    const std::string &                       suffix) const
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Generate parameters");

    ParamHandler params;

    // Extract keys from map and forward them to ParamHandler.
    {
      std::vector<std::string> nondefault_paths;

      std::transform(
        nondefault_params.begin(),
        nondefault_params.end(),
        std::back_inserter(nondefault_paths),
        [](const std::map<std::string, std::string>::value_type &param) {
          return param.first;
        });

      params.set_nondefault_paths(nondefault_paths);
    }

    declare_parameters(params);

    // Copy the map to be able of erasing elements.
    std::map<std::string, std::string> params_to_modify(nondefault_params);

    // Modify parameters.
    // The iterator is only incremented if "entry" has not been found.
    for (auto entry = params_to_modify.cbegin();
         entry != params_to_modify.cend();)
      {
        std::vector<std::string> param_path_split =
          Utilities::split_string_list(entry->first, "/");

        const std::string param_name = param_path_split.back();

        param_path_split.pop_back(); // Pop parameter name.

        if (params.subsection_path_exists(param_path_split))
          {
            // Convert path separator for compatibility with
            // {enter,leave}_subsection_path().
            std::string subsection_path;

            for (const auto &path : param_path_split)
              {
                subsection_path += path + "/";
              }

            params.enter_subsection_path(subsection_path);

            // Even if the subsection exists, the entry may not.
            // In this case, the call to params.set() throws an exception.
            try
              {
                params.set(param_name, entry->second);

                // If entry has been found, remove it from the map.
                entry = params_to_modify.erase(entry);
              }
            catch (const ParamHandler::ExcValueDoesNotMatchPattern &exception)
              {
                throw exception;
              }
            catch (...)
              {
                // Just ignore the exception.
                // Move to next entry rather than erasing the current one.
                ++entry;
              }

            params.leave_subsection_path();
          }
        else
          {
            ++entry;
          }
      }

    CoreModel::check_nondefault_parameters(params_to_modify);

    const std::vector<std::string> filename_split =
      Utilities::split_string_list(prm_filename, ".");

    // Filename without extension.
    std::string filename = prm_filename.substr(
      0, prm_filename.size() - filename_split.back().size() - 1);

    if (!suffix.empty())
      filename += "_" + suffix;

    params.save(filename + "." + filename_split.back());
  }

  void
  CoreModel::generate_parameters_from_json(
    const std::vector<std::string> &filenames_input,
    const std::string &             suffix) const
  {
    AssertThrow(!filenames_input.empty(),
                ExcMessage("At least one JSON filename must be provided."));

    // The Boost property tree structures are always converted to a
    // string-to-string map. This makes much easier to perform processes such as
    // merging or checking that all parameters are setup correctly
    // through @ref check_nondefault_parameters.

    // Create the initial map, empty by default.
    std::map<std::string, std::string> nondefault_params_map;

    // Loop over all initialization files.
    for (const std::string &filename_input : filenames_input)
      {
        // Read the JSON file and fill the map.
        boost::property_tree::ptree nondefault_params_tree;
        read_json("config/" + filename_input + ".json", nondefault_params_tree);

        utils::convert_ptree_to_map(nondefault_params_tree,
                                    nondefault_params_map);
      }

    // Generate parameters from std::map<std::string, std::string>.
    generate_parameters(nondefault_params_map, suffix);
  }

  void
  CoreModel::check_nondefault_parameters(
    const std::map<std::string, std::string> &nondefault_params)
  {
    std::string error_message;

    unsigned int n_wrong_entries = 0;

    for (const auto &param_not_found : nondefault_params)
      {
        ++n_wrong_entries;
        error_message += "    - " + param_not_found.first + "\n";
      }

    if (n_wrong_entries > 0)
      {
        if (n_wrong_entries == 1)
          error_message = "CoreModel::check_nondefault_parameters(): the "
                          "following entry was not declared:\n" +
                          error_message;
        else
          error_message = "CoreModel::check_nondefault_parameters(): the "
                          "following entries were not declared:\n" +
                          error_message;

        AssertThrow(n_wrong_entries == 0, ExcMessage(error_message));
      }
  }

  void
  CoreModel::write_map_as_json(
    const std::map<std::string, std::string> &input_map,
    const std::string &                       suffix) const
  {
    const std::vector<std::string> filename_split =
      Utilities::split_string_list(app_name, ".");

    // Filename without extension.
    std::string filename =
      app_name.substr(0, app_name.size() - filename_split.back().size() - 1);

    if (!suffix.empty())
      filename += "_" + suffix;

    // Convert map to a property tree and write JSON file.
    boost::property_tree::ptree output_tree;
    utils::convert_map_to_ptree(input_map, output_tree);
    write_json("config/" + filename + ".json", output_tree);
  }

  void
  CoreModel::write_json_diff(const std::string &suffix1,
                             const std::string &suffix2) const
  {
    const std::vector<std::string> filename_split =
      Utilities::split_string_list(app_name, ".");

    // Filename without extension.
    std::string filename = prm_filename.substr(
      0, prm_filename.size() - filename_split.back().size() - 1);

    AssertThrow(suffix1 != suffix2,
                ExcMessage("The two suffixes cannot be equal."));

    std::map<std::string, std::string> input_map1;
    {
      std::string filename_input1 = filename;

      if (!suffix1.empty())
        filename_input1 += "_" + suffix1;

      boost::property_tree::ptree input_tree1;
      read_json("config/" + filename_input1 + ".json", input_tree1);
      utils::convert_ptree_to_map(input_tree1, input_map1);
    }

    std::map<std::string, std::string> input_map2;
    {
      std::string filename_input2 = filename;

      if (!suffix2.empty())
        filename_input2 += "_" + suffix2;

      boost::property_tree::ptree input_tree2;
      read_json("config/" + filename_input2 + ".json", input_tree2);
      utils::convert_ptree_to_map(input_tree2, input_map2);
    }

    std::map<std::string, std::string> output_map(input_map2);

    // Erase parameters that are present both in output_map and in input_map1.
    for (const auto &entry : input_map1)
      {
        if (utils::contains(output_map, entry.first))
          {
            if (output_map.at(entry.first) == entry.second)
              {
                output_map.erase(entry.first);
              }
          }
      }

    // Overwrite "config/executable_name_suffix2.json" with the new content.
    write_map_as_json(output_map, suffix2);
  }
} // namespace lifex
