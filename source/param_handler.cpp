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

#include "source/param_handler.hpp"

#include <algorithm>

namespace lifex
{
  ParamHandler::ParamHandler()
    : ::dealii::ParameterHandler()
  {
    reset_verbosity();
  }

  void
  ParamHandler::declare_entry(const std::string &          entry,
                              const std::string &          default_value,
                              const Patterns::PatternBase &pattern,
                              const std::string &          documentation,
                              const bool &                 has_to_be_set)
  {
    if (Core::prm_generate_mode && is_default_entry(entry) &&
        verbosity > Core::prm_verbosity_param)
      return;

    this->::dealii::ParameterHandler::declare_entry(
      entry, default_value, pattern, documentation, has_to_be_set);
  }

  void
  ParamHandler::declare_entry_selection(const std::string &entry,
                                        const std::string &default_value,
                                        const std::string &options,
                                        const std::string &documentation_pre,
                                        const std::string &documentation_post,
                                        const bool &       has_to_be_set)
  {
    if (Core::prm_generate_mode && is_default_entry(entry) &&
        verbosity > Core::prm_verbosity_param)
      return;

    std::string documentation_with_description(documentation_pre);

    if (!documentation_with_description.empty())
      documentation_with_description += "\n";

    documentation_with_description += "Available options are: ";

    const std::vector<std::string> options_split =
      Utilities::split_string_list(options, "|");

    for (size_t k = 0; k < options_split.size(); ++k)
      {
        documentation_with_description += options_split[k];

        if (k < options_split.size() - 1)
          documentation_with_description += " | ";
        else
          documentation_with_description += ".";
      }

    if (!documentation_post.empty())
      documentation_with_description += "\n" + documentation_post;

    this->declare_entry(entry,
                        default_value,
                        Patterns::Selection(options),
                        documentation_with_description,
                        has_to_be_set);
  }

  void
  ParamHandler::enter_subsection(const std::string &subsection)
  {
    subsection_names.emplace_back(subsection);
    subsections_to_skip.emplace_back(Core::prm_generate_mode &&
                                     is_default_subsection(subsection) &&
                                     verbosity > Core::prm_verbosity_param);

    if (subsections_to_skip.back())
      return;

    this->::dealii::ParameterHandler::enter_subsection(subsection);
  }

  void
  ParamHandler::leave_subsection()
  {
    const bool to_skip = subsections_to_skip.back();

    subsection_names.pop_back();
    subsections_to_skip.pop_back();

    if (to_skip)
      return;

    this->::dealii::ParameterHandler::leave_subsection();
  }

  void
  ParamHandler::enter_subsection_path(const std::string &subsection_path)
  {
    current_subsection_paths.emplace_back(
      Utilities::split_string_list(subsection_path, "/"));

    for (const auto &subsection : current_subsection_paths.back())
      {
        this->enter_subsection(subsection);
      }
  }

  void
  ParamHandler::leave_subsection_path()
  {
    for (const auto &subsection : current_subsection_paths.back())
      {
        // Silence compiler warning.
        (void)subsection;
        this->leave_subsection();
      }

    current_subsection_paths.pop_back();
  }

  void
  ParamHandler::parse()
  {
    if (Core::prm_filename.empty())
      return;

    // Determine extension.
    const std::vector<std::string> prm_filename_split =
      Utilities::split_string_list(Core::prm_filename, ".");

    AssertThrow(prm_filename_split.back() == "prm" ||
                  prm_filename_split.back() == "json" ||
                  prm_filename_split.back() == "xml",
                ExcMessage("File " + Core::prm_filename +
                           " has a wrong extension."));

    if (prm_filename_split.back() == "prm")
      {
        this->::dealii::ParameterHandler::parse_input(Core::prm_filename);
      }
    else if (prm_filename_split.back() == "json")
      {
        std::ifstream param_file(Core::prm_filename);
        this->::dealii::ParameterHandler::parse_input_from_json(param_file);
      }
    else // if (prm_filename_split.back() == "xml")
      {
        std::ifstream param_file(Core::prm_filename);
        this->::dealii::ParameterHandler::parse_input_from_xml(param_file);
      }
  }

  void
  ParamHandler::save(const std::string &filename) const
  {
    if (Core::mpi_rank == 0)
      {
        const std::vector<std::string> filename_split =
          Utilities::split_string_list(filename, ".");

        // If no extension is provided, use that of @ref Core::prm_filename.
        if (filename_split.size() == 1 ||
            (filename_split.back() != "prm" &&
             filename_split.back() != "json" && filename_split.back() != "xml"))
          {
            save(filename + "." +
                 Utilities::split_string_list(Core::prm_filename, ".").back());
            return;
          }

        std::ofstream param_ofstream(Core::prm_output_directory + filename);

        if (filename_split.back() == "prm")
          {
            this->::dealii::ParameterHandler::print_parameters(
              param_ofstream,
              ParamHandler::Text | ParamHandler::KeepDeclarationOrder);
          }
        else if (filename_split.back() == "json")
          {
            this->::dealii::ParameterHandler::print_parameters(
              param_ofstream,
              ParamHandler::JSON | ParamHandler::KeepDeclarationOrder);
          }
        else if (filename_split.back() == "xml")
          {
            this->::dealii::ParameterHandler::print_parameters(
              param_ofstream,
              ParamHandler::XML | ParamHandler::KeepDeclarationOrder);
          }
      }
  }

  bool
  ParamHandler::is_default_entry(const std::string &entry) const
  {
    for (const auto &nondefault_path : nondefault_paths)
      {
        Utilities::split_string_list(nondefault_path, "/").back();

        std::vector<std::string> nondefault_path_split =
          Utilities::split_string_list(nondefault_path, "/");

        const std::string nondefault_entry_name = nondefault_path_split.back();

        nondefault_path_split.pop_back(); // Pop parameter name.

        // Look for matching names.
        if (nondefault_entry_name == entry)
          {
            // Skip entries with the same name not belonging to current
            // subsection.
            if (subsection_names.empty() ||
                subsection_names == nondefault_path_split)
              {
                return false;
              }
          }
      }

    return true;
  }

  bool
  ParamHandler::is_default_subsection(const std::string &subsection) const
  {
    for (const auto &nondefault_path : nondefault_paths)
      {
        std::vector<std::string> nondefault_path_split =
          Utilities::split_string_list(nondefault_path, "/");

        nondefault_path_split.pop_back(); // Pop parameter name.

        // Look for subsection in nondefault_path_split.
        if (std::find(nondefault_path_split.begin(),
                      nondefault_path_split.end(),
                      subsection) != nondefault_path_split.end())
          {
            // Skip subsections with the same name belonging to other
            // sections by determining whether the intersection between the
            // two sets is empty or not.
            const size_t min_size =
              std::min(subsection_names.size(), nondefault_path_split.size());

            std::vector<std::string> subsection_names_cut(
              subsection_names.begin(), subsection_names.begin() + min_size);

            std::vector<std::string> nondefault_path_split_cut(
              nondefault_path_split.begin(),
              nondefault_path_split.begin() + min_size);

            if (subsection_names_cut == nondefault_path_split_cut)
              {
                return false;
              }
          }
      }

    return true;
  }
} // namespace lifex
