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
 */

#include "source/core.hpp"

#include "source/io/csv_writer.hpp"

#include <boost/io/ios_state.hpp>

namespace lifex::utils
{
  CSVWriter::CSVWriter(const bool &active_)
    : active(active_)
  {}

  CSVWriter::~CSVWriter()
  {
    if (file.is_open())
      {
        file.close();
      }
  }

  void
  CSVWriter::declare_entries(const std::vector<std::string> &entries)
  {
    AssertThrow(!file.is_open(),
                ExcMessage("\"" + filename + "\" already open."));

    for (const auto &entry : entries)
      {
        // If entry was not already set.
        AssertThrow(!active || std::find(header.begin(), header.end(), entry) ==
                                 header.end(),
                    ExcMessage("Entry \"" + entry + "\" already declared."));

        if (active)
          header.push_back(entry);
      }
  }

  void
  CSVWriter::open(const std::string &filename_, const char &delimiter_)
  {
    if (file.is_open())
      {
        AssertThrow(filename == (Core::prm_output_directory + filename_) &&
                      delimiter == delimiter_,
                    ExcMessage("Filename and delimiter already set."));
      }
    else
      {
        filename  = Core::prm_output_directory + filename_;
        delimiter = delimiter_;

        // Active rank opens and writes the file, other ranks append to it.
        file.open(filename, active ? std::ofstream::out : std::ofstream::app);

        AssertThrow(file.is_open(), ExcFileNotOpen(filename));

        if (active)
          {
            for (const auto &entry : header)
              {
                file << entry << delimiter;
              }

            file << std::endl;
          }
      }
  }

  void
  CSVWriter::set_entries(const std::map<std::string, value_type> &entries)
  {
    AssertThrow(file.is_open(), ExcFileNotOpen(filename));

    for (const auto &entry : entries)
      {
        // Look for key "entry.first" in header.
        AssertThrow(
          !active || std::find(header.begin(), header.end(), entry.first) !=
                       header.end(),
          ExcMessage("Entry \"" + entry.first + "\" was not declared."));

        if (active)
          line_values[entry.first] = entry.second;
      }
  }

  void
  CSVWriter::write_line()
  {
    AssertThrow(file.is_open(), ExcFileNotOpen(filename));

    // Write line only if all entries have been inserted in the current
    // line.
    if (active && line_values.size() == header.size())
      {
        value_type value;

        // Values are printed in the same order they are declared in the
        // header.
        for (const auto &entry : header)
          {
            value = line_values[entry];

            // Backup stream flags and manipulators.
            const boost::io::ios_all_saver iostream_backup(file);

            // Print each type accordingly.
            if (std::holds_alternative<int>(value))
              {
                file << std::setiosflags(int_flags)
                     << std::setprecision(int_precision) << std::get<int>(value)
                     << delimiter;
              }
            else if (std::holds_alternative<unsigned int>(value))
              {
                file << std::setiosflags(int_flags)
                     << std::setprecision(int_precision)
                     << std::get<unsigned int>(value) << delimiter;
              }
            else if (std::holds_alternative<double>(value))
              {
                file << std::setiosflags(double_flags)
                     << std::setprecision(double_precision)
                     << std::get<double>(value) << delimiter;
              }
            else // if (std::holds_alternative<std::string>(value))
              {
                file << "\"" << std::get<std::string>(value) << "\""
                     << delimiter;
              }
          }

        file << std::endl;

        line_values.clear();
      }
  }

} // namespace lifex::utils
