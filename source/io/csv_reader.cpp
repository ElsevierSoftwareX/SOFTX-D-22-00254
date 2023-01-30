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
 * @author Francesco Regazzoni <francesco.regazzoni@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/io/csv_reader.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::utils
{
  std::vector<std::string>
  csv_read_header(const std::string &filename, const char &delimiter)
  {
    std::ifstream file(filename, std::ifstream::in);

    AssertThrow(file.is_open(), ExcFileNotOpen(filename));
    AssertThrow(file.peek() != EOF, ExcMessage(filename + " is empty."));

    std::string line;
    std::getline(file, line);

    return Utilities::split_string_list(line, delimiter);
  }

  unsigned int
  csv_find_time_row(const std::string &filename,
                    const double &     time,
                    const char &       delimiter,
                    const bool &       skip_header)
  {
    std::optional<unsigned int> row = {};

    const std::vector<std::vector<double>> csv =
      utils::csv_read(filename, delimiter, skip_header);

    for (unsigned int r = 0; r < csv.size(); ++r)
      {
        if (utils::is_equal(csv[r][0], time))
          {
            row = r;
            break;
          }
      }

    AssertThrow(row.has_value(),
                ExcMessage("Timestep " + std::to_string(time) +
                           " is not present in the " + filename + " file."));

    return row.value();
  }

  std::map<std::string, double>
  csv_read_time_variables(const std::string &             filename,
                          const double &                  time,
                          const std::vector<std::string> &variable_names,
                          const char &                    delimiter)
  {
    const std::vector<std::string> csv_header =
      utils::csv_read_header(filename, delimiter);

    const std::vector<std::string> variables =
      !variable_names.empty() ?
        variable_names :
        // Use the simulation CSV header. The first column (time) is ignored.
        std::vector<std::string>(csv_header.begin() + 1, csv_header.end());

    std::vector<unsigned int> col;

    // For each variable, find the corresponding column index in the CSV file.
    for (const auto &var : variables)
      {
        const auto it = std::find(csv_header.begin(), csv_header.end(), var);

        AssertThrow(it != csv_header.end(),
                    ExcMessage("Variable " + var + " is not present in the " +
                               filename + " file."));

        col.push_back(it - csv_header.begin());
      }

    // Determine row corresponding to current time.
    const unsigned int row =
      utils::csv_find_time_row(filename, time, delimiter);

    const std::vector<std::vector<double>> csv =
      utils::csv_read(filename, delimiter, true);

    std::map<std::string, double> output;
    for (unsigned int i = 0; i < variables.size(); ++i)
      output[variables[i]] = csv[row][col[i]];

    return output;
  }

  double
  csv_read_time_variable(const std::string &filename,
                         const double &     time,
                         const std::string &variable_name,
                         const char &       delimiter)
  {
    // csv_read_time_variables returns a map with only one entry: we simply
    // return its value.
    return csv_read_time_variables(filename, time, {variable_name}, delimiter)
      .begin()
      ->second;
  }
} // namespace lifex::utils
