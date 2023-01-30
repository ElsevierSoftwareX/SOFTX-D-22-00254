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
 */

#ifndef LIFEX_UTILS_CSV_READER_HPP_
#define LIFEX_UTILS_CSV_READER_HPP_

#include "source/lifex.hpp"

#include "source/param_handler.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * Read a CSV (Comma-Separated Values) file.
   * @param[in] filename    Name of the file to be read.
   * @param[in] delimiter   File delimiter (defaulted to comma).
   * @param[in] skip_header Bool to specify whether the first row in the file
   *                        contains header information to be ignored.
   * @return A vector of vectors. The first index corresponds to the row,
   * whereas the second one to the column.
   */
  template <class ValueType = double>
  std::vector<std::vector<ValueType>>
  csv_read(const std::string &filename,
           const char &       delimiter   = ',',
           const bool &       skip_header = true)
  {
    std::ifstream file(filename, std::ifstream::in);

    AssertThrow(file.is_open(), ExcFileNotOpen(filename));
    AssertThrow(file.peek() != EOF, ExcMessage(filename + " is empty."));

    std::vector<std::vector<ValueType>> matrix;

    std::string line;
    std::string item;

    if (skip_header)
      {
        // Read the first row and ignore it.
        std::getline(file, line);
      }

    // Parse file.
    while (std::getline(file, line))
      {
        // Trim CR characters.
        if (line.back() == '\r')
          line.pop_back();

        matrix.emplace_back();

        // Parse line.
        std::istringstream line_stream(line);

        while (std::getline(line_stream, item, delimiter))
          {
            matrix.back().push_back(string_to<ValueType>(item));
          }
      }

    file.close();

    return matrix;
  }

  /**
   * Read the header of a CSV file, i.e. its first line, and return it as a
   * vector of strings.
   */
  std::vector<std::string>
  csv_read_header(const std::string &filename, const char &delimiter = ',');

  /**
   * Find the index of the row corresponding to a given time in a CSV file.
   */
  unsigned int
  csv_find_time_row(const std::string &filename,
                    const double &     time,
                    const char &       delimiter   = ',',
                    const bool &       skip_header = true);

  /**
   * Extract values from the row corresponding to a given time in a CSV file.
   * Returns a map whose keys are the variable names and whose values are the
   * corresponding values in the CSV file at the requested time.
   *
   * Optionally, only a subset of the variables can be requested by passing
   * their names to the variable_names argument. If the argument is left empty,
   * all variables are returned.
   */
  std::map<std::string, double>
  csv_read_time_variables(const std::string &             filename,
                          const double &                  time,
                          const std::vector<std::string> &variable_names = {},
                          const char &                    delimiter      = ',');

  /**
   * Wrapper around @ref csv_read_time_variables for a single variable.
   */
  double
  csv_read_time_variable(const std::string &filename,
                         const double &     time,
                         const std::string &variable_name,
                         const char &       delimiter = ',');

  /**
   * Transpose data read by @ref csv_read.
   *
   * @return A vector of vectors. The first index corresponds to the column,
   * whereas the second one to the row.
   *
   * @note All the input rows are assumed to have the same length as the first one.
   * An exception is thrown if shorter rows are present.
   */
  template <class ValueType = double>
  std::vector<std::vector<ValueType>>
  csv_transpose(const std::vector<std::vector<ValueType>> &data)
  {
    // .at() throws an exception if entry does not exist.
    std::vector<std::vector<ValueType>> data_t(
      data.at(0).size(), std::vector<ValueType>(data.size()));

    for (size_t i = 0; i < data_t.size(); ++i)
      for (size_t j = 0; j < data.size(); ++j)
        data_t[i][j] = data[j].at(i);

    return data_t;
  }

  /**
   * Read a vector contained in a CSV (Comma-Separated Values) file.
   * N.B. The vector must be stored as a row vector in the file.
   * @param[in] filename    Name of the file to be read.
   * @param[in] delimiter   File delimiter (defaulted to comma).
   * @param[in] skip_header Bool to specify whether the first row in the file
   *                        contains header information to be ignored.
   * @return A vector of <kbd>ValueType</kbd>.
   */
  template <class ValueType = double>
  Vector<ValueType>
  csv_read_to_Vector(const std::string &filename,
                     const char &       delimiter   = ',',
                     const bool &       skip_header = true)
  {
    const std::vector<std::vector<ValueType>> raw_data =
      csv_read<ValueType>(filename, delimiter, skip_header);

    AssertThrow(raw_data.size() == 1, ExcDimensionMismatch(raw_data.size(), 1));

    Vector<ValueType> vec(raw_data[0].size());
    for (size_t i = 0; i < raw_data[0].size(); ++i)
      {
        vec(i) = raw_data[0][i];
      }

    return vec;
  }

  /**
   * Read full matrix contained in a CSV (Comma-Separated Values) file.
   * @param[in] filename    Name of the file to be read.
   * @param[in] delimiter   File delimiter (defaulted to comma).
   * @param[in] skip_header Bool to specify whether the first row in the file
   *                        contains header information to be ignored.
   * @return A full matrix of <kbd>ValueType</kbd>.
   */
  template <class ValueType = double>
  FullMatrix<ValueType>
  csv_read_to_FullMatrix(const std::string &filename,
                         const char &       delimiter   = ',',
                         const bool &       skip_header = true)
  {
    const std::vector<std::vector<ValueType>> raw_data =
      csv_read<ValueType>(filename, delimiter, skip_header);

    const size_t n_rows = raw_data.size();
    const size_t n_cols = raw_data[0].size();

    FullMatrix<ValueType> mat(n_rows, n_cols);

    for (size_t i = 0; i < n_rows; ++i)
      {
        AssertThrow(raw_data[i].size() == n_cols,
                    ExcDimensionMismatch(raw_data[i].size(), n_cols));

        for (size_t j = 0; j < n_cols; ++j)
          {
            mat(i, j) = raw_data[i][j];
          }
      }

    return mat;
  }

  /**
   * Read a \f$N \times 2\f$ matrix contained in a CSV (Comma-Separated Values)
   * file.
   * @param[in] filename    Name of the file to be read.
   * @param[in] delimiter   File delimiter (defaulted to comma).
   * @param[in] skip_header Bool to specify whether the first row in the file
   *                        contains header information to be ignored.
   * @return A map of <kbd><KeyType, ValueType></kbd>.
   *
   * @note If the file contains more than two columns,
   * the additional ones are discarded.
   */
  template <class KeyType = unsigned int, class ValueType = double>
  std::map<KeyType, ValueType>
  csv_read_to_map(const std::string &filename,
                  const char &       delimiter   = ',',
                  const bool &       skip_header = true)
  {
    std::ifstream file(filename, std::ifstream::in);

    AssertThrow(file.is_open(), ExcFileNotOpen(filename));
    AssertThrow(file.peek() != EOF, ExcMessage(filename + " is empty."));

    std::map<KeyType, ValueType> map;

    std::string line;
    std::string item;

    if (skip_header)
      {
        // Read the first row and ignore it.
        std::getline(file, line);
      }

    // Parse file.
    while (std::getline(file, line))
      {
        // Trim CR characters.
        if (line.back() == '\r')
          line.pop_back();

        KeyType   key;
        ValueType value;

        // Parse line.
        std::istringstream line_stream(line);

        std::getline(line_stream, item, delimiter);
        key = string_to<KeyType>(item);

        std::getline(line_stream, item, delimiter);
        value = string_to<ValueType>(item);

        map[key] = value;
      }

    file.close();

    return map;
  }

} // namespace lifex::utils

#endif /* LIFEX_UTILS_CSV_READER_HPP_ */
