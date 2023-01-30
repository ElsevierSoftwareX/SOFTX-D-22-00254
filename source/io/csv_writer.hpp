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

#ifndef LIFEX_UTILS_CSV_WRITER_HPP_
#define LIFEX_UTILS_CSV_WRITER_HPP_

#include "source/lifex.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace lifex::utils
{
  /// @brief Class to write a CSV (Comma-Separated Values) file.
  ///
  /// This class is supposed to be used as in the following example.
  /// @code{.cpp}
  /// // Declare a CSV writer and the corresponding entries to be exported.
  /// CSVWriter csv_writer;
  ///
  /// // The method declare_entries() may be called
  /// // multiple times from different scopes.
  /// csv_writer.declare_entries({"time", "pressure", ...});
  ///
  /// // Open file and print header.
  /// csv_writer.open("log.csv", ',');
  ///
  /// // Perform some computations.
  /// ...
  ///
  /// // For each new line, set new entries to be printed.
  /// // The method set_entries() may be called
  /// // multiple times from different scopes.
  /// csv_writer.set_entries({{"time", time}});
  /// csv_writer.set_entries({{"pressure", pressure}});
  /// ...
  ///
  /// // Notice that multiple entries can be assigned in one shot:
  /// csv_writer.set_entries({{"time", time},
  ///                         {"pressure", pressure},
  ///                         {"n_iter", it},
  ///                         {"phase", "phase 1"});
  ///
  /// // Once all entries have been set, print current line.
  /// csv_writer.write_line();
  /// @endcode
  ///
  /// @note The supported types of values are
  /// <kbd>double</kbd>, <kbd>int</kbd>,
  /// <kbd>unsigned int</kbd> and <kbd>std::string</kbd>.
  class CSVWriter
  {
  public:
    /// Since C++17.
    using value_type = std::variant<double, int, unsigned int, std::string>;

    /// Constructor.
    /// @param[in] active_ The condition based on which writes are actually
    ///                    performed, @a e.g. <kbd>mpi_rank == 0</kbd>.
    ///                    The object is active by default.
    CSVWriter(const bool &active_ = true);

    /// Destructor. Close the file.
    ~CSVWriter();

    /// Depending on the input flag, set @ref active to
    /// <kbd>true</kbd> or <kbd>false</kbd>.
    void
    set_condition(const bool &active_)
    {
      active = active_;
    }

    /// Getter method for @ref active.
    bool
    is_active() const
    {
      return active;
    }

    /// Set output format for <kbd>double</kbd> entries.
    void
    set_format_double(const std::ios_base::fmtflags &double_flags_,
                      const std::streamsize &        double_precision_ = 10)
    {
      AssertThrow(!file.is_open(),
                  ExcMessage("\"" + filename + "\" already open."));

      double_flags     = double_flags_;
      double_precision = double_precision_;
    }

    /// Set output format for <kbd>double</kbd> entries.
    void
    set_format_int(const std::ios_base::fmtflags &int_flags_,
                   const std::streamsize &        int_precision_ = 6)
    {
      AssertThrow(!file.is_open(),
                  ExcMessage("\"" + filename + "\" already open."));

      int_flags     = int_flags_;
      int_precision = int_precision_;
    }

    /// Declare a set of entries (@a i.e. variable names)
    /// to write to the CSV file.
    ///
    /// @param[in] entries Entry names.
    ///
    /// @note It may be convenient to prefix or suffix variable names
    /// with a relevant keyword (@a e.g. `"fluid_pressure"`), so to prevent that
    /// variables with the same name but different meaning are declared in
    /// multiple scopes.
    ///
    /// @warning This method must be called before calling @ref open.
    void
    declare_entries(const std::vector<std::string> &entries);

    /// Set the filename and the delimiter, open the file
    /// and print the header line.
    ///
    /// @param[in] filename_  the name of the CSV file to write;
    /// @param[in] delimiter_ the delimiter.
    ///
    /// @warning This method must be called before the first call to the
    /// @ref set_entries method and after @ref declare_entries.
    void
    open(const std::string &filename_, const char &delimiter_ = ',');

    /// Set entries for the current line.
    ///
    /// @param[in] entries Name-value map of the new entries.
    ///
    /// @note The keys in the map must match the entry names specified
    /// with @ref declare_entries.
    void
    set_entries(const std::map<std::string, value_type> &entries);

    /// If all the entries have been set in the current line,
    /// then write the current @ref line_values (i.e. all the values stored via
    /// the @ref set_entries method) to the CSV file and clear @ref line_values.
    ///
    /// Otherwise, do nothing (@a i.e. wait for the other missing values to be
    /// set).
    ///
    /// In this way, among different possible calls to this method, the current
    /// line is actually printed only once, as soon as all the entries have been
    /// set.
    void
    write_line();

  private:
    /// Whether this object is in active state or not.
    /// All write operations inside this class are performed only if
    /// <kbd>active</kbd> is <kbd>true</kbd>.
    bool active;

    /// CSV filename.
    std::string filename;

    /// The CSV file stream.
    std::ofstream file;

    /// CSV delimiter.
    char delimiter;

    /// CSV header.
    std::vector<std::string> header;

    /// Name-value map of the current CSV line.
    std::map<std::string, value_type> line_values;

    /// Output flags for <kbd>double</kbd> types.
    std::ios_base::fmtflags double_flags = std::ios::scientific;

    /// Output precision for <kbd>double</kbd> types.
    std::streamsize double_precision = 10;

    /// Output flags for <kbd>int</kbd> and <kbd>unsigned int</kbd> types.
    std::ios_base::fmtflags int_flags = std::ios::fixed;

    /// Output precision for <kbd>int</kbd> and <kbd>unsigned int</kbd> types.
    std::streamsize int_precision = 6;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_CSV_WRITER_HPP_ */
