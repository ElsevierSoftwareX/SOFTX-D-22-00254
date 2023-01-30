/********************************************************************************
  Copyright (C) 2022 by the lifex authors.

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

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/io/csv_reader.hpp"
#include "source/io/csv_test.hpp"
#include "source/io/csv_writer.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::tests
{
  /// Test class for I/O in <kbd>CSV</kbd> format using @lifex functionalities.
  class TestCSV : public CoreModel
  {
  public:
    /// Constructor.
    TestCSV(const std::string &subsection)
      : CoreModel(subsection)
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler & /* params */) const override
    {}

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.parse();
    }

    /// Run the test.
    virtual void
    run() override
    {
      const std::string filename = get_output_csv_filename();

      // Generate an arbitrary dataset.
      const size_t n = 101;

      std::vector<std::array<double, 4>> data(n);

      for (size_t i = 0; i < n; ++i)
        {
          data[i][0] = 0.1 * i;
          data[i][1] = i;
          data[i][2] = i * i;
          data[i][3] = -2.0 * i;
        }

      // Write a CSV.
      {
        utils::CSVWriter csv_writer;
        csv_writer.set_condition(mpi_rank == 0);
        csv_writer.declare_entries({"time", "x", "y", "z"});
        csv_writer.open(filename);

        for (size_t i = 0; i < n; ++i)
          {
            csv_writer.set_entries(
              {{"time", data[i][0]},
               {"x", static_cast<unsigned int>(data[i][1])},
               {"y", static_cast<double>(data[i][2])},
               {"z", static_cast<int>(data[i][3])}});
            csv_writer.write_line();
          }
      }

      // Read a CSV.
      for (unsigned int rank = 0; rank < mpi_size; ++rank)
        {
          if (rank == Core::mpi_rank)
            {
              const double tol = 1e-14;

              const std::vector<std::vector<double>> data_read =
                utils::csv_read(filename);

              const std::vector<std::vector<double>> data_transpose =
                utils::csv_transpose(data_read);

              const FullMatrix<double> data_matrix =
                utils::csv_read_to_FullMatrix(filename);

              for (size_t i = 0; i < n; ++i)
                {
                  AssertThrow(utils::is_equal(data_read[i][0], data[i][0], tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_read[i][1], data[i][1], tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_read[i][2], data[i][2], tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_read[i][3], data[i][3], tol),
                              ExcTestFailed());

                  AssertThrow(data_transpose[0][i] == data_read[i][0],
                              ExcTestFailed());
                  AssertThrow(data_transpose[1][i] == data_read[i][1],
                              ExcTestFailed());
                  AssertThrow(data_transpose[2][i] == data_read[i][2],
                              ExcTestFailed());
                  AssertThrow(data_transpose[3][i] == data_read[i][3],
                              ExcTestFailed());

                  AssertThrow(utils::is_equal(data_matrix[i][0],
                                              data[i][0],
                                              tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_matrix[i][1],
                                              data[i][1],
                                              tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_matrix[i][2],
                                              data[i][2],
                                              tol),
                              ExcTestFailed());
                  AssertThrow(utils::is_equal(data_matrix[i][3],
                                              data[i][3],
                                              tol),
                              ExcTestFailed());
                }

              const std::vector<std::string> header =
                utils::csv_read_header(filename);
              AssertThrow(header.size() == 4, ExcTestFailed());
              AssertThrow(header[0] == "time", ExcTestFailed());
              AssertThrow(header[1] == "x", ExcTestFailed());
              AssertThrow(header[2] == "y", ExcTestFailed());
              AssertThrow(header[3] == "z", ExcTestFailed());

              const double time = 1.0;

              const unsigned int idx = utils::csv_find_time_row(filename, time);
              AssertThrow(idx == 10, ExcTestFailed());

              const std::map<std::string, double> time_variables =
                utils::csv_read_time_variables(filename, time, {"x", "z"});
              AssertThrow(time_variables.at("x") ==
                            static_cast<unsigned int>(data[idx][1]),
                          ExcTestFailed());
              AssertThrow(time_variables.at("z") ==
                            static_cast<int>(data[idx][3]),
                          ExcTestFailed());
            }

          MPI_Barrier(mpi_comm);
        }
    }

    /// Return the name of the file the CSV output has been written to.
    std::string
    get_output_csv_filename() const
    {
      return "output.csv";
    }
  };
}; // namespace lifex::tests

/// Run CSV function test.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      using namespace lifex::utils;

      CSVTest<lifex::tests::TestCSV, WithLinAlg::No> test("Test CSV");
      test.main_run_generate_from_json({"nondefault"});
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
