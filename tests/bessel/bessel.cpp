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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/core_model.hpp"
#include "source/init.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::tests
{
  /// Test class for Bessel functions.
  class TestBessel : public CoreModel
  {
  public:
    /// Constructor.
    TestBessel(const std::string &subsection)
      : CoreModel(subsection)
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.enter_subsection(prm_subsection_path);
      params.declare_entry("Tolerance",
                           "1e-12",
                           Patterns::Double(0),
                           "Tolerance used for Bessel function computation and "
                           "to determine if the test passes.");
      params.leave_subsection();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.parse();

      params.enter_subsection(prm_subsection_path);
      prm_tolerance = params.get_double("Tolerance");
      params.leave_subsection();
    }

    /// Run the test.
    virtual void
    run() override
    {
      using namespace std::complex_literals;

      auto test_value = [this](const unsigned int &        n,
                               const std::complex<double> &in,
                               const std::complex<double> &out) {
        auto val = utils::bessel_j(n, in, prm_tolerance);
        pcout << "J_" << n << " " << in << " = " << std::setprecision(12) << val
              << std::endl;

        AssertThrow(std::abs(val - out) < prm_tolerance,
                    ExcMessage("Error in Bessel function computation"));
      };

      test_value(0, 0.0, 1.0);
      test_value(1, 0.0, 0.0);
      test_value(2, 0.0, 0.0);
      test_value(0, 1.0, 0.765197686557967);
      test_value(1, 1.0, 0.440050585744934);
      test_value(2, 1.0, 0.114903484931901);
      test_value(0, 1.0i, 1.266065877752008);
      test_value(1, 1.0i, 0.565159103992485i);
      test_value(2, 1.0i, -0.135747669767038);

      // Just to be sure...
      test_value(4,
                 M_PI + std::exp(1) * 1.0i,
                 -0.297659009487745 + 0.665297573904011i);
    }

  protected:
    /// Test tolerance.
    double prm_tolerance;
  };
}; // namespace lifex::tests

/// Run Bessel function test.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::tests::TestBessel test("Test Bessel");
      test.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
