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

#include "source/init.hpp"

#include <string>

namespace lifex
{
  lifex_init::lifex_init(int &               argc,
                         char **&            argv,
                         const unsigned int  max_num_threads,
                         const bool &        lifex_options,
                         const clipp::group &cli_in)
    : mpi_init_finalize(argc, argv, max_num_threads)
    , parser(argc, argv, lifex_options, cli_in)
    , core(parser.app_name,
           parser.param_filename,
           parser.generate_mode,
           parser.dry_run,
           parser.verbosity_param,
           parser.output_directory,
           parser.log_filename)
  {
    std::string app_command = "*** Run: " + parser.app_name + " ";

    // Parse executed command.
    for (int i = 1; i < argc; ++i)
      {
        app_command += argv[i];

        if (i < argc - 1)
          app_command += " ";
      }

    timer_output.enter_subsection(app_command);
  }

  lifex_init::~lifex_init()
  {
    timer_output.leave_subsection();

    // Make sure that the timer summary is always printed.
    timer_output.print_summary();
    timer_output.disable_output();
  }

} // namespace lifex
