/********************************************************************************
  Copyright (C) 2019 - 2023 by the lifex authors.

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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/command_line_parser.hpp"

#include <boost/filesystem.hpp>

namespace lifex
{
  CommandLineParser::CommandLineParser(int &               argc,
                                       char **&            argv,
                                       const bool &        lifex_options,
                                       const clipp::group &cli_in)
    : app_name(Utilities::split_string_list(argv[0], "/").back())
  {
    clipp::group cli = cli_in;

    if (lifex_options)
      cli = (cli, declare_lifex_options());

    cli = (cli, declare_misc_options());

    parse_cli(argc, argv, cli);

    if (!lifex_options)
      return;

    // Canonicalize output directory and add trailing '/'.
    // weakly_canonical() + absolute() are used instead of
    // canonical(), which assumes that the directory already exists.
    output_directory = boost::filesystem::weakly_canonical(
                         boost::filesystem::absolute(output_directory))
                         .string() +
                       "/";

    // Create output directory, if it does not already exist.
    boost::filesystem::create_directories(output_directory);

    // Check file extension.
    const std::vector<std::string> filename_split =
      Utilities::split_string_list(param_filename, ".");

    AssertThrow(filename_split.size() > 1 && (filename_split.back() == "prm" ||
                                              filename_split.back() == "json" ||
                                              filename_split.back() == "xml"),
                ExcMessage("Parameter filename \"" + param_filename +
                           "\" must provide a .prm, .json or .xml extension."));

    if (generate_mode)
      {
        // Parse verbosity.
        if (verbosity_param_minimal)
          verbosity_param = VerbosityParam::Minimal;
        else if (verbosity_param_full)
          verbosity_param = VerbosityParam::Full;
        else
          verbosity_param = VerbosityParam::Standard;
      }
  }

  void
  CommandLineParser::parse_cli(int &argc, char **&argv, const clipp::group &cli)
  {
    if (!clipp::parse(argc, argv, cli) || show_help)
      {
        // App name.
        const std::string app_name =
          Utilities::split_string_list(argv[0], "/").back();

        const clipp::doc_formatting format = clipp::doc_formatting{}
                                               .first_column(0)
                                               .indent_size(2)
                                               .doc_column(6)
                                               .last_column(80);

        // Core::mpi_rank has not been initialized yet.
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::cout << "Usage:" << std::endl
                      << clipp::usage_lines(cli, app_name, format) << std::endl
                      << std::endl
                      << "Options:" << std::endl
                      << clipp::documentation(cli, format) << std::endl
                      << std::endl
                      << "License: LGPLv3" << std::endl;
          }

        // Prevent MPI from abnormal termination.
        MPI_Finalize();

        std::exit(!show_help);
      }
  }

  clipp::group
  CommandLineParser::declare_misc_options()
  {
    // Set default values.
    show_help = false;

    const clipp::group cli =
      (("Miscellaneous:") %
       ((clipp::option("-h", "--help").set(show_help)) %
          ("display this help message and exit"),
        (clipp::option("-v", "--version").call([] {
          // Core::mpi_rank has not been initialized yet.
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::cout << "lifex v" << lifex_version << std::endl
                        << std::endl
                        << "License: LGPLv3" << std::endl;
            }

          // Prevent MPI from abnormal termination.
          MPI_Finalize();

          std::exit(0);
        })) %
          ("display version information and exit")));

    return cli;
  }

  clipp::group
  CommandLineParser::declare_lifex_options()
  {
    // Set default values.
    param_filename          = app_name + ".prm";
    generate_mode           = false;
    dry_run                 = false;
    verbosity_param_minimal = false;
    verbosity_param_full    = false;
    output_directory        = "./";
    log_filename            = "log_params";

    const clipp::group cli =
      (("Parameter files control:") %
         ((clipp::option("-g", "--generate-params").set(generate_mode) &
           (clipp::option("minimal").set(verbosity_param_minimal) |
            clipp::option("full").set(verbosity_param_full))) %
            ("If specified, only generate parameter file(s) and exit."
             "\nOptionally, increase/decrease the parameter file verbosity.\n"),

          (clipp::option("-f", "--params-filename") &
           clipp::value("param filename", param_filename)) %
            ("Param filename to load (or generate)\n[default: " +
             param_filename +
             "].\nExtensions supported: \".prm\", \".json\", \".xml\"."),

          (clipp::option("-d", "--dry-run").set(dry_run) %
           ("If specified, only declare, parse parameters, save parameter "
            "log file and exit."))),


       ("Output control:") %
         ((clipp::option("-o", "--output-directory") &
           clipp::value("output directory", output_directory)) %
            ("Output directory [default: " + output_directory + "]."),
          (clipp::option("-l", "--log-file") &
           clipp::value("log file filename", log_filename)) %
            ("Parameter log file filename [default: " + log_filename +
             "].\nExtensions supported: \".prm\", \".json\", \".xml\"."))
#if defined(LIN_ALG_PETSC)
         ,
       // This option is declared but never parsed by this class.
       // It is rather parsed by PETScInitialize(), called by the constructor of
       // dealii::Utilities:MPI_InitFinalize, in turn invoked by the constructor
       // of lifex_init. The following declaration makes the argument parser
       // aware that this option exists, so that it gets forwarded to PETSc.
       ("PETSc related options:") %
         ((clipp::option("-options_file") &
           clipp::value("PETSc options file")) %
            ("Filename for PETSc options (please refer to PETSc users "
             "manual)."),
          // Explicitly create a group by adding a second (empty) parameter.
          clipp::parameter())
#endif
      );

    return cli;
  }
} // namespace lifex
