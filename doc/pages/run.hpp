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
 */

/// @page run Run a lifex app
/// # Table of contents
/// - [Step 0 - Set the parameter file](#generate)
/// - [Step 1 - Run!](#run)
///   - [Parallel run](#run-parallel)
///   - [Dry run](#run-dry)
/// - [Step 2 - Restart](#restart)
///
/// <a name="generate"></a>
/// # Step 0 - Set the parameter file
/// Each @lifex application or example defines a set of parameters that
/// are required in order to be run. They involve problem-specific
/// parameters (such as constitutive relations, geometry, time interval,
/// boundary conditions, ...) as well as numerical parameters (types of
/// linear/non-linear solvers, tolerances, maximum number of iterations, ...) or
/// output-related options.
///
/// In case an application has sub-dependencies
/// (such as @ref lifex::utils::LinearSolverHandler for the
/// @ref lifex::tutorials::Tutorial01 class), also the related parameters are
/// included.
///
/// Each executable exploits the functionalities of the
/// @ref lifex::CommandLineParser class, that enables to parse a set of
/// pre-defined command line flags. They are used to specify the filename and
/// the corresponding subsection for all of the parameters associated with the
/// executable and its dependencies.
///
/// The available command line options are printed with the `-h`
/// (or `--help`) flag:
/// ~~~{.sh}
/// ./executable_name -h
/// ~~~
///
/// The first step before running an executable is to generate the parameter
/// file(s) containing all the default parameter values. This is done via the
/// `-g` (or `--generate-params`) flag:
/// ~~~{.sh}
/// ./executable_name -g
/// ~~~
/// that by default generates a (number of) parameter file(s) named after the
/// executable (followed by optional suffixes), in `.prm` format.
///
/// By default, only parameters considered @a standard are printed. The
/// parameter file verbosity can be decreased or increased by passing an
/// optional flag `minimal` or `full` to the `-g` flag, respectively:
/// ~~~{.sh}
/// ./executable_name -g minimal
/// ~~~
/// Application-specific non-default parameters are always printed, disregarding
/// the verbosity flag (see @ref lifex::CoreModel::generate_parameters and
/// @ref lifex::ParamHandler::set_verbosity).
///
/// The parameter basename to generate can be customized with the `-f` (or
/// `--params-filename`) option:
/// ~~~{.sh}
/// ./executable_name -g -f custom_param_file.ext
/// ~~~
/// where the file extension `ext` can be chosen among `prm`, `json` or `xml`
/// (the latter format can be modified through a convenient <a
/// href="https://github.com/dealii/parameter_gui">graphical user interface</a>
/// or via the command line tool <a
/// href="http://xmlsoft.org/xmllint.html">`xmllint`</a>).
///
/// Absolute or relative paths can be specified.
///
/// Once generated, the user can modify, copy, move or rename the parameter file
/// depending on their needs.
///
/// <a name="run"></a>
/// # Step 1 - Run!
/// To run an executable, the `-g` flag has simply to be omitted
/// whereas the `-f` option is used to specify the
/// parameter file to be @a read (as opposed to @a written, in generation mode),
/// @a e.g.:
/// ~~~{.sh}
/// ./executable_name -f custom_param_file.ext [option...]
/// ~~~
///
/// If no <kbd>-f</kbd> flag is provided, a file named `executable_name.prm` is
/// assumed to be available in the directory where the executable is run from.
///
/// The path to the directory where all the app output files will be saved to
/// can be selected via the `-o` (or `--output-directory`)
/// flag:
/// ~~~{.sh}
/// ./executable_name -o ./results/
/// ~~~
/// If the specified directory does not already exist, it will be created.
/// By default, the current working directory is used.
///
/// Absolute or relative paths can be specified for both the input parameter
/// file and the output directory.
///
/// <a name="run-parallel"></a>
/// ## Parallel run
/// To run an app in parallel, use the `mpirun` or `mpiexec` wrapper commands
/// (which may vary depending on the MPI implementation available on your
/// machine), @a e.g.:
/// ~~~{.sh}
/// mpirun -n <N_PROCS> ./executable_name [option...]
/// ~~~
/// where `<N_PROCS>` is the desired number of parallel processes to run on.
///
/// As a rule of thumb, 10000 to 100000 degrees of freedom per process should
/// lead to the best performance. Also, beware that if the mesh has a
/// small number of cells then its parallel partitioning may not cover all the
/// processes, thus the program may freeze: in such case please consider using a
/// smaller `N_PROCS` or a finer mesh.
///
/// <a name="run-dry"></a>
/// ## Dry run and parameter file conversion
/// Upon running, a parameter log file is automatically generated in the output
/// directory, that can be used later to retrieve which parameters had been used
/// for a specific run.
///
/// By default, <kbd>log_params.ext</kbd> will be used as its filename. This can
/// be changed via the <kbd>-l</kbd> (or `--log-file`) flag, @a e.g.:
/// ~~~{.sh}
/// ./executable_name -l my_log_file.ext [option...]
/// ~~~
///
/// The extension is not mandatory: if unspecified, the same extension as the
/// input parameter file will be used.
///
/// If the <b>dry run</b> option is enabled via the <kbd>-d</kbd> (or
/// `--dry-run`) flag, the execution terminates right after the parameter log
/// file generation. This has a two-fold purpose:
/// 1. checking the correctness of the parameters being declared and parsed @a
/// before running the actual simulation (if any of the parameters did not
/// match the specified pattern or has a wrong name or has not been declared in
/// a given subsection then a runtime exception is thrown);
/// 2. @b converting a parameter file between two different
/// formats/extensions. For example, the following command converts
/// <kbd>input.xml</kbd> to <kbd>output.json</kbd>:
/// ~~~{.sh}
/// ./executable_name -f input.xml -d -l output.json [option...]
/// ~~~
///
/// <a name="restart"></a>
/// # Step 2 - Restart
/// Many modules in @lifex support a @a restart feature, @a i.e. the possibility
/// to run a simulation starting from a previously computed solution.
///
/// In order to enable the possibility of a future restart you need to set
/// - <kbd>"* / Output / Serialize solution" = true</kbd>
///
/// in the parameter file.
///
/// In order to restart a simulation from a serialized solution you need to set
/// the following parameters:
/// - <kbd>"* / Time solver / Initial time"</kbd>: the initial time
/// associated to the serialized solution;
/// - <kbd>"* / Time solver / Restart" = true</kbd>;
/// - <kbd>"* / Time solver / Restart basename"</kbd>: path (absolute
/// or relative to the specified output directory) pointing to the serialized
/// solution, with basename (@a i.e. without suffix, trailing underscore or
/// timestep indices);
/// - <kbd>"* / Time solver / Restart initial timestep"</kbd>: the
/// timestep to restart from, with leading zeros (@a e.g. <kbd>000010</kbd>).
///
/// You may also want to change the <kbd>"* / Output /
/// Filename"</kbd> in order not to overwrite already existing files, or to set
/// a different output directory via command line as specified above.
///
/// The restarted simulation is supposed to be run using the
/// <b>same number of MPI processes</b> that was used in the original execution.
/// Using a different number will likely result in a program crash.
///
