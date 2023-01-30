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

#ifndef __LIFEX_CORE_HPP_
#define __LIFEX_CORE_HPP_

#include "source/lifex.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <memory>
#include <string>

namespace lifex
{
  /// Enumeration of the possible verbosity for parameter file generation.
  enum class VerbosityParam
  {
    /// Only minimal parameters (@a i.e. mesh info, time discretization), as
    /// well the non-default ones (set through @ref CoreModel::generate_parameters).
    Minimal,
    /// Default parameters (@ref VerbosityParam::Minimal +
    /// standard physical/numerical parameters).
    Standard,
    /// All parameters (@ref VerbosityParam::Standard +
    /// advanced physical/numerical parameters).
    Full
  };

  /**
   * @brief Helper class implementing the core of @lifex.
   *
   * All members are declared <kbd>static</kbd> so they get initialized once
   * for all the possible instances of this class.
   *
   * Since C++17, the <kbd>inline</kbd> specifier is used to initialize
   * <kbd>static</kbd> members in the header file
   * (see <a
   * href="https://stackoverflow.com/questions/45183324/whats-the-difference-between-static-constexpr-and-static-inline-variables-in-c">this
   * link</a> for further information).
   *
   * Members that are not supposed to be modified within derived classes
   * are declared <kbd>private</kbd> and aliased as
   * <kbd>public</kbd> <kbd>constexpr</kbd> references.
   *
   * @see @ref run for further details.
   */
  class Core
  {
  private:
    /// Check if all class members are initialized.
    static inline bool _initialized = false;

    /// Aliased by public member @ref mpi_comm.
    static inline MPI_Comm _comm;

    /// Aliased by public member @ref mpi_rank.
    static inline unsigned int _rank;

    /// Aliased by public member @ref mpi_size.
    static inline unsigned int _size;

    /// Aliased by public member @ref app_name.
    static inline std::string _app_name;

    /// Aliased by public member @ref prm_filename.
    static inline std::string _prm_filename;

    /// Aliased by public member @ref prm_generate_mode.
    static inline bool _prm_generate_mode;

    /// Aliased by public member @ref prm_dry_run.
    static inline bool _prm_dry_run;

    /// Aliased by public member @ref prm_verbosity_param.
    static inline VerbosityParam _prm_verbosity_param;

    /// Aliased by public member @ref prm_output_directory.
    static inline std::string _prm_output_directory;

    /// Aliased by public member @ref prm_log_filename.
    static inline std::string _prm_log_filename;

  public:
    /// Constructor.
    Core(const std::string &   app_name_             = "",
         const std::string &   prm_filename_         = "",
         const bool &          prm_generate_mode_    = false,
         const bool &          prm_dry_run_          = false,
         const VerbosityParam &prm_verbosity_param_  = VerbosityParam::Standard,
         const std::string &   prm_output_directory_ = "",
         const std::string &   prm_log_filename_     = "");

    /// MPI communicator.
    static inline constexpr MPI_Comm &mpi_comm = _comm;

    /// MPI rank.
    static inline constexpr unsigned int &mpi_rank = _rank;

    /// MPI size.
    static inline constexpr unsigned int &mpi_size = _size;

    /// Conditional stream for parallel standard output.
    /// @note This is a pointer because class <kbd>ConditionalOStream</kbd>
    /// has no default constructor, copy constructor or assignment operator,
    /// therefore its definition could not be delegated to Core constructor
    /// body. Also, being declared as <kbd>static inline</kbd>, it is
    /// initialized before Core constructor is called.
    static inline std::unique_ptr<ConditionalOStream> pcout_ptr;

/// Shorthand for dereferenced @ref ::lifex::Core::pcout_ptr.
#define pcout (*::lifex::Core::pcout_ptr)

    /// Conditional stream for parallel error output.
    /// @note This is a pointer because class <kbd>ConditionalOStream</kbd>
    /// has no default constructor, copy constructor or assignment operator,
    /// therefore its definition could not be delegated to Core constructor
    /// body. Also, being declared as <kbd>static inline</kbd>, it is
    /// initialized before Core constructor is called.
    static inline std::unique_ptr<ConditionalOStream> pcerr_ptr;

/// Shorthand for dereferenced @ref ::lifex::Core::pcerr_ptr.
#define pcerr (*::lifex::Core::pcerr_ptr)

    /// Timer.
    /// @note This is a pointer because class <kbd>TimerOutput</kbd>
    /// has no default constructor, copy constructor or assignment operator,
    /// therefore its definition could not be delegated to Core constructor
    /// body. Also, being declared as <kbd>static inline</kbd>, it is
    /// initialized before Core constructor is called.
    static inline std::unique_ptr<TimerOutput> timer_output_ptr;

/// Shorthand for dereferenced @ref ::lifex::Core::timer_output_ptr.
#define timer_output (*::lifex::Core::timer_output_ptr)

    /// App name, parsed via @ref CommandLineParser.
    static inline constexpr std::string &app_name = _app_name;

    /// Parameter filename, parsed via @ref CommandLineParser.
    static inline constexpr std::string &prm_filename = _prm_filename;

    /// Bool to specify whether to run in generate mode,
    /// parsed via @ref CommandLineParser.
    static inline constexpr bool &prm_generate_mode = _prm_generate_mode;

    /// Bool to toggle dry run, parsed via @ref CommandLineParser.
    static inline constexpr bool &prm_dry_run = _prm_dry_run;

    /// Verbosity level for generating parameter files.
    /// Any of @ref VerbosityParam::Minimal, @ref VerbosityParam::Standard
    /// or @ref VerbosityParam::Full.
    static inline constexpr VerbosityParam &prm_verbosity_param =
      _prm_verbosity_param;

    /// Directory for exporting output, such as simulation results, etc.,
    /// parsed via @ref CommandLineParser. It is '/' terminated.
    static inline constexpr std::string &prm_output_directory =
      _prm_output_directory;

    /// Parameter log filename.
    static inline constexpr std::string &prm_log_filename = _prm_log_filename;

    /// Number of digits used for filling output filenames with leading zeros.
    static inline constexpr unsigned int output_n_digits = 6;
  };

/// Shorthand for printing exceptions to @ref pcerr.
#define LIFEX_CATCH_EXC() LIFEX_CATCH_EXC_BASE(pcerr)

} // namespace lifex

#endif /* __LIFEX_CORE_HPP_ */
