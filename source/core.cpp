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

#include "source/core.hpp"

namespace lifex
{
  Core::Core(const std::string &   app_name_,
             const std::string &   prm_filename_,
             const bool &          prm_generate_mode_,
             const bool &          prm_dry_run_,
             const VerbosityParam &prm_verbosity_param_,
             const std::string &   prm_output_directory_,
             const std::string &   prm_log_filename_)
  {
    // Prevent the assignment operator from destructing the unique
    // pointers.
    if (!_initialized)
      {
        _initialized = true;

        // Static data members cannot be initialized in the initializer list.
        _comm = MPI_COMM_WORLD;
        _rank = Utilities::MPI::this_mpi_process(mpi_comm);
        _size = Utilities::MPI::n_mpi_processes(mpi_comm);

        pcout_ptr =
          std::make_unique<ConditionalOStream>(std::cout, mpi_rank == 0);
        pcerr_ptr =
          std::make_unique<ConditionalOStream>(std::cerr, mpi_rank == 0);

        timer_output_ptr = std::make_unique<TimerOutput>(
          mpi_comm, pcout, TimerOutput::summary, TimerOutput::wall_times);

        _app_name             = app_name_;
        _prm_filename         = prm_filename_;
        _prm_generate_mode    = prm_generate_mode_;
        _prm_dry_run          = prm_dry_run_;
        _prm_verbosity_param  = prm_verbosity_param_;
        _prm_output_directory = prm_output_directory_;
        _prm_log_filename     = prm_log_filename_;
      }
  }
} // namespace lifex
