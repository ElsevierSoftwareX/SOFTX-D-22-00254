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

#ifndef __LIFEX_INIT_HPP_
#define __LIFEX_INIT_HPP_

#include "source/command_line_parser.hpp"
#include "source/core.hpp"

namespace lifex
{
  /**
   * @brief Class used to initialize @lifex @ref Core functionalities.
   *
   * This class should be instantiated at the beginning of a <kbd>main</kbd>
   * function whenever such application makes use of <kbd>MPI</kbd> or Core
   * functionalities. In this way we make sure that this object is not
   * destructed until the program ends.
   *
   * @note The Core object is constructed explicitly, so to make
   * sure its constructor is executed before anything else, right after
   * <kbd>MPI</kbd> initialization. Without such explicit instantiation, Core
   * would be constructed upon the first instantiation of an object of any
   * class inheriting from it: if no such class was instantiated (which can
   * happen, e.g., in non-class-dependent tests), Core constructor would never
   * be called and its members never initialized.
   */
  class lifex_init
  {
  public:
    /// Constructor.
    lifex_init(
      int &               argc,
      char **&            argv,
      const unsigned int  max_num_threads = numbers::invalid_unsigned_int,
      const bool &        lifex_options   = true,
      const clipp::group &cli_in          = clipp::group());

    /// Destructor.
    ~lifex_init();

  private:
    /// Handler to a <kbd>MPI_InitFinalize</kbd> object.
    const Utilities::MPI::MPI_InitFinalize mpi_init_finalize;

    /// Handler to a @ref CommandLineParser object.
    const CommandLineParser parser;

    /// Handler to a @ref Core object.
    const Core core;
  };

} // namespace lifex

#endif /* __LIFEX_INIT_HPP_ */
