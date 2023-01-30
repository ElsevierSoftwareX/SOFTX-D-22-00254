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

#ifndef __LIFEX_EXCEPTIONS_HPP_
#define __LIFEX_EXCEPTIONS_HPP_

#include <deal.II/base/exceptions.h>

#include <exception>
#include <iostream>

/// Macro to define common <kbd>catch</kbd> blocks
/// used for exception handling.
#define LIFEX_CATCH_EXC_BASE(OUTPUT_STREAM)                               \
  catch (const std::exception &exc)                                       \
  {                                                                       \
    OUTPUT_STREAM                                                         \
      << std::endl                                                        \
      << std::endl                                                        \
      << "--------------------------------------------------------------" \
      << std::endl                                                        \
      << "Exception on processing: " << std::endl                         \
      << exc.what() << std::endl                                          \
      << "Aborting!" << std::endl                                         \
      << "--------------------------------------------------------------" \
      << std::endl;                                                       \
                                                                          \
    return EXIT_FAILURE;                                                  \
  }                                                                       \
  catch (...)                                                             \
  {                                                                       \
    OUTPUT_STREAM                                                         \
      << std::endl                                                        \
      << std::endl                                                        \
      << "--------------------------------------------------------------" \
      << std::endl                                                        \
      << "Unknown exception!" << std::endl                                \
      << "Aborting!" << std::endl                                         \
      << "--------------------------------------------------------------" \
      << std::endl;                                                       \
                                                                          \
    return EXIT_FAILURE;                                                  \
  }

/// Exception for generic internal error.
DeclExceptionMsg(
  ExcLifexInternalError,
  "This exception usually indicates that some condition which "
  "the author of the code thought must be satisfied at a "
  "certain point in an algorithm is not fulfilled or that you are "
  "trying an operation that is not defined for this object.");

/// Exception for generic not implemented error.
DeclExceptionMsg(
  ExcLifexNotImplemented,
  "You are trying to use a functionality in lifex that is currently not "
  "implemented. In many cases, this indicates that there simply didn't appear "
  "much of a need for it, or that the author of the original code did not have "
  "the time to implement a particular case. If you hit this exception, it is "
  "therefore worth the time to look into the code to find out whether you may "
  "be able to implement the missing functionality. If you do, please consider "
  "providing a patch to the lifex development sources.");

/// Exception for methods that can be run only in standalone mode.
DeclExceptionMsg(ExcStandaloneOnly,
                 "The functionality you are trying to access is only available "
                 "in standalone mode.");

/// Exception for methods that cannot be run in standalone mode.
DeclExceptionMsg(ExcNotStandalone,
                 "The functionality you are trying to access is not available "
                 "in standalone mode.");

/// Exception for operating on a non-locally owned cell.
DeclExceptionMsg(
  ExcCellNonLocallyOwned,
  "You are trying an operation on a cell that is only allowed if "
  "the cell is locally owned by the current process.");

/// Exception for non-converging non-linear solver.
DeclExceptionMsg(ExcNonlinearNotConverged,
                 "Non-linear solver did not converge.");

/// Exception for parallel vectors with ghost elements.
DeclExceptionMsg(
  ExcParallelNonGhosted,
  "You are trying an operation on a vector that is only allowed if "
  "the vector has ghost elements, or if the executable is run in "
  "serial.");

/// Exception for failing test.
DeclExceptionMsg(ExcTestFailed, "Test failed.");

#endif /* __LIFEX_EXCEPTIONS_HPP_ */
