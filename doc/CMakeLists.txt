## ---------------------------------------------------------------------
## Copyright (C) 2019 - 2022 by the lifex authors.
##
## This file is part of lifex.
##
## lifex is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## lifex is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with lifex.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

# Author: Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.

find_package(Doxygen COMPONENTS dot)

if (DOXYGEN_FOUND)
  set(Doxygen_FIND_QUIETLY TRUE)

  set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                 ${DOXYFILE} @ONLY)

  add_custom_target(
    doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation"
  )
endif()
