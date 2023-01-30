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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(LIFEX_CORE_STANDALONE TRUE)
else()
  set(LIFEX_CORE_STANDALONE FALSE)
endif()

if(LIFEX_CORE_STANDALONE)
  cmake_minimum_required(VERSION 3.12.0)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/init.cmake)
  
  project(${LIFEX_NAME} VERSION ${LIFEX_VERSION} LANGUAGES CXX C)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_dependencies.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/indent.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_lifex.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_lifex_tests.cmake)
endif()

# Generate documentation.
if(LIFEX_CORE_STANDALONE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/doc/)
endif()

# Generate core.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/)

# Generate tests.
if(LIFEX_CORE_STANDALONE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/)
endif()

# Generate apps.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/apps/)

# Generate examples.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/)
