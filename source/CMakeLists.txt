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

# Create lifex.hpp.
configure_file(lifex.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/lifex.hpp @ONLY)

# Core.
lifex_find_sources(SRCS_CORE)

# Utilities.
lifex_find_sources_recursively(SRCS_UTILS "geometry" "io" "numerics")

lifex_set(LIB_CORE core)
lifex_add_library(${LIB_CORE} ${SRCS_CORE} ${SRCS_UTILS})
lifex_link_libraries(LIB ${LIB_CORE} ${LIB_LIFEX} ${Boost_LIBRARIES} ${VTK_LIBRARIES})
