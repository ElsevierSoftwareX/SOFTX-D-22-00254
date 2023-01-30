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

lifex_find_sources(SRCS)
lifex_add_executable(EXAMPLE multidomain_stokes ${SRCS})
lifex_link_libraries(EXAMPLE multidomain_stokes)

lifex_setup_example(multidomain_stokes)
