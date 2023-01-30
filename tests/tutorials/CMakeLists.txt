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

lifex_find_sources(TUTORIALS)

foreach(tutorial ${TUTORIALS})
  # Strip directory name and file extension.
  get_filename_component(executable ${tutorial} NAME_WE)

  lifex_add_executable(TEST ${executable} ${tutorial})
  lifex_link_libraries(TEST ${executable})

  lifex_setup_test(${executable})
  lifex_add_test_soft_mpi(${executable} ${executable} -o output_${executable})
endforeach()

lifex_create_symlink(TEST "tutorial_07_ad" "config")
