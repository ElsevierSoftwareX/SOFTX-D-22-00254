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
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/init.hpp"

#include "source/io/vtk_preprocess.hpp"

/// Pre-processing of a generic VTK file.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      lifex::utils::VTKPreprocess app("VTK preprocess");

      app.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
