/********************************************************************************
  Copyright (C) 2021 - 2022 by the lifex authors.

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
 * @author Nicolas Alejandro Barnafi <nicolas.barnafi@unipv.it>.
 */

#include "source/geometry/rigid_motion.hpp"

#include <deal.II/grid/grid_tools.h>

#include <vector>

namespace lifex::utils
{
  RigidMotion::RigidMotion(const unsigned int &type_)
    : Function<dim>(dim)
    , type(type_)
  {
    Assert(dim == 3, ExcMessage("Rigid motions supported only in 3D."));
    Assert(type <= 5, ExcIndexRange(type, 0, 6));
  }

  double
  RigidMotion::value(const Point<dim> &p, const unsigned int component) const
  {
    if (type == 0)
      {
        return (component == 0);
      }
    else if (type == 1)
      {
        return (component == 1);
      }
    else if (type == 2)
      {
        return (component == 2);
      }
    else if (type == 3)
      {
        if (component == 0)
          return 0.0;
        else if (component == 1)
          return p[2];
        else
          return -p[1];
      }
    else if (type == 4)
      {
        if (component == 0)
          return -p[2];
        else if (component == 1)
          return 0.0;
        else
          return p[0];
      }
    else // if (type == 5)
      {
        if (component == 0)
          return p[1];
        else if (component == 1)
          return -p[0];
        else
          return 0.0;
      }
  }

} // namespace lifex::utils
