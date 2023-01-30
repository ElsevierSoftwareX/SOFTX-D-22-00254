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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "source/numerics/fixed_point_acceleration.hpp"

namespace lifex::utils
{
  /// Explicit instantiations.
  template class FixedPointRelaxation<LinAlg::MPI::Vector>;
  template class FixedPointRelaxation<LinAlg::MPI::BlockVector>;
  template class FixedPointRelaxation<Vector<double>>;

  template class AitkenAcceleration<LinAlg::MPI::Vector>;
  template class AitkenAcceleration<LinAlg::MPI::BlockVector>;
  template class AitkenAcceleration<Vector<double>>;

  template class AndersonAcceleration<LinAlg::MPI::Vector>;
  template class AndersonAcceleration<LinAlg::MPI::BlockVector>;
  template class AndersonAcceleration<Vector<double>>;

} // namespace lifex::utils
