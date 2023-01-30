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

#ifndef LIFEX_UTILS_RIGID_MOTION_HPP_
#define LIFEX_UTILS_RIGID_MOTION_HPP_

#include "source/geometry/mesh_handler.hpp"

namespace lifex::utils
{
  /// @brief Function representing all 6 rigid motions, hard-coded for 3D.
  ///
  /// It is used to create an orthogonal base spanning the rigid motions
  /// space, which is fundamental to obtain an efficient (AMG or BDDC)
  /// preconditioner. AMG preconditioners use it to create kernel-preserving
  /// restriction/extension operators, whereas BDDC uses it to enrich the
  /// primal space used for strong continuity between processors.
  class RigidMotion : public Function<dim>
  {
  public:
    /// Constructor.
    RigidMotion(const unsigned int &type_);

    /// Return function value.
    virtual double
    value(const Point<dim> &p, const unsigned int component) const override;

  private:
    /// Type of rigid motion. In order: translation along @f$x@f$, @f$y@f$,
    /// @f$z@f$ and counterclockwise rotations of @f$\frac{\pi}{2}@f$ with
    /// respect to @f$x@f$, @f$y@f$, @f$z@f$ axes).
    const unsigned int type;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_RIGID_MOTION_HPP_ */
