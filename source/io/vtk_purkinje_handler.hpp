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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 * @author Michele Barucca <michele.barucca@mail.polimi.it>.
 */

#ifndef LIFEX_UTILS_VTK_PURKINJE_HANDLER_HPP_
#define LIFEX_UTILS_VTK_PURKINJE_HANDLER_HPP_

#include "source/lifex.hpp"

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * This class reads and handles a Purkinje VTP file, storing in private
   * vectors all the coordinates of the Purkinje Muscle Junctions (PMJs) and
   * their activation times read from a point-data array defined on the VTP
   * file. The PMJs are the terminal points of the Purkinje network where the
   * electric signal passes into the cardiac muscle. Thus, they can be used to
   * model the physiological way to stimulate the myocardium.
   */
  class VTKPurkinjeHandler
  {
  public:
    /**
     * Constructor that reads the Purkinje VTP files and stores the coordinates
     * and the activation times of the Purkinje muscle junctions in private
     * vectors.
     *
     * @param[in] filename_ name of the Purkinje VTP file;
     * @param[in] array_name_ name of the point-data array of the activation times;
     * @param[in] mesh_scaling_factor_ scaling factor for the Purkinje
     *              network;
     * @param[in] array_scaling_factor_ scaling factor for the array
     *              storing the activation time values;
     **/
    VTKPurkinjeHandler(const std::string &filename_,
                       const std::string &array_name_,
                       const double &     mesh_scaling_factor_  = 1.0,
                       const double &     array_scaling_factor_ = 1.0);

    /// Getter of the Purkinje muscle junctions coordinates.
    const std::vector<Point<dim>> &
    get_pmj_coordinates() const
    {
      return pmj_coordinates;
    }

    /// Getter of the Purkinje muscle junctions activation times.
    const std::vector<double> &
    get_pmj_activation_times() const
    {
      return pmj_times;
    }


  private:
    /// Name of the VTP file.
    std::string filename;
    /// Name of the activation times point-data array.
    std::string array_name;
    /// The points coordinates are scaled by this factor.
    double mesh_scaling_factor;
    /// The array values are scaled by this factor.
    double array_scaling_factor;
    /// The reference VTK surface.
    vtkSmartPointer<vtkPolyData> purkinje;

    /// Coordinates of the Purkinje muscles junctions.
    std::vector<Point<dim>> pmj_coordinates;
    /// Activation times of the Purkinje muscles junctions.
    std::vector<double> pmj_times;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_VTK_PURKINJE_HANDLER_HPP_ */
