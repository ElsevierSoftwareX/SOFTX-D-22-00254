/********************************************************************************
  Copyright (C) 2019 - 2023 by the lifex authors.

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

#include "source/io/vtk_purkinje_handler.hpp"

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataReader.h>

#include <boost/filesystem.hpp>

#include <array>

namespace lifex::utils
{
  VTKPurkinjeHandler::VTKPurkinjeHandler(const std::string &filename_,
                                         const std::string &array_name_,
                                         const double &mesh_scaling_factor_,
                                         const double &array_scaling_factor_)
    : filename(filename_)
    , array_name(array_name_)
    , mesh_scaling_factor(mesh_scaling_factor_)
    , array_scaling_factor(array_scaling_factor_)
  {
    AssertThrow(boost::filesystem::exists(filename),
                ExcMessage("File " + filename + " does not exist."));

    vtkSmartPointer<vtkXMLPolyDataReader> reader =
      vtkSmartPointer<vtkXMLPolyDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();
    purkinje = reader->GetOutput();

    vtkSmartPointer<vtkDataArray> activation_times_array =
      purkinje->GetPointData()->GetArray(array_name.c_str());

    AssertThrow(activation_times_array->GetNumberOfComponents() == 1,
                ExcMessage("The point-data array with the activation times "
                           "must be a scalar array."));

    // The point is a PMJ (i.e. a terminal point) if it belongs only to a cell.
    // The for loop starts from id 1 to exclude the first point of the network
    // (i.e the origin) from PMJs.
    // TODO: find a more robust way to exclude the starting point of the
    // Purkinje network.
    for (unsigned int id = 1; id < purkinje->GetNumberOfPoints(); ++id)
      {
        // Copy point coordinates into user provided array x[3] for specified
        // point id.
        std::array<double, 3> x;
        purkinje->GetPoint(id, x.data());

        // Get all cells that vertex "id" belongs to.
        purkinje->BuildLinks();
        vtkSmartPointer<vtkIdList> n = vtkSmartPointer<vtkIdList>::New();
        purkinje->GetPointCells(id, n);

        // Find the points that only belong to one cell.
        if (n->GetNumberOfIds() == 1)
          {
            pmj_coordinates.push_back(Point<dim>(mesh_scaling_factor * x[0],
                                                 mesh_scaling_factor * x[1],
                                                 mesh_scaling_factor * x[2]));
            pmj_times.push_back(array_scaling_factor *
                                activation_times_array->GetComponent(0, id));
          }
      }
  }

} // namespace lifex::utils
