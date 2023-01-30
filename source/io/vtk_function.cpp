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
 * @author Simone Di Gregorio <simone.digregorio@polimi.it>.
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/core.hpp"

#include "source/geometry/finders.hpp"
#include "source/geometry/mesh_handler.hpp"

#include "source/io/vtk_function.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/numerics/fe_field_function.h>

#include <vtkArrayCalculator.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkGenericCell.h>
#include <vtkPointData.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkWarpVector.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <array>

namespace lifex::utils
{
  VTKFunction::VTKFunction(const std::string &filename_,
                           const VTKDataType &data_type_,
                           const double &     scaling_factor_,
                           const double &     geometry_scaling_factor_,
                           const bool &       data_is_vectorial)
    : Function<dim>(data_is_vectorial ? dim : 1)
    , filename(filename_)
    , data_type(data_type_)
    , scaling_factor(scaling_factor_)
    , geometry_scaling_factor(geometry_scaling_factor_)
    , return_infinity(false)
  {
    AssertThrow(boost::filesystem::exists(filename),
                ExcMessage("File " + filename + " does not exist."));

    read_file();
  }

  void
  VTKFunction::read_file()
  {
    if (data_type == VTKDataType::UnstructuredGrid)
      {
        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
          vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();

        if (!utils::is_equal(geometry_scaling_factor, 1.0))
          {
            vtkSmartPointer<vtkTransform> transform =
              vtkSmartPointer<vtkTransform>::New();

            transform->Scale(geometry_scaling_factor,
                             geometry_scaling_factor,
                             geometry_scaling_factor);

            vtkSmartPointer<vtkTransformFilter> transform_filter =
              vtkSmartPointer<vtkTransformFilter>::New();
            transform_filter->SetInputConnection(reader->GetOutputPort());
            transform_filter->SetTransform(transform);
            transform_filter->Update();

            vtk_mesh = transform_filter->GetUnstructuredGridOutput();
          }
        else
          {
            vtk_mesh = reader->GetOutput();
          }
      }
    else // if (data_type == VTKDataType::PolyData)
      {
        vtkSmartPointer<vtkXMLPolyDataReader> reader =
          vtkSmartPointer<vtkXMLPolyDataReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();

        if (!utils::is_equal(geometry_scaling_factor, 1.0))
          {
            vtkSmartPointer<vtkTransform> transform =
              vtkSmartPointer<vtkTransform>::New();

            transform->Scale(geometry_scaling_factor,
                             geometry_scaling_factor,
                             geometry_scaling_factor);

            vtkSmartPointer<vtkTransformFilter> transform_filter =
              vtkSmartPointer<vtkTransformFilter>::New();
            transform_filter->SetInputConnection(reader->GetOutputPort());
            transform_filter->SetTransform(transform);
            transform_filter->Update();

            vtk_surface = transform_filter->GetPolyDataOutput();
          }
        else
          {
            vtk_surface = reader->GetOutput();
          }
      }
  }

  void
  VTKFunction::setup_as_closest_point_projection(
    const std::string &     vtk_arrayname_,
    const VTKArrayDataType &array_data_type_)
  {
    mode = Mode::ClosestPointProjection;

    vtk_arrayname   = vtk_arrayname_;
    array_data_type = array_data_type_;

    counter_value_calls = 0;

    if (data_type == VTKDataType::UnstructuredGrid)
      {
        if (array_data_type == VTKArrayDataType::CellData)
          {
            vtk_cell_locator = vtkSmartPointer<vtkCellLocator>::New();
            vtk_cell_locator->SetDataSet(vtk_mesh);
            vtk_cell_locator->BuildLocator();

            vtk_array =
              vtk_mesh->GetCellData()->GetArray(vtk_arrayname.c_str());

            AssertThrow(vtk_array,
                        ExcMessage("No cell data array \"" + vtk_arrayname +
                                   "\" found in file \"" + filename + "\"."));
          }
        else // PointData
          {
            vtk_point_locator = vtkSmartPointer<vtkPointLocator>::New();
            vtk_point_locator->SetDataSet(vtk_mesh);
            vtk_point_locator->BuildLocator();

            vtk_array =
              vtk_mesh->GetPointData()->GetArray(vtk_arrayname.c_str());

            AssertThrow(vtk_array,
                        ExcMessage("No point data array \"" + vtk_arrayname +
                                   "\" found in file \"" + filename + "\"."));
          }
      }
    else // if (data_type == VTKDataType::PolyData)
      {
        if (array_data_type == VTKArrayDataType::CellData)
          {
            vtk_cell_locator = vtkSmartPointer<vtkCellLocator>::New();
            vtk_cell_locator->SetDataSet(vtk_surface);
            vtk_cell_locator->BuildLocator();

            vtk_array =
              vtk_surface->GetCellData()->GetArray(vtk_arrayname.c_str());

            AssertThrow(vtk_array,
                        ExcMessage("No cell data array \"" + vtk_arrayname +
                                   "\" found in file \"" + filename + "\"."));
          }
        else
          {
            vtk_point_locator = vtkSmartPointer<vtkPointLocator>::New();
            vtk_point_locator->SetDataSet(vtk_surface);
            vtk_point_locator->BuildLocator();

            vtk_array =
              vtk_surface->GetPointData()->GetArray(vtk_arrayname.c_str());

            AssertThrow(vtk_array,
                        ExcMessage("No point data array \"" + vtk_arrayname +
                                   "\" found in file \"" + filename + "\"."));
          }
      }
  }

  void
  VTKFunction::setup_as_linear_projection(const std::string &vtk_arrayname_)
  {
    mode = Mode::LinearProjection;

    vtk_arrayname   = vtk_arrayname_;
    array_data_type = VTKArrayDataType::PointData;

    counter_value_calls        = 0;
    counter_lin_int_correction = 0;

    vtk_probe_filter  = vtkSmartPointer<vtkProbeFilter>::New();
    vtk_cell_locator  = vtkSmartPointer<vtkCellLocator>::New();
    vtk_point_locator = vtkSmartPointer<vtkPointLocator>::New();

    if (data_type == VTKDataType::UnstructuredGrid)
      {
        vtk_probe_filter->SetSourceData(vtk_mesh);
        vtk_cell_locator->SetDataSet(vtk_mesh);
        vtk_point_locator->SetDataSet(vtk_mesh);
        vtk_array = vtk_mesh->GetPointData()->GetArray(vtk_arrayname.c_str());

        AssertThrow(vtk_array,
                    ExcMessage("No point data array \"" + vtk_arrayname +
                               "\" found in file \"" + filename + "\"."));
      }
    else // if (data_type == VTKDataType::PolyData)
      {
        vtk_probe_filter->SetSourceData(vtk_surface);
        vtk_cell_locator->SetDataSet(vtk_surface);
        vtk_point_locator->SetDataSet(vtk_surface);
        vtk_array =
          vtk_surface->GetPointData()->GetArray(vtk_arrayname.c_str());

        AssertThrow(vtk_array,
                    ExcMessage("No point data array \"" + vtk_arrayname +
                               "\" found in file \"" + filename + "\"."));
      }

    vtk_probe_filter->PassCellArraysOff();
    vtk_probe_filter->PassPointArraysOn();
    vtk_probe_filter->PassFieldArraysOff();

    vtk_cell_locator->BuildLocator();
    vtk_point_locator->BuildLocator();
  }

  void
  VTKFunction::setup_as_signed_distance(
    const std::optional<double> &cutoff_distance_,
    const double &               cutoff_value_)
  {
    AssertThrow(data_type != VTKDataType::UnstructuredGrid,
                ExcLifexInternalError());
    AssertThrow(this->Function<dim>::n_components == 1,
                ExcLifexInternalError());

    mode = Mode::SignedDistance;

    counter_value_calls = 0;

    vtk_signed_distance = vtkSmartPointer<vtkImplicitPolyDataDistance>::New();
    vtk_signed_distance->SetInput(vtk_surface);

    cutoff_distance = cutoff_distance_;
    cutoff_value    = cutoff_value_;

    if (cutoff_distance.has_value())
      compute_bounding_box();
  }

  double
  VTKFunction::value(const Point<dim> &p, const unsigned int component) const
  {
    ++counter_value_calls;

    if (return_infinity)
      return std::numeric_limits<double>::infinity();

    double value = 0;

    if (mode == Mode::ClosestPointProjection)
      {
        std::array<double, dim> test_point;
        for (unsigned int d = 0; d < dim; ++d)
          test_point[d] = p[d];

        // CellData: ID of the cell containing the closest point.
        // PointData: ID of the closest point.
        vtkIdType id;

        if (array_data_type == VTKArrayDataType::CellData)
          {
            // The coordinates of the closest point.
            std::array<double, dim> closest_point;
            // The squared distance to the closest point.
            double closest_point_dist_squared;
            // This is rarely used (in triangle strips only?)
            int sub_id;

            vtk_cell_locator->FindClosestPoint(test_point.data(),
                                               closest_point.data(),
                                               id,
                                               sub_id,
                                               closest_point_dist_squared);
          }
        else // if (array_data_type == VTKArrayDataType::PointData)
          {
            id = vtk_point_locator->FindClosestPoint(test_point.data());
          }

        value = scaling_factor * vtk_array->GetComponent(id, component);
      }
    else if (mode == Mode::LinearProjection)
      {
        vtkSmartPointer<vtkPolyData> tmp_poly_data;
        vtkSmartPointer<vtkPoints>   tmp_points;

        tmp_poly_data = vtkSmartPointer<vtkPolyData>::New();
        tmp_points    = vtkSmartPointer<vtkPoints>::New();

        std::array<double, dim> test_point;
        for (unsigned int d = 0; d < dim; ++d)
          test_point[d] = p[d];

        tmp_points->InsertPoint(0, test_point.data());
        tmp_poly_data->SetPoints(tmp_points);

        vtk_probe_filter->SetInputData(tmp_poly_data);
        vtk_probe_filter->Update(); // alternative: DoProbing? Probe?

        value = scaling_factor * vtk_probe_filter->GetOutput()
                                   ->GetPointData()
                                   ->GetArray(vtk_arrayname.c_str())
                                   ->GetComponent(0, component);

        // If default locator fails probing, do the interpolation "by hands"
        // using cell locator and cell interpolation.
        // TODO: vtkCellLocator will be settable as locator of the probe
        //       filter in VTK 9.0 (on developing).
        //       This code will be updated consequently.
        if (vtk_probe_filter->GetValidPoints()->GetNumberOfTuples() != 1)
          {
            ++counter_lin_int_correction;

            vtkIdType               id;
            std::array<double, dim> closest_point;
            double                  dist_squared;
            int                     sub_id;
            std::array<double, dim> p_coords;

            vtkSmartPointer<vtkGenericCell> generic_cell;
            generic_cell = vtkSmartPointer<vtkGenericCell>::New();

            vtkSmartPointer<vtkPointData> ref_point_data =
              (data_type == VTKDataType::UnstructuredGrid) ?
                vtk_mesh->GetPointData() :
                vtk_surface->GetPointData();

            vtkSmartPointer<vtkPointData> curr_point_data =
              tmp_poly_data->GetPointData();

            unsigned int n_points = tmp_poly_data->GetNumberOfPoints();

            curr_point_data->InterpolateAllocate(ref_point_data, n_points);

            // vtk_mesh or vtk_surface BuildCells() ?

            vtk_cell_locator->FindClosestPoint(test_point.data(),
                                               closest_point.data(),
                                               generic_cell,
                                               id,
                                               sub_id,
                                               dist_squared);

            std::vector<double> weights(generic_cell->GetNumberOfPoints());

            generic_cell->EvaluatePosition(closest_point.data(),
                                           NULL,
                                           sub_id,
                                           p_coords.data(),
                                           dist_squared,
                                           weights.data());

            curr_point_data->InterpolatePoint(ref_point_data,
                                              0,
                                              generic_cell->GetPointIds(),
                                              weights.data());

            value =
              scaling_factor * curr_point_data->GetArray(vtk_arrayname.c_str())
                                 ->GetComponent(0, component);
          }
      }
    else // if (mode == Mode::SignedDistance)
      {
        // If we set a cutoff threshold, we first check if we're inside the
        // bounding box: if we're not, we just return the cutoff threshold.
        if (cutoff_distance.has_value())
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                if (p[d] < bounding_box[d].first ||
                    p[d] > bounding_box[d].second)
                  {
                    return cutoff_value;
                  }
              }
          }

        // Note: a distance is always a scalar.
        value =
          scaling_factor * vtk_signed_distance->FunctionValue(p[0], p[1], p[2]);
      }

    return value;
  }

  void
  VTKFunction::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    ++counter_value_calls;

    if (return_infinity)
      {
        values = std::numeric_limits<double>::infinity();
        return;
      }

    if (mode == Mode::ClosestPointProjection)
      {
        std::array<double, dim> test_point;
        for (unsigned int d = 0; d < dim; ++d)
          test_point[d] = p[d];

        // CellData: ID of the cell containing the closest point.
        // PointData: ID of the closest point.
        vtkIdType id;

        if (array_data_type == VTKArrayDataType::CellData)
          {
            // The coordinates of the closest point.
            std::array<double, dim> closest_point;
            // The squared distance to the closest point.
            double closest_point_dist_squared;
            // This is rarely used (in triangle strips only?)
            int sub_id;

            vtk_cell_locator->FindClosestPoint(test_point.data(),
                                               closest_point.data(),
                                               id,
                                               sub_id,
                                               closest_point_dist_squared);
          }
        else // if (array_data_type == VTKArrayDataType::PointData)
          {
            id = vtk_point_locator->FindClosestPoint(test_point.data());
          }

        for (unsigned int component = 0; component < values.size(); ++component)
          {
            values[component] =
              scaling_factor * vtk_array->GetComponent(id, component);
          }
      }
    else if (mode == Mode::LinearProjection)
      {
        vtkSmartPointer<vtkPolyData> tmp_poly_data;
        vtkSmartPointer<vtkPoints>   tmp_points;

        tmp_poly_data = vtkSmartPointer<vtkPolyData>::New();
        tmp_points    = vtkSmartPointer<vtkPoints>::New();

        std::array<double, dim> test_point;
        for (unsigned int d = 0; d < dim; ++d)
          test_point[d] = p[d];

        tmp_points->InsertPoint(0, test_point.data());
        tmp_poly_data->SetPoints(tmp_points);

        vtk_probe_filter->SetInputData(tmp_poly_data);
        vtk_probe_filter->Update(); // alternative: DoProbing? Probe?

        for (unsigned int component = 0; component < values.size(); ++component)
          {
            values[component] =
              scaling_factor * vtk_probe_filter->GetOutput()
                                 ->GetPointData()
                                 ->GetArray(vtk_arrayname.c_str())
                                 ->GetComponent(0, component);
          }

        // If default locator fails probing, do the interpolation "by hands"
        // using cell locator and cell interpolation.
        // TODO: vtkCellLocator will be settable as locator of the probe
        //       filter in VTK 9.0 (on developing).
        //       This code will be updated consequently.
        if (vtk_probe_filter->GetValidPoints()->GetNumberOfTuples() != 1)
          {
            ++counter_lin_int_correction;

            vtkIdType               id;
            std::array<double, dim> closest_point;
            double                  dist_squared;
            int                     sub_id;
            std::array<double, dim> p_coords;

            vtkSmartPointer<vtkGenericCell> generic_cell;
            generic_cell = vtkSmartPointer<vtkGenericCell>::New();

            vtkSmartPointer<vtkPointData> ref_point_data =
              (data_type == VTKDataType::UnstructuredGrid) ?
                vtk_mesh->GetPointData() :
                vtk_surface->GetPointData();

            vtkSmartPointer<vtkPointData> curr_point_data =
              tmp_poly_data->GetPointData();

            unsigned int n_points = tmp_poly_data->GetNumberOfPoints();

            curr_point_data->InterpolateAllocate(ref_point_data, n_points);

            // vtk_mesh or vtk_surface BuildCells() ?

            vtk_cell_locator->FindClosestPoint(test_point.data(),
                                               closest_point.data(),
                                               generic_cell,
                                               id,
                                               sub_id,
                                               dist_squared);

            std::vector<double> weights(generic_cell->GetNumberOfPoints());

            generic_cell->EvaluatePosition(closest_point.data(),
                                           NULL,
                                           sub_id,
                                           p_coords.data(),
                                           dist_squared,
                                           weights.data());

            curr_point_data->InterpolatePoint(ref_point_data,
                                              0,
                                              generic_cell->GetPointIds(),
                                              weights.data());

            for (unsigned int component = 0; component < values.size();
                 ++component)
              {
                values[component] =
                  scaling_factor *
                  curr_point_data->GetArray(vtk_arrayname.c_str())
                    ->GetComponent(0, component);
              }
          }
      }
    else // if (mode == Mode::SignedDistance)
      {
        values = scaling_factor * value(p);
      }
  }

  void
  VTKFunction::warp_by_array_combination(
    const std::map<const std::string, const double> &arraynames_and_scalings)
  {
    AssertThrow(data_type == VTKDataType::PolyData, ExcLifexNotImplemented());

    // Create calculator to compute the warping field.
    std::string warping_function =
      "0 * " + arraynames_and_scalings.begin()
                 ->first; // Initialization to ensure well definition.
    vtkSmartPointer<vtkArrayCalculator> arrayCalculator =
      vtkSmartPointer<vtkArrayCalculator>::New();
    arrayCalculator->SetInputData(vtk_surface);

    for (const auto &name_scal : arraynames_and_scalings)
      {
        vtkSmartPointer<vtkDataArray> array =
          vtk_surface->GetPointData()->GetArray(name_scal.first.c_str());

        AssertThrow(array,
                    ExcMessage("No \"" + name_scal.first +
                               "\" PointData array found."));

        // Check if dimension == 3.
        // Required by VTK, irrespectively of @ref dim.
        AssertThrow(array->GetNumberOfComponents() == 3,
                    ExcMessage(
                      "This operation works only for arrays of 3 components."));

        arrayCalculator->AddVectorArrayName(name_scal.first.c_str());
        warping_function +=
          " + " + name_scal.first + " * " + std::to_string(name_scal.second);
      }

    arrayCalculator->SetFunction(warping_function.c_str());
    std::string actual_warping_field_name = "warping_field";
    arrayCalculator->SetResultArrayName(actual_warping_field_name.c_str());
    arrayCalculator->Update();
    vtkSmartPointer<vtkDataArray> actual_warping_field =
      arrayCalculator->GetPolyDataOutput()->GetPointData()->GetArray(
        actual_warping_field_name.c_str());
    vtk_surface->GetPointData()->AddArray(actual_warping_field);

    // Set warping_field as active: required by vtkWarpVector
    vtk_surface->GetPointData()->SetActiveVectors(
      actual_warping_field_name.c_str());

    // Apply warping.
    vtkSmartPointer<vtkWarpVector> warpVector =
      vtkSmartPointer<vtkWarpVector>::New();
    warpVector->SetInputData(vtk_surface);
    warpVector->Update();
    vtk_surface = warpVector->GetPolyDataOutput();

    if (cutoff_distance.has_value())
      compute_bounding_box();
  }

  void
  VTKFunction::compute_bounding_box()
  {
    const unsigned int n_points = vtk_surface->GetNumberOfPoints();

    for (unsigned int d = 0; d < dim; ++d)
      {
        bounding_box[d].first  = std::numeric_limits<double>::max();
        bounding_box[d].second = std::numeric_limits<double>::lowest();
      }

    double tmp_point_coords[3];

    for (unsigned int i = 0; i < n_points; ++i)
      {
        vtk_surface->GetPoint(i, tmp_point_coords);

        for (unsigned int d = 0; d < dim; ++d)
          {
            bounding_box[d].first =
              std::min(bounding_box[d].first, tmp_point_coords[d]);
            bounding_box[d].second =
              std::max(bounding_box[d].second, tmp_point_coords[d]);
          }
      }

    if (cutoff_distance.has_value())
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            bounding_box[d].first -= cutoff_distance.value();
            bounding_box[d].second += cutoff_distance.value();
          }
      }
  }

  std::vector<std::vector<std::pair<types::global_dof_index, double>>>
  VTKFunction::find_closest_owned_dofs(
    const DoFHandler<dim> &dof_handler,
    const std::vector<std::map<types::global_dof_index, Point<dim>>>
      &support_points) const
  {
    const unsigned int n_points = vtk_surface->GetNumberOfPoints();

    std::vector<Point<dim>> points(n_points);
    std::array<double, 3>   tmp_point;

    for (unsigned int i = 0; i < n_points; ++i)
      {
        vtk_surface->GetPoint(i, tmp_point.data());

        for (unsigned int d = 0; d < dim; ++d)
          points[i][d] = tmp_point[d];
      }

    return utils::find_closest_owned_dofs(dof_handler, support_points, points);
  }

  void
  VTKFunction::warp_by_pointwise_vectors(
    const std::vector<Vector<double>> &input_vector,
    const double &                     scaling_factor)
  {
    const unsigned int n_points = vtk_surface->GetNumberOfPoints();

    Assert(input_vector.size() == n_points,
           ExcDimensionMismatch(input_vector.size(), n_points));

    vtkNew<vtkDoubleArray> warp_data;
    warp_data->SetNumberOfComponents(dim);
    warp_data->SetName("warp_data");

    for (unsigned int i = 0; i < n_points; ++i)
      {
        if (dim == 2)
          warp_data->InsertNextTuple2(scaling_factor * input_vector[i][0],
                                      scaling_factor * input_vector[i][1]);
        else if (dim == 3)
          warp_data->InsertNextTuple3(scaling_factor * input_vector[i][0],
                                      scaling_factor * input_vector[i][1],
                                      scaling_factor * input_vector[i][2]);
        else
          Assert(false, ExcLifexNotImplemented());
      }

    // Apply warp.
    vtk_surface->GetPointData()->AddArray(warp_data);
    vtk_surface->GetPointData()->SetActiveVectors(warp_data->GetName());

    vtkNew<vtkWarpVector> warp_vector;
    warp_vector->SetInputData(vtk_surface);
    warp_vector->Update();

    vtk_surface = warp_vector->GetPolyDataOutput();
  }

  std::vector<Vector<double>>
  VTKFunction::extract_nearest_neighbor_values(
    const std::vector<std::vector<std::pair<types::global_dof_index, double>>>
      &                        closest_dofs,
    const LinAlg::MPI::Vector &input)
  {
    const unsigned int          n_points     = closest_dofs.size();
    const unsigned int          n_components = closest_dofs[0].size();
    std::vector<Vector<double>> result(n_points, Vector<double>(n_components));

    for (unsigned int i = 0; i < n_points; ++i)
      {
        // Find the process owning the closest DoF. We assume that if one
        // process owns one component (the first), then it owns all of them.
        const unsigned int closest_dof_owner =
          utils::MPI::minloc(closest_dofs[i][0].second, Core::mpi_comm).rank;

        // If this process owns the closest DoF, extract the corresponding
        // warp vector.
        if (Core::mpi_rank == closest_dof_owner)
          {
            for (unsigned int d = 0; d < n_components; ++d)
              result[i][d] = input[closest_dofs[i][d].first];
          }

        // Broadcast to all processes.
        result[i] = Utilities::MPI::broadcast(Core::mpi_comm,
                                              result[i],
                                              closest_dof_owner);
      }

    return result;
  }

  InterpolatedSignedDistance::InterpolatedSignedDistance(
    const std::string &                  filename,
    const DoFHandler<dim> &              dof_handler_,
    const std::shared_ptr<Mapping<dim>> &mapping,
    const std::optional<double> &        cutoff_distance,
    const double &                       cutoff_value)
    : VTKFunction(filename, utils::VTKDataType::PolyData)
    , dof_handler(dof_handler_)
    , mapping(mapping)
    , needs_interpolation(true)
  {
    setup_as_signed_distance(cutoff_distance, cutoff_value);
  }

  InterpolatedSignedDistance::InterpolatedSignedDistance(
    const InterpolatedSignedDistance &other)
    : VTKFunction(other.filename, utils::VTKDataType::PolyData)
    , dof_handler(other.dof_handler)
    , interpolated_owned(other.interpolated_owned)
    , interpolated(other.interpolated)
  {}

  InterpolatedSignedDistance &
  InterpolatedSignedDistance::operator=(const InterpolatedSignedDistance &other)
  {
    Assert(&dof_handler == &other.dof_handler,
           ExcMessage("Instances of InterpolatedSignedDistance can be copied "
                      "only if they share the same DoF handler."));

    filename           = other.filename;
    interpolated_owned = other.interpolated_owned;
    interpolated       = other.interpolated;

    return *this;
  }

  void
  InterpolatedSignedDistance::update()
  {
    setup_as_signed_distance(cutoff_distance, cutoff_value);
  }

  void
  InterpolatedSignedDistance::reset()
  {
    // Read again the surface from file.
    read_file();
    update();

    needs_interpolation = true;
  }

  void
  InterpolatedSignedDistance::interpolate() const
  {
    if (needs_interpolation)
      {
        // Setup interpolation vectors and apply the interpolation.
        IndexSet owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

        interpolated_owned.reinit(owned_dofs, Core::mpi_comm);
        interpolated.reinit(owned_dofs, relevant_dofs, Core::mpi_comm);

        if (mapping)
          VectorTools::interpolate(*mapping,
                                   dof_handler,
                                   *this,
                                   interpolated_owned);
        else
          VectorTools::interpolate(dof_handler, *this, interpolated_owned);

        interpolated        = interpolated_owned;
        needs_interpolation = false;
      }
  }

  void
  InterpolatedSignedDistance::compress(const VectorOperation::values &operation)
  {
    interpolated_owned.compress(operation);
    interpolated = interpolated_owned;
  }
} // namespace lifex::utils
