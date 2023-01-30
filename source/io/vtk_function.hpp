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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#ifndef LIFEX_UTILS_VTK_FUNCTION_HPP_
#define LIFEX_UTILS_VTK_FUNCTION_HPP_

#include "source/lifex.hpp"


#define HZ_OLD \
  HZ ///< Prevent compiler error when including vtkPointLocator.h and @dealii
     ///< (PETSc).
#undef HZ ///< Temporarily overwritten by VTK. Re-defined later.

#include <vtkCellLocator.h>
#include <vtkImplicitPolyDataDistance.h>
#include <vtkPointLocator.h>
#include <vtkPolyData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /// Enumeration for the usable array data type of VTK.
  enum class VTKArrayDataType
  {
    PointData, ///< an array defined on the points of the mesh/surface.
    CellData   ///< an array defined on the cells of the mesh/surface.
  };

  /// Enumeration for the usable data structure of VTK.
  enum class VTKDataType
  {
    PolyData,        ///< Polygonal surface.
    UnstructuredGrid ///< Unstructured mesh.
  };

  /**
   * @brief Function that gives as output a result depending on a VTK mesh or
   * surface exploiting some VTK functionalities.
   *
   * This function inherits from standard @dealii Function overriding the method
   * value, that returns a scalar value in function of the coordinates of an
   * input point. The overridden method exploits some VTK functionalities. In
   * particular, the class owns a reference VTK data read from a filename during
   * construction, that can be a vtkUnstructuredGrid - @a i.e. an unstructured
   * mesh - or a vtkPolyData - @a i.e. a polygonal surface. The output of the
   * function depends on a flag selectable among those
   * available in the enumeration @ref Mode.
   * In detail, the following three modes are available:
   *     - @ref Mode::ClosestPointProjection finds the closest point or cell on
   *       the reference mesh or surface from the input point and returns the
   *       value assumed on this point or cell by an input array previously
   *       initialized by the user
   *       (see method @ref setup_as_closest_point_projection);
   *     - @ref Mode::LinearProjection performs a linear interpolation of the
   *       point data array of the reference mesh or surface on the input point
   *       (see method @ref setup_as_linear_projection);
   *     - @ref Mode::SignedDistance returns the signed distance from the input
   *       point to the reference surface
   *       (see method @ref setup_as_signed_distance).
   *
   * The class is constructed passing the filename and the @ref VTKDataType of the
   * reference surface or mesh, that can not be modified after construction.
   * Then, user has to select the mode calling the corresponding setup method
   * that prepares the private VTK objects necessary for running @ref value or
   * @ref vector_value.
   *
   * Here is an example of usage:
   * @code{.cpp}
   * // Define a deal.II point.
   * Point p(0.0, 0.0, 0.0);
   *
   * // Construct a VTKFunction with a reference mesh.
   * VTKFunction vtk_function("mesh.vtu", VTKDataType::UnstructuredGrid);
   *
   * // Setup the function with ClosestPointProjection mode.
   * vtk_function.setup_as_closest_point_projection("u",
   *                                                VTKArrayDataType::CellData);
   *
   * // Use the value method.
   * double val = vtk_function.value(p);
   * @endcode
   *
   * Note that this class has the flexibility to work with a reference surface
   * or a reference mesh and with a point data array or a cell data array, but
   * this flexibility is not guarantee for all the modes. Thus, some modes can
   * generate runtime errors if used with incompatible
   * @ref VTKDataType or @ref VTKArrayDataType.
   *
   * ### Warping according to a finite element field.
   * This class exposes methods allowing to warp the underlying surface
   * following a finite element field. To do so, the vector-valued function
   * represented by a finite element vector is interpolated onto the nodes of
   * the surface (using nearest neighbor interpolation).
   *
   * The procedure to do so is as follows:
   * @code{.cpp}
   * std::unique_ptr<VTKFunction> surface;
   *
   * // Setup surface...
   *
   * // Generate the maps of support points for the DoF handler used to
   * // represent the warping field, for each component.
   * std::vector<std::map<types::global_dof_index, Point<dim>>>
   *   support_points(dim);
   *
   * for (unsigned int d = 0; d < dim; ++d)
   *   {
   *     ComponentMask mask(dim, false);
   *     mask.set(d, true);
   *
   *     DoFTools::map_dofs_to_support_points(*mapping,
   *                                          dof_handler,
   *                                          support_points[d],
   *                                          mask);
   *  }
   *
   * // Generate the nearest-neighbor interpolation pattern by finding the
   * // closest (owned) DoFs to each of the surface points.
   * surface_closest_dofs =
   *   surface->find_closest_owned_dofs(dof_handler, support_points);
   *
   * // Retrieve values of the warp vector on the nearest-neighbor DoFs.
   * std::vector<Vector<double>> warp_values =
   *   utils::VTKFunction::extract_nearest_neighbor_values(
   *     surface_closest_dofs, warp_vector);
   *
   * // Actually apply the warp vector.
   * surface->warp_by_pointwise_vectors(warp_values, 1.0);
   * @endcode
   * This multi-step procedure may appear cumbersome, but it allows to compute
   * everything only when needed, for efficiency. For example, one might compute
   * the support points only once at simulation start (as opposed to recomputing
   * them whenever the surface needs to be warped).
   *
   * For a basic example, see @ref examples::ExampleVTKWarpByFE.
   */
  class VTKFunction : public Function<dim>
  {
  public:
    /**
     * Constructor.
     *
     * @param[in] filename_ the name of the mesh or surface file,
     * @param[in] data_type_ the VTKDataType of the input file.
     * @param[in] scaling_factor_ file data is scaled by this factor.
     * @param[in] data_is_vectorial flag to select if the input function is vectorial or scalar.
     * @param[in] geometry_scaling_factor_ file geometry is scaled by this factor.
     */
    VTKFunction(const std::string &filename_,
                const VTKDataType &data_type_ = VTKDataType::UnstructuredGrid,
                const double &     scaling_factor_          = 1.0,
                const double &     geometry_scaling_factor_ = 1.0,
                const bool &       data_is_vectorial        = false);

    /// Enumeration with the possible modes.
    enum class Mode
    {
      ClosestPointProjection, ///< Closest point projection mode.
      LinearProjection,       ///< Linear projection mode.
      SignedDistance          ///< Signed distance mode.
    };

    /**
     * Method to setup the function as closest point projection.
     *
     * Set the array to be processed reading a PointData or CellData array.
     * @param[in] vtk_arrayname_ the name of the new array to process.
     * @param[in] array_data_type_ the VTKArrayDataType of the array to process.
     */
    void
    setup_as_closest_point_projection(
      const std::string &     vtk_arrayname_,
      const VTKArrayDataType &array_data_type_ = VTKArrayDataType::CellData);

    /**
     * Method to setup the function as linear interpolation.
     *
     * Set the array to be processed reading an array from the reference surface
     * or mesh.
     * Note that this method is conceived to work only with Point Data array.
     * @param[in] vtk_arrayname_ the name of the new array to process.
     */
    void
    setup_as_linear_projection(const std::string &vtk_arrayname_);

    /**
     * Method to setup the function as signed distance.
     *
     * This mode works only with a reference PolyData surface and can not work
     * with a reference UnstructuredGrid.
     * There is no need of an array to process.
     *
     * ### Cutoff distance
     * Accurate evaluations of the signed distance are often needed only close
     * to the surface, whereas, far from the surface, it often suffices to know
     * that we are far away. Therefore, interpolating the signed distance over
     * the whole domain (see @ref InterpolatedSignedDistance) might perform
     * a large number of unneeded evaluations to compute the signed distance far
     * from the surface. These evaluations have a high computational cost.
     *
     * To reduce the overall cost, a cutoff distance can be specified as an
     * optional argument to this method. In that case, evaluation of the signed
     * distance has the following steps:
     * 1. when calling setup_as_signed_distance or warp_by_array_combination,
     * the axis-aligned bounding box of the surface is computed;
     * 2. the bounding box is enlarged by the cutoff distance along all
     * directions;
     * 3. when calling value or vector_value, we first check if the evaluation
     * point lies within the bounding box; if it does, we evaluate the signed
     * distance; if it does not, we return the cutoff distance.
     *
     * Therefore, the cutoff distance has the meaning of a distance beyond which
     * we won't be needing the signed distance function. Since usually most of
     * the points will lie outside the bounding box, this will likely lead to a
     * large computational saving (since the cost of checking if a point is
     * inside a box is significantly smaller than that of evaluating the signed
     * distance).
     */
    void
    setup_as_signed_distance(
      const std::optional<double> &cutoff_distance_ = {},
      const double &cutoff_value_ = std::numeric_limits<double>::infinity());

    /**
     * Overridden method that works differently depending on the flag @ref mode.
     *
     * This method is invoked if the input data is scalar.
     * Otherwise, see @ref vector_value.
     *
     * @param[in] p         Point at which the function is evaluated.
     * @param[in] component Component to evaluate.
     */
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    /**
     * Overridden method that works differently depending on the flag @ref mode.
     *
     * This method is invoked if the input data is vectorial.
     * Otherwise, see @ref value.
     *
     * @param[in]  p      Point at which the function is evaluated.
     * @param[out] values Data values, depending on the flag @ref mode,
     *                    multiplied by @ref scaling_factor.
     */
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override;

    /**
     * Move points of the reference surface according to the input array
     * @ref vtk_arrayname times @ref scaling_factor.
     *
     * The input array is the one passed to
     * @ref setup_as_closest_point_projection()
     * or @ref setup_as_linear_projection(), and used by @ref vector_value().
     * This method does not work for the mode @ref Mode::SignedDistance.
     *
     * This method works only with a reference PolyData surface.
     * Note that this method is conceived to work only with Point Data array.
     *
     * The array this class refers to must be a 3D vectorial field
     * (cf. documentation of vktWarpVector).
     */
    void
    warp_by_input_array()
    {
      warp_by_array_combination({{vtk_arrayname, scaling_factor}});
    }

    /**
     * Move points of the reference surface according to a linear combination
     * of given arrays.
     *
     * This method works only with a reference PolyData surface.
     * Note that this method is conceived to work only with Point Data array.
     *
     * The surface is warped by a field of the form
     * <tt>array1 * scaling1 + array2 * scaling2 + ...</tt>
     *
     * @param[in] arraynames_and_scalings Map with names of the arrays and
     * corresponding scaling factors.
     */
    void
    warp_by_array_combination(
      const std::map<const std::string, const double> &arraynames_and_scalings);

    /**
     * @brief Warp the surface by a vector defined for each of its points.
     *
     * @param[in] input_vector A vector with one element for each point of the
     * surface. Each element indicates the warp vector to be applied to the
     * corresponding point.
     * @param[in] scaling_factor A scaling factor applied to the warp.
     */
    void
    warp_by_pointwise_vectors(const std::vector<Vector<double>> &input_vector,
                              const double &scaling_factor = 1.0);

    /**
     * @brief Find nearest owned DoFs to surface points.
     *
     * @param[in] dof_handler The DoF handler.
     * @param[in] support_points A vector that for each component stores the
     * support points of the associated DoFs. Should be computed by means of the
     * function DoFTools::map_dofs_to_support_points.
     */
    std::vector<std::vector<std::pair<types::global_dof_index, double>>>
    find_closest_owned_dofs(
      const DoFHandler<dim> &dof_handler,
      const std::vector<std::map<types::global_dof_index, Point<dim>>>
        &support_points) const;

    /**
     * @brief Extract values of nearest neighbor DoFs to the surface from a FE
     * vector.
     *
     * @param[in] closest_dofs The neirest-neighbor pattern to use. It should be
     * constructed using @ref find_closest_owned_dofs for a given surface.
     * @param[in] input The input finite element vector.
     *
     * @return A vector with one entry for each entry of closest_dofs. Each
     * entry is a Vector of doubles with as many components as those described
     * by closest_dofs, containing the corresponding values of the input vector.
     */
    static std::vector<Vector<double>>
    extract_nearest_neighbor_values(
      const std::vector<std::vector<std::pair<types::global_dof_index, double>>>
        &                        closest_dofs,
      const LinAlg::MPI::Vector &input);

    /// Make the function return
    /// <kbd>std::numeric_limits<double>::infinity()</kbd> at all points.
    ///
    /// @param[in] return_infinity_ Toggles returning infinity.
    void
    set_to_infinity(const bool &return_infinity_)
    {
      return_infinity = return_infinity_;
    }

    /// Getter of @ref counter_value_calls.
    const unsigned int &
    get_counter_value_calls() const
    {
      return counter_value_calls;
    }

    /// Getter of @ref counter_lin_int_correction.
    const unsigned int &
    get_counter_lin_int_correction() const
    {
      return counter_lin_int_correction;
    }

    /// Getter of @ref vtk_surface.
    const vtkSmartPointer<vtkPolyData> &
    get_vtk_surface() const
    {
      return vtk_surface;
    }

  protected:
    /// Read the input file.
    void
    read_file();

    /// Name of the file containing the input VTK mesh or surface.
    std::string filename;
    /// Name of the array to process defined on the input mesh or surface.
    std::string vtk_arrayname;

    /// The reference VTK mesh.
    vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh;
    /// The reference VTK surface.
    vtkSmartPointer<vtkPolyData> vtk_surface;
    /// The array to process.
    vtkSmartPointer<vtkDataArray> vtk_array;
    /// Locator used to find the closest point on the reference mesh/surface.
    vtkSmartPointer<vtkPointLocator> vtk_point_locator;
    /// Locator used to find the closest cell on the reference mesh/surface.
    vtkSmartPointer<vtkCellLocator> vtk_cell_locator;
    /// Filter used to evaluate an array of an input data in a point.
    vtkSmartPointer<vtkProbeFilter> vtk_probe_filter;
    /// The VTK object that compute the signed distance from a PolyData.
    vtkSmartPointer<vtkImplicitPolyDataDistance> vtk_signed_distance;

    Mode              mode; ///< Flag to choose the mode of the VTKFunction.
    const VTKDataType data_type; ///< VTK data type of the reference VTK file.
    VTKArrayDataType  array_data_type; ///< Type of the array to process.

    double scaling_factor;          ///< File data is scaled by this factor.
    double geometry_scaling_factor; ///< File geometry is scaled by this factor.

    /// Toggle returning infinity at all points, instead of actual value.
    ///
    /// Used for RIIS on-off behavior.
    bool return_infinity;

    /// @name Cutoff of signed distance.
    /// @{

    /// Cutoff threshold for signed distance.
    std::optional<double> cutoff_distance = {};

    /// Cutoff value (i.e. signed distance value beyond the cutoff distance).
    double cutoff_value;

    /// Bounding box. For each dimension, stores lower and upper bounds in a
    /// pair. Updated by calling compute_bounding_box.
    std::array<std::pair<double, double>, dim> bounding_box;

    /// Compute the bounding box for the surface.
    void
    compute_bounding_box();

    /// @}

    /// Number of calls to @ref vector_value, reset at each setup call.
    mutable unsigned int counter_value_calls;
    /// Number of linear interpolation correction, reset at each
    /// setup call.
    mutable unsigned int counter_lin_int_correction;
  };


  /**
   * @brief Interpolated signed distance function.
   *
   * This class inherits from VTK function and stores interpolated signed
   * distance vectors, handling the interpolation operations.
   *
   * For maximum efficiency, interpolation is done only if surface position has
   * changed since last call.
   *
   * If needed, interpolation is done automatically when accessing the
   * interpolated vectors (through get_owned, get_ghosted or throught the []
   * operator), so that in most cases users do not need to call the interpolate
   * method explicitly. A relevant exception to this is when said methods are
   * called only by some processes in the parallel pool: since interpolation is
   * a parallel operation, this may lead to deadlocks. In those cases, the user
   * should call interpolate beforehand on all processes. Notice that if
   * interpolation is not necessary the call is completely inexpensive.
   */
  class InterpolatedSignedDistance : public VTKFunction
  {
  public:
    /// Load the VTK file and compute the interpolation.
    InterpolatedSignedDistance(
      const std::string &                  filename,
      const DoFHandler<dim> &              dof_handler_,
      const std::shared_ptr<Mapping<dim>> &mapping         = nullptr,
      const std::optional<double> &        cutoff_distance = {},
      const double &cutoff_value = std::numeric_limits<double>::infinity());

    /// Copy-constructor. The VTK function is left to nullptr, while the
    /// interpolated vectors are copied.
    InterpolatedSignedDistance(const InterpolatedSignedDistance &other);

    /// Copy-assignment operator. The VTK function is left to nullptr, while
    /// the interpolated vectors are copied.
    InterpolatedSignedDistance &
    operator=(const InterpolatedSignedDistance &other);

    /// Destructor.
    ~InterpolatedSignedDistance() = default;

    /// Update, setting up the signed distance.
    void
    update();

    /// Reset: reload the file and interpolate.
    void
    reset();

    /// Access operator for elements within the stored vector
    /// (without ghost elements).
    LinAlg::MPI::Vector::reference
    operator[](const LinAlg::MPI::Vector::size_type &idx)
    {
      interpolate();
      return interpolated_owned[idx];
    }

    /// Access operator for elements within the stored vector
    /// (without ghost elements), <kbd>const</kbd> overload.
    LinAlg::MPI::Vector::value_type
    operator[](const LinAlg::MPI::Vector::size_type &idx) const
    {
      interpolate();
      return interpolated_owned[idx];
    }

    /// Compresses the vector without ghost elements, and updates the one with
    /// ghost elements.
    void
    compress(const VectorOperation::values &operation);

    /// Get the interpolated vector, with ghosted elements.
    const LinAlg::MPI::Vector &
    get_ghosted() const
    {
      interpolate();
      return interpolated;
    }

    /// Get the interpolated vector, with ghosted elements.
    const LinAlg::MPI::Vector &
    get_owned() const
    {
      interpolate();
      return interpolated_owned;
    }

    /// Calls the function in the base class, setting the needs_interpolation
    /// flag to true.
    void
    warp_by_array_combination(
      const std::map<const std::string, const double> &arraynames_and_scalings)
    {
      VTKFunction::warp_by_array_combination(arraynames_and_scalings);
      needs_interpolation = true;
    }

    /// Calls the function in the base class, setting the needs_interpolation
    /// flag to true.
    void
    warp_by_pointwise_vectors(const std::vector<Vector<double>> &input_vector,
                              const double &scaling_factor = 1.0)
    {
      VTKFunction::warp_by_pointwise_vectors(input_vector, scaling_factor);
      needs_interpolation = true;
    }

    /// Interpolate the signed distance onto the finite element vectors. This
    /// method is declared const because it is called by getter methods
    /// internally, to update interpolated vectors if needed.
    void
    interpolate() const;

  protected:
    /// DoF handler used for interpolation.
    const DoFHandler<dim> &dof_handler;

    /// Mapping used for interpolation.
    std::shared_ptr<const Mapping<dim>> mapping;

    /// Flag indicating whether interpolation is needed.
    mutable bool needs_interpolation;

    /// Interpolated distance function, without ghost elements.
    mutable LinAlg::MPI::Vector interpolated_owned;

    /// Interpolated distance function, with ghost elements.
    mutable LinAlg::MPI::Vector interpolated;
  };

} // namespace lifex::utils


#define HZ HZ_OLD ///< Restore previous value.


#endif /* LIFEX_UTILS_VTK_FUNCTION_HPP_ */
