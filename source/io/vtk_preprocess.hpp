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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 */

#ifndef LIFEX_UTILS_VTK_PREPROCESS_HPP_
#define LIFEX_UTILS_VTK_PREPROCESS_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/io/vtk_function.hpp"

#include <string>

namespace lifex::utils
{
  /// @brief Class used to preprocess VTK data.
  ///
  /// This class is used to preprocess a VTK mesh using a
  /// @ref VTKFunction object:
  /// when running in parallel, every process would store
  /// the full input data rather than only the information
  /// needed to interpolate on the local mesh.
  ///
  /// Therefore, one would preprocess the VTK data through
  /// this class with, @a e.g., a serial run. Then the
  /// data are saved in a serialized format (see @ref serialize)
  /// that can be read and partitioned by multiple processes
  /// without redundant overlap.
  /// The preprocess output is also saved in a .vtu file
  /// for visualization (and a .vtp file if the input is a .vtp).
  class VTKPreprocess : public CoreModel
  {
  public:
    /// Constructor.
    VTKPreprocess(const std::string &subsection);

    /// Run VTK preprocess.
    virtual void
    run() override;

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

  private:
    /// Mesh.
    utils::MeshHandler triangulation;

    /// @name Parameters read from file.
    /// @{

    unsigned int prm_fe_degree; ///< FE space degree.

    std::string prm_vtk_filename;  ///< VTK filename.
    std::string prm_vtk_arrayname; ///< VTK field name.

    double prm_vtk_array_scaling_factor;    ///< VTK array scaling factor.
    double prm_vtk_geometry_scaling_factor; ///< Scaling factor for the input
                                            ///< geometry.

    /// Toggle reading a scalar or a vectorial array.
    bool prm_vtk_data_is_vectorial;

    VTKDataType       prm_vtk_datatype;      ///< @ref VTKDataType.
    VTKArrayDataType  prm_vtk_arraydatatype; ///< @ref VTKArrayDataType.
    VTKFunction::Mode prm_mode;              ///< @ref VTKFunction::Mode.

    std::string prm_output_filename; ///< Output filename.
    bool        prm_VTK_move_mesh;   ///< Toggle moving the mesh or not.
    /// Toggle moving the input surface points
    /// (if input is a vectorial PointData of a PolyData).
    bool prm_VTK_move_input_surface;

    /// @}
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_VTK_PREPROCESS_HPP_ */
