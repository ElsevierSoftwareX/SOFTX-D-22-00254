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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#ifndef LIFEX_UTILS_MESH_INFO_HPP_
#define LIFEX_UTILS_MESH_INFO_HPP_

#include "source/core.hpp"

#include <deal.II/distributed/tria_base.h>

#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Class to store geometrical information of a
   * parallel distributed triangulation.
   *
   * This class also contains some mesh utilities, such as
   * @ref compute_mesh_volume and @ref compute_surface_area.
   */
  class MeshInfo : public Core
  {
  public:
    /// Constructor.
    MeshInfo(const Triangulation<dim> &triangulation_);

    /// Compute diameters, volume IDs and face/line boundary IDs of the input
    /// triangulation.
    void
    initialize();

    /// Clear all data members and reset to a non-initialized state.
    void
    clear();

    /// Print the following mesh information to standard output:
    /// - a user-specified label;
    /// - maximum cell diameter;
    /// - average cell diameter;
    /// - minimum cell diameter;
    /// - a user-specified string info about the number of degrees of freedom
    /// stored by a DoFHandler attached to the current mesh;
    /// - if the last parameter is <kbd>true</kbd>, the volume IDs and face/line
    ///   boundary IDs.
    void
    print(const std::string &label,
          const std::string &n_dofs_info,
          const bool &       print_ids) const;

    /// Same as the function above, where the second input is the number of dofs
    /// to print.
    void
    print(const std::string & label,
          const unsigned int &n_dofs,
          const bool &        print_ids) const;

    /// Mesh quality info structure.
    class MeshQualityInfo
    {
    public:
      /// Constructor.
      MeshQualityInfo(const double &edge_ratio_min_,
                      const double &edge_ratio_max_,
                      const double &jacobian_min_,
                      const double &jacobian_max_)
        : edge_ratio_min(edge_ratio_min_)
        , edge_ratio_max(edge_ratio_max_)
        , jacobian_min(jacobian_min_)
        , jacobian_max(jacobian_max_)
      {}

      double edge_ratio_min; ///< Minimum edge ratio.
      double edge_ratio_max; ///< Maximum edge ratio.
      double jacobian_min;   ///< Minimum jacobian.
      double jacobian_max;   ///< Maximum jacobian.
    };

    /// Compute quality metrics of the mesh and print them to standard output.
    /// Computed metrics are:
    /// - **edge ratio**: for every element @f$E@f$, whose edges have lengths
    /// @f$l_i@f$, @f$i = 1, \dots, n_\text{edges}@f$, computes @f$\frac{\max_i
    /// l_i}{\min_i l_i}@f$, and reports maximum and minimum ratio across all
    /// elements;
    /// - **jacobians**: computes the jacobian of the mapping from the reference
    /// to the physical element and prints minimum and maximum value.
    ///
    /// @param[in] verbose If set to true, also print information about
    /// individual elements that have bad quality (either edge ratio greater
    /// than 10, or negative jacobian). For such elements we report the material
    /// ID as well as the boundary IDs of their faces, if they are on the
    /// boundary.
    ///
    /// @note For simplex meshes, the minimum and maximum jacobians are
    /// estimated, not computed exactly.
    ///
    /// @todo Check proper metrics on tetrahedra.
    MeshQualityInfo
    print_mesh_quality_info(const bool &verbose = false) const;

    /// Save diameter vector @ref diameters to output file,
    /// sorted by parallel rank and cell index.
    void
    save_diameters(const std::string &filename) const;

    /// Getter for @ref _initialized.
    bool
    initialized() const
    {
      return _initialized;
    }

    /// Get total mesh diameter.
    const double &
    get_diameter_tot() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return diameter_tot;
    }

    /// Get minimum cell diameter.
    const double &
    get_diameter_min() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return diameter_min;
    }

    /// Get maximum cell diameter.
    const double &
    get_diameter_max() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return diameter_max;
    }

    /// Get average cell diameter.
    const double &
    get_diameter_avg() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return diameter_avg;
    }

    /// Get volume IDs.
    const std::set<types::material_id> &
    get_ids_volume() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return ids_volume;
    }

    /// Get face boundary IDs.
    const std::set<types::boundary_id> &
    get_ids_face() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return ids_face;
    }

    /// Get line boundary IDs.
    const std::set<types::boundary_id> &
    get_ids_line() const
    {
      AssertThrow(_initialized, ExcNotInitialized());

      return ids_line;
    }

    /// Compute the volume of a mesh.
    double
    compute_mesh_volume() const;

    /// Compute the volume of a subportion of a mesh, given a set of material
    /// IDs that define it.
    double
    compute_submesh_volume(
      const std::set<types::material_id> &material_ids) const;

    /// Compute the volume of a subportion of a mesh, given its material ID.
    double
    compute_submesh_volume(const types::material_id &material_id) const;

    /// Compute the area of a surface, given its boundary ID.
    double
    compute_surface_area(const types::boundary_id &boundary_id) const;

    /// Compute the normal vector to a @b flat boundary surface, given its
    /// boundary ID.
    Tensor<1, dim, double>
    compute_flat_boundary_normal(const types::boundary_id &boundary_id) const;

    /// Compute the barycenter of the mesh.
    Point<dim>
    compute_mesh_barycenter() const;

    /// Compute the barycenter of a set of @b flat boundary surfaces, given
    /// their boundary IDs.
    Point<dim>
    compute_surface_barycenter(
      const std::set<types::boundary_id> &boundary_ids) const;

    /// Compute the barycenter of a @b flat boundary surface, given its boundary
    /// ID.
    Point<dim>
    compute_surface_barycenter(const types::boundary_id &boundary_id) const;

    /// Compute the moment of inertia of the mesh w.r.t. Cartesian directions.
    Tensor<1, dim, double>
    compute_mesh_moment_inertia() const;

  private:
    /// <kbd>true</kbd> if @ref initialize was called, <kbd>false</kbd> otherwise.
    bool _initialized;

    /// Reference to a triangulation object.
    const Triangulation<dim> &triangulation;

    /// Vector containing the diameter of each locally owned active cell.
    std::vector<double> diameters;

    /// Global number of active cells.
    unsigned int n_cells;

    double diameter_tot; ///< Total mesh diameter.
    double diameter_min; ///< Minimum cell diameter.
    double diameter_max; ///< Maximum cell diameter.
    double diameter_avg; ///< Average cell diameter.

    std::set<types::material_id> ids_volume; ///< Volume IDs.
    std::set<types::boundary_id> ids_face;   ///< Face boundary IDs.
    std::set<types::boundary_id> ids_line;   ///< Line boundary IDs.
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_MESH_INFO_HPP_ */
