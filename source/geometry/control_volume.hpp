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

#ifndef LIFEX_UTILS_CONTROL_VOLUME_HPP_
#define LIFEX_UTILS_CONTROL_VOLUME_HPP_

#include "source/core_model.hpp"

#include "source/io/vtk_function.hpp"

#include <memory>
#include <string>

namespace lifex::utils
{
  /**
   * @brief Control volume class.
   *
   * Represents a control volume within the domain, defined as an analytical
   * sphere or by a surface read from file (see @ref ControlVolume::Type), that
   * can be used to compute the average solution within the volume for
   * postprocessing purposes.
   */
  class ControlVolume : public CoreModel
  {
  public:
    /// Quadrature evaluator.
    class QuadratureEvaluation : public QuadratureEvaluationFEMScalar
    {
    public:
      /// Constructor.
      QuadratureEvaluation(const DoFHandler<dim> &dof_handler,
                           const Quadrature<dim> &quadrature,
                           const ControlVolume &  volume_);

      /// Call operator.
      ///
      /// @note This evaluates the distance using the interpolated vector, rather
      /// than the signed distance VTKFunction directly. This way, if the mesh
      /// is moved, the control volume moves with it.
      virtual double
      operator()(const unsigned int &q,
                 const double &      t   = 0,
                 const Point<dim> &  x_q = Point<dim>()) override;

    protected:
      /// Reference to the control volume.
      const ControlVolume &volume;

      /// Distance, for preallocation.
      double distance_q;
    };

    /// Type of control volume.
    enum class Type
    {
      Sphere, ///< Spherical control volume.
      File    ///< Volume read from a VTP file.
    };

    /// Constructor.
    ControlVolume(const std::string &subsection, const unsigned int &index_);

    /**
     * @brief Declare parameters.
     *
     * Since the number of volumes is unknown a priori, this method declares
     * parameters for an arbitrary number of volumes (i.e. as lists). Then,
     * when parsing with parse_parameters, only the relevant entries in the
     * lists are kept, according to the index of the volume as given to the
     * constructor.
     */
    void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    void
    parse_parameters(ParamHandler &params) override;

    /**
     * @brief Initialize the control volume.
     *
     * @param[in] mapping A pointer to a mapping used in interpolation. Leave to
     * nullptr to use the default mapping. Beware that the default mapping might
     * not be appropriate for moving meshes (@a e.g. in fluid dynamics problems
     * with ALE formulation), especially after restarting a simulation from a
     * deformed configuration.
     * @param[in] dof_handler The scalar DoF handler used to interpolate the
     * control volume distance on.
     */
    void
    initialize(const std::shared_ptr<Mapping<dim>> &mapping,
               const DoFHandler<dim> &              dof_handler);

    /// Initialize the control volume using the default mapping.
    void
    initialize(const DoFHandler<dim> &dof_handler)
    {
      initialize(nullptr, dof_handler);
    }

    /// @name Getters.
    /// @{

    /// Get the label.
    const std::string &
    get_label() const
    {
      return prm_label;
    }

    /// Get owned signed distance vector.
    const LinAlg::MPI::Vector &
    get_owned() const
    {
      if (prm_type == Type::File)
        return signed_distance->get_owned();
      else // if (prm_type == Type::Sphere)
        return sphere_distance_owned;
    }

    /// Get ghosted signed distance vector.
    const LinAlg::MPI::Vector &
    get_ghosted() const
    {
      if (prm_type == Type::File)
        return signed_distance->get_ghosted();
      else // if (prm_type == Type::Sphere)
        return sphere_distance;
    }

    /// @}

  protected:
    /// Index identifying the control volume.
    unsigned int index;

    /// Interpolated signed distance for the spherical control volume, without
    /// ghost elements.
    LinAlg::MPI::Vector sphere_distance_owned;

    /// Interpolated signed distance for the spherical control volume, with
    /// ghost elements.
    LinAlg::MPI::Vector sphere_distance;

    /// Signed distance function from the control volume (used it type is
    /// Type::File).
    std::unique_ptr<utils::InterpolatedSignedDistance> signed_distance;

    /// @name Parameters read from file.
    /// @{

    /// Type of control volume.
    Type prm_type;

    /// Label identifying the control volume.
    std::string prm_label;

    /// Center of spherical control volume.
    Point<dim> prm_center;

    /// Radius of spherical control point.
    double prm_radius;

    /// Filename to read the signed distance from if type is Type::File.
    std::string prm_filename;

    /// @}
  };
} // namespace lifex::utils

#endif
