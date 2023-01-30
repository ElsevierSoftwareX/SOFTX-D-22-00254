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

#ifndef LIFEX_UTILS_MESH_OPTIMIZATION_HPP_
#define LIFEX_UTILS_MESH_OPTIMIZATION_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_handler.hpp"

#include "source/numerics/non_linear_conjugate_gradient.hpp"

#include <functional>
#include <memory>
#include <string>

namespace lifex::utils
{
  /**
   * @brief Mesh optimization helper.
   *
   * This class solves an optimization problems to remove invalid elements from
   * a mesh @f$\mathcal{T}@f$ by displacing its nodes. To do so, it minimizes
   * the functional
   * @f[
   * L = \sum_{K \in \mathcal{T}}\frac{1}{4}\left(\frac{|V_K - \gamma\bar{V}| -
   * (V_K - \gamma\bar{V})}{\bar{V}}\right)^2 + \alpha\|\nabla(\mathbf x -
   * \mathbf x_0)\|^2
   * @f]
   * with respect to the position vector of internal nodes @f$\mathbf x@f$.
   * Boundary nodes are not displaced. @f$\bar{V}@f$ is the average element
   * volume (which equals the total volume of @f$\mathcal{T}@f$ devided by the
   * number of elements, and is therefore independent of @f$\mathbf{x}@f$),
   * @f$\gamma@f$ is a user-defined parameter expressing the minimum target
   * volume as a fraction of @f$\bar{V}@f$, and @f$\alpha > 0@f$ weights a
   * regularization on the gradient of the displacement with respect to the
   * initial position vector @f$\mathbf x_0@f$.
   *
   * The loss functional penalizes inverted elements (i.e. elements such that
   * @f$V < \gamma\bar{V}@f$, including elements with negative volume), so that
   * its minimization eliminates those elements.
   *
   * Minimization is achieved by means of the non-linear conjugate gradient
   * method as implemented by @ref NonLinearConjugateGradient.
   *
   * **Reference**: @refcite{knupp2001hexahedral, Knupp (2001)}.
   */
  class MeshOptimization : public CoreModel
  {
  public:
    /**
     * @brief Constructor.
     */
    MeshOptimization(const std::string &subsection);

    /**
     * @brief Declare parameters.
     */
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /**
     * @brief Parse parameters.
     */
    virtual void
    parse_parameters(ParamHandler &params) override;

    /**
     * @brief Initialize.
     *
     * @param[in] dof_handler_ A pointer to a DoFHandler that is initialized with
     * the mesh to be optimized, and with the finite element space used to
     * represent mesh position and displacement.
     *
     * @note The finite element space underlying the DoF handler must represent
     * exactly dim components (since it must represent a displacement field).
     */
    void
    setup_system(const std::shared_ptr<DoFHandler<dim>> &dof_handler_);

    /**
     * @brief Run mesh optimization.
     */
    virtual void
    run() override;

    /**
     * @brief Run mesh optimization, providing an initial position vector.
     */
    void
    run(const LinAlg::MPI::Vector &x0_owned_);

    /**
     * @brief Compute the loss function associated to mesh quality.
     *
     * @param[in] x Finite element vector that describes the position of the
     * mesh nodes (including ghost elements).
     */
    double
    compute_loss(const LinAlg::MPI::Vector &x);

    /**
     * @brief Compute the gradient of the loss function associated to mesh quality.
     *
     * @param[in] x Finite element vector that describes the position of the
     * mesh nodes (including ghost elements).
     * @param[out] loss_gradient Gradient of the loss function. The vector
     * should be initialized and have a parallel partitioning consistent with
     * that of x, but excluding ghost elements. Previous content is overwritten.
     */
    void
    compute_loss_gradient(const LinAlg::MPI::Vector &x,
                          LinAlg::MPI::Vector &      loss_gradient);

    /// @name Getters.
    /// @{

    /// Get incremental displacement.
    const LinAlg::MPI::Vector &
    get_displacement_owned() const
    {
      return d_incr_owned;
    }

    /// @}

    /// @name Setters.
    /// @{

    /// Set initial position.
    void
    set_x0(const LinAlg::MPI::Vector &x0_owned_)
    {
      x0_owned = x0_owned_;
      x0       = x0_owned;
    }

    /// @}

  protected:
    /// DoF handler.
    std::shared_ptr<DoFHandler<dim>> dof_handler;

    /// Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    /// Boundary DoF indices.
    IndexSet boundary_dofs;

    /// Position vector, owned elements.
    LinAlg::MPI::Vector x_owned;

    /// Position vector.
    LinAlg::MPI::Vector x;

    /// Initial position vector, owned elements.
    LinAlg::MPI::Vector x0_owned;

    /// Initial position vector.
    LinAlg::MPI::Vector x0;

    /// Incremental displacement, owned elements.
    LinAlg::MPI::Vector d_incr_owned;

    /// Non-linear conjugate gradient solver.
    NonLinearConjugateGradient<LinAlg::MPI::Vector> nlcg;

    /// Average element volume.
    double average_element_volume;

    /// Absolute minimum element volume.
    double minimum_element_volume;

    /// @name Parameters read from file.
    /// @{

    /// Minimum volume, as a fraction of average element volume.
    double prm_min_volume;

    /// Weight @f$\alpha@f$ of displacement gradient term in the loss function.
    double prm_weight_gradient;

    /// @}
  };
} // namespace lifex::utils

#endif
