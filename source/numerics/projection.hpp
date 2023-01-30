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

#ifndef LIFEX_UTILS_PROJECTION_HPP_
#define LIFEX_UTILS_PROJECTION_HPP_

#include "source/lifex.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::utils
{
  /**
   * @brief Compute the @f$L^2@f$ projection of a function
   * to a given finite element space.
   *
   * Given a function @f$f(\textbf{x})@f$, this class computes
   * a finite element function @f$f_h(\textbf{x})=\sum_{j=1}^{N_h} F_j
   * \varphi_j(\textbf{x})@f$ whose vector of finite element coefficients
   * @f$\textbf{F}@f$ satisfies
   * @f[
   * (f_h, \varphi_i)_{\Omega_h} = (f, \varphi_i)_{\Omega_h},
   * \quad \forall i = 1, \dots, N_h,
   * @f]
   * which is equivalent to solving the following linear system associated with
   * the mass matrix of the finite element space considered:
   * @f[
   * \sum_{j=1}^{N_h} F_j (\varphi_j, \varphi_i)_{\Omega_h} = (f,
   * \varphi_i)_{\Omega_h}, \quad \forall i = 1, \dots, N_h.
   * @f]
   *
   * An example of usage can be found in @ref lifex::tests::TestProjection::project.
   */
  class ProjectionL2
  {
  public:
    /**
     * Constructor for a general regularized projector.
     *
     * This method allocates the system and fills the mass matrix.
     * The mass matrix can be optionally lumped to prevent numerical
     * instabilities, @a e.g. when interpolating the gradient of a FEM solution.
     *
     * @param[in] dof_handler_        Dof handler associated to the space of
     *                                projection.
     * @param[in] quadrature_formula_ Quadrature formula used to integrate
     *                                the projection problem.
     * @param[in] lumping             Enable/disable mass lumping.
     * @param[in] regularization_absolute Enable a regularization term
     *                                @f$(r^2\nabla\varphi_j,
     *                                \nabla\varphi_i)_{\Omega_h}@f$ to the mass
     *                                matrix, where @f$r@f$ is the absolute
     *                                regularization coefficient.
     * @param[in] regularization_relative Enable a regularization term
     *                                @f$((k\,h)^2\nabla\varphi_j,
     *                                \nabla\varphi_i)_{\Omega_h}@f$ to the mass
     *                                matrix, where @f$k@f$ is the relative
     *                                regularization coefficient and @f$h@f$ is
     *                                the cell diameter.
     */
    ProjectionL2(const DoFHandler<dim> &dof_handler_,
                 const Quadrature<dim> &quadrature_formula_,
                 const bool &           lumping                 = false,
                 const double &         regularization_absolute = 0,
                 const double &         regularization_relative = 0);

    /**
     * Assemble the rhs and solve the projection problem.
     *
     * @param[in]  func                The function to project.
     * @param[out] sol_projected_owned The projection result vector,
     *                                 without ghost entries. It must already be
     *                                 initialized with the proper size.
     *
     * @tparam FunctionType the type of function to interpolate.
     *
     * <kbd>FunctionType</kbd> can be one of the following:
     * - @ref QuadratureEvaluationScalar;
     * - @ref QuadratureEvaluationVector;
     * - <kbd>std::vector<std::vector<double>></kbd>;
     * - <kbd>std::vector<std::vector<Tensor<1, dim, double>>></kbd>;
     * - <kbd>Function<dim></kbd>.
     */
    template <class FunctionType>
    void
    project(const FunctionType &func, LinAlg::MPI::Vector &sol_projected_owned);

  private:
    /// Dof handler.
    const DoFHandler<dim> &dof_handler;

    /// Finite element space.
    const FiniteElement<dim> &fe;

    /// Quadrature formula.
    const Quadrature<dim> &quadrature_formula;

    /// System matrix.
    LinAlg::MPI::SparseMatrix mat;

    /// System rhs.
    LinAlg::MPI::Vector rhs;

    /// Preconditioner.
    LinAlg::MPI::PreconditionJacobi preconditioner;
  };
} // namespace lifex::utils

#endif /* LIFEX_UTILS_PROJECTION_HPP_ */
