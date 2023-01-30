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
 * @author Marco Fedele <marco.fedele@polimi.it>.
 */

#ifndef LIFEX_UTILS_TIME_HANDLER_HPP_
#define LIFEX_UTILS_TIME_HANDLER_HPP_

#include "source/lifex.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Handler class for BDF time advancing schemes.
   *
   * The class implements BDF1, BDF2 and BDF3 methods for first derivatives and
   * the corresponding extrapolation formulas.
   *
   * Given the BDF order @f$\sigma_\mathrm{BDF}@f$, the first derivative
   * @f$\frac{\partial \mathbf{u}}{\partial t}(\mathbf{x}, t)@f$ is approximated
   * as:
   * @f[
   * \frac{\partial \mathbf{u}}{\partial t}(\mathbf{x}, t) \approx
   * \frac{\alpha_{\mathrm{BDF}\sigma} \mathbf{u}^{n+1} -
   * \mathbf{u}_{\mathrm{BDF}\sigma}^n}{\Delta t},
   * @f]
   *
   * where @f$\Delta t@f$ is the time step.
   *
   * The extrapolation is computed as a linear combination of the previous time
   * steps:
   * @f[
   * \mathbf{u}^{n + 1}_{\mathrm{EXT}\sigma} \approx \sum_{i =
   * 0}^{\sigma_\mathrm{BDF}} \beta_i \mathbf{u}^{n - i}.
   * @f]
   *
   * The formulas implemented are taken from
   * https://doi.org/10.1016/j.compfluid.2015.05.011.
   *
   * @tparam VectorType Type of solution vector.
   * @note <kbd>VectorType</kbd> can be one of the following:
   * - <kbd>double</kbd>
   * - <kbd>LinAlg::MPI::Vector</kbd>
   * - <kbd>LinAlg::MPI::BlockVector</kbd>
   * - <kbd>LinearAlgebra::distributed::Vector<double></kbd>
   * - <kbd>std::vector<std::vector<double>></kbd>
   * - <kbd>std::vector<std::vector<Tensor<1, dim, double>>></kbd>
   * - <kbd>std::vector<std::vector<Tensor<2, dim, double>>></kbd>
   * .
   * The <kbd>std::vector<std::vector<T>></kbd> specializations can be used
   * for fields evaluated at cell quadrature points:
   * such fields are indicized as <kbd>v[c][q]</kbd> where
   * <kbd>c</kbd> is the index of a cell (local to current processor) and
   * <kbd>q</kbd> is the index of a quadrature node on such cell.
   */
  template <class VectorType>
  class BDFHandler
  {
  public:
    /// Empty constructor.
    BDFHandler();

    /// Constructor.
    /// @param[in] order_            Order of the BDF time advancing scheme.
    /// @param[in] initial_solutions Initial guesses, without ghost entries.
    BDFHandler(const unsigned int &           order_,
               const std::vector<VectorType> &initial_solutions);

    /// Shallow copy from another BDFHandler.
    void
    copy_from(const BDFHandler<VectorType> &other);

    /// Initialize @ref solutions, @ref sol_bdf and @ref sol_extrapolation
    /// with the input initial guesses.
    ///
    /// @param[in] order_            Order of the BDF time advancing scheme.
    /// @param[in] initial_solutions Initial guesses, without ghost entries.
    void
    initialize(const unsigned int &           order_,
               const std::vector<VectorType> &initial_solutions);

    /// Given a new solution vector, this method
    /// - updates @ref solutions, <kbd>push_back</kbd>ing the newest solution;
    /// - updates @ref sol_bdf and @ref sol_extrapolation;
    /// - updates @ref solutions, <kbd>pop_front</kbd>ing the oldest solution.
    ///
    /// @param[in] sol_new              New solution vector, whithout ghost entries.
    /// @param[in] update_extrapolation Bool to specify whether to update
    ///                                 @f$\mathbf{u}_{\mathrm{EXT}\sigma}^{n+1}@f$
    ///                                 or not.
    ///
    /// @note This method is supposed to be called at the @b beginning of the time loop,
    /// before the system assembly.
    void
    time_advance(const VectorType &sol_new,
                 const bool &      update_extrapolation = true);

    /// Get the BDF order.
    const unsigned int &
    get_order() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return order;
    };

    /// Get @f$\alpha_{\mathrm{BDF}\sigma}@f$.
    const double &
    get_alpha() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return alpha;
    };

    /// Get reference to @f$\mathbf{u}_{\mathrm{BDF}\sigma}^n@f$, without ghost
    /// entries.
    const VectorType &
    get_sol_bdf() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return *sol_bdf;
    };

    /// Get reference to @f$\mathbf{u}_{\mathrm{EXT}\sigma}^{n+1}@f$, without
    /// ghost entries.
    const VectorType &
    get_sol_extrapolation() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return *sol_extrapolation;
    };

    /// Get pointer to @f$\mathbf{u}_{\mathrm{BDF}\sigma}^n@f$, without ghost
    /// entries.
    std::shared_ptr<const VectorType>
    get_sol_bdf_ptr() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return sol_bdf;
    };

    /// Get pointer to @f$\mathbf{u}_{\mathrm{EXT}\sigma}^{n+1}@f$, without
    /// ghost entries.
    std::shared_ptr<const VectorType>
    get_sol_extrapolation_ptr() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return sol_extrapolation;
    };

    /// Get @ref solutions.
    const std::deque<VectorType> &
    get_solutions() const
    {
      AssertThrow(initialized, ExcNotInitialized());

      return solutions;
    };

  private:
    /// Flag to tell whether @ref order and @ref solutions have been initialized.
    bool initialized;

    /// Order of the BDF scheme used.
    unsigned int order;

    /// At time step @f$n@f$, deque with the solutions at time steps @f$t_{n -
    /// \sigma_\mathrm{BDF} + 1}, \dots, t_{n-1}@f$.
    std::deque<VectorType> solutions;

    /// @f$\alpha@f$ coefficient multiplying the term @f$u_{n+1}@f$.
    double alpha;

    /// @f$\mathbf{u}_\mathrm{BDF}@f$, without ghost entries.
    std::shared_ptr<VectorType> sol_bdf;

    /// @f$\mathbf{u}_\mathrm{extrapolated}@f$, without ghost entries.
    std::shared_ptr<VectorType> sol_extrapolation;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_TIME_HANDLER_HPP_ */
