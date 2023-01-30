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

#ifndef LIFEX_UTILS_NON_LINEAR_CONJUGATE_GRADIENT_HPP_
#define LIFEX_UTILS_NON_LINEAR_CONJUGATE_GRADIENT_HPP_

#include "source/core_model.hpp"

#include <functional>
#include <string>

namespace lifex::utils
{
  /**
   * @brief Solver for minimization problems using the non-linear conjugate
   * gradient method.
   *
   * Given a loss functional @f$L(\mathbf{x})@f$, this class is used to compute
   * @f$\hat{\mathbf{x}} = \text{arg}\min_{\mathbf{x}} L(\mathbf{x})@f$ with the
   * following iterative procedure: given @f$\mathbf{x}_0@f$, for @f$n = 0, 1,
   * \dots, n_\max@f$:
   * 1. compute @f$\Delta\mathbf{x}_n = -\nabla L(\mathbf{x}_n)@f$;
   * 2. compute the descent direction @f$\mathbf{s}_n@f$:
   *   1. if @f$n = 0@f$, set @f$\mathbf{s}_n = \Delta\mathbf{x}_n@f$;
   *   2. else, compute @f$\beta_n@f$ according to one of the formulas below,
   *      and set @f$\mathbf{s}_n = \beta_n \mathbf{s}_{n-1} +
   *      \Delta\mathbf{x}_n@f$;
   * 3. use backtracking line search to solve @f$\hat{\alpha}^n =
   * \text{arg}\min_\alpha L(\mathbf{x}_n + \alpha \mathbf{s}_n)@f$;
   * 4. set @f$\mathbf{x}_{n+1} = \mathbf{x}_n + \hat{\alpha}^n\mathbf{s}_n@f$.
   *
   * Iterations stop when either
   * @f$\displaystyle\frac{L(\mathbf{x}_n)}{L(\mathbf{x}_0} \leq
   * \varepsilon_1@f$ or @f$\displaystyle\frac{L(\mathbf{x}_n) -
   * L(\mathbf{x}_{n-1})}{L(\mathbf{x}_1) - L(\mathbf{x}_{0})} \leq
   * \varepsilon_2@f$.
   *
   * ## Usage
   * The class exposes an interface similar to that of @ref NonLinearSolverHandler.
   * Use the @ref initialize method to setup the solver, passing pointers to the
   * solution vector (both with and without ghost entries). To compute the
   * minimum (and update the solution vector accordingly), call the @ref solve
   * method, providing functionals to evaluate both the loss function and its
   * gradient.
   *
   * For more details refer to the documentation of the mentioned methods. For
   * an example, refer to @ref MeshOptimization.
   *
   * ## Choice of @f$\beta_n@f$
   * There are three different strategies for updating the descent direction,
   * selectable through the parameter file.
   * - **Fletcher-Reeves** (FR): @f$\displaystyle\beta_n =
   * \frac{\|\Delta\mathbf{x}_n\|^2}{\|\Delta\mathbf{x}_{n-1}\|^2}@f$;
   * - **Polak-Ribière** (PR, the default): @f$\displaystyle\beta_n =
   * \frac{\Delta\mathbf{x}_n^T (\Delta\mathbf{x}_n -
   * \Delta\mathbf{x}_{n-1})}{\|\Delta\mathbf{x}_{n-1}\|^2}@f$;
   * - **Hestenes-Stiefeil** (HS): @f$\displaystyle\beta_n =
   * \frac{\Delta\mathbf{x}_n^T (\Delta\mathbf{x}_n -
   * \Delta\mathbf{x}_{n-1})}{-\mathbf{s}_{n-1}^T(\Delta\mathbf{x}_n -
   * \Delta\mathbf{x}_{n-1})}@f$.
   *
   * ## Backtracking line search
   * To solve step 3 of the conjugate gradient algorithm above, given
   * @f$\alpha_0^n@f$, @f$c \in [0, 1]@f$ and @f$\tau \in (0, 1)@f$, for @f$j =
   * 0, 1, \dots@f$:
   * 1. compute @f$L_j = L(\mathbf{x}_n + \alpha_j^n\mathbf{s}_n)@f$;
   * 2. if @f$L_j \leq L(\mathbf{x}_n) + \alpha_j^n c \Delta\mathbf{x}_n^T
   * \mathbf{s}_n@f$, then set
   * @f$\hat\alpha^n = \alpha_j^n@f$ and terminate the procedure;
   * 3. else, set @f$\alpha_{j+1}^n = \tau \alpha_j@f$.
   *
   * For the first conjugate gradient iteration (i.e. for @f$n = 0@f$) the value
   * of @f$\alpha_0^0@f$ must be provided by the user. Subsequent CG iterations
   * choose @f$\displaystyle\alpha_0^n = \alpha_0^{n-1}
   * \frac{\Delta\mathbf{x}_{n-1}^T \mathbf{s}_{n-1}}{\Delta\mathbf{x}_{n}^T
   * \mathbf{s}_{n}}@f$.
   *
   * **Reference**: @refcite{wright1999numerical, Nocedal and Wright (1999)}.
   */
  template <class VectorType>
  class NonLinearConjugateGradient : public CoreModel
  {
  public:
    /// Constructor.
    NonLinearConjugateGradient(const std::string &subsection)
      : CoreModel(subsection)
      , initialized(false)
    {}

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /**
     * @brief Initialize the solver.
     *
     * This method initializes @ref sol, @ref sol_owned and all auxiliary
     * vectors.
     *
     * @note All input pointers are stored inside this class. This method should
     * be called after <kbd>sol_</kbd> and <kbd>sol_owned_</kbd> have been
     * properly allocated and initialized.
     */
    void
    initialize(VectorType *sol_owned_, VectorType *sol_);

    /**
     * @brief Solve the optimization problem.
     *
     * This method should be called after @ref initialize. The solution is
     * stored in the vectors passed upon calling initialize.
     *
     * @param[in] loss_fun A function to compute the loss @f$L(\mathbf{x}@f$
     * @param[in] loss_gradient_fun A function that computes the loss gradient
     * @f$\nabla L(\mathbf{x})@f$ and stores it in its second argument.
     */
    void
    solve(const std::function<double(const VectorType &)> &loss_fun,
          const std::function<void(const VectorType &, VectorType &)>
            &loss_gradient_fun);

  protected:
    /// Flag indicating whether initialize has been called.
    bool initialized;

    /// Pointer to the solution vector, without ghost entries.
    VectorType *sol_owned;

    /// Pointer to the solution vector.
    VectorType *sol;

    /// Loss gradient.
    VectorType loss_gradient;

    /// Loss gradient at previous iteration.
    VectorType loss_gradient_old;

    /// Descent direction.
    VectorType descent;

    /// Temporary vector for intermediate computations, without ghost entries.
    VectorType tmp_owned;

    /// Temporary vector for intermediate computations.
    VectorType tmp;

    /// Iterations count.
    unsigned int n_iter;

    /// @name Parameters read from file.
    /// @{

    /// Log frequency.
    unsigned int prm_log_frequency;

    /// Method used to compute @f$beta@f$.
    std::string prm_descent_update_method;

    /// @name Stopping criterion.
    /// @{

    /// Maximum number of iterations.
    unsigned int prm_n_max_iterations;

    /// Relative tolerance on the loss reduction.
    double prm_tolerance_loss_reduction;

    /// Relative tolerance on solution increment.
    double prm_tolerance_increment;

    /// @}

    /// @name Line search.
    /// @{

    /// Initial line search step length for the first iteration of the method.
    double prm_linesearch_initial_step_length;

    /// Reduction of step length upon failed line search iteration.
    double prm_linesearch_step_reduction;

    /// Minimum loss improvement between iterations, used as optimality
    /// criterion for line search.
    double prm_linesearch_loss_improvement;

    /// @}

    /// @}
  };
} // namespace lifex::utils

#endif
