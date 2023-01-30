/********************************************************************************
  Copyright (C) 2020 - 2022 by the lifex authors.

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

#ifndef LIFEX_UTILS_NON_LINEAR_SOLVER_HANDLER_HPP_
#define LIFEX_UTILS_NON_LINEAR_SOLVER_HANDLER_HPP_

#include "source/core_model.hpp"

#include "source/io/csv_writer.hpp"

#include "source/numerics/fixed_point_acceleration.hpp"
#include "source/numerics/linear_solver_handler.hpp"

#include <deal.II/base/timer.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace lifex::utils
{
  /// @brief Helper class to declare a non-linear solver.
  ///
  /// This class enables the solution of an arbitrary equation of
  /// the type:
  /// @f[
  /// \mathbf{f}(\mathbf{x}) = \mathbf{0}.
  /// @f]
  ///
  /// The previous equation is linearized via Newton's method that, for
  /// a given initial guess @f$\mathbf{x}^{(0)}@f$, consists of solving for
  /// @f$k = 1, \dots, N_\mathrm{max}@f$ until convergence the following linear
  /// systems:
  /// @f[
  /// \left\{
  /// \begin{aligned}
  /// \frac{\mathrm{d}\mathbf{f}}{\mathrm{d}\mathbf{x}}\left(\mathbf{x}^{(k)}\right)
  /// \delta\mathbf{x}^{(k+1)} & = \mathbf{f}\left(\mathbf{x}^{(k)}\right),
  /// \\ \mathbf{x}^{(k+1)} & = \mathbf{x}^{(k)} - \delta\mathbf{x}^{(k+1)}.
  /// \end{aligned}
  /// \right.
  /// @f]
  ///
  /// The user is required to provide a function that assembles, at each
  /// non-linear step, the jacobian matrix @f$\mathfrak{J}^{(k)} =
  /// \frac{\mathrm{d}\mathbf{f}}{\mathrm{d}\mathbf{x}}\left(\mathbf{x}^{(k)}\right)@f$
  /// and the residual vector
  /// @f$\mathfrak{r}^{(k)} = \mathbf{f}\left(\mathbf{x}^{(k)}\right)@f$, and a
  /// function that solves the linearized system for
  /// @f$\delta\mathbf{x}^{(k+1)}@f$.
  ///
  /// The algorithms is stopped as soon as one of the following criteria is met:
  /// - the absolute residual norm @f$\left\|\mathbf{r}^{(k)}\right\|@f$ is
  /// smaller than a specified tolerance;
  /// - the relative residual norm @f$\left\|\mathbf{r}^{(k)}\right\| /
  /// \left\|\mathbf{r}^{(0)}\right\|@f$ is smaller than a specified tolerance;
  /// - the relative increment norm @f$\left\|\delta\mathbf{x}^{(k+1)}\right\| /
  /// \left\|\mathbf{x}^{(k)}\right\|@f$ is smaller than a specified tolerance;
  /// - the maximum number of iterations is reached.
  template <class VectorType>
  class NonLinearSolverHandler : public CoreModel
  {
  public:
    /// Constructor.
    ///
    /// @param[in] subsection Parameter subsection path.
    NonLinearSolverHandler(const std::string &subsection);

    /// Override of @ref lifex::CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref lifex::CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// This method initializes @ref sol, @ref sol_owned and @ref incr.
    ///
    /// @note All input pointers are stored inside this class. This method should
    /// be called after <kbd>sol_</kbd> and <kbd>sol_owned_</kbd> have been
    /// properly allocated and initialized.
    void
    initialize(VectorType *sol_owned_, VectorType *sol_);

    /// This method solves the non-linear equation until convergence.
    ///
    /// After calling this function, both the vectors pointed by
    /// @ref sol and @ref sol_owned are updated.
    ///
    /// @param[in] assemble_fun A function that assembles the jacobian matrix
    ///                         and the residual vector, with an input parameter
    ///                         corresponding to a boolean value used to specify
    ///                         whether to re-assemble the matrix or not.
    ///                         It returns a user-defined residual norm.
    /// @param[in] solve_fun A function that assembles the preconditioner and
    ///                      solves the non-linear iteration.
    ///                      The input parameter corresponds to a boolean
    ///                      value used to specify whether to re-assemble
    ///                      the preconditioner or not, and the increment vector
    ///                      @ref incr to be updated. It returns the solution
    ///                      norm computed at the previous non-linear iteration,
    ///                      the increment norm computed after solving the
    ///                      current step and the number of iterations performed
    ///                      by the linear solver.
    ///
    /// @return Whether the method has reached convergence or not.
    ///
    /// Prototype examples for <kbd>assemble_fun</kbd>,
    /// <kbd>precondition_fun</kbd> and <kbd>solve_fun</kbd> are:
    /// @code{.cpp}
    /// auto assemble_fun = [...](const bool &assemble_jac) {
    ///   assemble_system(assemble_jac);
    ///
    ///   return res.l2_norm();
    /// };
    ///
    /// auto solve_fun = [preconditioner, ...](const bool &assemble_prec,
    ///                                        LinAlg::MPI::Vector &incr) {
    ///   // Solution norm at previous iteration.
    ///   const double norm_sol = sol_owned.l2_norm();
    ///
    ///   if (assemble_prec)
    ///     preconditioner.initialize(jac);
    ///
    ///   linear_solver.solve(jac, incr, res, preconditioner);
    ///
    ///   // Increment norm at current iteration.
    ///   const double norm_incr = incr.l2_norm();
    ///
    ///   // Number of linear solver iterations.
    ///   const unsigned int n_iterations_linear =
    ///     linear_solver.get_n_iterations();
    ///
    ///   return std::make_tuple(norm_sol, norm_incr, n_iterations_linear);
    /// };
    /// @endcode
    virtual bool
    solve(const std::function<double(const bool &assemble_jac)> &assemble_fun,
          const std::function<std::tuple<double, double, unsigned int>(
            const bool &assemble_prec,
            VectorType &incr)> &                                 solve_fun);

    /// Getter method for @ref n_iterations.
    const unsigned int &
    get_n_iterations() const
    {
      return n_iterations;
    }

    /// Declare entries to an external CSV writer.
    void
    declare_entries_csv(CSVWriter &csv_writer, const std::string &prefix) const;

    /// Set entries to an external CSV writer.
    void
    set_entries_csv(CSVWriter &csv_writer) const;

  protected:
    /// Boolean value to specify whether @ref initialize has been called.
    bool initialized;

    /// Pointer to solution vector, without ghost entries.
    VectorType *sol_owned;

    /// Pointer to solution vector, with ghost entries.
    VectorType *sol;

    /// Increment vector, without ghost entries.
    VectorType incr;

    /// Last number of iterations performed.
    unsigned int n_iterations;


    /// @name Linear solver statistics.
    /// @{

    /// Prefix for CSV entries.
    mutable std::string csv_prefix;

    /// Current number of linear solver iterations.
    unsigned int n_iterations_linear;

    /// Minimum number of linear solver iterations.
    unsigned int n_iterations_linear_min;

    /// Maximum number of linear solver iterations.
    unsigned int n_iterations_linear_max;

    /// Total number of linear solver iterations.
    unsigned int n_iterations_linear_tot;

    /// Average number of linear solver iterations.
    unsigned int n_iterations_linear_avg;

    /// Total solving time.
    double time_solve;

    /// @}

    /// Fixed point acceleration scheme.
    std::unique_ptr<utils::FixedPointAcceleration<VectorType>>
      fixed_point_acceleration;

    /// Internal timer used to keep track of the solving time.
    Timer timer_solve;

    /// @name Parameters read from file.
    /// @{

    /// If true, the following parameters will be ignored: the max. number of
    /// iterations will implicitly be set to 1 and tolerances to 0.
    bool prm_linearized;

    /// Maximum number of iterations.
    unsigned int prm_max_iterations;

    /// Tolerance on absolute residual norm.
    double prm_absolute_res_tol;

    /// Tolerance on relative residual norm.
    double prm_relative_res_tol;

    /// Tolerance on relative increment norm.
    double prm_relative_incr_tol;

    /// Reassemble jacobian every n-th iteration.
    unsigned int prm_jacobian_lag;

    /// Type of acceleration to use at each non-linear step.
    std::string prm_acceleration_type;

    /// @}
  };

  /**
   * @brief Utility function for quasi-Newton implicit Runge-Kutta solver.
   *
   * This function is supposed to be used as a second input to the
   * <kbd>evolve_one_time_step</kbd> method of @dealii implicit Runge-Kutta
   * solvers when the Jacobian matrix is not known but rather approximated
   * via forward finite differences.
   *
   * @param[in] f The function @f$f(t, y)@f$ defining the ODE problem.
   * @param[in] y The initial guess for the current timestep.
   * @param[in] step_rel The relative step increment for the finite difference formula.
   * @param[in] step_min The minimum allowed step increment for the finite difference formula.
   */
  std::function<
    Vector<double>(const double, const double, const Vector<double> &)>
  id_minus_tau_J_inverse_quasinewton(
    const std::function<Vector<double>(const double, const Vector<double> &)>
      &                   f,
    const Vector<double> &y,
    const double &        step_rel = 1e-6,
    const double &        step_min = 1e-10);

  /**
   * @brief Newton-Krylov solver class.
   *
   * Consider a non-linear problem solved with Newton's method as described by
   * @ref NonLinearSolverHandler, in which at every iteration the linearized
   * problem is solved with an iterative method as implemented by the
   * @ref LinearSolverHandler class.
   *
   * During the first iterations, there is no need to solve the linear system
   * with good accuracy, since we are far from the solution of the non-linear
   * problem. This class provides a way to adjust the relative tolerance for the
   * linear solver during Newton iterations, starting with a large tolerance and
   * reducing it as the Newton loop gets closer to the solution. This approach
   * is referred to as "inexact" in the sense that the linearized system is not
   * solved exactly (due to the higher tolerance).
   *
   * @note In this context, it is recommended to use a very small absolute tolerance
   * for the linear solver. Indeed, the absolute tolerance safeguards from
   * trying to reach unrealistically small residual reductions when the absolute
   * residual is already small, but should not play a role in other cases.
   *
   * Refer to @ref InexactNonLinearSolverHandler::Method for implemented methods.
   *
   * **Reference:** @refcite{Eisenstat1996, Eisenstat and Walker (1996)}.
   */
  template <class VectorType>
  class InexactNonLinearSolverHandler
    : public NonLinearSolverHandler<VectorType>
  {
  public:
    /// Enumeration for the different methods implemented.
    ///
    /// @note Of the two choices presented in @refcite{Eisenstat1996, Eisenstat
    /// and Walker (1996)}, only Choice 2 is implemented. The reason is that
    /// Choice 1, while reportedly better according to the original paper,
    /// appears to result in loss of convergence in many cases. This is
    /// something that might be worth going back to in the future.
    enum class Method
    {
      /// Newton method: falls back on the behavior of NonLinearSolverHandler.
      Newton,

      /// Use a power law for the relative tolerance. At @f$k@f$-th iteration,
      /// set the relative tolerance to @f$\eta_k = \alpha^{\beta k+\gamma}@f$,
      /// with @f$\alpha, \beta@f$ and @f$\gamma@f$ suitable parameters.
      ///
      /// **Reference**: @refcite{Brown1990, Brown and Saad (1990)}
      PowerLawReduction,

      /// Eisenstat-Walker inexact Newton method: the relative tolerance of the
      /// linear solver is adjusted during Newton iterations according to the
      /// "Choice 2" rule proposed in @refcite{Eisenstat1996, Eisenstat and
      /// Walker (1996)}, that is
      /// @f[ \eta_k = \gamma\left(\frac{\left\|\mathbf f(\mathbf
      /// x^{(k)})\right\|}{\left\|\mathbf f(\mathbf
      /// x^{(k-1)})\right\|}\right)^\alpha @f]
      /// with @f$\gamma@f$ and @f$\alpha@f$ user-defined parameters.
      EisenstatWalker
    };

    /// Constructor.
    InexactNonLinearSolverHandler(
      const std::string &              subsection,
      LinearSolverHandler<VectorType> &linear_solver_handler_)
      : NonLinearSolverHandler<VectorType>(subsection)
      , linear_solver_handler(linear_solver_handler_)
    {}

    /// Solve non-linear problem.
    virtual bool
    solve(const std::function<double(const bool &assemble_jac)> &assemble_fun,
          const std::function<std::tuple<double, double, unsigned int>(
            const bool &assemble_prec,
            VectorType &incr)> &solve_fun) override;

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

  protected:
    /// Reference to the linear solver handler.
    LinearSolverHandler<VectorType> &linear_solver_handler;

    /// Norm of the residual at current iteration.
    double residual_norm;

    /// Norm of the residual at previous iteration.
    double residual_norm_old;

    /// Relative tolerance for the linear solver.
    double rel_tolerance;

    /// @name Parameters read from file.
    /// @{

    /// Method.
    Method prm_method;

    /// Power-law alpha parameter.
    double prm_power_law_alpha;

    /// Power law beta parameter.
    double prm_power_law_beta;

    /// Power law n parameter.
    double prm_power_law_gamma;

    /// Maximum relative tolerance for Eisenstat-Walker method.
    double prm_ew_tolerance_max;

    /// Eisenstat-Walker alpha parameter.
    double prm_ew_alpha;

    /// Eisenstat-Walker gamma parameter.
    double prm_ew_gamma;

    /// @}
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_NON_LINEAR_SOLVER_HANDLER_HPP_ */
