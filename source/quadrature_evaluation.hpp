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
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 */

#ifndef __LIFEX_QUADRATURE_EVALUATION_HPP_
#define __LIFEX_QUADRATURE_EVALUATION_HPP_

#include "source/lifex.hpp"

#include <memory>
#include <vector>

namespace lifex
{
  /**
   * @brief Abstract functor that evaluates a field at quadrature nodes.
   *
   * Quadrature nodes are defined on each cell by a given quadrature rule.
   *
   * A @ref QuadratureEvaluation object is supposed to be used inside
   * an assembly loop: it needs to be initialized using @ref init before the loop starts
   * and is evaluated on current cell using @ref reinit.
   */
  template <class NumberType>
  class QuadratureEvaluation
  {
  public:
    /// Default constructor.
    QuadratureEvaluation() = default;

    /// Destructor.
    virtual ~QuadratureEvaluation() = default;

    /// Initialize object before the assembly loop.
    virtual void
    init()
    {}

    /// Reinitialize object on current cell.
    virtual void
    reinit(const DoFHandler<dim>::active_cell_iterator & /*cell_other*/)
    {}

    /**
     * @brief Call operator.
     * @param[in] q_index Index of the quadrature node to evaluate the function at.
     * @param[in] t       Current simulation time.
     * @param[in] x_q     Coordinates of current quadrature node (only used to evaluate analytical functions).
     * @return the function evaluation.
     */
    virtual NumberType
    operator()(const unsigned int &q_index,
               const double &      t   = 0,
               const Point<dim> &  x_q = Point<dim>()) = 0;
  };

  /// Alias to evaluate a scalar function.
  using QuadratureEvaluationScalar = QuadratureEvaluation<double>;

  /// Alias to evaluate a vector function.
  using QuadratureEvaluationVector =
    QuadratureEvaluation<Tensor<1, dim, double>>;

  /// Alias to evaluate a tensor function.
  using QuadratureEvaluationTensor =
    QuadratureEvaluation<Tensor<2, dim, double>>;

  /**
   * @brief Represents a constant function
   */
  template <class NumberType>
  class QuadratureEvaluationConstant : public QuadratureEvaluation<NumberType>
  {
  public:
    /**
     * @brief Constructor.
     * @param[in] value_   Constant value returned by the function.
     */
    QuadratureEvaluationConstant(const NumberType &value_)
      : value(value_)
    {}

    /**
     * @brief Call operator.
     * @return the function evaluation.
     */
    virtual NumberType
    operator()(const unsigned int & /*q_index*/,
               const double & /*t*/       = 0,
               const Point<dim> & /*x_q*/ = Point<dim>()) override
    {
      return value;
    }

    /// Constant value returned by the function.
    NumberType value;
  };

  /**
   * @brief Abstract functor that evaluates finite elements fields at quadrature nodes.
   *
   * Quadrature nodes are defined on each cell by a given quadrature rule.
   */
  template <class NumberType>
  class QuadratureEvaluationFEM : public QuadratureEvaluation<NumberType>
  {
  public:
    /**
     * @brief Default constructor.
     */
    QuadratureEvaluationFEM()
      : dof_handler(nullptr)
      , fe_values(nullptr)
      , dof_indices(0)
    {}

    /**
     * @brief Constructor.
     * @param[in] dof_handler_ DoF handler associated to the field to be evaluated.
     * @param[in] quadrature   Quadrature rule to evaluate the field at.
     * @param[in] update_flags Update flags.
     */
    QuadratureEvaluationFEM(
      const DoFHandler<dim> &dof_handler_,
      const Quadrature<dim> &quadrature   = Quadrature<dim>(1),
      const UpdateFlags &    update_flags = update_default)
    {
      setup(dof_handler_, quadrature, update_flags);
    }

    /**
     * Same as the constructor above, for (re-)initializing this object
     * after construction.
     */
    void
    setup(const DoFHandler<dim> &dof_handler_,
          const Quadrature<dim> &quadrature   = Quadrature<dim>(1),
          const UpdateFlags &    update_flags = update_default)
    {
      dof_handler = &dof_handler_;

      fe_values = std::make_unique<FEValues<dim>>(dof_handler->get_fe(),
                                                  quadrature,
                                                  update_flags);

      dof_indices.resize(dof_handler->get_fe().dofs_per_cell);
    }

    /// Destructor.
    virtual ~QuadratureEvaluationFEM() = default;

    /// Initialize @ref cell_next and @ref endc as
    /// <kbd>begin</kbd> and <kbd>end</kbd> active iterators.
    /// @ref cell_next points to the first locally owned active cell.
    virtual void
    init() override final
    {
      AssertThrow(dof_handler != nullptr, ExcNotInitialized());
      AssertThrow(fe_values != nullptr, ExcNotInitialized());

      endc = dof_handler->end();

      for (cell_next = dof_handler->begin_active(); cell_next != endc;
           ++cell_next)
        if (cell_next->is_locally_owned())
          {
            break;
          }

      post_init_callback();
    }

    /// Update @ref fe_values and @ref dof_indices on current cell
    /// and increment @ref cell_next to get ready for the next iteration.
    /// @warning The input cell is only used for asserting correctness:
    /// since the current @ref fe_values is associated to a different DoFHandler,
    /// it cannot be reinitialized on the the cell used in the assembly loop
    /// (@a i.e. the one passed in input).
    virtual void
    reinit(
      const DoFHandler<dim>::active_cell_iterator &cell_other) override final
    {
      cell = cell_next;

      fe_values->reinit(cell);
      cell->get_dof_indices(dof_indices);

      // Get next locally owned cell.
      do
        {
          ++cell_next;

          if (cell_next == endc)
            {
              break;
            }
        }
      while (!cell_next->is_locally_owned());

      AssertThrow(cell->center() == cell_other->center(),
                  ExcLifexInternalError());

      post_reinit_callback(cell_other);
    }

    /**
     * @brief Call operator.
     * @param[in] q_index Index of the quadrature node to evaluate the function at.
     * @param[in] t       Current simulation time.
     * @param[in] x_q     Coordinates of current quadrature node (only used to evaluate analytical functions).
     * @return the function evaluation.
     */
    virtual NumberType
    operator()(const unsigned int &q_index,
               const double &      t   = 0,
               const Point<dim> &  x_q = Point<dim>()) override
    {
      // Silence "unused parameter" warnings while still allowing Doxygen
      // documentation of the input parameters.
      (void)q_index;
      (void)t;
      (void)x_q;

      AssertThrow(false, ExcLifexInternalError());

      return NumberType();
    };

  protected:
    /// This method is called at the end of @ref init.
    /// This is meant to allow derived classes to add
    /// features to the @ref init method. Empty by default.
    virtual void
    post_init_callback()
    {}

    /// This method is called at the end of @ref reinit.
    /// This is meant to allow derived classes to add
    /// features to the @ref reinit method. Empty by default.
    virtual void
    post_reinit_callback(
      const DoFHandler<dim>::active_cell_iterator & /*cell_other*/)
    {}

    /// DoF handler.
    const DoFHandler<dim> *dof_handler;

    /// FEValues object that stores values of the FE space at current cell
    /// dofs.
    std::unique_ptr<FEValues<dim>> fe_values;

    /// Vector to store dof indices of current cell.
    std::vector<types::global_dof_index> dof_indices;

    /// Iterator to current cell.
    DoFHandler<dim>::active_cell_iterator cell;
    /// Iterator to next locally owned cell.
    DoFHandler<dim>::active_cell_iterator cell_next;
    /// Iterator to last active cell.
    DoFHandler<dim>::active_cell_iterator endc;
  };

  /// Alias to evaluate a scalar finite element field.
  using QuadratureEvaluationFEMScalar = QuadratureEvaluationFEM<double>;

  /// Alias to evaluate a vector finite element field.
  using QuadratureEvaluationFEMVector =
    QuadratureEvaluationFEM<Tensor<1, dim, double>>;

  /// Alias to evaluate a tensor finite element field.
  using QuadratureEvaluationFEMTensor =
    QuadratureEvaluationFEM<Tensor<2, dim, double>>;

  /// @brief Evaluate a scalar solution given as a FE vector.
  class QuadratureFEMSolution : public QuadratureEvaluationFEMScalar
  {
  public:
    /**
     * Constructor.
     * @param[in] sol_        Solution vector, with ghost entries.
     * @param[in] dof_handler Dof handler (scalar).
     * @param[in] quadrature_ Quadrature rule to evaluate solution at.
     */
    QuadratureFEMSolution(const LinAlg::MPI::Vector &sol_,
                          const DoFHandler<dim> &    dof_handler,
                          const Quadrature<dim> &    quadrature_);

    /// Call operator.
    virtual double
    operator()(const unsigned int &q,
               const double & /*t*/       = 0,
               const Point<dim> & /*x_q*/ = Point<dim>()) override;

  private:
    /// Solution vector.
    const LinAlg::MPI::Vector &sol;

    /// Used for pre-allocation.
    double sol_q;
  };

  /// @brief Evaluate the gradient of a scalar solution given as a FE vector.
  class QuadratureFEMGradient : public QuadratureEvaluationFEMVector
  {
  public:
    /**
     * Constructor.
     * @param[in] sol_        Solution vector, with ghost entries.
     * @param[in] dof_handler Dof handler (scalar).
     * @param[in] quadrature_ Quadrature rule to evaluate gradient at.
     * @param[in] component_  The selected component.
     */
    QuadratureFEMGradient(const LinAlg::MPI::Vector &sol_,
                          const DoFHandler<dim> &    dof_handler,
                          const Quadrature<dim> &    quadrature_,
                          const unsigned int &       component_ = 0);

    /// Call operator.
    virtual Tensor<1, dim, double>
    operator()(const unsigned int &q,
               const double & /*t*/       = 0,
               const Point<dim> & /*x_q*/ = Point<dim>()) override;

  private:
    /// Solution vector.
    const LinAlg::MPI::Vector &sol;

    /// Selected component.
    const unsigned int component;

    /// Used for pre-allocation.
    Tensor<1, dim, double> grad_q;
  };

  /// @brief Evaluate the divergence of a vector solution given as a FE vector.
  class QuadratureFEMDivergence : public QuadratureEvaluationFEMScalar
  {
  public:
    /**
     * Constructor.
     * @param[in] sol_        Solution vector, with ghost entries.
     * @param[in] dof_handler Dof handler (vectorial).
     * @param[in] quadrature_ Quadrature rule to evaluate divergence at.
     */
    QuadratureFEMDivergence(const LinAlg::MPI::Vector &sol_,
                            const DoFHandler<dim> &    dof_handler,
                            const Quadrature<dim> &    quadrature_);

    /// Call operator.
    virtual double
    operator()(const unsigned int &q,
               const double & /*t*/       = 0,
               const Point<dim> & /*x_q*/ = Point<dim>()) override;

  private:
    /// Solution vector.
    const LinAlg::MPI::Vector &sol;

    /// Used for pre-allocation.
    double div_q;
  };
} // namespace lifex

#endif /* __LIFEX_QUADRATURE_EVALUATION_HPP_ */
