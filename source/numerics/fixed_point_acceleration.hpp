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

#ifndef LIFEX_UTILS_FIXED_POINT_ACCELERATION_HPP_
#define LIFEX_UTILS_FIXED_POINT_ACCELERATION_HPP_

#include "source/core_model.hpp"
#include "source/generic_factory.hpp"

#include "source/numerics/numbers.hpp"

#include <deal.II/lac/qr.h>

#include <memory>
#include <string>

namespace lifex::utils
{
  /**
   * @brief Base class for acceleration and relaxation schemes for fixed point
   * iterations.
   *
   * Given a fixed-point problem @f$x = g(x)@f$ (where @f$x@f$ may be a vector),
   * the class provides an interface for relaxation and acceleration schemes
   * that aim at improving the convergence of the fixed-point iterations
   * @f$x^{k+1} = g(x^k)@f$.
   *
   * A class using FixedPointAcceleration can use the declare_parameters and
   * parse_parameters static methods of this class to allow the acceleration
   * scheme to be specified in the parameter file.
   */
  template <class VectorType>
  class FixedPointAcceleration : public CoreModel
  {
  public:
    /// Alias for the factory.
    using Factory =
      GenericFactory<FixedPointAcceleration<VectorType>, const std::string &>;

    /// Constructor.
    FixedPointAcceleration(const std::string &subsection)
      : CoreModel(subsection)
    {}

    /// Virtual destructor.
    virtual ~FixedPointAcceleration() = default;

    /// Get the iterate @f$x^{k+1}@f$, given the evaluation of the fixed
    /// point iteration function @f$g(x^k)@f$.
    virtual VectorType
    get_next_element(const VectorType & /*fun_eval*/) = 0;

    /// Reset the sequence to the given initial guess @f$x^0@f$.
    virtual void
    reset(const VectorType & /*initial_guess*/) = 0;
  };

  /**
   * @brief Relaxation scheme with constant relaxation parameter.
   *
   * Given @f$\omega \in (0,1]@f$, implements the scheme: @f$x^{k+1} = (1 -
   * \omega)x^k + \omega g(x^k)@f$.
   */
  template <class VectorType>
  class FixedPointRelaxation : public FixedPointAcceleration<VectorType>
  {
  public:
    /// Label for this scheme.
    static inline constexpr auto label = "Static relaxation";

    /// Constructor.
    FixedPointRelaxation(const std::string &subsection)
      : FixedPointAcceleration<VectorType>(subsection + " / " + label)
    {}

    /// Get next element from the sequence.
    virtual VectorType
    get_next_element(const VectorType &fun_eval) override
    {
      current *= 1.0 - relaxation_coefficient;
      current.add(relaxation_coefficient, fun_eval);
      return current;
    }

    /// Reset.
    virtual void
    reset(const VectorType &initial_guess) override
    {
      current = initial_guess;
    }

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.set_verbosity(VerbosityParam::Full);
      params.enter_subsection_path(this->prm_subsection_path);
      params.declare_entry("Relaxation coefficient",
                           "0.25",
                           Patterns::Double(0, 1),
                           "Relaxation coefficient.");
      params.leave_subsection_path();
      params.reset_verbosity();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.enter_subsection_path(this->prm_subsection_path);
      relaxation_coefficient = params.get_double("Relaxation coefficient");
      params.leave_subsection_path();
    }

  protected:
    double     relaxation_coefficient; ///< Relaxation coefficient.
    VectorType current;                ///< Current element in the sequence.
  };

  /// Aitken acceleration. Reference: https://doi.org/10.1007/s00466-008-0255-5
  template <class VectorType>
  class AitkenAcceleration : public FixedPointAcceleration<VectorType>
  {
  public:
    /// Label for this scheme.
    static inline constexpr auto label = "Aitken acceleration";

    /// Constructor.
    AitkenAcceleration(const std::string &subsection)
      : FixedPointAcceleration<VectorType>(subsection + " / " + label)
      , iter(0)
    {}

    /// Get the iterate @f$x^{k+1}@f$, given the evaluation of the fixed
    /// point iteration function @f$g(x^k)@f$.
    virtual VectorType
    get_next_element(const VectorType &fun_eval) override
    {
      if (iter <= 1)
        {
          current_relaxation = initial_relaxation;
        }

      // If enough sequence elements have been computed, compute dynamic
      // relaxation coefficient.
      else // if (iter > 1)
        {
          VectorType tmp_vector = fun_eval;
          tmp_vector -= current;
          tmp_vector -= old_fun_eval;
          tmp_vector += old;

          const double tmp_vector_norm_sqr = tmp_vector.norm_sqr();

          if (!utils::is_zero(tmp_vector_norm_sqr, 1e-24))
            {
              old_fun_eval -= old;
              current_relaxation *=
                (-1.0) * (old_fun_eval * tmp_vector) / tmp_vector_norm_sqr;
            }
          else
            current_relaxation = 0.0;
        }

      // Update "old" values with current values.
      old          = current;
      old_fun_eval = fun_eval;
      ++iter;

      // Compute new current value and return it.
      current *= 1.0 - current_relaxation;
      current.add(current_relaxation, fun_eval);
      return current;
    }

    /// Reset the sequence: dynamic relaxation coefficient is reset to the
    /// initial one.
    virtual void
    reset(const VectorType &initial_guess) override
    {
      current = initial_guess;
      iter    = 0;
    }

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.set_verbosity(VerbosityParam::Full);
      params.enter_subsection_path(this->prm_subsection_path);
      params.declare_entry("Initial relaxation",
                           "1.0",
                           Patterns::Double(0, 1),
                           "Relaxation coefficient used for the first two "
                           "fixed-point iterations.");
      params.leave_subsection_path();
      params.reset_verbosity();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.enter_subsection_path(this->prm_subsection_path);
      initial_relaxation = params.get_double("Initial relaxation");
      params.leave_subsection_path();
    }

  protected:
    double       initial_relaxation; ///< Initial relaxation coefficient.
    double       current_relaxation; ///< Current relaxation coefficient.
    unsigned int iter;               ///< Current iteration index.

    VectorType current;      ///< Current iterate.
    VectorType old;          ///< Previous iterate.
    VectorType old_fun_eval; ///< Evaluation of the fixed point iteration
                             ///< function at previous iterate.
  };


  /// Anderson acceleration. Reference: https://doi.org/10.1137/10078356X
  template <class VectorType>
  class AndersonAcceleration : public FixedPointAcceleration<VectorType>
  {
  public:
    /// Label for this scheme.
    static inline constexpr auto label = "Anderson acceleration";

    /// Constructor.
    AndersonAcceleration(const std::string &subsection)
      : FixedPointAcceleration<VectorType>(subsection + " / " + label)
      , k(0)
    {}

    /// Get the iterate @f$x^{k+1}@f$, given the evaluation of the fixed
    /// point iteration function @f$g(x^k)@f$.
    virtual VectorType
    get_next_element(const VectorType &fun_eval) override
    {
      VectorType delta_fk = fk;
      VectorType delta_xk = xk;

      fk = fun_eval;
      fk -= xk;

      if (k > 0)
        {
          delta_fk *= -1;
          delta_fk += fk;

          if (utils::is_zero(delta_fk.l2_norm()))
            {
              xk = fun_eval;
            }
          else
            {
              F_k->append_column(delta_fk);
              if (F_k->size() > m)
                F_k->remove_column();

              Vector<double> gamma(F_k->size());
              Vector<double> y(F_k->size());
              F_k->multiply_with_QT(y, fk);
              F_k->solve(gamma, y);

              xk.add(relaxation, fk);
              F_k->multiply_with_A(delta_fk, gamma);
              xk.add(-relaxation, delta_fk);
              X_k->multiply_with_A(delta_fk, gamma);
              xk -= delta_fk;
            }
        }
      else
        {
          xk = fun_eval;
        }

      delta_xk *= -1;
      delta_xk += xk;

      X_k->append_column(delta_xk);
      if (X_k->size() > m)
        X_k->remove_column();

      ++k;

      return xk;
    }

    /// Reset the sequence.
    virtual void
    reset(const VectorType &initial_guess) override
    {
      k   = 0;
      xk  = initial_guess;
      F_k = std::make_unique<QR<VectorType>>();
      X_k = std::make_unique<QR<VectorType>>();
    }

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override
    {
      params.set_verbosity(VerbosityParam::Full);
      params.enter_subsection_path(this->prm_subsection_path);
      {
        params.declare_entry(
          "Number of steps",
          "3",
          Patterns::Integer(1),
          "Number of previous iterations used to compute the next one.");

        params.declare_entry("Relaxation",
                             "1.0",
                             Patterns::Double(0.0, 1.0),
                             "Relaxation parameter.");
      }
      params.leave_subsection_path();
      params.reset_verbosity();
    }

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override
    {
      params.enter_subsection_path(this->prm_subsection_path);
      m          = params.get_integer("Number of steps");
      relaxation = params.get_double("Relaxation");
      params.leave_subsection_path();
    }

  protected:
    unsigned int k;          ///< Current iteration.
    unsigned int m;          ///< Number of steps.
    double       relaxation; ///< Relaxation coefficient.
    VectorType   xk;         ///< Current iteration.
    VectorType   fk;         ///< Current residual.

    std::unique_ptr<QR<VectorType>>
      F_k; ///< QR factorization of the matrix of residuals.
    std::unique_ptr<QR<VectorType>>
      X_k; ///< QR factorization of the matrix of solution increments.
  };

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Relaxation_MPI_Vector,
    FixedPointAcceleration<LinAlg::MPI::Vector>,
    FixedPointRelaxation<LinAlg::MPI::Vector>,
    FixedPointRelaxation<LinAlg::MPI::Vector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Relaxation_MPI_BlockVector,
    FixedPointAcceleration<LinAlg::MPI::BlockVector>,
    FixedPointRelaxation<LinAlg::MPI::BlockVector>,
    FixedPointRelaxation<LinAlg::MPI::BlockVector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Relaxation_Vector,
    FixedPointAcceleration<Vector<double>>,
    FixedPointRelaxation<Vector<double>>,
    FixedPointRelaxation<Vector<double>>::label,
    const std::string &);


  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Aitken_MPI_Vector,
    FixedPointAcceleration<LinAlg::MPI::Vector>,
    AitkenAcceleration<LinAlg::MPI::Vector>,
    AitkenAcceleration<LinAlg::MPI::Vector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Aitken_MPI_BlockVector,
    FixedPointAcceleration<LinAlg::MPI::BlockVector>,
    AitkenAcceleration<LinAlg::MPI::BlockVector>,
    AitkenAcceleration<LinAlg::MPI::BlockVector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Aitken_Vector,
    FixedPointAcceleration<Vector<double>>,
    AitkenAcceleration<Vector<double>>,
    AitkenAcceleration<Vector<double>>::label,
    const std::string &);


  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Anderson_MPI_Vector,
    FixedPointAcceleration<LinAlg::MPI::Vector>,
    AndersonAcceleration<LinAlg::MPI::Vector>,
    AndersonAcceleration<LinAlg::MPI::Vector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Anderson_MPI_BlockVector,
    FixedPointAcceleration<LinAlg::MPI::BlockVector>,
    AndersonAcceleration<LinAlg::MPI::BlockVector>,
    AndersonAcceleration<LinAlg::MPI::BlockVector>::label,
    const std::string &);

  LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(
    FixedPointAcceleration_From_Anderson_Vector,
    FixedPointAcceleration<Vector<double>>,
    AndersonAcceleration<Vector<double>>,
    AndersonAcceleration<Vector<double>>::label,
    const std::string &);

} // namespace lifex::utils

#endif /* LIFEX_UTILS_FIXED_POINT_ACCELERATION_HPP_ */
