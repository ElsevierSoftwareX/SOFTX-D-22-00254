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
 * @author Ivan Fumagalli <ivan.fumagalli@polimi.it>.
 */

#ifndef LIFEX_UTILS_BC_HANDLER_HPP_
#define LIFEX_UTILS_BC_HANDLER_HPP_

#include "source/core.hpp"

#include <memory>
#include <tuple>
#include <variant>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Class representing an analytical function for Dirichlet boundary conditions.
   *
   * This is essentially a @dealii <kbd>Function</kbd>, but with a
   * different type to prevent wrong conversions between a FunctionDirichlet and
   * a FunctionNeumann.
   *
   * @note This class is supposed to be used in conjunction with a @ref BCHandler.
   *
   * @warning In the case on @b flat at boundaries, the user has the possibility
   * to access the normal vector to that boundary, to impose @a e.g. conditions
   * like @f$\mathbf{u} = 5\mathbf{n}@f$ or @f$\mathbf{u} \cdot \mathbf{n} =
   * 5@f$.
   */
  class FunctionDirichlet : public Function<dim>
  {
  public:
    /**
     * Constructor.
     * May take an initial value for the number of components (which defaults to
     * a scalar function), and the time variable (which defaults to zero).
     */
    FunctionDirichlet(const unsigned int n_components = 1,
                      const time_type    initial_time = 0.0)
      : Function<dim>(n_components, initial_time)
    {}

    /**
     * Set normal vector at current boundary.
     *
     * This method is invoked within the @ref BCHandler::set_time
     * method and is able to update the normal in case of a moving mesh.
     */
    virtual void
    set_normal_vector(const Tensor<1, dim, double> &normal_vector_)
    {
      normal_vector = normal_vector_;
    }

    /**
     * Get normal vector at current boundary.
     *
     * @warning If this function is associated to a curved boundary
     * rather than a flat one, the information provided by this method
     * will likely be inaccurate.
     */
    const Tensor<1, dim, double> &
    get_normal_vector() const
    {
      return normal_vector;
    }

    /**
     * Set surface area of the current boundary.
     *
     * This method is invoked within the @ref BCHandler::set_time
     * method and is able to update the normal in case of a moving mesh.
     */
    virtual void
    set_surface_area(const double &surface_area_)
    {
      surface_area = surface_area_;
    }

    /**
     * Get surface area of the current boundary.
     *
     * @warning If this function is associated to a curved boundary
     * rather than a flat one, the information provided by this method
     * will likely be inaccurate.
     */
    const double &
    get_surface_area() const
    {
      return surface_area;
    }

    /**
     * Set barycenter of the current boundary.
     *
     * This method is invoked within the @ref BCHandler::set_time
     * method and is able to update the barycenter in case of a moving mesh.
     */
    virtual void
    set_barycenter(const Point<dim> &barycenter_)
    {
      barycenter = barycenter_;
    }

    /**
     * Get barycenter of the current boundary.
     *
     * @warning If this function is associated to a curved boundary
     * rather than a flat one, the information provided by this method
     * will likely be inaccurate.
     */
    const Point<dim> &
    get_barycenter() const
    {
      return barycenter;
    }

  private:
    Tensor<1, dim, double> normal_vector; ///< Normal vector.
    double                 surface_area;  ///< Surface area.
    Point<dim>             barycenter;    ///< Barycenter.
  };

  /**
   * @brief Class representing an analytical function for Neumann boundary conditions.
   *
   * This is a specialization of @dealii <kbd>Function</kbd>, which can
   * be used for Neumann data depending on the normal vector.
   *
   * @note This class is supposed to be used in conjunction with a @ref BCHandler.
   */
  class FunctionNeumann : public Function<dim>
  {
  public:
    /**
     * Constructor.
     * May take an initial value for the number of components (which defaults to
     * a scalar function), and the time variable (which defaults to zero).
     */
    FunctionNeumann(const unsigned int n_components = 1,
                    const time_type    initial_time = 0.0)
      : Function<dim>(n_components, initial_time)
    {}

    /**
     * Set normal vector at current quadrature node.
     * This method is invoked within the assembly loop used to
     * assemble Neumann contributions (see @ref BCHandler::apply_neumann).
     */
    void
    set_normal_vector(const Tensor<1, dim, double> &normal_vector_)
    {
      normal_vector = normal_vector_;
    }

    /**
     * Get normal vector at current quadrature node.
     */
    const Tensor<1, dim, double> &
    get_normal_vector() const
    {
      return normal_vector;
    }

  private:
    Tensor<1, dim, double> normal_vector; ///< Normal vector.
  };

  /**
   * @brief Specialization for a constant FunctionDirichlet or FunctionNeumann.
   */
  class ConstantBCFunction : public FunctionDirichlet, public FunctionNeumann
  {
  public:
    /**
     * Constructor.
     * Set values of all components to the provided one.
     * The default number of components corresponds to a scalar function.
     */
    ConstantBCFunction(const double &      value,
                       const unsigned int &n_components = 1)
      : FunctionDirichlet(n_components)
      , FunctionNeumann(n_components)
      , v(value)
    {}

    /// Set the constant value returned by the function.
    void
    set_value(const double &new_value_)
    {
      v = new_value_;
    }

    /// Get the constant value returned by the function.
    virtual double
    get_value() const
    {
      return v;
    }

    /// Return the constant value @ref v.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/) const override final
    {
      return v;
    }

  private:
    double v; ///< The constant value.
  };

  /**
   * @brief Specialization for a zero FunctionDirichlet or FunctionNeumann.
   */
  class ZeroBCFunction : public ConstantBCFunction
  {
  public:
    /**
     * Constructor.
     * Set values of all components to zero.
     * The default number of components corresponds to a scalar function.
     */
    ZeroBCFunction(const unsigned int n_components = 1)
      : ConstantBCFunction(0, n_components)
    {}
  };

  /**
   * @brief Helper class used to represent an arbitrary boundary condition.
   *
   * @tparam FunctionType The type of function associated with current boundary condition.
   * The possible values are <kbd>FunctionDirichlet</kbd> and
   * <kbd>FunctionNeumann</kbd>.
   */
  template <class FunctionType>
  class BC
  {
  public:
    /// This class is not default constructible.
    BC() = delete;

    /**
     * Constructor.
     *
     * @param[in] boundary_id_     Tag of the boundary where a Dirichlet/Neumann
     *                             boundary condition is imposed.
     * @param[in] function_        Dirichlet (Neumann) datum to be imposed on
     *                             that boundary.
     * @param[in] component_mask_  Component mask of the solution to which the
     *                             boundary condition is referred to.
     *                             By default all the components are selected.
     *
     * @note Whether a component mask is specified or not,
     * the number of components of @p function_ must match that of the finite element used.
     * If, for example, you are solving the Navier-Stokes equations in 3D
     * with components @f$(u_x, u_y, u_z, p)@f$ but only want to assign a
     * boundary condition on the first 3 components (discarding the pressure),
     * you still need to provide a <kbd>Function</kbd> object that has 4
     * components.
     */
    BC(const types::boundary_id &           boundary_id_,
       const std::shared_ptr<FunctionType> &function_,
       const ComponentMask &                component_mask_ = ComponentMask());

    /// The boundary id.
    types::boundary_id boundary_id;
    /// The function to impose on the boundary.
    std::shared_ptr<FunctionType> function;
    /// The component mask the function is referred to.
    ComponentMask component_mask;
  };

  /**
   * @brief Class handling Dirichlet, Neumann, normal and tangential flux boundary conditions.
   *
   * This class is used to impose Dirichlet, Neumann, normal and tangential flux
   * boundary conditions.
   *
   * In case of boundaries with shared dofs, among two possible boundary
   * conditions, the one that will be imposed on those dofs is the first being
   * set in the corresponding BC vector.
   *
   * Consider for example a computational domain @f$ \Omega
   * \in \mathbb R^d @f$ with two adjacent Dirichlet boundaries @f$ \Gamma_A \in
   * \mathbb R^{d-1} @f$ and @f$ \Gamma_B \in \mathbb R^{d-1} @f$ (hence they
   * share some dofs). Dirichlet boundary conditions are prescribed: @f$ u = g_A
   * \text{ on } \Gamma_A @f$, @f$ u = g_B \text{ on } \Gamma_B @f$ and stored
   * in the BC objects <kbd>bc_A</kbd> and <kbd>bc_B</kbd>.
   * If @ref dirichlet_bcs is constructed as <kbd>{bc_A, bc_B}</kbd>,
   * then the datum @f$ g_A @f$ will be prescribed on the shared dofs.
   * The same will hold for the other types of boundary conditions
   * handled by this class.
   */
  class BCHandler : public Core
  {
  public:
    /**
     * Constructor.
     */
    BCHandler(const DoFHandler<dim> &dof_handler_);

    /**
     * Set boundary conditions and update their geometry info
     * (see @ref update_bc_geometry_info).
     *
     * @param[in] dirichlet_bcs_ Vector of BC containing Dirichlet boundary conditions (ID, function, component mask).
     * @param[in] neumann_bcs_   Vector of BC containing Neumann boundary conditions (ID, function, component mask).
     * @param[in] face_quadrature_formula A quadrature formula to be used for Neumann boundary conditions.
     * @param[in] dirichlet_normal_flux_bcs_     Vector of BC containing normal flux boundary conditions (ID, function, component mask).
     * @param[in] dirichlet_tangential_flux_bcs_ Vector of BC containing tangential flux boundary conditions (ID, function, component mask).
     *
     * @warning The functions in <kbd>dirichlet_normal_flux_bcs_</kbd> and
     * in <kbd>dirichlet_tangential_flux_bcs_</kbd> need to have exactly @ref dim components.
     */
    void
    initialize(
      const std::vector<BC<FunctionDirichlet>> &dirichlet_bcs_ =
        std::vector<BC<FunctionDirichlet>>(),
      const std::vector<BC<FunctionNeumann>> &neumann_bcs_ =
        std::vector<BC<FunctionNeumann>>(),
      const Quadrature<dim - 1> &face_quadrature_formula =
        Quadrature<dim - 1>(),
      const std::vector<BC<FunctionDirichlet>> &dirichlet_normal_flux_bcs_ =
        std::vector<BC<FunctionDirichlet>>(),
      const std::vector<BC<FunctionDirichlet>> &dirichlet_tangential_flux_bcs_ =
        std::vector<BC<FunctionDirichlet>>());

    /// Call @a set_time on all the private members.
    /// If <kbd>update_geometry_info</kbd> is <kbd>true</kbd>, also updates
    /// normal vector, surface area and barycenter
    /// for Dirichlet and flux boundaries.
    void
    set_time(const double &t, const bool &update_geometry_info);

    /**
     * Build constraints where Dirichlet boundary conditions are applied.
     *
     * The parameter <kbd>homogeneous_constraints</kbd> can be used for
     * non-linear problems linearized by means of a Newton method: at each
     * timestep we usually set homogeneous Dirichlet BCs for the solution
     * increment, whereas the initial guess
     * is initialized by imposing the proper BCs (as specified in @ref dirichlet_bcs)
     * in correspondence of Dirichlet degrees of freedom.
     *
     * @param[in, out] constraints             The affine constraints.
     * @param[in]      homogeneous_constraints If <kbd>true</kbd>, set
     *                                         homogeneous Dirichlet BCs instead
     *                                         of the functions specified in @ref dirichlet_bcs.
     *
     * @note The <kbd>constraints</kbd> object is assumed to be initialized
     * with the locally relevant dofs and will be returned in a non-closed
     * status. Consider calling <kbd>constraints.close()</kbd> after invoking
     * this method.
     */
    void
    apply_dirichlet(AffineConstraints<double> &constraints,
                    const bool &               homogeneous_constraints = false);

    /**
     * Apply Dirichlet BCs to a vector.
     *
     * This function loops over the elements of @ref dirichlet_bcs
     * (in reverse order, as explained in the main documentation of this class),
     * extracts the corresponding boundary dofs, interpolates the corresponding
     * function and imposes the interpolated values on <kbd>vector_owned</kbd>.
     *
     * The non-ghosted <kbd>vector</kbd> is used to get the relevant dofs
     * in order to communicate interpolated values among the different parallel
     * processes. It will be returned with the proper boundary values set on
     * both the locally owned and relevant dofs.
     *
     * @param[in, out] vector_owned The non-ghosted vector where Dirichlet boundary conditions have to be applied.
     * @param[in, out] vector       The ghosted vector where Dirichlet boundary conditions have to be applied.
     * @param[in]      reinit       Bool to specify whether vectors and boundary dofs have to re-initialized. This is usually set to <kbd>true</kbd> at the first timestep iteration or whenever the mesh connectivity changes.
     *
     * @tparam VectorType Type of distributed vector.
     */
    template <class VectorType = LinAlg::MPI::Vector>
    void
    apply_dirichlet(VectorType &vector_owned,
                    VectorType &vector,
                    const bool &reinit);

    /**
     * Assemble Neumann BCs.
     *
     * This function imposes Neumann boundary conditions of the type
     * @f[
     * \nabla u \cdot \mathbf{n} = g_i, \quad \text{on } \Gamma_i^\mathrm{N},
     * @f]
     * or, for a more general operator with natural Neumann term
     * @f$\boldsymbol{\sigma}(\mathbf{u})@f$,
     * @f[
     * \boldsymbol{\sigma}(\mathbf{u}) \mathbf{n} = \mathbf{g}_i, \quad \text{on
     * } \Gamma_i^\mathrm{N},
     * @f]
     * where @f$\Gamma_i^\mathrm{N}@f$ and @f$g_i@f$ are the Neumann boundary
     * (with outward normal @f$\mathbf{n}@f$) and the Neumann datum
     * corresponding to the @f$i@f$-th element in @ref neumann_bcs, respectively.
     *
     * The @f$i@f$-th integral contribution for a given FE test function
     * @f$\varphi_k@f$ is approximated through a given @f$N_q@f$-points
     * quadrature formula
     * (represented by @p face_quadrature_formula,
     * with nodes @f$\left\{\mathbf{x}_q\right\}_{q=1}^{N_q}@f$
     * and weights @f$\left\{\omega_q\right\}_{q=1}^{N_q}@f$) as follows:
     * @f[
     * \int_\mathrm{\Gamma_i^\mathrm{N}} g_i(\mathbf{x}) \varphi_k(\mathbf{x})
     * \mathrm{d}\,\mathbf{x} \approx \sum_{q=1}^{N_q} g_i(\mathrm{x}_q)
     * \varphi_k(\mathbf{x}_q) \omega_q,
     * @f]
     * and @b subtracted from the @f$k@f$-th element of <kbd>cell_rhs</kbd>.
     *
     * @param[in, out] cell_rhs                The cell right-hand side vector where to apply Neumann boundary conditions.
     * @param[in]      cell                    Iterator to the present cell.
     */
    void
    apply_neumann(Vector<double> &                             cell_rhs,
                  const DoFHandler<dim>::active_cell_iterator &cell);

    /**
     * Same as @ref apply_dirichlet(AffineConstraints<double> &, const bool &)
     * "apply_dirichlet", but for normal flux conditions of the type
     * @f$\mathbf{u}\cdot\mathbf{n} = \mathbf{u}_\Gamma\cdot\mathbf{n}@f$.
     */
    void
    apply_dirichlet_normal_flux(AffineConstraints<double> &constraints,
                                const bool &homogeneous_constraints = false);

    /**
     * Same as @ref apply_dirichlet(AffineConstraints<double> &, const bool &)
     * "apply_dirichlet", but for tangential flux conditions of the type
     * @f$\mathbf{u}\times\mathbf{n} = \mathbf{u}_\Gamma\times\mathbf{n}@f$.
     */
    void
    apply_dirichlet_tangential_flux(
      AffineConstraints<double> &constraints,
      const bool &               homogeneous_constraints = false);

    /**
     * Apply Dirichlet or normal/tangential flux constraints to a FE vector.
     *
     * @param[in, out] vector_owned The non-ghosted vector where the constraints have to be imposed.
     * @param[in, out] vector       The ghosted vector where the constraints have to be imposed.
     * @param[in]      constraints  An affine constraint object.
     *
     * @tparam VectorType Type of distributed vector.
     */
    template <class VectorType = LinAlg::MPI::Vector>
    static void
    apply_dirichlet_constraints(VectorType &                     vector_owned,
                                VectorType &                     vector,
                                const AffineConstraints<double> &constraints);

  private:
    /// Update geometry info for a Dirichlet and flux BC boundary.
    void
    update_bc_geometry_info(BC<FunctionDirichlet> &bc);

    const DoFHandler<dim> &dof_handler; ///< Dof handler.

    std::vector<BC<FunctionDirichlet>>
                                     dirichlet_bcs; ///< Vector with Dirichlet BCs.
    std::vector<BC<FunctionNeumann>> neumann_bcs; ///< Vector with Neumann BCs.
    std::vector<BC<FunctionDirichlet>>
      dirichlet_normal_flux_bcs; ///< Vector with normal flux BCs.
    std::vector<BC<FunctionDirichlet>>
      dirichlet_tangential_flux_bcs; ///< Vector with tangential flux BCs.

    /// IndexSet of Dirichlet dofs for each Dirichlet BC.
    std::vector<IndexSet> dirichlet_dofs;

    /// Vector with Dirichlet data, without ghost entries (since C++17).
    /// Pre-allocated here for use in
    /// @ref apply_dirichlet(VectorType &, VectorType &, const bool &)
    /// "apply_dirichlet".
    std::variant<LinAlg::MPI::Vector, LinAlg::MPI::BlockVector>
      vec_dirichlet_owned;

    /// Vector with Dirichlet data (since C++17).
    /// Pre-allocated here for use in
    /// @ref apply_dirichlet(VectorType &, VectorType &, const bool &)
    /// "apply_dirichlet".
    std::variant<LinAlg::MPI::Vector, LinAlg::MPI::BlockVector> vec_dirichlet;

    /// Evaluator for shape functions used when assembling Neumann conditions.
    std::unique_ptr<FEFaceValues<dim>> fe_face_values;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_BC_HANDLER_HPP_ */
