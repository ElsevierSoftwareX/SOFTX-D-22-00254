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

#ifndef LIFEX_UTILS_INTERFACE_HANDLER_HPP_
#define LIFEX_UTILS_INTERFACE_HANDLER_HPP_

#include "source/core.hpp"

#include "source/geometry/finders.hpp"

#include "source/numerics/numbers.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /// Enumeration to define the behavior of @ref make_interface_constraints when
  /// encountering interface DoFs that are already constrained. Suppose that
  /// dof_0 and dof_1 are two corresponding interface DoFs, and that one or both
  /// of them are already constrained by some Dirichlet condition.
  enum class AlreadyConstrainedBehavior
  {
    keep_constraints_0 =
      1 << 0, ///< Keep existing constraints only on subdomain 0: if
              ///< dof_0 is already constrained, keep that constraint
              ///< and add the interface constraint dof_1 = dof_0
              ///< (ignoring possible constraints on dof_1).
    keep_constraints_1 =
      1 << 1, ///< Keep existing constraints only on subdomain 1:
              ///< if dof_1 is already constrained, keep that
              ///< constraint and add the interface constraint dof_0 =
              ///< dof_1 (ignoring possible constraints on dof_0).
    keep_constraints_both =
      keep_constraints_0 |
      keep_constraints_1 ///< Keep existing constraints on both subdomains: if
                         ///< only one DoF is already constrained, keep that
                         ///< constraint and add the interface constraint for
                         ///< the other DoF (e.g. if dof_0 is constrained, add
                         ///< dof_1 = dof_0); if both DoFs are already
                         ///< constrained, keep both constaints and don't add
                         ///< any interface constraint.
  };

  /// Class to store information on an interface DoF.
  class InterfaceDoF
  {
  public:
    /// Default constructor.
    InterfaceDoF() = default;

    /// Constructor.
    InterfaceDoF(const types::global_dof_index &interface_index_,
                 const types::global_dof_index &other_subdomain_index_,
                 const unsigned int &           component_,
                 const unsigned int &           other_component_,
                 const unsigned int &           owner_,
                 const unsigned int &           other_owner_);

    /// Getter for interface index.
    types::global_dof_index
    get_interface_index() const
    {
      return interface_index;
    }

    /// Getter for index in other subdomain.
    types::global_dof_index
    get_other_subdomain_index() const
    {
      return other_subdomain_index;
    }

    /// Getter for component in this subdomain.
    unsigned int
    get_component() const
    {
      return component;
    }

    /// Getter for component in the other subdomain.
    unsigned int
    get_other_component() const
    {
      return other_component;
    }

    /// Get owner in this subdomain.
    unsigned int
    get_owner() const
    {
      return owner;
    }

    /// Get owner in other subdomain.
    unsigned int
    get_other_owner() const
    {
      return other_owner;
    }

    /// Serialization to a Boost Archive.
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &interface_index;
      ar &other_subdomain_index;
      ar &component;
      ar &other_component;
      ar &owner;
      ar &other_owner;
    }

  protected:
    /// Index local to the interface.
    types::global_dof_index interface_index;

    /// Index on the other subdomain.
    types::global_dof_index other_subdomain_index;

    /// Component in this subdomain.
    unsigned int component;

    /// Component on other subdomain.
    unsigned int other_component;

    /// Owner in this subdomain.
    unsigned int owner;

    /// Owner in other subdomain.
    unsigned int other_owner;
  };

  /// Interface map: for every DoF on the interface, maps its global
  /// index onto the corresponding InterfaceDoF.
  using InterfaceMap = std::map<types::global_dof_index, InterfaceDoF>;

  /**
   * @brief Compute the interface maps for DoFs between the two given DoF handlers.
   *
   * Refer to @ref interface_maps and to the general documentation of
   * @ref InterfaceHandler for a description of what interface maps are and what
   * they are used for.
   *
   * @param[in] dof_handler_0   DoF handler of the first domain.
   * @param[in] dof_handler_1   DoF handler of the second domain.
   * @param[in] interface_tags  For each subdomain, the set of boundary tags of the interface.
   * @param[in] component_map   A map that contains pairs of components that
   * should be associated on the two domains. For instance, the pair {0, 2}
   * means that the component 0 on subdomain 0 must be associated to the
   * component 2 in subdomain 1. Leaving the map empty is equivalent to passing
   * {{0, 0}, {1, 1}, {2, 2}, ...}}. If the map is not empty, components that
   * are not present are not mapped (e.g. if passing {{0, 2}}, components 0 and
   * 1 on the second subdomain are not mapped).
   *
   * @return A tuple with the following elements: an array of two maps, one for
   * each subdomain, associating a global DoF index on that subdomain to the
   * corresponding interface DoF index; the set of interface DoFs owned by
   * this subdomain; the set of interface DoFs relevant to this subdomain.
   *
   * @note For efficiency, this function runs only minor checks on the
   * conformity of the provided meshes. In particular, it checks that interface
   * DoFs (DoFs on boundaries corresponding to interface_tags) are in the same
   * number. When running in debug, it checks that no duplicate entries exist in
   * the map. No check is done to make sure that corresponding DoFs are located
   * in the same physical point (DoFs are matched on a "nearest point" basis).
   *
   * @note This function is costly, and its cost increases as the number of
   * parallel processes increase. For repeated runs, consider using
   * @ref serialize_interface_maps and @ref deserialize_interface_maps to save
   * some computational time.
   */
  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  compute_interface_maps(
    const DoFHandler<dim> &                            dof_handler_0,
    const DoFHandler<dim> &                            dof_handler_1,
    const std::array<std::set<types::boundary_id>, 2> &interface_tags,
    const std::map<unsigned int, unsigned int> &       component_map = {});

  /**
   * @brief Compute the interface map for DoFs between the two given DoF handlers.
   *
   * Same as the function above, but taking only a single interface tag for each
   * subdomain (for backwards compatibility).
   */
  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  compute_interface_maps(
    const DoFHandler<dim> &                     dof_handler_0,
    const DoFHandler<dim> &                     dof_handler_1,
    const std::array<types::boundary_id, 2> &   interface_tags,
    const std::map<unsigned int, unsigned int> &component_map = {});

  /**
   * @brief Serialize interface maps to a file.
   *
   * Interface maps from all processes are gathered, then rank 0 takes care
   * of writing them to file.
   *
   * @param[in] filename The name of the file to serialize to.
   * @param[in] interface_maps The interface maps, as returned by compute_interface_maps.
   */
  void
  serialize_interface_maps(
    const std::string &                                 filename,
    const std::array<std::shared_ptr<InterfaceMap>, 2> &interface_maps);

  /**
   * @brief Deserialize interface maps from file.
   *
   * Reads in a map as serialized by the corresponding serialize method. Each
   * process reads the map for the whole interface, but then they only keep
   * those DoFs that are either owned by or relevant to that process.
   *
   * @warning Serialization and deserialization of interface maps relies on the
   * fact that both subdomains are partitioned in the same way when serializing
   * and when deserializing (in particular, this is required to read correctly
   * the members owner and other_owner of InterfaceDoF). Therefore, this works
   * as long as deserialization is done with the same number of cores as
   * serialization. Such number of cores is written into the serialization file
   * and checked when deserializing.
   *
   * @param[in] filename The name of the file to deserialize from. Beware that
   * it is not automatically prefixed with the global output directory.
   * @param[in] owned_dofs For each subdomain, the index set of owned degrees of
   * freedom for that subdomain.
   * @param[in] relevant_dofs For each subdomain, the index set of relevant degrees of
   * freedom for that subdomain.
   *
   * @return A tuple with the following elements: an array of two maps, one for
   * each subdomain, associating a global DoF index on that subdomain to the
   * corresponding interface DoF index; the set of interface DoFs owned by
   * this subdomain; the set of interface DoFs relevant to this subdomain.
   */
  std::tuple<std::array<std::shared_ptr<InterfaceMap>, 2>, IndexSet, IndexSet>
  deserialize_interface_maps(const std::string &            filename,
                             const std::array<IndexSet, 2> &owned_dofs,
                             const std::array<IndexSet, 2> &relevant_dofs);


  /**
   * @brief Communicate interface relevant constraints across processes.
   *
   * Consider a problem defined on two subdomains coupled through a continuity
   * condition at a common interface. It is possible that a DoF dof_0 is owned
   * by one process rank_0 on first subdomain, but the corresponding DoF dof_1
   * on second subdomain is owned by some other process rank_1. When
   * constructing interface constraints (see @ref make_interface_constraints),
   * rank_1 will need to be aware of any constraints that exist on dof_0,
   * although dof_0 is neither owned nor relevant for rank_1, if only the second
   * subdomain is considered (as done e.g. by @ref BCHandler::apply_dirichlet).
   * Therefore, in general, rank_1 will not know about constraints of dof_0.
   * This function performs communication between processes so that every
   * process stores the constraints for the DoFs that it needs.
   *
   * @param[in,out] constraints The constraints for the subdomain. Will be modified
   * adding constraints for DoFs that are relevant due to interface conditions.
   * @param[in] interface_map The interface map from current subdomain to the other.
   * @param[in] offset DoF indices for the other subdomain will be shifted by this offset.
   */
  void
  communicate_interface_constraints(AffineConstraints<double> &constraints,
                                    const InterfaceMap &       interface_map,
                                    const unsigned int &       offset = 0);

  /**
   * @brief Add constraints due to interface conditions to existent constraint objects.
   *
   * For two DoFs @f$u_0@f$ and @f$u_1@f$ that correspond to the same physical
   * point on the interface, this function adds the constraints @f$ u_0 =
   * u_1@f$.
   *
   * @param[in,out] constraints, interface_constraints  The constraints objects. One
   * is meant to contain the constraints for both subdomains being coupled, the
   * other is meant to contain only the constraints describing interface
   * continuity conditions (see the return value of @ref make_interface_constraints);
   * for the purpose of this function, however, there is no difference between
   * the two.
   * @param[in] interface_map The interface map.
   * @param[in] subdomain Index of the subdomain the interface map refers to (0 or 1).
   * @param[in] offset Offset applied to DoFs coming from the second subdomain.
   */
  void
  add_interface_constraints(AffineConstraints<double> &constraints,
                            AffineConstraints<double> &interface_constraints,
                            const InterfaceMap &       interface_map,
                            const unsigned int &       subdomain,
                            const unsigned int &       offset);

  /**
   * @brief Make constraints for the monolithic system of a multidomain problem.
   *
   * Given the affine constraints for the individual subdomains, the function
   * creates a new AffineConstraints object that contains both those
   * constraints (the DoF indices of the second subdomain are shifted by an
   * offset), and adds the constraints that couple the two subdomains as
   * described by the interface map (see @ref add_interface_constraints).
   *
   * @param[in] constraints_0, constraints_1  The constraints for the subdomains
   * (without any interface coupling condition).
   * @param[in] interface_map_0, interface_map_1   The interface maps for the two
   * subdomains.
   * @param[in] relevant_dofs     The DoFs that are relevant for the monolithic
   * system (i.e. the index set of relevant DoFs in both subdomains).
   * @param[in] offset            The offset applied to DoF indices from second
   * subdomain; typically, it will correspond to the number of DoFs in the
   * first subdomain.
   * @param[in] already_constrained_behavior Behavior when already constrained interface DoFs are met.
   *
   * @return A pair of two affine constraints objects: the first contains
   * all the constraints obtained by merging constraints_0, constraints_1 and
   * adding the interface constraints; the second contains only the interface
   * constraints.
   */
  std::pair<AffineConstraints<double>, AffineConstraints<double>>
  make_interface_constraints(
    const AffineConstraints<double> & constraints_0,
    const AffineConstraints<double> & constraints_1,
    const InterfaceMap &              interface_map_0,
    const InterfaceMap &              interface_map_1,
    const IndexSet &                  relevant_dofs,
    const unsigned int &              offset,
    const AlreadyConstrainedBehavior &already_constrained_behavior =
      AlreadyConstrainedBehavior::keep_constraints_both);

  /**
   * @brief Make the sparsity pattern of the monolithic system of a multidomain
   * problem.
   *
   * Given the DoF handlers of the two subproblems, constructs the sparsity
   * pattern of the monolithic system.
   *
   * @param[in] dof_handler_0, dof_handler_1  DoF handlers of the two subproblems.
   * @param[in] n_dofs_0, n_dofs_1   for each subproblem, a vector whose entries
   * are the number of DoFs in each block of that subproblem. For example, for a
   * problem with a single block, the vector will have only one element with the
   * number of DoFs of that subproblem. For a problem with two blocks for
   * velocity and pressure, the vector will contain the number of velocity DoFs
   * and the number of pressure DoFs.
   * @param[in] constraints   Constraints for the monolithic problem, as returned
   * by make_interface_constraints.
   * @param[out] dest   Sparsity pattern to be filled.
   */
  void
  make_interface_sparsity_pattern(
    const DoFHandler<dim> &                     dof_handler_0,
    const DoFHandler<dim> &                     dof_handler_1,
    const std::vector<types::global_dof_index> &n_dofs_0,
    const std::vector<types::global_dof_index> &n_dofs_1,
    const AffineConstraints<double> &           constraints,
    BlockDynamicSparsityPattern &               dest);

  /**
   * @brief Handler for interface conditions between conforming triangulations.
   *
   * # Table of contents
   * - @ref interface_handler
   * - @ref interface_maps
   *   - @ref interface_maps_parallel
   * - @ref interface_data
   *   - @ref interface_data_dirichlet
   *   - @ref interface_data_neumann
   *   - @ref interface_data_robin
   *   - @ref interface_data_custom
   * - @ref interface_handler_examples
   *
   * @section interface_handler Interface handler
   *
   * Consider two subdomains @f$\Omega_0@f$ and @f$\Omega_1@f$, sharing a common
   * interface @f$\Sigma@f$, with conforming discretizations. Let @f$u_0@f$ and
   * @f$u_1@f$ be finite element functions defined on the two subdomains,
   * typically representing solutions to differential problems defined on the
   * two subdomains. Suppose that the problem defined on @f$\Omega_0@f$ involves
   * conditions on @f$\Sigma@f$ that depend on @f$u_1@f$, such as (but not
   * limited to) @f$u_0 = u_1@f$ on @f$\Sigma@f$.
   *
   * The InterfaceHandler class manages, for one of the subdomains,
   * from here on @f$\Omega_0@f$, the extraction of interface data (i.e. data on
   * @f$\Sigma@f$) from the other subdomain @f$\Omega_1@f$, and its application
   * as a boundary condition to @f$\Omega_0@f$ on @f$\Sigma@f$.
   *
   * In order to use the @ref InterfaceHandler for some differential problem,
   * the following setup steps must be taken:
   * - the interface maps must be constructed as described in @ref interface_maps;
   * - `std::vector`s of @ref InterfaceData must be constructed (see
   * @ref interface_data for how to initialize them), to specify which kind of
   * interface conditions the problem is under;
   * - the `std::vector`s of @ref InterfaceData are passed to the
   * constructor of the @ref InterfaceHandler instance.
   *
   * This way the interface conditions are set up, and can be used to provide
   * boundary data to the problem in @f$\Omega_0@f$. To do so:
   * - interface data must be *extracted*, i.e. read from the "source" subdomain
   * @f$\Omega_1@f$ and stored in the internal data structures; this is done
   * by calling the @ref InterfaceHandler::extract method before the system
   * assembly;
   * - interface data must be *applied* as a boundary condition: this is done
   * differently according to whether the conditions are essential (Dirichlet)
   * or natural (Neumann or Robin). Essential conditions are applied by calling
   * @ref InterfaceHandler::apply_dirichlet, whenever you would apply any other
   * Dirichlet condition (typically before the assembly loop). Natural
   * conditions must be applied *within* the assembly loop, by calling
   * @ref InterfaceHandler::apply_current_subdomain, and at the end of the
   * assembly loop, by calling @ref InterfaceHandler::apply_other_subdomain.
   *
   * @section interface_maps Interface maps
   *
   * In order to communicate interface data, *interface maps* are needed. These
   * are objects of type @ref InterfaceMap, one for each subdomain @f$\Omega_i@f$,
   * that to each degree of freedom of @f$\Omega_i@f$ that lies on @f$\Sigma@f$
   * associates an index (the *interface index*) from 0 to the number of
   * interface degrees of freedom.
   *
   * Corresponding interface degrees of the two subdomains (i.e. degrees of
   * freedom of @f$\Omega_0@f$ and @f$\Omega_1@f$ on @f$\Sigma@f$ that have the
   * same support point) are associated by the respective maps to the same
   * interface index. This way, corresponding DoFs are indirectly associated by
   * the two interface maps.
   *
   * Construction of interface maps is done by calling the function
   * @ref compute_interface_maps, that returns the map for both subdomains.
   *
   * @subsection interface_maps_parallel Parallel simulations
   *
   * When running in parallel, each interface map only contains the DoFs that
   * are relevant to the current process. Moreover, ownership of interface
   * indices is partitioned across processes: an interface index is owned by
   * a process if the corresponding DoF is owned by that process on the first
   * subdomain, and it is relevant to a process if the corresponding DoF on
   * either subdomain is relevant to that process.
   *
   * Two index sets containing locally owned and relevant interface indices are
   * returned by @ref compute_interface_maps, and should be stored and kept in
   * memory, as they are needed for most interface conditions related classes
   * and functions.
   *
   * @section interface_data Interface data
   *
   * The data that is read from one subdomain to be passed as boundary condition
   * to the other is referred to as *interface data*, and is implemented by
   * classes derived from @ref InterfaceData.
   *
   * @ref InterfaceData works *unidirectionally*, meaning that it reads data
   * from one subdomain (say @f$\Omega_1@f$) and passes it to the other
   * (@f$\Omega_0@f$), but not the other way around.
   *
   * Concerning interface data, we define two operations (and function arguments
   * names are consistent with the following):
   * - the *extraction*: reading data from subdomain @f$\Omega_1@f$;
   * - the *application*: using that data to provide boundary conditions to
   * @f$\Omega_0@f$.
   *
   * Internally, @ref InterfaceData stores a vector containing only (locally
   * relevant) interface degrees of freedom. The only operation the base class
   * defines is the *extraction* of interface data, that is the copy of
   * interface degrees of freedom from a finite element vector @f$u_1@f$ defined
   * on the whole subdomain @f$\Omega_1@f$ onto the internal vector.
   *
   * The use of the extracted data for the application of boundary conditions to
   * @f$\Omega_0@f$ is demanded to the derived classes
   * (@ref InterfaceDataDirichlet, @ref InterfaceDataNeumann,
   * @ref InterfaceDataRobinLinear and @ref InterfaceDataRobinNonLinear).
   *
   * @subsection interface_data_dirichlet Dirichlet interface data
   *
   * These represent conditions of the type @f$u_0 = u_1@f$ on @f$\Sigma@f$.
   * They are treated using @dealii's AffineConstraints (i.e. they are imposed
   * strongly).
   *
   * To construct @ref InterfaceDataDirichlet instances, you must provide a
   * reference to the finite element vector representing @f$u_1@f$ (the data we
   * want to read from), as well as the interface maps and the index sets for
   * interface indices, all returned from @ref compute_interface_maps. Optionally,
   * you can provide a ComponentMask that specifies which components the
   * interface condition is applied to.
   *
   * @subsection interface_data_neumann Neumann interface data
   *
   * Let @f$\sigma(u)@f$ be the flow, or stress, associated to the natural
   * conditions for the problem at hand. Neumann interface data corresponds to
   * conditions of the type @f$\sigma(u_0)\mathbf{n} = \sigma(u_1)\mathbf{n}@f$
   * on @f$\Sigma@f$.
   *
   * @subsubsection interface_data_neumann_residual Evaluation of Neumann data
   *
   * For stability and accuracy reasons, @f$\sigma(u_1)\mathbf{n}@f$ should not
   * be evaluated directly in strong form, but rather in residual form. To do
   * so, you must assemble in some way the *interface residual* of the
   * differential system defined on domain @f$\Omega_1@f$, that is the residual
   * of the linear system associated to the problem but removing all conditions
   * on @f$\Sigma@f$.
   *
   * The obtained residual is then passed upon construction to the
   * @ref InterfaceDataNeumann instance, together with interface maps and index
   * sets.
   *
   * @subsubsection interface_data_neumann_app Application of Neumann conditions
   *
   * Neumann interface conditions are applied by calling
   * @ref InterfaceHandler::apply_other_subdomain at the end of the assembly
   * loop. This operation basically adds the interface residual to the residual
   * of the @f$\Omega_0@f$ problem for interface degrees of freedom.
   *
   * @subsection interface_data_robin Robin interface data
   *
   * Robin conditions are a linear combination of Dirichlet and Neumann
   * conditions, i.e. @f$\sigma(u_0)\mathbf{n} + \alpha u_0 =
   * \sigma(u_1)\mathbf{n} + \alpha u_1@f$ on @f$\Sigma@f$. Since the stresses
   * @f$\sigma(u_1)@f$ are still involved, you need to evaluate and provide the
   * interface residual as described in @ref interface_data_neumann_residual.
   *
   * If the problem on @f$\Omega_0@f$ is solved with Newton's method (i.e.
   * through @ref utils::NonLinearSolverHandler), the class to be used is the
   * class @ref InterfaceDataRobinNonLinear. Otherwise, you can use the class
   * @ref InterfaceDataRobinNonLinear.
   *
   * Either way, you need to provide to the interface data class upon
   * construction both the interface residual and the vector @f$u_1@f$, as well
   * as the interface maps and index sets, and the Robin coefficient
   * @f$\alpha@f$.
   *
   * Like Neumann conditions, Robin conditions are natural and must be applied
   * during assembly. However, they entail terms on both the left and right hand
   * sides of the system: therefore, you need to call both
   * @ref InterfaceHandler::apply_current_subdomain within the assembly loop and
   * @ref InterfaceHandler::apply_other_subdomain at the end of the assembly loop.
   *
   * @subsection interface_data_custom Custom interface data
   *
   * You can derive from @ref InterfaceDataDirichlet, @ref InterfaceDataNeumann,
   * @ref InterfaceDataRobinLinear and @ref InterfaceDataRobinNonLinear to
   * create custom interface conditions for your problems.
   *
   * @section interface_handler_examples Examples of usage
   *
   * One typical example of use is in fixed-point iterations between two coupled
   * problems on two domains. In that case, each of the subproblem has its own
   * @ref InterfaceHandler instance, to read data from the other subdomain. This
   * is done iteratively until satisfaction of some convergence criterion.
   * Minimal examples of this are found in @ref tests::TestMultidomainLaplace,
   * @ref examples::ExampleMultidomainHeat, @ref examples::ExampleMultidomainHeatNonLinear
   * and @ref examples::ExampleMultidomainStokes.
   *
   * A simpler use case is the one-way coupling of one problem with another.
   */
  template <class ExtractVectorType, class ApplyVectorType = ExtractVectorType>
  class InterfaceHandler : public Core
  {
  public:
    /**
     * @brief Generic interface data class.
     *
     * Manages the allocation of interface data vectors.
     *
     * @note Interface data vectors are always Trilinos vectors, regardless of
     * the chosen linear algebra backend, because we need them to be not
     * contiguous across processes, and Trilinos vectors allow that whereas
     * PETSc and deal.II distributed vectors do not.
     */
    class InterfaceData
    {
    public:
      /**
       * Constructor. Initializes data vectors.
       *
       * @param[in] n_data_vectors_       Number of interface data vectors to be stored.
       * @param[in] owned                 Owned interface data indices.
       * @param[in] relevant              Relevant interface data indices.
       * @param[in] map_apply_            Map used for the application of interface data, i.e.
       * a map from the subdomain the data is applied to onto the interface.
       * @param[in] map_extract_          Map used for the extraction of interface data, i.e.
       * a map from the subdomain the data is extracted from onto the interface.
       */
      InterfaceData(const unsigned int &                      n_data_vectors_,
                    const IndexSet &                          owned,
                    const IndexSet &                          relevant,
                    const std::shared_ptr<const InterfaceMap> map_apply_,
                    const std::shared_ptr<const InterfaceMap> map_extract_)
        : n_data_vectors(n_data_vectors_)
        , data(n_data_vectors_)
        , data_owned(n_data_vectors_)
        , map_apply(map_apply_)
        , map_extract(map_extract_)
      {
        for (unsigned int i = 0; i < n_data_vectors; ++i)
          {
            data[i].reinit(owned, relevant, Core::mpi_comm);
            data_owned[i].reinit(owned, Core::mpi_comm);
            data[i] = data_owned[i] = 0.0;
          }
      }

      /// Destructor.
      virtual ~InterfaceData() = default;

      /// Return the stored interface vector.
      const LinAlgTrilinos::Wrappers::MPI::Vector &
      get_data_owned(const unsigned int &i) const
      {
        return data_owned[i];
      }

    protected:
      /**
       * @brief Interface data evaluation on quadrature points.
       *
       * @param[in] face_values A FEFaceValues object used for evaluation; must already have been initialized
       * for the current face.
       * @param[in] quadrature The face quadrature that was used to initialize face_values.
       * @param[in] data_index The index of the data vector to be evaluated.
       * @param[in] local_dof_indices Global DoF indices corresponding to the DoFs supported on current face.
       * @param[out] dest A vector that will be filled with the function values at quadrature points. It must
       * have quadrature.size() entries, and such entries must have already been
       * initialized with the proper number of components (i.e.
       * face_values.get_fe().n_components()).
       */
      void
      get_other_function_values(
        const FEFaceValues<dim> &                  face_values,
        const Quadrature<dim - 1> &                quadrature,
        const unsigned int &                       data_index,
        const std::vector<types::global_dof_index> local_dof_indices,
        std::vector<Vector<double>> &              dest) const
      {
        const unsigned int n_q_points    = quadrature.size();
        const unsigned int dofs_per_cell = face_values.get_fe().dofs_per_cell;

        Assert(dest.size() == n_q_points,
               ExcDimensionMismatch(dest.size(), n_q_points));

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            dest[q] = 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  face_values.get_fe().system_to_component_index(i).first;
                const auto interface_idx =
                  map_apply->find(local_dof_indices[i]);

                if (interface_idx != map_apply->end())
                  dest[q](component_i) +=
                    data[data_index]
                        [interface_idx->second.get_interface_index()] *
                    face_values.shape_value(i, q);
              }
          }
      }

      /**
       * @brief Copy interface data from a vector into interface data vector.
       *
       * @param[in] i The index of data vector to copy into.
       * @param[in] src The vector to copy from.
       */
      void
      extract_data(const unsigned int &i, const ExtractVectorType &src)
      {
        for (const auto &dof : *map_extract)
          if (src.locally_owned_elements().is_element(dof.first))
            this->data_owned[i][dof.second.get_interface_index()] =
              src[dof.first];
      }

      /// Compresses owned interface data and copies it into the vector with
      /// ghost entries.
      void
      compress()
      {
        for (unsigned int i = 0; i < n_data_vectors; ++i)
          {
            data_owned[i].compress(VectorOperation::insert);
            data[i] = data_owned[i];
          }
      }

      /// Number of stored interface data vectors.
      unsigned int n_data_vectors;

      /// Vectors containing the interface data.
      std::vector<LinAlgTrilinos::Wrappers::MPI::Vector> data;

      /// Vectors containing the owned interface data.
      std::vector<LinAlgTrilinos::Wrappers::MPI::Vector> data_owned;

      /// Map for the application of interface data.
      std::shared_ptr<const InterfaceMap> map_apply;

      /// Map for the extraction of interface data.
      std::shared_ptr<const InterfaceMap> map_extract;
    };

    /**
     * @brief Dirichlet interface data.
     */
    class InterfaceDataDirichlet : public InterfaceData
    {
    public:
      /**
       * Constructor.
       *
       * @param[in] dirichlet_vector_ The vector from which Dirichlet data will be extracted.
       * @param[in] owned_dofs      Owned interface degrees of freedom.
       * @param[in] relevant_dofs   Relevant interface degrees of freedom.
       * @param[in] map_apply       Map used for the application of interface data, i.e.
       * a map from the subdomain the data is applied to onto the interface.
       * @param[in] map_extract     Map used for the extraction of interface data, i.e.
       * a map from the subdomain the data is extracted from onto the interface.
       * @param[in] mask_           Component mask. The interface condition will be
       * applied only to those components matching the mask.
       */
      InterfaceDataDirichlet(
        const ExtractVectorType &                  dirichlet_vector_,
        const IndexSet &                           owned_dofs,
        const IndexSet &                           relevant_dofs,
        const std::shared_ptr<const InterfaceMap> &map_apply,
        const std::shared_ptr<const InterfaceMap> &map_extract,
        const ComponentMask &                      mask_ = ComponentMask())
        : InterfaceData(1, owned_dofs, relevant_dofs, map_apply, map_extract)
        , dirichlet_vector(dirichlet_vector_)
        , mask(mask_)
      {}

      /// Destructor.
      virtual ~InterfaceDataDirichlet() = default;

      /// Extracts the interface data from given dof_handler.
      /// Copies into the elements of interface data vector the values in the
      /// corresponding positions of dirichlet_vector.
      virtual void
      extract()
      {
        this->extract_data(0, dirichlet_vector);
        this->compress();
      }

      /// Apply Dirichlet conditions to a distributed vector.
      void
      apply(ApplyVectorType &solution_owned,
            ApplyVectorType &solution,
            const bool &     homogeneous = false) const
      {
        for (const auto &dof : *(this->map_apply))
          if (this->mask[dof.second.get_component()])
            {
              if (!homogeneous)
                solution_owned[dof.first] = compute_dirichlet_value(dof);
              else
                solution_owned[dof.first] = 0.0;
            }

        solution_owned.compress(VectorOperation::insert);
        solution = solution_owned;
      }

      /// Apply Dirichlet conditions to DoF map.
      void
      apply(std::map<types::global_dof_index, double> &boundary_values,
            const bool &homogeneous = false) const
      {
        for (const auto &dof : *(this->map_apply))
          if (this->mask[dof.second.get_component()])
            {
              if (!homogeneous)
                boundary_values[dof.first] = compute_dirichlet_value(dof);
              else
                boundary_values[dof.first] = 0.0;
            }
      }

      /// Apply Dirichlet conditions to an AffineConstraints object.
      void
      apply(AffineConstraints<double> &constraints,
            const bool &               homogeneous = false) const
      {
        for (const auto &dof : *(this->map_apply))
          if (this->mask[dof.second.get_component()])
            {
              constraints.add_line(dof.first);
              if (!homogeneous)
                constraints.set_inhomogeneity(dof.first,
                                              compute_dirichlet_value(dof));
            }
      }

    protected:
      /// Compute the Dirichlet value corresponding to a given interface DoF.
      virtual double
      compute_dirichlet_value(const InterfaceMap::value_type &dof) const
      {
        return this->data[0][dof.second.get_interface_index()];
      }

      /// Reference to the vector Dirichlet data is extracted from.
      const ExtractVectorType &dirichlet_vector;

      /// Component mask.
      ComponentMask mask;
    };

    /**
     * @brief Neumann interface data.
     *
     * Implements Neumann-like interface conditions.
     */
    class InterfaceDataNeumann : public InterfaceData
    {
    public:
      /**
       * Constructor.
       *
       * @param[in] other_subdomain_residual_ Reference to the interface residual of the other subdomain.
       * @param[in] owned_dofs Owned interface DoFs.
       * @param[in] relevant_dofs Relevant interface DoFs.
       * @param[in] map_apply Map used for the application of interface data, i.e.
       * a map from the subdomain the data is applied to onto the interface.
       * @param[in] map_extract Map used for the extraction of interface data, i.e.
       * a map from the subdomain the data is extracted from onto the interface.
       */
      InterfaceDataNeumann(
        const ExtractVectorType &                  other_subdomain_residual_,
        const IndexSet &                           owned_dofs,
        const IndexSet &                           relevant_dofs,
        const std::shared_ptr<const InterfaceMap> &map_apply,
        const std::shared_ptr<const InterfaceMap> &map_extract)
        : InterfaceData(1, owned_dofs, relevant_dofs, map_apply, map_extract)
        , other_subdomain_residual(other_subdomain_residual_)
      {}

      /// Destructor.
      virtual ~InterfaceDataNeumann() = default;

      /// Extracts the interface data.
      virtual void
      extract()
      {
        this->extract_data(0, other_subdomain_residual);
        this->compress();
      }

      /// Apply the interface data.
      void
      apply_other_subdomain(ApplyVectorType &                rhs,
                            const AffineConstraints<double> &constraints =
                              AffineConstraints<double>())
      {
        for (const auto &dof : *(this->map_apply))
          {
            if (rhs.locally_owned_elements().is_element(dof.first) &&
                !constraints.is_constrained(dof.first))
              rhs[dof.first] += this->data[0][dof.second.get_interface_index()];
          }
      }

    protected:
      /// Reference to the residual vector on the other subdomain.
      const ExtractVectorType &other_subdomain_residual;
    };

    /**
     * @brief Robin interface data for linear problems.
     *
     * Implements Robin-like interface conditions for linear problems.
     */
    class InterfaceDataRobinLinear : public InterfaceData
    {
    public:
      /**
       * Constructor.
       *
       * @param[in] other_subdomain_residual_ Reference to the interface residual of the other subdomain.
       * @param[in] other_solution_ Reference to the solution vector of the other subdomain.
       * @param[in] owned_dofs Owned interface DoFs.
       * @param[in] relevant_dofs Relevant interface DoFs.
       * @param[in] map_apply Map used for the application of interface data, i.e.
       * a map from the subdomain the data is applied to onto the interface.
       * @param[in] map_extract Map used for the extraction of interface data, i.e.
       * a map from the subdomain the data is extracted from onto the interface.
       * @param[in] current_interface_tags_ Interface tags of the subdomain the data is applied to.
       * @param[in] robin_coefficient_ The coefficient for the Robin condition.
       * @param[in] components_current_subdomain_ Component mask representing the components
       * on current subdomain the Robin condition must be applied to.
       */
      InterfaceDataRobinLinear(
        const ExtractVectorType &                  other_subdomain_residual_,
        const ExtractVectorType &                  other_solution_,
        const IndexSet &                           owned_dofs,
        const IndexSet &                           relevant_dofs,
        const std::shared_ptr<const InterfaceMap> &map_apply,
        const std::shared_ptr<const InterfaceMap> &map_extract,
        const std::set<types::boundary_id> &       current_interface_tags_,
        const double &                             robin_coefficient_,
        const ComponentMask &components_current_subdomain_)
        : InterfaceData(2, owned_dofs, relevant_dofs, map_apply, map_extract)
        , other_subdomain_residual(other_subdomain_residual_)
        , other_solution(other_solution_)
        , current_interface_tags(current_interface_tags_)
        , robin_coefficient(robin_coefficient_)
        , components_current_subdomain(components_current_subdomain_)
      {}

      /// Destructor.
      virtual ~InterfaceDataRobinLinear() = default;

      /// Extracts the interface data from given dof_handler.
      virtual void
      extract()
      {
        this->extract_data(0, other_solution);
        this->extract_data(1, other_subdomain_residual);
        this->compress();
      }

      /// Assembles for one element the terms related to this interface
      /// condition and to variables coming from this subdomain.
      virtual void
      apply_current_subdomain(
        FullMatrix<double> &                         cell_matrix,
        Vector<double> &                             cell_rhs,
        const DoFHandler<dim>::active_cell_iterator &cell,
        const Quadrature<dim - 1> &                  face_quadrature_formula,
        const bool &                                 assemble_jac = true) const
      {
        const FiniteElement<dim> &fe            = cell->get_fe();
        const unsigned int        dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_face_q_points = face_quadrature_formula.size();
        FEFaceValues<dim>  face_values(fe,
                                      face_quadrature_formula,
                                      update_values | update_JxW_values);

        std::vector<Vector<double>> other_solution_loc(
          n_face_q_points, Vector<double>(components_current_subdomain.size()));

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
          {
            if (!cell->face(face_number)->at_boundary() ||
                !contains(current_interface_tags,
                          cell->face(face_number)->boundary_id()))
              continue;

            face_values.reinit(cell, face_number);
            this->get_other_function_values(face_values,
                                            face_quadrature_formula,
                                            0,
                                            local_dof_indices,
                                            other_solution_loc);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                  if (!components_current_subdomain[component_i])
                    continue;

                  if (assemble_jac)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const unsigned int component_j =
                            fe.system_to_component_index(j).first;

                          if (!components_current_subdomain[component_j])
                            continue;

                          if (component_i == component_j)
                            cell_matrix(i, j) += robin_coefficient *
                                                 face_values.shape_value(i, q) *
                                                 face_values.shape_value(j, q) *
                                                 face_values.JxW(q);
                        }
                    }

                  cell_rhs(i) +=
                    robin_coefficient * other_solution_loc[q](component_i) *
                    face_values.shape_value(i, q) * face_values.JxW(q);
                }
          }
      }

      /// Apply the interface data.
      void
      apply_other_subdomain(ApplyVectorType &                rhs,
                            const AffineConstraints<double> &constraints =
                              AffineConstraints<double>()) const
      {
        for (const auto &dof : *(this->map_apply))
          {
            if (rhs.locally_owned_elements().is_element(dof.first) &&
                !constraints.is_constrained(dof.first))
              rhs[dof.first] += this->data[1][dof.second.get_interface_index()];
          }
      }

    protected:
      /// Reference to the residual vector on the other subdomain.
      const ExtractVectorType &other_subdomain_residual;

      /// Solution vector on the other subdomain.
      const ExtractVectorType &other_solution;

      /// Tags of the interface on the current subdomain.
      std::set<types::boundary_id> current_interface_tags;

      double robin_coefficient; ///< Coefficient for the Robin condition.

      /// Components to which the Robin condition is applied.
      ComponentMask components_current_subdomain;
    };

    /**
     * @brief Robin interface data for non-linear problems.
     *
     * Implements Robin-like interface conditions for non-linear problems.
     */
    class InterfaceDataRobinNonLinear : public InterfaceData
    {
    public:
      /**
       * Constructor.
       *
       * @param[in] other_subdomain_residual_ Reference to the interface residual of the other subdomain.
       * @param[in] other_solution_ Reference to the solution vector of the other subdomain.
       * @param[in] current_solution_ Reference to the solution vector of current subdomain.
       * @param[in] owned_dofs Owned interface DoFs.
       * @param[in] relevant_dofs Relevant interface DoFs.
       * @param[in] map_apply Map used for the application of interface data, i.e.
       * a map from the subdomain the data is applied to onto the interface.
       * @param[in] map_extract Map used for the extraction of interface data, i.e.
       * a map from the subdomain the data is extracted from onto the interface.
       * @param[in] current_interface_tags_ Interface tags of the subdomain the data is applied to.
       * @param[in] robin_coefficient_ The coefficient for the Robin condition.
       * @param[in] components_current_subdomain_ Component mask representing the components
       * on current subdomain the Robin condition must be applied to.
       */
      InterfaceDataRobinNonLinear(
        const ExtractVectorType &                  other_subdomain_residual_,
        const ExtractVectorType &                  other_solution_,
        const ApplyVectorType &                    current_solution_,
        const IndexSet &                           owned_dofs,
        const IndexSet &                           relevant_dofs,
        const std::shared_ptr<const InterfaceMap> &map_apply,
        const std::shared_ptr<const InterfaceMap> &map_extract,
        const std::set<types::boundary_id> &       current_interface_tags_,
        const double &                             robin_coefficient_,
        const ComponentMask &components_current_subdomain_)
        : InterfaceData(2, owned_dofs, relevant_dofs, map_apply, map_extract)
        , other_subdomain_residual(other_subdomain_residual_)
        , other_solution(other_solution_)
        , current_solution(current_solution_)
        , current_interface_tags(current_interface_tags_)
        , robin_coefficient(robin_coefficient_)
        , components_current_subdomain(components_current_subdomain_)
      {}

      /// Extracts the interface data from given dof_handler.
      virtual void
      extract()
      {
        this->extract_data(0, other_solution);
        this->extract_data(1, other_subdomain_residual);
        this->compress();
      }

      /// Assembles for one element the terms related to this interface
      /// condition and to variables coming from this subdomain.
      /// Assembles for one element the terms related to this interface
      /// condition and to variables coming from this subdomain.
      virtual void
      apply_current_subdomain(
        FullMatrix<double> &                         cell_matrix,
        Vector<double> &                             cell_rhs,
        const DoFHandler<dim>::active_cell_iterator &cell,
        const Quadrature<dim - 1> &                  face_quadrature_formula,
        const bool &                                 assemble_jac = true) const
      {
        const FiniteElement<dim> &fe            = cell->get_fe();
        const unsigned int        dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_face_q_points = face_quadrature_formula.size();
        FEFaceValues<dim>  face_values(fe,
                                      face_quadrature_formula,
                                      update_values | update_JxW_values);

        std::vector<Vector<double>> current_solution_loc(
          n_face_q_points, Vector<double>(components_current_subdomain.size()));
        std::vector<Vector<double>> other_solution_loc(
          n_face_q_points, Vector<double>(components_current_subdomain.size()));

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
          {
            if (!cell->face(face_number)->at_boundary() ||
                !contains(current_interface_tags,
                          cell->face(face_number)->boundary_id()))
              continue;

            face_values.reinit(cell, face_number);
            face_values.get_function_values(current_solution,
                                            current_solution_loc);
            this->get_other_function_values(face_values,
                                            face_quadrature_formula,
                                            0,
                                            local_dof_indices,
                                            other_solution_loc);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                  if (!components_current_subdomain[component_i])
                    continue;

                  if (assemble_jac)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const unsigned int component_j =
                            fe.system_to_component_index(j).first;

                          if (!components_current_subdomain[component_j])
                            continue;

                          if (component_i == component_j)
                            {
                              cell_matrix(i, j) +=
                                robin_coefficient *
                                face_values.shape_value(i, q) *
                                face_values.shape_value(j, q) *
                                face_values.JxW(q);
                            }
                        }
                    }

                  cell_rhs(i) += robin_coefficient *
                                 (current_solution_loc[q](component_i) -
                                  other_solution_loc[q](component_i)) *
                                 face_values.shape_value(i, q) *
                                 face_values.JxW(q);
                }
          }
      }

      /// Apply the interface data.
      void
      apply_other_subdomain(ApplyVectorType &                rhs,
                            const AffineConstraints<double> &constraints =
                              AffineConstraints<double>()) const
      {
        for (const auto &dof : *(this->map_apply))
          {
            if (rhs.locally_owned_elements().is_element(dof.first) &&
                !constraints.is_constrained(dof.first))
              rhs[dof.first] += this->data[1][dof.second.get_interface_index()];
          }
      }

    protected:
      /// Reference to the residual vector on the other subdomain.
      const ExtractVectorType &other_subdomain_residual;

      /// Solution vector on the other subdomain.
      const ExtractVectorType &other_solution;

      /// Solution vector on current subdomain.
      const ApplyVectorType &current_solution;

      /// Tag of the interface on the current subdomain.
      std::set<types::boundary_id> current_interface_tags;

      double robin_coefficient; ///< Coefficient for the Robin condition.

      /// Components to which the Robin condition is applied.
      ComponentMask components_current_subdomain;
    };

    /// Constructor.
    InterfaceHandler(
      const DoFHandler<dim> &other_dof_handler_,
      const std::vector<std::shared_ptr<InterfaceDataDirichlet>>
        &interfaces_dirichlet_,
      const std::vector<std::shared_ptr<InterfaceDataNeumann>>
        &interfaces_neumann_,
      const std::vector<std::shared_ptr<InterfaceDataRobinLinear>>
        &interfaces_robin_linear_,
      const std::vector<std::shared_ptr<InterfaceDataRobinNonLinear>>
        &interfaces_robin_nonlinear_)
      : other_dof_handler(other_dof_handler_)
      , interfaces_dirichlet(interfaces_dirichlet_)
      , interfaces_neumann(interfaces_neumann_)
      , interfaces_robin_linear(interfaces_robin_linear_)
      , interfaces_robin_nonlinear(interfaces_robin_nonlinear_)
    {}

    /// Apply the specified Dirichlet conditions.
    void
    apply_dirichlet(AffineConstraints<double> &constraints,
                    const bool &               homogeneous = false) const
    {
      TimerOutput::Scope timer_section(timer_output,
                                       "InterfaceHandler / Apply Dirichlet");

      for (const auto &condition : interfaces_dirichlet)
        condition->apply(constraints, homogeneous);
    }

    /// Apply the specified Dirichlet conditions.
    void
    apply_dirichlet(ApplyVectorType &solution_owned,
                    ApplyVectorType &solution,
                    const bool &     homogeneous = false) const
    {
      TimerOutput::Scope timer_section(timer_output,
                                       "InterfaceHandler / Apply Dirichlet");

      for (const auto &condition : interfaces_dirichlet)
        condition->apply(solution_owned, solution, homogeneous);
    }

    /// Apply the specified Dirichlet conditions.
    void
    apply_dirichlet(std::map<types::global_dof_index, double> &boundary_values,
                    const bool &homogeneous = false) const
    {
      TimerOutput::Scope timer_section(timer_output,
                                       "InterfaceHandler / Apply Dirichlet");

      for (const auto &condition : interfaces_dirichlet)
        condition->apply(boundary_values, homogeneous);
    }

    /// Add to the system RHS the contributions related to Neumann/Robin
    /// interface conditions and to the other subdomain.
    void
    apply_other_subdomain(ApplyVectorType &                rhs,
                          const AffineConstraints<double> &constraints =
                            AffineConstraints<double>()) const
    {
      for (const auto &condition : interfaces_neumann)
        condition->apply_other_subdomain(rhs, constraints);

      for (const auto &condition : interfaces_robin_linear)
        condition->apply_other_subdomain(rhs, constraints);

      for (const auto &condition : interfaces_robin_nonlinear)
        condition->apply_other_subdomain(rhs, constraints);
    }

    /// Assemble the contributions related to Robin interface conditions and to
    /// the current subdomain.
    void
    apply_current_subdomain(FullMatrix<double> &cell_matrix,
                            Vector<double> &    cell_rhs,
                            const DoFHandler<dim>::active_cell_iterator &cell,
                            const Quadrature<dim - 1> &face_quadrature_formula,
                            const bool &assemble_jac = true) const
    {
      if (!cell->at_boundary())
        return;

      for (const auto &condition : interfaces_robin_linear)
        condition->apply_current_subdomain(
          cell_matrix, cell_rhs, cell, face_quadrature_formula, assemble_jac);

      for (const auto &condition : interfaces_robin_nonlinear)
        condition->apply_current_subdomain(
          cell_matrix, cell_rhs, cell, face_quadrature_formula, assemble_jac);
    }

    /// Extract all conditions.
    void
    extract()
    {
      TimerOutput::Scope timer_section(timer_output,
                                       "InterfaceHandler / Extract conditions");

      for (const auto &condition : interfaces_dirichlet)
        condition->extract();

      for (auto &condition : interfaces_neumann)
        condition->extract();

      for (auto &condition : interfaces_robin_linear)
        condition->extract();

      for (auto &condition : interfaces_robin_nonlinear)
        condition->extract();
    }

  protected:
    /// DoF handler of the other subdomain, from which interface conditions
    /// are read.
    const DoFHandler<dim> &other_dof_handler;

    /// Dirichlet interface conditions.
    std::vector<std::shared_ptr<InterfaceDataDirichlet>> interfaces_dirichlet;

    /// Neumann interface conditions.
    std::vector<std::shared_ptr<InterfaceDataNeumann>> interfaces_neumann;

    /// Robin interface conditions for linear problems.
    std::vector<std::shared_ptr<InterfaceDataRobinLinear>>
      interfaces_robin_linear;

    /// Robin interface conditions for non-linear problems.
    std::vector<std::shared_ptr<InterfaceDataRobinNonLinear>>
      interfaces_robin_nonlinear;
  };

  /**
   * @brief Compute the norm of the interface residual.
   *
   * Given the solution vectors @f$\mathbf x_i@f$ and interface residuals
   * @f$\mathbf r_i@f$ of two subproblems (@f$i = 0,\,1@f$) coupled at an
   * interface, computes @f$||\alpha_D(x_0 - x_1) + \alpha_N\,(r_0 +
   * r_1)||_2@f$, where @f$\alpha_D@f$ and @f$\alpha_N@f$ are positive weights
   * for the discontinuity in the solution and in its normal derivative
   * respectively. The norm can be used as stopping criterion for fixed-point
   * multidomain iterative schemes (as implemented by @ref utils::InterfaceHandler).
   */
  template <class VectorType0, class VectorType1>
  double
  compute_interface_residual_norm(
    const IndexSet &    owned_interface_dofs,
    const VectorType0 & solution_0,
    const VectorType0 & residual_0,
    const InterfaceMap &interface_map_0,
    const VectorType1 & solution_1,
    const VectorType1 & residual_1,
    const InterfaceMap &interface_map_1,
    const double &      weight_dirichlet,
    const double &      weight_neumann,
    const std::shared_ptr<const LinAlgTrilinos::Wrappers::SparseMatrix>
      &interface_mass_matrix = nullptr)
  {
    LinAlgTrilinos::Wrappers::MPI::Vector residual_dirichlet(
      owned_interface_dofs, Core::mpi_comm);
    LinAlgTrilinos::Wrappers::MPI::Vector residual_neumann(owned_interface_dofs,
                                                           Core::mpi_comm);

    for (const auto &dof : interface_map_0)
      {
        if (solution_0.locally_owned_elements().is_element(dof.first))
          {
            residual_dirichlet(dof.second.get_interface_index()) =
              solution_0(dof.first);
            residual_neumann(dof.second.get_interface_index()) =
              residual_0(dof.first);
          }
      }
    residual_dirichlet.compress(VectorOperation::insert);
    residual_neumann.compress(VectorOperation::insert);

    for (const auto &dof : interface_map_1)
      {
        if (solution_1.locally_owned_elements().is_element(dof.first))
          {
            residual_dirichlet(dof.second.get_interface_index()) -=
              solution_1(dof.first);
            residual_neumann(dof.second.get_interface_index()) +=
              residual_1(dof.first);
          }
      }
    residual_dirichlet.compress(VectorOperation::add);
    residual_neumann.compress(VectorOperation::add);

    residual_dirichlet *= weight_dirichlet;
    residual_neumann *= weight_neumann;

    if (interface_mass_matrix)
      interface_mass_matrix->vmult_add(residual_neumann, residual_dirichlet);
    else
      residual_neumann += residual_dirichlet;

    return residual_neumann.l2_norm();
  }


  /**
   * @brief Compute interface residual norm for Dirichlet data.
   *
   * Given the solution vectors @f$\mathbf x_i@f$ of two subproblems (@f$i =
   * 0,\,1@f$) coupled at an interface, computes @f$||(x_0 - x_1)||_2@f$. The
   * norm can be used as stopping criterion for fixed-point
   * multidomain iterative schemes (as implemented by @ref utils::InterfaceHandler).
   */
  template <class VectorType0, class VectorType1>
  double
  compute_interface_residual_norm_Dirichlet(
    const IndexSet &           owned_interface_dofs,
    const VectorType0 &        solution_0,
    const utils::InterfaceMap &interface_map_0,
    const VectorType1 &        solution_1,
    const utils::InterfaceMap &interface_map_1)
  {
    LinAlgTrilinos::Wrappers::MPI::Vector residual_dirichlet(
      owned_interface_dofs, Core::mpi_comm);

    for (const auto &dof : interface_map_0)
      {
        if (solution_0.locally_owned_elements().is_element(dof.first))
          {
            residual_dirichlet(dof.second.get_interface_index()) =
              solution_0(dof.first);
          }
      }
    residual_dirichlet.compress(VectorOperation::insert);

    for (const auto &dof : interface_map_1)
      {
        if (solution_1.locally_owned_elements().is_element(dof.first))
          {
            residual_dirichlet(dof.second.get_interface_index()) -=
              solution_1(dof.first);
          }
      }
    residual_dirichlet.compress(VectorOperation::add);

    return residual_dirichlet.l2_norm();
  }

  /**
   * @brief Compute interface mass matrix.
   *
   * Let @f$\Omega_0@f$ and @f$\Omega_1@f$ be two domains and @f$\Sigma@f$ be
   * their common, conforming interface. Let @f$N_\Sigma@f$ be the number of
   * DoFs associated to @f$\Sigma@f$. This constructs the interface mass matrix
   * @f$M \in \mathbb{R}^{N_\Sigma\times N_\Sigma}@f$ such that @f$
   * M_{ij} = \int_{\Sigma} \varphi_i \varphi_j d\gamma @f$
   * where @f$\varphi_i@f$ are the basis functions associated to interface
   * degrees of freedom.
   *
   * The matrix is indexed following interface indices (see @ref interface_maps)
   * and distributed according to the parallel distribution interface degrees of
   * freedom (see @ref interface_maps_parallel).
   *
   * @param[in] interface_map_0 the interface map for the first subdomain.
   * @param[in] owned_interface_dofs, relevant_interface_dofs Owned and relevant
   * interface DoFs.
   * @param[in] dof_handler_0 DoFHandler for the first subdomain.
   * @param[in] interface_tags_0 Set of interface tags for the first subdomain.
   * @param[out] matrix The matrix in which the mass matrix will be stored; any
   * previous content (including the sparsity pattern) will be overwritten.
   * @param[in] mask_0 A component mask for the components represented by the
   * interface map.
   *
   * @note We use Trilinos as backend for the matrix, regardless of the general
   * backend, because we need to use non-contiguous partitioning (similar to
   * what we do for @ref InterfaceHandler::InterfaceData).
   */
  void
  compute_interface_mass_matrix(
    const utils::InterfaceMap &             interface_map_0,
    const IndexSet &                        owned_interface_dofs,
    const IndexSet &                        relevant_interface_dofs,
    const DoFHandler<dim> &                 dof_handler_0,
    const std::set<types::boundary_id> &    interface_tags_0,
    LinAlgTrilinos::Wrappers::SparseMatrix &matrix,
    const ComponentMask &                   mask_0 = ComponentMask());
} // namespace lifex::utils

#endif /* LIFEX_UTILS_INTERFACE_HANDLER_HPP_ */
