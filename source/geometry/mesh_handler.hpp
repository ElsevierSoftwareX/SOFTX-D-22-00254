/********************************************************************************
  Copyright (C) 2019 - 2023 by the lifex authors.

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

#ifndef LIFEX_UTILS_MESH_HANDLER_HPP_
#define LIFEX_UTILS_MESH_HANDLER_HPP_

#include "source/core_model.hpp"

#include "source/geometry/mesh_info.hpp"

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/reference_cell.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Class to handle mesh creation and refinement.
   *
   * This is a wrapper around @dealii
   * <kbd>parallel::DistributedTriangulationBase</kbd> classes and derived,
   * satisfying the <kbd>MeshType</kbd> concept from @dealii.
   *
   * This class has a twofold function:
   * - the mesh type and parameters can be parsed from file via
   *   @ref declare_parameters and @ref parse_parameters, such as in
   *   @ref tutorials::Tutorial01;
   * - the mesh type and parameters can be hard-coded (or parsed and later
   *   overridden), such as in @ref examples::ExampleLaplace.
   */
  class MeshHandler : public CoreModel
  {
  public:
    /// Enumeration of the possible element types.
    enum class ElementType
    {
      /// Hexahedral elements.
      Hex,

      /// Tetrahedral elements.
      Tet
    };

    /// Enumeration of the possible geometry types.
    enum class GeometryType
    {
      /// Read mesh from file.
      File,

      /// Generate a hypercube.
      ///
      /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
      /// otherwise they are numbered in lexicographical order
      /// (<kbd>x_min: 0</kbd>, <kbd>x_max: 1</kbd>,
      ///  <kbd>y_min: 2</kbd>, ...).
      Hypercube,

      /// Generate a cylinder.
      ///
      /// The cylinder is obtained by extruding a circle of a given radius
      /// centered at @f$(0, 0, 0)@f$ along the @f$z@f$ direction.
      ///
      /// Boundary IDs are numbered as follows:
      /// - bottom cylinder base (at @f$z = 0@f$): 1;
      /// - top cylinder base (at @f$z = L@f$): 2;
      /// - cylinder wall: 0.
      Cylinder,

      /// Generate a channel with cylinder.
      ///
      /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
      /// otherwise they are numbered as follows:
      /// - left boundary: 0;
      /// - right boundary: 1;
      /// - cylinder boundary: 2;
      /// - channel walls: 3.
      ChannelWithCylinder,

      /// Generate a hyper shell.
      ///
      /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
      /// otherwise they are numbered as follows:
      /// - inner boundary: 0;
      /// - outer boundary: 1.
      HyperShell,

      /// Mesh generated using user-defined methods.
      Other
    };

    /// Enumeration of the possible refinement types.
    enum class RefinementType
    {
      /// No refinement.
      None,
      /// Global refinement.
      Global,
      /// Load refinement information from a serialized file.
      Deserialize
    };

    /// @name Requirements to satisfy the <kbd>MeshType</kbd> concept.
    /// @{

    /// Alias to identify cell iterators.
    using cell_iterator =
      parallel::DistributedTriangulationBase<dim>::cell_iterator;

    /// Alias to identify active cell iterators.
    using active_cell_iterator =
      parallel::DistributedTriangulationBase<dim>::active_cell_iterator;

    /// Dimension of the manifold the triangulation lives in.
    static const unsigned int dimension =
      parallel::DistributedTriangulationBase<dim>::dimension;

    /// Spatial dimension the triangulation lives in.
    static const unsigned int space_dimension =
      parallel::DistributedTriangulationBase<dim>::space_dimension;

    /// Get the underlying triangulation object.
    parallel::DistributedTriangulationBase<dim> &
    get_triangulation()
    {
      return *triangulation;
    }

    /// Get the underlying triangulation object, <kbd>const</kbd> version.
    const parallel::DistributedTriangulationBase<dim> &
    get_triangulation() const
    {
      return *triangulation;
    }

    /// Iterator to the first active cell on level @p level.
    active_cell_iterator
    begin_active(const unsigned int level = 0) const
    {
      return triangulation->begin_active(level);
    }

    /// Return the first active cell iterator not on the given level.
    active_cell_iterator
    end_active(const unsigned int level = 0) const
    {
      return triangulation->end_active(level);
    }

    /// @}

    /**
     * @brief Constructor.
     *
     * The parameters are declared in subsection + " / Mesh and space
     * discretization".
     */
    MeshHandler(const std::string &           subsection,
                const MPI_Comm &              mpi_comm_,
                const std::set<GeometryType> &geometry_type_set_ = {
                  GeometryType::File});

    /**
     * @brief Constructor.
     *
     * The parameters are declared in subsection + " / " + subsubsection.
     */
    MeshHandler(const std::string &           subsection,
                const std::string &           subsubsection,
                const MPI_Comm &              mpi_comm_,
                const std::set<GeometryType> &geometry_type_set_ = {
                  GeometryType::File});

    /// Copy constructor.
    ///
    /// The parameters of the mesh handler passed as argument are copied,
    /// whereas the underlying mesh object is not.
    MeshHandler(const MeshHandler &other);

    /// Assignment operator.
    /// The parameters of the mesh handler passed as argument are copied,
    /// whereas the underlying mesh object is not.
    MeshHandler &
    operator=(const MeshHandler &other);

    /// Destructor.
    ~MeshHandler() = default;

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Get the underlying triangulation object.
    /// Alias for @ref get_triangulation.
    parallel::DistributedTriangulationBase<dim> &
    get()
    {
      return this->get_triangulation();
    }

    /// Get the underlying triangulation object, <kbd>const</kbd> version.
    /// Alias for @ref get_triangulation.
    const parallel::DistributedTriangulationBase<dim> &
    get() const
    {
      return this->get_triangulation();
    }

    /// Initialize a coarse mesh from file and set scaling factor.
    void
    initialize_from_file(const std::string &filename_,
                         const double &     scaling_factor_);

    /// Initialize a hypercube.
    /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
    /// otherwise they are numbered in lexicographical order
    /// (<kbd>x_min: 0</kbd>, <kbd>x_max: 1</kbd>,
    ///  <kbd>y_min: 2</kbd>, ...).
    void
    initialize_hypercube(const double &left     = 0,
                         const double &right    = 1,
                         const bool    colorize = true);

    /// Initialize a cylinder.
    ///
    /// The cylinder is obtained by extruding a circle of a given radius
    /// centered at @f$(0, 0, 0)@f$ along the @f$z@f$ direction.
    ///
    /// Boundary IDs are numbered as follows:
    /// - bottom cylinder base (at @f$z = 0@f$): 1;
    /// - top cylinder base (at @f$z = L@f$): 2;
    /// - cylinder wall: 0.
    void
    initialize_cylinder(const double       radius   = 0.01,
                        const double       length   = 0.1,
                        const unsigned int n_slices = 2);

    /// Initialize a channel with cylinder.
    /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
    /// otherwise they are numbered as follows:
    /// - left boundary: 0;
    /// - right boundary: 1;
    /// - cylinder boundary: 2;
    /// - channel walls: 3.
    void
    initialize_channel_with_cylinder(const double shell_region_width = 0.03,
                                     const unsigned int n_shells     = 2,
                                     const double       skewness     = 2.0,
                                     const bool         colorize     = true);

    /// Initialize a hyper shell.
    /// Boundary IDs are set to 0 if @p colorize is <kbd>false</kbd>,
    /// otherwise they are numbered as follows:
    /// - inner boundary: 0;
    /// - outer boundary: 1.
    void
    initialize_hyper_shell(const Point<dim>   center,
                           const double       inner_radius = 0.5,
                           const double       outer_radius = 1.5,
                           const unsigned int n_cells      = 0,
                           const bool         colorize     = true);

    /// Set element type.
    void
    set_element_type(const ElementType &element_type);

    /// Set number of refinements.
    void
    set_refinement_global(const unsigned int &n_refinements_);

    /// Set serialized filename to load refinement information from.
    /// @param[in] filename_to_deserialize_ Filename to load.
    void
    set_refinement_from_file(const std::string &filename_to_deserialize_);


    /// Create coarse mesh and, optionally, refine it.
    /// The second and third parameters toggle mesh smoothing and settings
    /// upon refinement (ignored if @ref Data::element_type is <kbd>Tet</kbd>).
    void
    create_mesh(
      const bool &                                      refine = true,
      const typename Triangulation<dim>::MeshSmoothing &smoothing =
        Triangulation<dim>::none,
      const std::optional<parallel::distributed::Triangulation<dim>::Settings>
        &settings = {});

    /// Getter for @ref mesh_info.
    const MeshInfo &
    get_info() const
    {
      AssertThrow(mesh_info != nullptr, ExcNotInitialized());

      if (!mesh_info->initialized())
        mesh_info->initialize();

      return *mesh_info;
    }

    /// Getter for @ref Data::element_type.
    const ElementType &
    get_element_type() const
    {
      return data.element_type;
    }

    /// Getter for @ref Data::geometry_type.
    const GeometryType &
    get_geometry_type() const
    {
      return data.geometry_type;
    }

    /// Getter for @ref Data::refinement_type.
    const RefinementType &
    get_refinement_type() const
    {
      return data.refinement_type;
    }

    /// Getter for @ref Data::filename.
    const std::string &
    get_filename() const
    {
      AssertThrow(
        data.geometry_type == GeometryType::File,
        ExcMessage(
          "This method can only be invoked for meshes read from file."));

      return data.filename;
    }

    /// Getter for @ref Data::scaling_factor.
    const double &
    get_scaling_factor() const
    {
      return data.scaling_factor;
    }

    /// Getter for @ref Data::n_refinements.
    const unsigned int &
    get_n_refinements() const
    {
      return data.n_refinements;
    }

    /// Getter for @ref Data::filename_to_deserialize.
    const std::string &
    get_refinement_filename() const
    {
      AssertThrow(data.refinement_type == RefinementType::Deserialize,
                  ExcMessage("This method can only be invoked for meshes with "
                             "refinement information deserialized from file."));

      return data.filename_to_deserialize;
    }

    /// Return whether the underlying triangulation only consists of
    /// quadrilateral (2D) or hexahedral (3D) cells.
    bool
    is_hex() const
    {
      return is_hex(*triangulation);
    }

    /// Return whether the input triangulation only consists of
    /// quadrilateral (2D) or hexahedral (3D) cells.
    static bool
    is_hex(const Triangulation<dim> &triangulation)
    {
      const std::vector<ReferenceCell> &reference_cells =
        triangulation.get_reference_cells();

      Assert(reference_cells.size() == 1, ExcLifexNotImplemented());

      return reference_cells[0].is_hyper_cube();
    }

    /// Return whether the underlying triangulation only consists of
    /// triangular (2D) or tetrahedral (3D) cells.
    bool
    is_tet() const
    {
      return is_tet(*triangulation);
    }

    /// Return whether the input triangulation only consists of
    /// triangular (2D) or tetrahedral (3D) cells.
    static bool
    is_tet(const Triangulation<dim> &triangulation)
    {
      const std::vector<ReferenceCell> &reference_cells =
        triangulation.get_reference_cells();

      Assert(reference_cells.size() == 1, ExcLifexNotImplemented());

      return reference_cells[0].is_simplex();
    }

    /// Return the number of faces per cell, depending on the reference cell
    /// type.
    unsigned int
    n_faces_per_cell() const
    {
      const std::vector<ReferenceCell> &reference_cells =
        triangulation->get_reference_cells();

      Assert(reference_cells.size() == 1, ExcLifexNotImplemented());

      return reference_cells[0].n_faces();
    }

    /// Return a polymorphic object representing the standard Lagrange finite
    /// element space on this mesh, depending on the element type.
    std::unique_ptr<FiniteElement<dim>>
    get_fe_lagrange(const unsigned int &degree) const
    {
      return get_fe_lagrange(*triangulation, degree);
    }

    /// Return a polymorphic object representing the standard Lagrange finite
    /// element space on the input mesh, depending on the element type.
    static std::unique_ptr<FiniteElement<dim>>
    get_fe_lagrange(const Triangulation<dim> &triangulation,
                    const unsigned int &      degree)
    {
      if (is_hex(triangulation))
        return std::make_unique<FE_Q<dim>>(degree);
      else // if (is_tet(triangulation))
        return std::make_unique<FE_SimplexP<dim>>(degree);
    }

    /// Return a polymorphic object representing the standard DG finite
    /// element space on this mesh, depending on the element type.
    std::unique_ptr<FiniteElement<dim>>
    get_fe_dg(const unsigned int &degree) const
    {
      return get_fe_dg(*triangulation, degree);
    }

    /// Return a polymorphic object representing the standard DG finite
    /// element space on this mesh, depending on the element type.
    static std::unique_ptr<FiniteElement<dim>>
    get_fe_dg(const Triangulation<dim> &triangulation,
              const unsigned int &      degree)
    {
      if (is_hex(triangulation))
        return std::make_unique<FE_DGQ<dim>>(degree);
      else // if (is_tet(triangulation))
        return std::make_unique<FE_SimplexDGP<dim>>(degree);
    }

    /// Return a polymorphic object representing the standard Gaussian
    /// quadrature formula on this mesh, depending on the element type.
    template <unsigned int dim_quad = dim>
    std::unique_ptr<Quadrature<dim_quad>>
    get_quadrature_gauss(const unsigned int &n_points) const
    {
      return get_quadrature_gauss<dim_quad>(*triangulation, n_points);
    }

    /// Return a polymorphic object representing the standard Gaussian
    /// quadrature formula on the input mesh, depending on the element type.
    template <unsigned int dim_quad = dim>
    static std::unique_ptr<Quadrature<dim_quad>>
    get_quadrature_gauss(const Triangulation<dim> &triangulation,
                         const unsigned int &      n_points)
    {
      if (is_hex(triangulation))
        return std::make_unique<QGauss<dim_quad>>(n_points);
      else // if (is_tet(triangulation))
        return std::make_unique<QGaussSimplex<dim_quad>>(n_points);
    }

    /// Return a polymorphic object representing a quadrature formula defined at
    /// nodal points. Beware that weights of this formula are not initialized,
    /// and as such it should not be used for integration.
    std::unique_ptr<Quadrature<dim>>
    get_quadrature_nodal() const
    {
      return get_quadrature_nodal(*triangulation);
    }

    /// Return a polymorphic object representing a quadrature formula defined at
    /// nodal points. Beware that weights of this formula are not initialized,
    /// and as such it should not be used for integration.
    static std::unique_ptr<Quadrature<dim>>
    get_quadrature_nodal(const Triangulation<dim> &triangulation)
    {
      const std::vector<ReferenceCell> &reference_cells =
        triangulation.get_reference_cells();

      Assert(reference_cells.size() == 1, ExcLifexNotImplemented());

      return std::make_unique<Quadrature<dim>>(
        reference_cells[0].get_nodal_type_quadrature<dim>());
    }

    /// Return a polymorphic object representing the mapping of a given degree
    /// on the input mesh, depending on the element type.
    std::unique_ptr<Mapping<dim>>
    get_mapping(const unsigned int &degree) const
    {
      return get_mapping(*triangulation, degree);
    }

    /// Return a polymorphic object representing the mapping of a given degree
    /// on the input mesh, depending on the element type.
    static std::unique_ptr<Mapping<dim>>
    get_mapping(const Triangulation<dim> &triangulation,
                const unsigned int &      degree)
    {
      if (is_hex(triangulation))
        return std::make_unique<MappingQ<dim>>(degree);
      else // if (is_tet(triangulation))
        {
          static FE_SimplexP<dim> fe(degree);
          return std::make_unique<MappingFE<dim>>(fe);
        }
    }

    /// Return a polymorphic object representing the (bi-/tri-)linear mapping on
    /// the input mesh, depending on the element type.
    std::unique_ptr<Mapping<dim>>
    get_linear_mapping() const
    {
      return get_linear_mapping(*triangulation);
    }

    /// Return a polymorphic object representing the (bi-/tri-)linear mapping on
    /// the input mesh, depending on the element type.
    static std::unique_ptr<Mapping<dim>>
    get_linear_mapping(const Triangulation<dim> &triangulation)
    {
      return get_mapping(triangulation, 1);
    }

  private:
    /// Polymorphic pointer to the underlying parallel distributed triangulation
    /// object.
    std::unique_ptr<parallel::DistributedTriangulationBase<dim>> triangulation;

    /// Mesh information.
    std::unique_ptr<MeshInfo> mesh_info;

    /// @brief Structure to bind parameters read from file and additional data.
    ///
    /// This is used to make all such members trivially copyable at once.
    class Data
    {
    public:
      /// MPI communicator.
      MPI_Comm mpi_comm;

      /// MPI rank.
      unsigned int mpi_rank;

      /// MPI size.
      unsigned int mpi_size;

      /// Actual element type.
      ElementType element_type = ElementType::Hex;

      /**
       * @brief Reading group coefficient.
       *
       * Tet meshes are read in serial and then distributed. The processes are
       * divided in groups, whose size is controlled by this parameter. Only one
       * process per group creates the serial mesh, and then distributes it to
       * all other processes in its group.
       *
       * If this is set to 1, every process reads the mesh from file. This is
       * fast, but potentially memory intensive. Conversely, if this is equal to
       * 0, only one process reads the mesh. This is possibly slower, but
       * uses less memory.
       */
      unsigned int reading_group_size;

      /// Set of allowed geometry types.
      std::set<GeometryType> geometry_type_set;

      /// Map from geometry type to string.
      static inline const std::map<GeometryType, std::string>
        geometry_type_map = {{GeometryType::File, "File"},
                             {GeometryType::Hypercube, "Hypercube"},
                             {GeometryType::Cylinder, "Cylinder"},
                             {GeometryType::ChannelWithCylinder,
                              "Channel with cylinder"}};

      /// Actual geometry type.
      GeometryType geometry_type = GeometryType::Other;

      /// @name Mesh read from file.
      /// @{

      /// Coarse mesh filename.
      std::string filename;

      /// Scaling factor.
      double scaling_factor = 1;

      /// @}


      /// @name Hypercube.
      /// @{

      /// Hypercube left endpoint.
      double hypercube_left = 0.0;

      /// Hypercube right endpoint.
      double hypercube_right = 1.0;

      /// Assign different boundary IDs if set to <kbd>true</kbd>,
      /// as explained in @ref GeometryType::Hypercube.
      bool hypercube_colorize = true;

      /// @}


      /// @name Cylinder.
      /// @{

      /// Radius of the cylinder.
      double cylinder_radius = 0.01;

      /// Length of the cylinder along the @f$z@f$ axis.
      double cylinder_length = 0.1;

      /// Number of slices along the @f$z@f$ axis.
      unsigned int cylinder_n_slices = 3;

      /// @}


      /// @name Channel with cylinder.
      /// @{

      /// Width of the layer of shells around the cylinder.
      double channel_shell_region_width = 0.03;

      /// Number of shells around the cylinder.
      unsigned int channel_n_shells = 2;

      /// Parameter controlling how close the shells are to the cylinder.
      double channel_skewness = 2.0;

      /// Assign different boundary IDs if set to <kbd>true</kbd>,
      /// as explained in @ref GeometryType::ChannelWithCylinder.
      bool channel_colorize = true;

      /// @}


      /// @name Hyper shell.
      /// @{

      /// Center of the shell.
      Point<dim> shell_center;

      /// Radius of the internal shell.
      double shell_inner_radius;

      /// Radius of the external.
      double shell_outer_radius;

      /// Number of cells of the triangulation of the shell.
      unsigned int shell_n_cells = 0;

      /// Assign different boundary IDs if set to <kbd>true</kbd>,
      /// as explained in @ref GeometryType::HyperShell.
      bool shell_colorize = true;

      /// @}


      /// @name Refinement.
      /// @{

      /// Refinement type.
      RefinementType refinement_type = RefinementType::None;

      /// Number of refinements.
      unsigned int n_refinements = 0;

      /// Serialized filename to load refinement information from.
      std::string filename_to_deserialize;

      /// @}
    };

    /// Parameters and additional data.
    Data data;
  };

} // namespace lifex::utils

#endif /* LIFEX_UTILS_MESH_HANDLER_HPP_ */
