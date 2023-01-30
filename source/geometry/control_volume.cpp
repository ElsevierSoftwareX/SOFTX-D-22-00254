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

#include "source/geometry/control_volume.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex::utils
{
  ControlVolume::QuadratureEvaluation::QuadratureEvaluation(
    const DoFHandler<dim> &dof_handler,
    const Quadrature<dim> &quadrature,
    const ControlVolume &  volume_)
    : QuadratureEvaluationFEMScalar(dof_handler,
                                    quadrature,
                                    update_values | update_quadrature_points)
    , volume(volume_)
  {}

  double
  ControlVolume::QuadratureEvaluation::operator()(const unsigned int &q,
                                                  const double & /*t*/,
                                                  const Point<dim> & /*x_q*/)
  {
    distance_q = 0;

    for (unsigned int i = 0; i < dof_indices.size(); ++i)
      {
        distance_q +=
          volume.get_ghosted()[dof_indices[i]] * fe_values->shape_value(i, q);
      }

    return static_cast<double>(distance_q < 0);
  }

  ControlVolume::ControlVolume(const std::string & subsection,
                               const unsigned int &index_)
    : CoreModel(subsection)
    , index(index_)
  {}

  void
  ControlVolume::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Type",
                         "Sphere",
                         Patterns::Selection("Sphere|File"),
                         "Types of control volumes. Sphere | File.");

    params.declare_entry("Labels",
                         "",
                         Patterns::List(Patterns::Anything()),
                         "Labels used to identify the control volumes");

    params.enter_subsection("Sphere");
    {
      params.declare_entry(
        "Centers",
        "",
        Patterns::List(Patterns::List(Patterns::Double(), dim, dim, " "), 0),
        "Coordinates of the centers of the control volumes [m].");

      params.declare_entry("Radii",
                           "",
                           Patterns::List(Patterns::Double()),
                           "Radii of the control volumes [m].");
    }
    params.leave_subsection();

    params.enter_subsection("File");
    {
      params.declare_entry("Filenames",
                           "",
                           Patterns::List(Patterns::FileName(
                             Patterns::FileName::FileType::input)),
                           "VTP files to read the control volumes from.");
    }
    params.leave_subsection();

    params.leave_subsection_path();
  }

  void
  ControlVolume::parse_parameters(ParamHandler &params)
  {
    params.parse();

    params.enter_subsection_path(prm_subsection_path);

    const unsigned int n_volumes =
      Utilities::split_string_list(params.get("Labels")).size();

    const std::string type_str = params.get("Type");
    if (type_str == "Sphere")
      prm_type = Type::Sphere;
    else // if (type_str == "File")
      prm_type = Type::File;

    prm_label = params.get_vector_item("Labels", index, n_volumes);

    if (prm_type == Type::Sphere)
      {
        params.enter_subsection("Sphere");
        prm_center =
          params.get_vector_item<Point<dim>>("Centers", index, n_volumes);
        prm_radius = params.get_vector_item<double>("Radii", index, n_volumes);
        params.leave_subsection();
      }
    else // if (prm_type == Type::File)
      {
        params.enter_subsection("File");
        prm_filename = params.get_vector_item("Filenames", index, n_volumes);
        params.leave_subsection();
      }

    params.leave_subsection_path();
  }

  void
  ControlVolume::initialize(const std::shared_ptr<Mapping<dim>> &mapping,
                            const DoFHandler<dim> &              dof_handler)
  {
    if (prm_type == Type::File)
      {
        signed_distance =
          std::make_unique<utils::InterpolatedSignedDistance>(prm_filename,
                                                              dof_handler,
                                                              mapping);
      }
    else // if (prm_type == Type::Sphere)
      {
        // Function for signed distance from a sphere.
        class SphereDistanceFunction : public Function<dim>
        {
        public:
          // Constructor.
          SphereDistanceFunction(const Point<dim> &center_,
                                 const double &    radius_)
            : center(center_)
            , radius(radius_)
          {}

          // Evaluation.
          double
          value(const Point<dim> &p,
                const unsigned int /*component*/ = 0) const override
          {
            return (center - p).norm() - radius;
          }

        protected:
          // Center of the sphere.
          Point<dim> center;

          // Radius of the sphere.
          double radius;
        };

        IndexSet owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

        sphere_distance_owned.reinit(owned_dofs, mpi_comm);
        sphere_distance.reinit(owned_dofs, relevant_dofs, mpi_comm);

        if (mapping)
          {
            VectorTools::interpolate(*mapping,
                                     dof_handler,
                                     SphereDistanceFunction(prm_center,
                                                            prm_radius),
                                     sphere_distance_owned);
          }
        else
          {
            VectorTools::interpolate(dof_handler,
                                     SphereDistanceFunction(prm_center,
                                                            prm_radius),
                                     sphere_distance_owned);
          }

        sphere_distance = sphere_distance_owned;
      }

    // We check that the volume is actually inside the mesh by making sure that
    // the interpolated signed distance has at least one positive and one
    // negative element.
    AssertThrow(
      utils::vec_max(get_owned()) > 0 && utils::vec_min(get_owned()) < 0,
      ExcMessage("Control volume " + prm_label + " lies outside the domain."));
  }
} // namespace lifex::utils
