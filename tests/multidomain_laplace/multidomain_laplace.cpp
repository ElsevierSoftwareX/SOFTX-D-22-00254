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

#include "source/numerics/preconditioner_handler.hpp"

#include "tests/multidomain_laplace/multidomain_laplace.hpp"

#include <iomanip>
#include <set>

namespace lifex::tests
{
  TestMultidomainLaplace::DirichletData::DirichletData(
    const std::vector<double> &dirichlet_data_)
    : utils::FunctionDirichlet(dirichlet_data_.size())
    , dirichlet_data(dirichlet_data_)
  {}

  double
  TestMultidomainLaplace::DirichletData::value(
    const Point<dim> & /*p*/,
    const unsigned int component) const
  {
    return dirichlet_data[component];
  }

  void
  TestMultidomainLaplace::DirichletData::vector_value(
    const Point<dim> & /*p*/,
    Vector<double> &value) const
  {
    for (unsigned int i = 0; i < n_components; ++i)
      value[i] = dirichlet_data[i];
  }

  TestMultidomainLaplace::ExactSolution::ExactSolution(
    const unsigned int &n_components,
    std::vector<double> dirichlet_data_left_,
    std::vector<double> dirichlet_data_right_)
    : Function<dim>(n_components)
    , dirichlet_data_left(dirichlet_data_left_)
    , dirichlet_data_right(dirichlet_data_right_)
  {}

  double
  TestMultidomainLaplace::ExactSolution::value(
    const Point<dim> & p,
    const unsigned int component) const
  {
    double t = (p[0] + 0.5) / 2.0;
    return (1 - t) * dirichlet_data_left[component] +
           t * dirichlet_data_right[component];
  }

  void
  TestMultidomainLaplace::ExactSolution::vector_value(
    const Point<dim> &p,
    Vector<double> &  value) const
  {
    double t = (p[0] + 0.5) / 2.0;
    for (unsigned int i = 0; i < n_components; ++i)
      value[i] = (1 - t) * dirichlet_data_left[i] + t * dirichlet_data_right[i];
  }

  TestMultidomainLaplace::TestMultidomainLaplace(
    const std::string &subsection,
    const std::string &subsection_subproblem_0_,
    const std::string &subsection_subproblem_1_)
    : CoreModel(subsection)
    , interface_tag({{1, 0}})
    , subsection_subproblem_0(subsection_subproblem_0_)
    , subsection_subproblem_1(subsection_subproblem_1_)
    , linear_solver(prm_subsection_path + " / Monolithic / Linear solver",
                    {"GMRES", "BiCGStab"},
                    "GMRES")
    , preconditioner(prm_subsection_path + " / Block preconditioner",
                     system_matrix)
  {}

  void
  TestMultidomainLaplace::declare_parameters(lifex::ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry("Number of components",
                         "2",
                         Patterns::Integer(1),
                         "Number of components of the solution.");

    params.declare_entry(
      "Left Dirichlet data",
      "0.0, 1.0",
      Patterns::List(Patterns::Double()),
      "Dirichlet boundary condition on the left side (x = -0.5).");

    params.declare_entry(
      "Right Dirichlet data",
      "1.0, 0.0",
      Patterns::List(Patterns::Double()),
      "Dirichlet boundary condition on the right side (x = -1.5).");

    params.declare_entry("Transport coefficient",
                         "0, 0, 0",
                         Patterns::List(Patterns::Double(), dim, dim),
                         "Coefficient of the transport term.");

    params.declare_entry("Refinements",
                         "3",
                         Patterns::Integer(0),
                         "Number of refinements of the mesh.");

    params.declare_entry_selection("Scheme",
                                   "Fixed point",
                                   "Fixed point|Monolithic",
                                   "Scheme used for coupling the subdomains.");

    params.enter_subsection("Fixed point");
    {
      params.declare_entry(
        "Interface conditions",
        "Dirichlet, Neumann",
        Patterns::List(Patterns::Selection("Dirichlet|Neumann|Robin"), 2, 2),
        "Interface conditions applied to the subdomains "
        "(Dirichlet|Neumann|Robin).");

      params.enter_subsection("Relaxation");
      params.declare_entry_selection(
        "Scheme",
        utils::FixedPointRelaxation<LinAlg::MPI::Vector>::label,
        utils::FixedPointAcceleration<
          LinAlg::MPI::Vector>::Factory::get_registered_keys_prm(),
        "Scheme to be used for relaxation.");
      params.leave_subsection();

      params.declare_entry(
        "Tolerance",
        "1e-6",
        Patterns::Double(0),
        "Tolerance on the norm of the increment of interface "
        "data from second subdomain, to stop iterations.");

      params.declare_entry("Max iterations",
                           "50",
                           Patterns::Integer(0),
                           "Maximum number of iterations.");

      params.declare_entry(
        "Robin coefficients",
        "1.0, 0.5",
        Patterns::List(Patterns::Double(0)),
        "Coefficients for Robin interface conditions on the two subdomains.");
    }
    params.leave_subsection();

    params.declare_entry(
      "Error tolerance",
      "1e-3",
      Patterns::Double(0),
      "Tolerance on the L2 norm of the error for the test to pass");

    params.leave_subsection_path();

    linear_solver.declare_parameters(params);
    preconditioner.declare_parameters(params);

    TestLaplace tmp_laplace_problem_0(
      0,
      1,
      Tensor<1, dim>(),
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_0);
    tmp_laplace_problem_0.declare_parameters(params);

    TestLaplace tmp_laplace_problem_1(
      0,
      1,
      Tensor<1, dim>(),
      std::vector<utils::BC<utils::FunctionDirichlet>>(),
      subsection_subproblem_1);
    tmp_laplace_problem_1.declare_parameters(params);

    utils::FixedPointAcceleration<LinAlg::MPI::Vector>::Factory::
      declare_children_parameters(params,
                                  prm_subsection_path +
                                    " / Fixed point / Relaxation");
  }

  void
  TestMultidomainLaplace::parse_parameters(lifex::ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);

    prm_n_components = params.get_integer("Number of components");

    prm_dirichlet_data_left = params.get_vector<double>("Left Dirichlet data");
    prm_dirichlet_data_right =
      params.get_vector<double>("Right Dirichlet data");

    AssertThrow(
      prm_dirichlet_data_left.size() == prm_n_components,
      ExcMessage(
        "Incorrect number of Dirichlet values specified for the left side."));
    AssertThrow(
      prm_dirichlet_data_right.size() == prm_n_components,
      ExcMessage(
        "Incorrect number of Dirichlet values specified for the right side."));

    exact_solution = std::make_unique<ExactSolution>(prm_n_components,
                                                     prm_dirichlet_data_left,
                                                     prm_dirichlet_data_right);

    for (unsigned int i = 0; i < dim; ++i)
      prm_transport_coefficient[i] =
        params.get_vector<double>("Transport coefficient")[i];

    prm_n_refinements = params.get_integer("Refinements");

    prm_scheme = params.get("Scheme");

    params.enter_subsection("Fixed point");
    {
      const std::vector<std::string> &interface_type_str =
        params.get_vector<std::string>("Interface conditions");

      params.enter_subsection("Relaxation");
      prm_fixed_point_acceleration_type = params.get("Scheme");
      params.leave_subsection();

      prm_tolerance  = params.get_double("Tolerance");
      prm_n_max_iter = params.get_integer("Max iterations");

      const std::vector<double> &robin_coefficients =
        params.get_vector<double>("Robin coefficients");

      for (unsigned int i = 0; i < 2; ++i)
        {
          if (interface_type_str[i] == "Dirichlet")
            prm_fixed_point_ICs[i] = InterfaceType::Dirichlet;
          else if (interface_type_str[i] == "Neumann")
            prm_fixed_point_ICs[i] = InterfaceType::Neumann;
          else if (interface_type_str[i] == "Robin")
            prm_fixed_point_ICs[i] = InterfaceType::Robin;

          prm_robin_coefficient[i] = robin_coefficients[i];
        }
    }
    params.leave_subsection();

    prm_error_tolerance = params.get_double("Error tolerance");

    params.leave_subsection_path();

    linear_solver.parse_parameters(params);
    preconditioner.parse_parameters(params);

    {
      std::vector<utils::BC<utils::FunctionDirichlet>> dirichlet_bcs;
      dirichlet_bcs.emplace_back(
        0, std::make_shared<DirichletData>(prm_dirichlet_data_left));
      subproblems[0] = std::make_unique<TestLaplace>(0,
                                                     prm_n_components,
                                                     prm_transport_coefficient,
                                                     dirichlet_bcs,
                                                     subsection_subproblem_0);
      subproblems[0]->parse_parameters(params);
    }

    {
      std::vector<utils::BC<utils::FunctionDirichlet>> dirichlet_bcs;
      dirichlet_bcs.emplace_back(
        1, std::make_shared<DirichletData>(prm_dirichlet_data_right));
      subproblems[1] = std::make_unique<TestLaplace>(1,
                                                     prm_n_components,
                                                     prm_transport_coefficient,
                                                     dirichlet_bcs,
                                                     subsection_subproblem_1);
      subproblems[1]->parse_parameters(params);
    }

    fixed_point_acceleration =
      utils::FixedPointAcceleration<LinAlg::MPI::Vector>::Factory::
        parse_child_parameters(params,
                               prm_fixed_point_acceleration_type,
                               prm_subsection_path +
                                 " / Fixed point / Relaxation");
  }

  void
  TestMultidomainLaplace::run()
  {
    create_mesh(prm_n_refinements);
    setup_system();

    pcout << utils::log::separator_section << "\nSetting up interface maps"
          << std::endl;

    setup_interface();

    pcout << utils::log::separator_section << std::endl;

    if (prm_scheme == "Fixed point")
      {
        n_iter = 0;

        // Interface data vectors are always Trilinos vectors, regardless of the
        // chosen linear algebra backend, because we need them to be not
        // contiguous across processes, and Trilinos vectors allow that whereas
        // PETSc and deal.II distributed vectors do not.
        LinAlgTrilinos::Wrappers::MPI::Vector residual_dirichlet(
          owned_interface_dofs, mpi_comm);
        LinAlgTrilinos::Wrappers::MPI::Vector residual_neumann(
          owned_interface_dofs, mpi_comm);

        fixed_point_acceleration->reset(subproblems[1]->solution_owned);

        do
          {
            subproblems[0]->step();
            subproblems[0]->compute_interface_residual();

            subproblems[1]->step();
            subproblems[1]->set_solution(
              fixed_point_acceleration->get_next_element(
                subproblems[1]->solution_owned));
            subproblems[1]->compute_interface_residual();

            subproblems[0]->output_results(n_iter);
            subproblems[1]->output_results(n_iter);

            res_norm = utils::compute_interface_residual_norm(
              owned_interface_dofs,
              subproblems[0]->solution,
              subproblems[0]->residual_no_interface,
              *interface_maps[0],
              subproblems[1]->solution,
              subproblems[1]->residual_no_interface,
              *interface_maps[1],
              weight_dirichlet,
              weight_neumann,
              prm_fixed_point_ICs[0] == InterfaceType::Robin ?
                interface_mass_matrix :
                nullptr);

            compute_error();

            pcout << "Iteration " << std::setw(3) << n_iter
                  << "\tresidual = " << std::setprecision(8) << std::setw(13)
                  << res_norm << "\tL2 error = " << std::setprecision(8)
                  << std::setw(13) << error << std::endl;

            ++n_iter;
          }
        while (n_iter < prm_n_max_iter && res_norm > prm_tolerance);
      }
    else // if (prm_scheme == "Monolithic")
      {
        setup_system_monolithic();
        assemble_monolithic();
        solve_monolithic();

        subproblems[0]->output_results(0);
        subproblems[1]->output_results(0);

        compute_error();

        pcout << "L2 error = " << std::setprecision(8) << std::setw(13) << error
              << std::endl;
      }

    AssertThrow(error < prm_error_tolerance, ExcTestFailed());
  }

  void
  TestMultidomainLaplace::create_mesh(const unsigned int &n_refinements)
  {
    subproblems[0]->create_mesh(n_refinements);
    subproblems[1]->create_mesh(n_refinements);
  }

  void
  TestMultidomainLaplace::setup_system()
  {
    subproblems[0]->setup_system();
    subproblems[1]->setup_system();
  }

  void
  TestMultidomainLaplace::setup_interface()
  {
    {
      TimerOutput::Scope timer_section(timer_output,
                                       prm_subsection_path +
                                         " / Setup interface maps");

      std::tie(interface_maps, owned_interface_dofs, relevant_interface_dofs) =
        utils::compute_interface_maps(subproblems[0]->dof_handler,
                                      subproblems[1]->dof_handler,
                                      interface_tag,
                                      {});

      pcout << "Number of interface DoFs: " << owned_interface_dofs.size()
            << std::endl;
    }

    if (prm_scheme == "Fixed point")
      {
        TimerOutput::Scope timer_section(timer_output,
                                         prm_subsection_path +
                                           " / Setup interface handlers");

        for (unsigned int subdomain = 0; subdomain < 2; ++subdomain)
          {
            const unsigned int other_subdomain = (subdomain == 0 ? 1 : 0);

            // Vectors of interface data.
            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::Vector>::InterfaceDataDirichlet>>
              interface_data_dirichlet;

            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::Vector>::InterfaceDataNeumann>>
              interface_data_neumann;

            std::vector<std::shared_ptr<utils::InterfaceHandler<
              LinAlg::MPI::Vector>::InterfaceDataRobinLinear>>
              interface_data_robin;

            if (prm_fixed_point_ICs[subdomain] == InterfaceType::Dirichlet)
              interface_data_dirichlet.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::Vector>::InterfaceDataDirichlet>(
                  subproblems[other_subdomain]->solution,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain]));

            if (prm_fixed_point_ICs[subdomain] == InterfaceType::Neumann)
              interface_data_neumann.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::Vector>::InterfaceDataNeumann>(
                  subproblems[other_subdomain]->residual_no_interface,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain]));

            if (prm_fixed_point_ICs[subdomain] == InterfaceType::Robin)
              interface_data_robin.push_back(
                std::make_shared<utils::InterfaceHandler<
                  LinAlg::MPI::Vector>::InterfaceDataRobinLinear>(
                  subproblems[other_subdomain]->residual_no_interface,
                  subproblems[other_subdomain]->solution,
                  owned_interface_dofs,
                  relevant_interface_dofs,
                  interface_maps[subdomain],
                  interface_maps[other_subdomain],
                  std::set<types::boundary_id>({{interface_tag[subdomain]}}),
                  prm_robin_coefficient[subdomain],
                  ComponentMask(prm_n_components, true)));

            subproblems[subdomain]->setup_interface(
              subproblems[other_subdomain]->dof_handler,
              interface_data_dirichlet,
              interface_data_neumann,
              interface_data_robin);
          }

        if (prm_fixed_point_ICs[0] == InterfaceType::Dirichlet)
          {
            weight_dirichlet = 1.0;
            weight_neumann   = 0.0;
          }
        else if (prm_fixed_point_ICs[0] == InterfaceType::Neumann)
          {
            weight_dirichlet = 0.0;
            weight_neumann   = 1.0;
          }
        else if (prm_fixed_point_ICs[0] == InterfaceType::Robin)
          {
            weight_dirichlet = prm_robin_coefficient[0];
            weight_neumann   = 1.0;

            // For Robin conditions, we use the mass matrix to weigh the
            // Dirichlet residual.
            interface_mass_matrix =
              std::make_shared<LinAlgTrilinos::Wrappers::SparseMatrix>();
            compute_interface_mass_matrix(*interface_maps[0],
                                          owned_interface_dofs,
                                          relevant_interface_dofs,
                                          subproblems[0]->dof_handler,
                                          {interface_tag[0]},
                                          *interface_mass_matrix);
          }
      }
  }

  void
  TestMultidomainLaplace::compute_error()
  {
    Vector<double> difference_per_cell;

    const auto quadrature_error_0 =
      subproblems[0]->triangulation.get_quadrature_gauss(
        subproblems[0]->fe->degree + 2);

    VectorTools::integrate_difference(subproblems[0]->dof_handler,
                                      subproblems[0]->solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      *quadrature_error_0,
                                      VectorTools::L2_norm);
    double error_0 =
      VectorTools::compute_global_error(subproblems[0]->triangulation.get(),
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    const auto quadrature_error_1 =
      subproblems[1]->triangulation.get_quadrature_gauss(
        subproblems[1]->fe->degree + 2);

    VectorTools::integrate_difference(subproblems[1]->dof_handler,
                                      subproblems[1]->solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      *quadrature_error_1,
                                      VectorTools::L2_norm);
    double error_1 =
      VectorTools::compute_global_error(subproblems[1]->triangulation.get(),
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    error = std::sqrt(error_0 * error_0 + error_1 * error_1);
  }

  void
  TestMultidomainLaplace::assemble_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Assemble monolithic system");

    system_matrix = 0.0;
    system_rhs    = 0.0;

    for (unsigned int k = 0; k < 2; ++k)
      {
        const FiniteElement<dim> &fe            = *(subproblems[k]->fe);
        const DoFHandler<dim> &   dof_handler   = subproblems[k]->dof_handler;
        const unsigned int        dofs_per_cell = fe.dofs_per_cell;

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            subproblems[k]->assemble_cell(cell, cell_matrix, cell_rhs);
            cell->get_dof_indices(local_dof_indices);

            // DoF indices for the second subdomain must be shifted.
            if (k == 1)
              {
                for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                  local_dof_indices[i] += n_dofs[0];
              }

            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void
  TestMultidomainLaplace::setup_system_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Setup monolithic system");

    // Count DoFs on the two subdomains.
    n_dofs.resize(2);
    n_dofs[0] = subproblems[0]->dof_handler.n_dofs();
    n_dofs[1] = subproblems[1]->dof_handler.n_dofs();

    const unsigned int n_total_dofs = n_dofs[0] + n_dofs[1];

    // Construct index sets of owned and relevant DoFs.
    block_owned_dofs.resize(2);
    block_owned_dofs[0] = subproblems[0]->owned_dofs;
    block_owned_dofs[1] = subproblems[1]->owned_dofs;

    block_relevant_dofs.resize(2);
    block_relevant_dofs[0] = subproblems[0]->relevant_dofs;
    block_relevant_dofs[1] = subproblems[1]->relevant_dofs;

    for (const auto &dof : *interface_maps[0])
      block_relevant_dofs[1].add_index(dof.second.get_other_subdomain_index());

    for (const auto &dof : *interface_maps[1])
      block_relevant_dofs[0].add_index(dof.second.get_other_subdomain_index());

    owned_dofs_global.set_size(n_total_dofs);
    relevant_dofs_global.set_size(n_total_dofs);
    unsigned int offset = 0;
    for (unsigned int i = 0; i < 2; ++i)
      {
        owned_dofs_global.add_indices(block_owned_dofs[i], offset);
        relevant_dofs_global.add_indices(block_relevant_dofs[i], offset);
        offset += n_dofs[i];
      }

    /// Constructs constraints for the monolithic system.
    AffineConstraints<double> constraints_0(block_relevant_dofs[0]);
    subproblems[0]->bc_handler.apply_dirichlet(constraints_0);
    constraints_0.close();

    AffineConstraints<double> constraints_1(block_relevant_dofs[1]);
    subproblems[1]->bc_handler.apply_dirichlet(constraints_1);
    constraints_1.close();

    std::tie(constraints, std::ignore) =
      utils::make_interface_constraints(constraints_0,
                                        constraints_1,
                                        *interface_maps[0],
                                        *interface_maps[1],
                                        relevant_dofs_global,
                                        n_dofs[0]);
    constraints.close();

    // Construct the sparsity pattern.
    BlockDynamicSparsityPattern dsp;
    utils::make_interface_sparsity_pattern(subproblems[0]->dof_handler,
                                           subproblems[1]->dof_handler,
                                           {n_dofs[0]},
                                           {n_dofs[1]},
                                           constraints,
                                           dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs_global,
                                               mpi_comm,
                                               relevant_dofs_global);

    // Initialize the matrices and vectors.
    utils::initialize_matrix(system_matrix, block_owned_dofs, dsp);
    system_rhs.reinit(block_owned_dofs, mpi_comm);
    solution_owned.reinit(block_owned_dofs, mpi_comm);
    solution.reinit(block_owned_dofs, block_relevant_dofs, mpi_comm);
  }

  void
  TestMultidomainLaplace::solve_monolithic()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     prm_subsection_path +
                                       " / Solve monolithic system");

    // Construct the preconditioner.
    preconditioner.initialize(system_matrix);

    // Solve.
    linear_solver.solve(system_matrix,
                        solution_owned,
                        system_rhs,
                        preconditioner);
    constraints.distribute(solution_owned);
    solution = solution_owned;

    // Copy the solution to the subproblems.
    subproblems[0]->solution = solution.block(0);
    subproblems[1]->solution = solution.block(1);
  }
} // namespace lifex::tests
