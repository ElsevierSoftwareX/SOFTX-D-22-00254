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

#include "source/numerics/preconditioner_handler.hpp"

#include <map>

namespace lifex::utils
{
  PreconditionerHandler::PreconditionerHandler(const std::string &subsection,
                                               const bool &       elliptic_)
    : CoreModel(subsection)
    , Subscriptor()
    , initialized(false)
  {
    data.elliptic = elliptic_;
  }

  PreconditionerHandler::PreconditionerHandler(
    const PreconditionerHandler &other)
    : CoreModel(other.prm_subsection_path)
    , Subscriptor()
    , initialized(false)
  {
    data = other.data;
  }

  PreconditionerHandler &
  PreconditionerHandler::operator=(const PreconditionerHandler &other)
  {
    initialized = false;

    data = other.data;

    preconditioner.reset();

    return *this;
  }

  void
  PreconditionerHandler::declare_parameters(ParamHandler &params) const
  {
    params.set_verbosity(VerbosityParam::Full);
    params.enter_subsection_path(prm_subsection_path);
    {
      params.declare_entry_selection("Preconditioner type",
                                     "AMG",
                                     "None | AMG"
                                     " | BlockJacobi "
#if defined(LIN_ALG_TRILINOS)
                                     " | AdditiveSchwarz "
#endif
                                     ,
                                     "Determine which preconditioner to use.");

      params.enter_subsection("AMG parameters");
      declare_parameters_amg(params);
      params.leave_subsection();

      // BlockJacobi has no parameters to declare.

#if defined(LIN_ALG_TRILINOS)
      params.enter_subsection("AdditiveSchwarz parameters");
      declare_parameters_additive_schwarz(params);
      params.leave_subsection();
#endif
    }
    params.leave_subsection_path();
    params.reset_verbosity();
  }

  void
  PreconditionerHandler::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      data.prm_preconditioner_type = params.get("Preconditioner type");

      params.enter_subsection("AMG parameters");
      parse_parameters_amg(params);
      params.leave_subsection();

      // BlockJacobi has no parameters to parse.

#if defined(LIN_ALG_TRILINOS)
      params.enter_subsection("AdditiveSchwarz parameters");
      parse_parameters_additive_schwarz(params);
      params.leave_subsection();
#endif
    }
    params.leave_subsection_path();
  }

  void
  PreconditionerHandler::initialize(const LinAlg::MPI::SparseMatrix &matrix,
                                    const DoFHandler<dim> *dof_handler,
                                    const ComponentMask &  component_mask)
  {
    // Silence compiler warning.
    (void)dof_handler;
    (void)component_mask;

    AssertThrow(!initialized || preconditioner != nullptr, ExcNotInitialized());

    if (data.prm_preconditioner_type == "None")
      {
        create_preconditioner<LinAlg::Wrappers::
#if defined(LIN_ALG_TRILINOS)
                                PreconditionIdentity
#elif defined(LIN_ALG_PETSC)
                                PreconditionNone
#endif
                              >(matrix, data.prm_additional_data_identity);
      }
    else if (data.prm_preconditioner_type == "AMG")
      {
#if defined(LIN_ALG_TRILINOS)
        if (data.prm_constant_modes)
          {
            AssertThrow(dof_handler != nullptr, ExcNotInitialized());

            DoFTools::extract_constant_modes(
              *dof_handler,
              component_mask,
              data.prm_additional_data_amg.constant_modes);
          }
#endif
        create_preconditioner<LinAlg::MPI::PreconditionAMG>(
          matrix, data.prm_additional_data_amg);
      }
    else if (data.prm_preconditioner_type == "BlockJacobi")
      {
        create_preconditioner<LinAlg::Wrappers::PreconditionBlockJacobi>(
          matrix, data.prm_additional_data_block_jacobi);
      }
#if defined(LIN_ALG_TRILINOS)
    else if (data.prm_preconditioner_type == "AdditiveSchwarz")
      {
        if (data.prm_additive_schwarz_inner_solver == "SOR")
          create_preconditioner<LinAlg::Wrappers::PreconditionSOR>(
            matrix, data.prm_additional_data_sor);
        else if (data.prm_additive_schwarz_inner_solver == "SSOR")
          create_preconditioner<LinAlg::Wrappers::PreconditionSSOR>(
            matrix, data.prm_additional_data_ssor);
        else if (data.prm_additive_schwarz_inner_solver == "BlockSOR")
          create_preconditioner<LinAlg::Wrappers::PreconditionBlockSOR>(
            matrix, data.prm_additional_data_block_sor);
        else if (data.prm_additive_schwarz_inner_solver == "BlockSSOR")
          create_preconditioner<LinAlg::Wrappers::PreconditionBlockSSOR>(
            matrix, data.prm_additional_data_block_ssor);
        else if (data.prm_additive_schwarz_inner_solver == "ILU")
          create_preconditioner<LinAlg::Wrappers::PreconditionILU>(
            matrix, data.prm_additional_data_ilu);
        else // if (data.prm_additive_schwarz_inner_solver == "ILUT")
          create_preconditioner<LinAlg::Wrappers::PreconditionILUT>(
            matrix, data.prm_additional_data_ilut);
      }
#endif

    initialized = true;
  }

  void
  PreconditionerHandler::vmult(LinAlg::MPI::Vector &      dst,
                               const LinAlg::MPI::Vector &src) const
  {
    preconditioner->vmult(dst, src);
  }

  void
  PreconditionerHandler::declare_parameters_amg(ParamHandler &params) const
  {
#if defined(LIN_ALG_TRILINOS)
    const std::string smoother_or_coarse_solver_options =
      "Aztec | IFPACK | Jacobi | ML symmetric Gauss-Seidel"
      "| symmetric Gauss-Seidel | ML Gauss-Seidel | Gauss-Seidel"
      "| block Gauss-Seidel | symmetric block Gauss-Seidel | Chebyshev "
      "| MLS | Hiptmair | Amesos-KLU | Amesos-Superlu | Amesos-UMFPACK"
      "| Amesos-Superludist | Amesos-MUMPS | user-defined | SuperLU"
      "| IFPACK-Chebyshev | self | do-nothing | IC | ICT | ILU | ILUT"
      "| Block Chebyshev | IFPACK-Block Chebyshev";

    params.declare_entry("Elliptic",
                         data.elliptic ? "true" : "false",
                         Patterns::Bool(),
                         "Optimization for elliptic problem.");

    params.declare_entry("Higher order elements",
                         "false",
                         Patterns::Bool(),
                         "Optimization for higher order elements.");

    params.declare_entry("N-cycles",
                         "1",
                         Patterns::Integer(0),
                         "Number of multigrid cycles.");

    params.declare_entry("W-cycle",
                         "false",
                         Patterns::Bool(),
                         "Use of W-cycle instead of a V-cycle.");

    params.declare_entry("Aggregation threshold",
                         "1e-4",
                         Patterns::Double(0),
                         "Dropping threshold in aggregation.");

    params.declare_entry("Constant modes",
                         "false",
                         Patterns::Bool(),
                         "Specify the constant modes of the matrix.");

    params.declare_entry("Smoother sweeps",
                         "2",
                         Patterns::Integer(0),
                         "Number of the sweeps of the smoother.");

    params.declare_entry("Smoother overlap",
                         "0",
                         Patterns::Integer(0),
                         "Number of the overlap in the error smoother.");

    params.declare_entry_selection(
      "Smoother type",
      "Chebyshev",
      smoother_or_coarse_solver_options,
      "Determines which smoother to use for the AMG cycle.");

    params.declare_entry_selection(
      "Coarse type",
      "Amesos-KLU",
      smoother_or_coarse_solver_options,
      "Determines which solver to use on the coarsest "
      "level.");

#elif defined(LIN_ALG_PETSC)
    params.declare_entry("Symmetric operator",
                         data.elliptic ? "true" : "false",
                         Patterns::Bool());

    params.declare_entry("Strong threshold", "0.5", Patterns::Double(0.0, 1.0));

    params.declare_entry("Max row sum", "0.9", Patterns::Double(0.0, 1.0));

    params.declare_entry("Aggressive coarsening number of levels",
                         "0",
                         Patterns::Integer(0));
#endif

    params.declare_entry("Output details",
                         "false",
                         Patterns::Bool(),
                         "Flag to print AMG details.");
  }

  void
  PreconditionerHandler::parse_parameters_amg(ParamHandler &params)
  {
#if defined(LIN_ALG_TRILINOS)
    data.prm_additional_data_amg.elliptic = params.get_bool("Elliptic");
    data.prm_additional_data_amg.higher_order_elements =
      params.get_bool("Higher order elements");
    data.prm_additional_data_amg.n_cycles = params.get_integer("N-cycles");
    data.prm_additional_data_amg.w_cycle  = params.get_bool("W-cycle");
    data.prm_additional_data_amg.aggregation_threshold =
      params.get_double("Aggregation threshold");
    data.prm_additional_data_amg.smoother_sweeps =
      params.get_integer("Smoother sweeps");
    data.prm_additional_data_amg.smoother_overlap =
      params.get_integer("Smoother overlap");

    // Store in class members so that they do not go out of scope
    // before data.prm_additional_data_amg is destroyed
    // (Trilinos stores them as const char *).
    data.prm_smoother_type                     = params.get("Smoother type");
    data.prm_additional_data_amg.smoother_type = data.prm_smoother_type.c_str();

    data.prm_coarse_type                     = params.get("Coarse type");
    data.prm_additional_data_amg.coarse_type = data.prm_coarse_type.c_str();

    data.prm_constant_modes = params.get_bool("Constant modes");

#elif defined(LIN_ALG_PETSC)
    data.prm_additional_data_amg.symmetric_operator =
      params.get_bool("Symmetric operator");
    data.prm_additional_data_amg.strong_threshold =
      params.get_double("Strong threshold");
    data.prm_additional_data_amg.max_row_sum = params.get_double("Max row sum");
    data.prm_additional_data_amg.aggressive_coarsening_num_levels =
      params.get_integer("Aggressive coarsening number of levels");

#endif

    data.prm_additional_data_amg.output_details =
      params.get_bool("Output details");
  }

#if defined(LIN_ALG_TRILINOS)
  void
  PreconditionerHandler::declare_parameters_additive_schwarz(
    ParamHandler &params) const
  {
    params.declare_entry_selection(
      "Inner solver",
      "SOR",
      "SOR | SSOR | BlockSOR | BlockSSOR | ILU | ILUT",
      "Solver to be used on subdomains.");

    params.declare_entry("Overlap",
                         "0",
                         Patterns::Integer(0),
                         "Overlap between subdomains.");

    params.enter_subsection("SOR, SSOR, BlockSOR and BlockSSOR");
    params.declare_entry("Block size",
                         "1",
                         Patterns::Integer(1),
                         "Size of the blocks (BlockSOR and BlockSSOR only).");
    params.declare_entry("Omega",
                         "1.0",
                         Patterns::Double(0),
                         "Over-relaxation parameter.");
    params.declare_entry("Min diagonal",
                         "0.0",
                         Patterns::Double(0),
                         "Minimum diagonal value.");
    params.declare_entry("Number of sweeps",
                         "1",
                         Patterns::Integer(1),
                         "Number of sweeps per preconditioner application.");
    params.leave_subsection();

    params.enter_subsection("ILU and ILUT");
    params.declare_entry("Fill", "0", Patterns::Integer(0), "Fill-in level.");
    params.declare_entry("Drop threshold",
                         "0.0",
                         Patterns::Double(0),
                         "Threshold below which values are removed from the "
                         "factorization (ILUT only).");
    params.declare_entry("Absolute threshold",
                         "0.0",
                         Patterns::Double(0),
                         "Absolute threshold.");
    params.declare_entry("Relative threshold",
                         "1.0",
                         Patterns::Double(1.0),
                         "Relative threshold.");
    params.leave_subsection();
  }

  void
  PreconditionerHandler::parse_parameters_additive_schwarz(ParamHandler &params)
  {
    data.prm_additive_schwarz_inner_solver = params.get("Inner solver");

    data.prm_additional_data_sor.overlap        = params.get_integer("Overlap");
    data.prm_additional_data_ssor.overlap       = params.get_integer("Overlap");
    data.prm_additional_data_block_sor.overlap  = params.get_integer("Overlap");
    data.prm_additional_data_block_ssor.overlap = params.get_integer("Overlap");
    data.prm_additional_data_ilu.overlap        = params.get_integer("Overlap");
    data.prm_additional_data_ilut.overlap       = params.get_integer("Overlap");

    params.enter_subsection("SOR, SSOR, BlockSOR and BlockSSOR");
    {
      data.prm_additional_data_block_sor.block_size =
        params.get_integer("Block size");
      data.prm_additional_data_block_ssor.block_size =
        params.get_integer("Block size");

      data.prm_additional_data_sor.omega        = params.get_double("Omega");
      data.prm_additional_data_ssor.omega       = params.get_double("Omega");
      data.prm_additional_data_block_sor.omega  = params.get_double("Omega");
      data.prm_additional_data_block_ssor.omega = params.get_double("Omega");

      data.prm_additional_data_sor.min_diagonal =
        params.get_double("Min diagonal");
      data.prm_additional_data_ssor.min_diagonal =
        params.get_double("Min diagonal");
      data.prm_additional_data_block_sor.min_diagonal =
        params.get_double("Min diagonal");
      data.prm_additional_data_block_ssor.min_diagonal =
        params.get_double("Min diagonal");

      data.prm_additional_data_sor.n_sweeps =
        params.get_integer("Number of sweeps");
      data.prm_additional_data_ssor.n_sweeps =
        params.get_integer("Number of sweeps");
      data.prm_additional_data_block_sor.n_sweeps =
        params.get_integer("Number of sweeps");
      data.prm_additional_data_block_ssor.n_sweeps =
        params.get_integer("Number of sweeps");
    }
    params.leave_subsection();

    params.enter_subsection("ILU and ILUT");
    {
      data.prm_additional_data_ilu.ilu_fill   = params.get_integer("Fill");
      data.prm_additional_data_ilut.ilut_fill = params.get_integer("Fill");
      data.prm_additional_data_ilut.ilut_drop =
        params.get_double("Drop threshold");
      data.prm_additional_data_ilu.ilu_atol =
        params.get_double("Absolute threshold");
      data.prm_additional_data_ilut.ilut_atol =
        params.get_double("Absolute threshold");
      data.prm_additional_data_ilu.ilu_rtol =
        params.get_double("Relative threshold");
      data.prm_additional_data_ilut.ilut_rtol =
        params.get_double("Relative threshold");
    }
    params.leave_subsection();
  }
#endif


  BlockPreconditionerHandler::BlockPreconditionerHandler(
    const std::string &                   subsection,
    const LinAlg::MPI::BlockSparseMatrix &matrix_,
    const bool &                          elliptic)
    : CoreModel(subsection)
    , subsection_diagonal_block(prm_subsection_path +
                                " / Diagonal block parameters")
    , initialized(false)
    , preconditioner_dummy(subsection_diagonal_block, elliptic)
    , matrix(matrix_)
    , prm_block_solver_type("Jacobi")
  {}

  void
  BlockPreconditionerHandler::declare_parameters(ParamHandler &params) const
  {
    // Declare parameters.
    params.set_verbosity(VerbosityParam::Full);
    params.enter_subsection_path(prm_subsection_path);
    {
      params.declare_entry_selection(
        "Block solver type",
        "Jacobi",
        "None|Jacobi|Gauss-Seidel|Gauss-Seidel symmetric");
    }
    params.leave_subsection_path();
    params.reset_verbosity();

    declare_parameters_single_block(params);
  }

  void
  BlockPreconditionerHandler::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
    {
      prm_block_solver_type = params.get("Block solver type");
    }
    params.leave_subsection_path();

    parse_parameters_single_block(params);
  }

  void
  BlockPreconditionerHandler::declare_parameters_single_block(
    ParamHandler &params) const
  {
    preconditioner_dummy.declare_parameters(params);
  }

  void
  BlockPreconditionerHandler::parse_parameters_single_block(
    ParamHandler &params)
  {
    preconditioner_dummy.parse_parameters(params);
  }

  void
  BlockPreconditionerHandler::initialize(
    const LinAlg::MPI::BlockSparseMatrix &matrix_)
  {
    AssertThrow(matrix_.n_block_rows() == matrix_.n_block_cols(),
                ExcDimensionMismatch(matrix_.n_block_rows(),
                                     matrix_.n_block_cols()));

    AssertThrow(!initialized || matrix_.n_block_rows() == n_blocks,
                ExcDimensionMismatch(matrix_.n_block_rows(), n_blocks));

    if (!initialized)
      {
        AssertThrow(matrix.n_block_rows() == matrix.n_block_cols(),
                    ExcDimensionMismatch(matrix.n_block_rows(),
                                         matrix.n_block_cols()));

        AssertThrow(matrix_.n_block_rows() == matrix.n_block_rows(),
                    ExcDimensionMismatch(matrix_.n_block_rows(),
                                         matrix.n_block_rows()));

        n_blocks = matrix_.n_block_rows();
        diagonal_blocks.reserve(n_blocks);

        for (unsigned int i = 0; i < n_blocks; ++i)
          {
            diagonal_blocks.emplace_back(subsection_diagonal_block);

            // Use same parameters for all blocks, as the number of blocks
            // is only known at run-time.
            diagonal_blocks[i] = preconditioner_dummy;
          }
      }

    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        diagonal_blocks[i].initialize(matrix_.block(i, i));
      }

    initialized = true;
  }

  void
  BlockPreconditionerHandler::vmult(LinAlg::MPI::BlockVector &      dst,
                                    const LinAlg::MPI::BlockVector &src) const
  {
    if (prm_block_solver_type == "None")
      {
        dst = src;
      }
    else if (prm_block_solver_type == "Jacobi")
      {
        for (size_t i = 0; i < n_blocks; ++i)
          {
            diagonal_blocks[i].vmult(dst.block(i), src.block(i));
          }
      }
    else
      { // if (prm_block_solver_type == "Gauss-Seidel" ||
        //     prm_block_solver_type == "Gauss-Seidel symmetric")

        // Initialize tmp, if needed.
        if (tmp.size() == 0)
          {
            tmp.reinit(src.block(0));
          }

        for (size_t i = 0; i < n_blocks; ++i)
          {
            // A(i, i) * x(i) = b(i) - \sum_{j < i} A(i, j) * x(j)
            tmp = src.block(i);
            tmp *= -1.0;

            for (size_t j = 0; j < i; ++j)
              {
                matrix.block(i, j).vmult_add(tmp, dst.block(j));
              }

            tmp *= -1.0;

            diagonal_blocks[i].vmult(dst.block(i), tmp);
          }

        if (prm_block_solver_type == "Gauss-Seidel symmetric")
          {
            for (size_t i = n_blocks - 1; i > 0; --i)
              {
                // A(i, i) * x(i) = b(i) - \sum_{j > i} A(i, j) * x(j)
                tmp = src.block(i);
                tmp *= -1.0;

                for (size_t j = n_blocks - 1; j > i; --j)
                  {
                    matrix.block(i, j).vmult_add(tmp, dst.block(j));
                  }

                tmp *= -1.0;

                diagonal_blocks[i].vmult(dst.block(i), tmp);
              }
          }
      }
  }

} // namespace lifex::utils
