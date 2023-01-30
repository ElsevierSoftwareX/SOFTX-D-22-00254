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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#include "source/io/data_writer.hpp"

#include <deal.II/numerics/data_out_faces.h>

#include <vector>

namespace lifex::utils
{
  template <class DataOutType>
  void
  dataout_write_hdf5(const DataOutType &data_out,
                     const std::string &basename,
                     const bool &       filter_duplicate_vertices)
  {
    const std::string filename_h5   = basename + ".h5";
    const std::string filename_xdmf = basename + ".xdmf";

    DataOutBase::DataOutFilter data_filter(
      DataOutBase::DataOutFilterFlags(filter_duplicate_vertices, true));

    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter,
                                 Core::prm_output_directory + filename_h5,
                                 Core::mpi_comm);

    std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
      data_filter, filename_h5, 0.0, Core::mpi_comm)});
    data_out.write_xdmf_file(xdmf_entries,
                             Core::prm_output_directory + filename_xdmf,
                             Core::mpi_comm);
  }

  /// Explicit instantiation.
  template void
  dataout_write_hdf5(const DataOut<dim> &data_out,
                     const std::string & basename,
                     const bool &        filter_duplicate_vertices);

  /// Explicit instantiation.
  template void
  dataout_write_hdf5(const DataOutFaces<dim> &data_out,
                     const std::string &      basename,
                     const bool &             filter_duplicate_vertices);

  template <class DataOutType>
  void
  dataout_write_hdf5(const DataOutType & data_out,
                     const std::string & basename,
                     const unsigned int &time_step,
                     const unsigned int &time_step_mesh,
                     const double &      time,
                     const bool &        filter_duplicate_vertices)
  {
    const bool export_mesh = (time_step == time_step_mesh);

    const std::string filename_h5 =
      basename + "_" + utils::timestep_to_string(time_step) + ".h5";
    const std::string filename_xdmf =
      basename + "_" + utils::timestep_to_string(time_step) + ".xdmf";
    const std::string filename_mesh =
      basename + "_" + utils::timestep_to_string(time_step_mesh) + ".h5";

    DataOutBase::DataOutFilter data_filter(
      DataOutBase::DataOutFilterFlags(filter_duplicate_vertices, true));

    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter,
                                 export_mesh,
                                 Core::prm_output_directory + filename_mesh,
                                 Core::prm_output_directory + filename_h5,
                                 Core::mpi_comm);

    std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
      data_filter, filename_mesh, filename_h5, time, Core::mpi_comm)});

    data_out.write_xdmf_file(xdmf_entries,
                             Core::prm_output_directory + filename_xdmf,
                             Core::mpi_comm);
  }

  /// Explicit instantiation.
  template void
  dataout_write_hdf5(const DataOut<dim> &data_out,
                     const std::string & basename,
                     const unsigned int &time_step,
                     const unsigned int &time_step_mesh,
                     const double &      time,
                     const bool &        filter_duplicate_vertices);

  /// Explicit instantiation.
  template void
  dataout_write_hdf5(const DataOutFaces<dim> &data_out,
                     const std::string &      basename,
                     const unsigned int &     time_step,
                     const unsigned int &     time_step_mesh,
                     const double &           time,
                     const bool &             filter_duplicate_vertices);
} // namespace lifex::utils
