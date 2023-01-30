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

#ifndef LIFEX_UTILS_DATA_WRITER_HPP_
#define LIFEX_UTILS_DATA_WRITER_HPP_

#include "source/core.hpp"

#include <string>

namespace lifex::utils
{
  /// Convert timestep index to a string of length
  /// @ref Core::output_n_digits, with leading zeros.
  inline std::string
  timestep_to_string(const unsigned int &timestep)
  {
    return Utilities::int_to_string(timestep, Core::output_n_digits);
  }

  /**
   * @brief Write the content of a DataOut object in the HDF5/XDMF format.
   *
   * @param[in] data_out The object whose content is to be written.
   * @param[in] basename The basename of the output file, relative to directory and
   * without extension. The file will be suffixed with the timestep index and
   * with the extensions .h5 and .xdmf when writing the HDF5 and XDMF files.
   * @param[in] filter_duplicate_vertices Filter duplicate vertices and associated values. To be disabled if cellwise data are exported.
   */
  template <class DataOutType>
  void
  dataout_write_hdf5(const DataOutType &data_out,
                     const std::string &basename,
                     const bool &       filter_duplicate_vertices = true);

  /**
   * @brief Write the content of a DataOut object in HDF5/XDMF format.
   *
   * @param[in] data_out The object whose content is to be written.
   * @param[in] basename The basename of the output file, relative to directory and
   * without extension. The file will be suffixed with the timestep index and
   * with the extensions .h5 and .xdmf when writing the HDF5 and XDMF files.
   * @param[in] time_step The index of the timestep to be saved.
   * @param[in] time_step_mesh The index of the timeste where the mesh is saved.
   * If equal to time_step, the mesh will be saved in the output file.
   * @param[in] time The time to be stored in the XDMF entry.
   * @param[in] filter_duplicate_vertices Filter duplicate vertices and associated values. To be disabled if cellwise data are exported.
   */
  template <class DataOutType>
  void
  dataout_write_hdf5(const DataOutType & data_out,
                     const std::string & basename,
                     const unsigned int &time_step,
                     const unsigned int &time_step_mesh,
                     const double &      time,
                     const bool &        filter_duplicate_vertices = true);
} // namespace lifex::utils

#endif /* LIFEX_UTILS_DATA_WRITER_HPP_ */
