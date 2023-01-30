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

#include "source/init.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
  using namespace lifex;

  std::string
  copy_line(std::ifstream &file_in, std::ofstream &file_out)
  {
    std::string line;

    std::getline(file_in, line);
    file_out << line << std::endl;

    return line;
  }

  void
  convert_mesh(const std::string &filename_in, const std::string &filename_out)
  {
    std::ifstream file_in(filename_in, std::ifstream::in);
    std::ofstream file_out(filename_out, std::ifstream::out);

    AssertThrow(file_in.is_open(), ExcFileNotOpen(filename_in));
    AssertThrow(file_in.peek() != EOF, ExcMessage(filename_in + " is empty."));
    AssertThrow(file_out.is_open(), ExcFileNotOpen(filename_out));

    char delimiter = ' ';

    std::string       line;
    std::stringstream line_str;

    // First determine the file format.
    {
      std::string version, file_type, data_size;

      // $MeshFormat
      copy_line(file_in, file_out);

      // 2.2 0 8
      line_str = std::stringstream(copy_line(file_in, file_out));

      std::getline(line_str, version, delimiter);
      std::getline(line_str, file_type, delimiter);
      std::getline(line_str, data_size, delimiter);

      AssertThrow(version == "2.2",
                  ExcMessage("Input mesh is not in format 2.2."));
    }

    // $EndMeshFormat
    copy_line(file_in, file_out);

    // $Nodes
    line = copy_line(file_in, file_out);

    AssertThrow(line == "$Nodes",
                ExcMessage("$Nodes block expected in input file."));

    line = copy_line(file_in, file_out);

    unsigned int n_nodes = std::stoi(line);

    // Copy nodes.
    for (unsigned int i = 0; i < n_nodes; ++i)
      {
        pcout << "Printing node " << i + 1 << " / " << n_nodes << "\r";
        copy_line(file_in, file_out);
      }
    pcout << std::endl;

    // $EndNodes
    copy_line(file_in, file_out);

    // $Elements
    line = copy_line(file_in, file_out);

    AssertThrow(line == "$Elements",
                ExcMessage("$Elements block expected in input file."));

    line = copy_line(file_in, file_out);

    unsigned int n_elems = std::stoi(line);

    // Convert elements.
    for (unsigned int i = 0; i < n_elems; ++i)
      {
        pcout << "Converting element " << i + 1 << " / " << n_elems << "\r";

        std::string elem_id, elem_type, n_tags, phys_tag, elem_tag,
          connectivity;

        file_in >> elem_id >> elem_type >> n_tags >> phys_tag >> elem_tag;

        std::getline(file_in, connectivity);

        file_out << elem_id << " " << elem_type << " " << n_tags << " "
                 << elem_tag << " " << elem_tag << connectivity << std::endl;
      }
    pcout << std::endl;

    // $EndElements
    copy_line(file_in, file_out);

    file_in.close();
    file_out.close();

    pcout << "Done!" << std::endl
          << "Written to " + filename_out << "." << std::endl;
  }
} // namespace

/// Convert elementary entities tags to physical tags in a GMSH v2.2 mesh
/// file.
int
main(int argc, char **argv)
{
  // Declare and parse additional command-line options.
  std::string filename_in;
  std::string filename_out;

  const clipp::group cli =
    (clipp::opt_value("input file", filename_in) % ("Input filename.") &
     clipp::value("output file", filename_out) % ("Output filename."));

  lifex::lifex_init lifex_initializer(argc, argv, 1, false, cli);

  try
    {
      convert_mesh(filename_in, filename_out);
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
