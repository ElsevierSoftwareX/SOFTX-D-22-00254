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
 */

/// @page dealii-documentation deal.II documentation
/// # Table of contents
/// - [Tutorials on @dealii](#tutorials-on-dealii)
/// - [How to compile a @dealii step file](#how-to-compile-a-dealii-step-file)
///
/// <a name="tutorials-on-dealii"></a>
/// # Tutorials on @dealii
/// The @dealii documentation is available
/// [here](https://www.dealii.org/current/doxygen/deal.II/mainpage.html).
///
/// If you are not familiar with @dealii, here are a few
/// [tutorials](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html)
/// from its documentation covering most of the topics needed to develop @lifex,
/// sorted by increasing difficulty. Please be sure to main the techniques
/// described before getting your hands into the code.
///
/// 1. [Mesh
/// handling](https://www.dealii.org/current/doxygen/deal.II/step_1.html)
/// 2. [Finite element degrees of
/// freedom](https://www.dealii.org/current/doxygen/deal.II/step_2.html)
/// 3. [Laplace equation (part
/// 1)](https://www.dealii.org/current/doxygen/deal.II/step_3.html)
/// 4. [Laplace equation (part
/// 2)](https://www.dealii.org/current/doxygen/deal.II/step_4.html)
/// 5. [Vector
/// problems](https://www.dealii.org/current/doxygen/deal.II/step_8.html),
/// [mixed
/// formulations](https://www.dealii.org/current/doxygen/deal.II/step_20.html)
/// 6. [Non-linear
/// problems](https://www.dealii.org/current/doxygen/deal.II/step_15.html)
/// 7. [Time-dependent
/// problems](https://www.dealii.org/current/doxygen/deal.II/step_26.html)
/// 8. Parallelization [using
/// PETSc](https://www.dealii.org/current/doxygen/deal.II/step_17.html) and
/// [using both PETSc and
/// Trilinos](https://www.dealii.org/current/doxygen/deal.II/step_40.html)
/// 9. [Time advancing
/// schemes](https://www.dealii.org/current/doxygen/deal.II/step_52.html)
/// 10. [Assembling block
/// preconditioners](https://www.dealii.org/current/doxygen/deal.II/step_55.html)
/// 11. [Automatic
/// differentiation](https://www.dealii.org/current/doxygen/deal.II/step_33.html)
///
/// <a name="how-to-compile-a-dealii-step-file"></a>
/// # How to compile a @dealii step file
/// The instructions to compile and run a tutorial source file are the
/// following.
///
/// 1. Create a <kbd>CMakeLists.txt</kbd> file in the same folder as the source
/// file you want to compile, as explained
/// [here](https://www.dealii.org/9.1.1/users/cmakelists.html).
/// 2. Create a build folder and enter it with `mkdir build && cd build`.
/// 3. Configure with `cmake .. -DDEAL_II_DIR=/path/to/dealii/installation/`
/// (add the `-DCMAKE_BUILD_TYPE=Debug` flag for a debug build).
/// 4. Compile with `make -j<N>`, where `N` is the desired number of processes.
///
