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

/// @page download-and-install Download and install
/// > **Warning**: some background knowledge on the following topics is
/// > required. Before getting started, please make sure to get familiar with
/// > them (a few introductory references are listed, but many more resources
/// > can be easily found on the web).
/// > - Command line interface on <kbd>Unix/Linux</kbd>
/// > ([tutorial](https://ryanstutorials.net/linuxtutorial/), [cheat
/// > sheet](https://github.com/RehanSaeed/Bash-Cheat-Sheet)).
/// > - Command line text editors, such as
/// > [<kbd>Vim</kbd>](https://www.vim.org/) ([cheat
/// > sheet](https://devhints.io/vim)),
/// > [<kbd>Emacs</kbd>](https://www.gnu.org/software/emacs/) ([cheat
/// > sheet](https://www.gnu.org/software/emacs/refcards/pdf/refcard.pdf)), or
/// > [<kbd>nano</kbd>](https://www.nano-editor.org/) ([cheat
/// > sheet](https://www.nano-editor.org/dist/latest/cheatsheet.html)).
/// > - [<kbd>Git</kbd>](https://git-scm.com/) ([Pro Git
/// > book](https://git-scm.com/book/en/v2), [cheat
/// > sheet](https://training.github.com/downloads/github-git-cheat-sheet/),
/// > [visual cheat sheet](https://ndpsoftware.com/git-cheatsheet.html)).
///
/// # Table of contents
/// - [Dependencies](#dependencies)
/// - [Step 0 - Install @lifex dependencies](#step-0-install-lifex-dependencies)
///   - [<kbd>mk</kbd>](#mk)
///   - [<kbd>life<sup>x</sup>-env</kbd>](#lifex-env)
///   - [<kbd>Spack</kbd>](#spack)
///   - [<kbd>Docker</kbd>](#docker)
/// - [Step 1 - Install @lifex](#step-1-install-lifex)
///   - [Step 1.0 - Setup your <kbd>GitLab</kbd>
/// account](#step-1-0-setup-your-gitlab-account)
///   - [Step 1.1 - Get @lifex](#step-1-1-get-lifex)
///   - [Step 1.2 - Configure @lifex](#step-1-2-configure-lifex)
///   - [Step 1.3 - Compile @lifex](#step-1-3-compile-lifex)
///   - [Step 1.4 - Check @lifex installation](#step-1-4-check-lifex)
/// - [Step 2 - What's next?](#step-2-whats-next)
///
/// <a name="dependencies"></a>
/// # Dependencies
/// @lifex has the following dependencies:
/// - A <kbd>C++17</kbd>-compliant compiler and related standard library (such
/// as [<kbd>GCC</kbd>](https://gcc.gnu.org/))
/// - An <kbd>MPI</kbd> installation (such as
/// [<kbd>OpenMPI</kbd>](https://www.open-mpi.org/) or
/// [<kbd>MPICH</kbd>](https://www.mpich.org/))
/// - [<kbd>CMake</kbd>](https://cmake.org/) ≥ <kbd>3.12.0</kbd>
/// - [@dealii](https://www.dealii.org/) ≥ <kbd>9.3.1</kbd>
/// - [<kbd>VTK</kbd>](https://vtk.org/) ≥ <kbd>9.0.3</kbd>
/// - [<kbd>Boost</kbd>](https://www.boost.org/) ≥ <kbd>1.76.0</kbd>
/// - (Optional) [<kbd>Doxygen</kbd>](http://www.doxygen.nl/) and
/// [<kbd>Graphviz</kbd>](https://www.graphviz.org/)
///
/// You need to install the previous dependencies in order to be able to
/// successfully configure and build @lifex.
///
/// We recommended you to follow [step 0](#step-0-install-lifex-dependencies)
/// for a pain-free procedure to get all the dependencies you need. Then you can
/// safely go through [step 1](#step-1-install-lifex).
///
/// > **Note** (for advanced users): you may also consider to install all of the
/// > dependencies manually. Please refer to the above links for further
/// > instructions.
///
/// <a name="step-0-install-lifex-dependencies"></a>
/// # Step 0 - Install @lifex dependencies
/// Different procedure are available depending on the operating system you
/// are using.
///
/// The defaut installation relies upon the
/// - [<kbd>mk</kbd>](#mk) (<kbd>Linux</kbd>, recommended for
/// MOX developers)
/// .
/// package.
///
/// @b Only if the previous option does @b not work for you, you may consider
/// one of the following alternatives:
/// - [<kbd>life<sup>x</sup>-env</kbd>](#lifex-env)
/// (<kbd>Linux</kbd>, <kbd>macOS</kbd>)
/// - [<kbd>Spack</kbd>](#spack) (<kbd>Linux</kbd>,
/// <kbd>macOS</kbd>)
/// - [<kbd>Docker</kbd>](#docker) (<kbd>Linux</kbd>,
/// <kbd>macOS</kbd>, <kbd>Windows</kbd>)
///
/// > **Note**: <kbd>Windows 10</kbd> users may consider installing
/// > <a href="https://docs.microsoft.com/en-US/windows/wsl/about"><kbd>Windows
/// > Subsystem for Linux (WSL)</kbd></a>, a compatibility layer for running
/// > <kbd>Linux</kbd> binaries on a <kbd>Windows</kbd> host (version <kbd>WSL
/// > 2</kbd> recommended). All the above <kbd>Linux</kbd> options, including
/// > [<kbd>mk</kbd>](#mk), are available for <kbd>WSL</kbd>.
///
/// <a name="mk"></a>
/// ## <kbd>mk</kbd>
/// For <kbd>Linux</kbd> systems, we have packed a number of scientific
/// computing softwares, including all @lifex dependencies, in a
/// <kbd>x86-64</kbd> binary package called <kbd>mk</kbd>.
///
/// The recommended procedure is to install <kbd>mk</kbd> directly on
/// your system as described below.
///
/// > **Note**: in case you have a previous (< <kbd>v2022.0</kbd>) installation
/// > of <kbd>mk</kbd>, delete it with `sudo rm -rf /u/`.
///
/// > **Note**: the following steps 1 and 2 are @b not needed on MOX clusters.
///
/// 1. Download the <kbd>mk</kbd> archive from [this
/// link](https://github.com/elauksap/mk/releases/download/v2022.0/mk-2022.0-lifex.tar.gz).
/// 2. Open a terminal in the folder containing the archive
/// <kbd>mk-2022.0-lifex.tar.gz</kbd> just downloaded and type the following
/// command:
/// ~~~{.sh}
/// sudo tar xvzf mk-2022.0-lifex.tar.gz -C /
/// ~~~
/// 3. Add the following lines to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent):
/// ~~~{.sh}
/// # mk.
/// source /u/sw/etc/bash.bashrc
/// module load gcc-glibc/11.2.0 dealii vtk
/// ~~~
/// 4. Restart the shell.
///
/// You can now proceed to [step 1](#step-1-install-lifex).
///
/// > **Note**: the <kbd>mk</kbd> modules are compatible with any @a recent
/// > enough <kbd>Linux</kbd> distribution. On some (old) distros, you may
/// > experience errors like <kbd>"version GLIBC_X.YY not found"</kbd>. If this
/// > is your case, we recommend you to update your operating system to a more
/// > recent version, possibly shipping a
/// > [<kbd>glibc</kbd>](https://www.gnu.org/software/libc/) version not older
/// > than `X.YY`.
///
/// <a name="lifex-env"></a>
/// ## <kbd>life<sup>x</sup>-env</kbd>
/// > **Note**: Skip this step if you have already installed @lifex dependencies
/// > using one of the other methods in this section.
///
/// <kbd>life<sup>x</sup>-env</kbd> is a set of shell scripts that
/// download, configure, build, and install @lifex
/// dependencies on <kbd>Linux</kbd>-based systems.
///
/// 1. Follow the instructions available
/// [here](https://gitlab.com/lifex/lifex-env) to install
/// <kbd>life<sup>x</sup>-env</kbd> (this step may require a long
/// time to complete).
/// 2. Add the following line to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent), replacing `/path/to/lifex-env/` with the prefix where
/// <kbd>life<sup>x</sup>-env</kbd> has been installed:
/// ~~~{.sh}
/// # lifex-env.
/// source /path/to/lifex-env/configuration/enable_lifex.sh
/// ~~~
///
/// You can now proceed to [step 1](#step-1-install-lifex).
///
/// <a name="spack"></a>
/// ## <kbd>Spack</kbd>
/// > **Note**: Skip this step if you have already installed @lifex dependencies
/// > using one of the other methods in this section.
///
/// @lifex dependencies are also available on
/// [<kbd>Spack</kbd>](https://spack.io/), a package manager for
/// <kbd>Linux</kbd> and <kbd>macOS</kbd>.
///
/// Make sure that a <kbd>C/C++</kbd> compiler, <kbd>python</kbd>,
/// <kbd>make</kbd>, <kbd>git</kbd>, <kbd>curl</kbd> are installed.
///
/// More details about @dealii on <kbd>Spack</kbd> can be found
/// [here](https://github.com/dealii/dealii/wiki/deal.II-in-Spack#quick-installation-on-the-desktop).
///
/// 1. Add the following lines to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent), replacing `/path/to/spack/` with the path where you
/// want to install <kbd>Spack</kbd>:
/// ~~~{.sh}
/// export SPACK_ROOT=/path/to/spack/
/// export PATH=${SPACK_ROOT}/bin:${PATH}
/// ~~~
///
/// 2. Clone <kbd>Spack</kbd> with:
/// ~~~{.sh}
/// mkdir -p ${SPACK_ROOT}
/// cd ${SPACK_ROOT}
/// git clone https://github.com/spack/spack.git ./
/// ~~~
/// 3. Add the following line to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent):
/// ~~~{.sh}
/// source ${SPACK_ROOT}/share/spack/setup-env.sh
/// ~~~
///
/// 4. Configure <kbd>Spack</kbd> as an environment
/// module manager with
/// ~~~{.sh}
/// spack install environment-modules
/// ~~~
/// 5. Add the following line to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent):
/// ~~~{.sh}
/// source $(spack location -i environment-modules)/init/bash
/// ~~~
///
/// 6. Install @dealii and <kbd>VTK</kbd> (this step may require a long
/// time to run) with
/// ~~~{.sh}
/// spack install dealii@9.3.1
/// spack install vtk@9.0.3 ^mesa~llvm
/// ~~~
///
/// 7. Add the following lines to your <kbd>${HOME}/.bashrc</kbd> file (or
/// equivalent):
/// ~~~{.sh}
/// spack load openmpi
/// spack load cmake
/// spack load boost
/// spack load dealii
/// spack load vtk
/// ~~~
///
/// You can now proceed to [step 1](#step-1-install-lifex).
///
/// <a name="docker"></a>
/// ## <kbd>Docker</kbd>
/// > **Note**: Skip this step if you have already installed @lifex dependencies
/// > using one of the other methods in this section.
///
/// Two minimal [<kbd>Ubuntu</kbd>](https://www.ubuntu-it.org/) distributions,
/// with [<kbd>life<sup>x</sup>-env</kbd>](#lifex-env) and [<kbd>mk</kbd>](#mk)
/// installed respectively, are available for use with
/// [<kbd>Docker</kbd>](https://www.docker.com/).
///
/// First you need to install <kbd>Docker</kbd> following these
/// [instructions](https://docs.docker.com/install/#supported-platforms).
///
/// The @lifex image(s) can be downloaded using the following commands (root
/// privileges are required):
/// ~~~{.sh}
/// docker login registry.gitlab.com
/// docker pull registry.gitlab.com/lifex/lifex/mk
/// # Or: docker pull registry.gitlab.com/lifex/lifex-env
/// ~~~
///
/// Then a corresponding container can be run through
/// ~~~{.sh}
/// docker run -it registry.gitlab.com/lifex/lifex/mk
/// # Or: docker run -it registry.gitlab.com/lifex/lifex/env
/// ~~~
///
/// Please refer to [<kbd>Docker</kbd>
/// documentation](https://docs.docker.com/reference/) for further instructions
/// on how to use <kbd>Docker</kbd>.
///
/// You can now proceed to [step 1](#step-1-install-lifex).
///
/// <a name="step-1-install-lifex"></a>
/// # Step 1 - Install @lifex
///
/// <a name="step-1-0-setup-your-gitlab-account"></a>
/// ## Step 1.0 - Setup your <kbd>GitLab</kbd> account
/// It is strongly recommended that you associate an
/// [<kbd>SSH</kbd> key](https://docs.gitlab.com/ee/ssh/) to your
/// <kbd>GitLab</kbd> account in order to be able to use all of the
/// [<kbd>git</kbd>](https://git-scm.com/) functionalities without being
/// prompted to enter your credentials each time.
///
/// 1. Generate an <kbd>SSH</kbd> key on your system following [these
/// steps](https://docs.gitlab.com/ee/ssh/#generating-a-new-ssh-key-pair).
/// 2. Add it to your <kbd>GitLab</kbd> account following [these
/// instructions](https://docs.gitlab.com/ee/ssh/#adding-an-ssh-key-to-your-gitlab-account).
/// 3. Configure <kbd>git</kbd> with your real name and email address:
/// ~~~{.sh}
/// git config --global user.name "Name(s) Surname(s)"
/// git config --global user.email "my.email@address.com"
/// ~~~
/// > **Note**: the settings above @b must be formatted properly, @a i.e.
/// > names and surname(s) should start with an uppercase letter and be
/// > separated by a (single) space character.
///
/// > **Note**: please make sure that these settings are consistent on @b all
/// > computers you are using @lifex on.
///
///
/// <a name="step-1-1-get-lifex"></a>
/// ## Step 1.1 - Get @lifex
/// Move to a target folder (make sure its path does not contain special
/// characters such as <kbd>&</kbd>, <kbd>$</kbd> or brackets) and run:
/// ~~~{.sh}
/// git clone --recursive git@gitlab.com:lifex/lifex.git
/// ~~~
///
/// > **Note**: if you are unable to use the <kbd>SSH</kbd> protocol, a
/// > standard-authentication <kbd>HTTPS</kbd> URL is available:
/// > ~~~{.sh}
/// > git clone --recursive https://your_username@gitlab.com/lifex/lifex.git
/// > ~~~
///
/// <a name="step-1-2-configure-lifex"></a>
/// ## Step 1.2 - Configure @lifex
/// First create a build folder and enter it:
/// ~~~{.sh}
/// cd /path/to/lifex
/// mkdir build
/// cd build
/// ~~~
/// Then run @lifex configuration with
/// ~~~{.sh}
/// cmake ..
/// ~~~
///
/// Add the optional `-DCMAKE_BUILD_TYPE=Debug` flag to build with debug symbols
/// and no compile-time optimization.
///
/// > **Note** (for developers): @lifex can use either <kbd>Trilinos</kbd> or
/// > <kbd>PETSc</kbd> as linear algebra backends. The backend can be customized
/// > with `-DLIN_ALG=Trilinos` (the default value) or `-DLIN_ALG=PETSc`.
///
/// > **Note** (for developers): in order to enable <kbd>Doxygen</kbd> and
/// > <kbd>Graphviz</kbd>, make sure that the `doxygen` and `dot` executables
/// > are in your system `PATH` environment variable
/// > (see also @ref coding-guidelines).
///
/// > **Note** (for advanced users): if you compiled and installed
/// > @dealii, <kbd>VTK</kbd> or <kbd>Boost</kbd> manually in any other way than
/// > [<kbd>mk</kbd>](#mk), [<kbd>Spack</kbd>](#spack) or
/// > [<kbd>Docker</kbd>](#docker), specify the installation directories with
/// > the `cmake` flags `-DDEAL_II_DIR=/path/to/dealii/`,
/// > `-DVTK_DIR=/path/to/vtk/` and `-DBOOST_DIR=/path/to/boost/`.
///
/// > **Note** (for [<kbd>Docker</kbd>](#docker) users): please specify the
/// > `cmake` flag `-DMPIEXEC_PREFLAGS="--allow-run-as-root"`
/// > to enable the container to run parallel executables.
///
/// <a name="step-1-3-compile-lifex"></a>
/// ## Step 1.3 - Compile @lifex
/// Run the compilation with
/// ~~~{.sh}
/// make -j<N>
/// ~~~
/// where `<N>` is the desired number of parallel processes. By default, this
/// builds all the targets available.
///
/// If you only need to compile part of the library, specific targets can be
/// selected with, @a e.g.:
/// ~~~{.sh}
/// make -j<N> electrophysiology mechanics
/// ~~~
/// (`make help` prints the list of all possible targets).
///
/// <a name="step-1-4-check-lifex"></a>
/// ## Step 1.4 - Check @lifex installation
/// Large additional data and meshes are required by automatic tests and
/// available externally. Run
/// ~~~{.sh}
/// cd /path/to/lifex
/// git submodule update --init
/// ~~~
/// to download (or update) them.
///
/// Finally, to check whether everything works successfully, setup and run the
/// testing platform with
/// ~~~{.sh}
/// make -j<N> setup_tests
/// ctest -L test_soft
/// ~~~
///
/// > **Note**: by default, parallel tests related are run in parallel using
/// > <kbd>MPI</kbd> on all the processors detected on the host system.
/// > Configure @lifex with the `cmake` flag `-DMPIEXEC_MAX_NUMPROCS=<N>` to set
/// > a custom number of parallel processes.
///
/// > **Note** (for developers): more computationally intensive tests can be run
/// > with `ctest -L test_hard`.
///
/// You did it, have fun with @lifex!
///
/// <a name="step-2-whats-next"></a>
/// # Step 2 - What's next?
/// 1. Learn how to run a @lifex app by visiting @ref run.
/// 2. If you are a new developer, make sure to read and follow our @ref coding-guidelines.
///
