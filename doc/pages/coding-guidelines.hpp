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

/// @page coding-guidelines Coding guidelines
/// # Table of contents
/// - [Configure <kbd>git</kbd>](#configure-git)
///   - [How to fix <kbd>Invalid author</kbd> errors](#invalid-author)
/// - [Development rules](#development-rules)
/// - [Archived branches](#archived-branches)
/// - [Coding conventions](#coding-conventions)
///   - [@lifex coding rules](#lifex-coding-rules)
///   - [Install <kbd>clang-format</kbd>](#install-clang-format)
///   - [@lifex coding style](#lifex-coding-style)
/// - [Configure your editor](#configure-your-editor)
///   - [Use with <kbd>clang-format</kbd>](#use-with-clang-format)
///   - [Syntax highlight for parameter
///   files](#syntax-highlight-for-parameter-files)
///
/// <a name="configure-git"></a>
/// # Configure <kbd>git</kbd>
/// First of all, configure <kbd>git</kbd> with your real name and email
/// address, as already described
/// [here](download-and-install.html#step-1-0-setup-your-gitlab-account):
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
/// <a name="invalid-author"></a>
/// ## How to fix <kbd>Invalid author</kbd> errors
///
/// One of the most common problems experienced by @lifex developers is the
/// `check indentation` job failing for your branch with the following error.
/// ```{.sh}
/// $ scripts/format/check_indentation
/// Invalid author 'Nickname91' without firstname and lastname
/// ```
/// This is because you probably skipped the <kbd>git</kbd> configuration steps.
///
/// The recommended way to solve the issue consists of:
/// - running the <kbd>git config [...]</kbd> commands reported above;
/// - using <kbd>git filter-branch</kbd> command. It allows you to batch-process
/// all commits in your branch with a script. Run the below sample script on
/// your branch (filling in real values for the old and new email and name).
///
/// > **WARNING**: the following procedure rewrites the history of your branch,
/// > creating new commit objects along the way! Use it with extreme care and
/// > only if you're aware of the side effects. Make sure to **create a backup**
/// > of your branch modifications before proceeding.
///
/// ~~~{.sh}
/// git filter-branch --force --env-filter '
/// OLD_NAME="Nickname91"
/// CORRECT_NAME="Name Surname"
/// CORRECT_EMAIL="name.surname@polimi.it"
///
/// if [ "${GIT_COMMITTER_NAME}" = "${OLD_NAME}" ]
/// then
///     export GIT_COMMITTER_NAME="${CORRECT_NAME}
///     export GIT_COMMITTER_EMAIL="${CORRECT_EMAIL}
/// fi
/// if [ "${GIT_AUTHOR_NAME}" = "${OLD_NAME}" ]
/// then
///     export GIT_AUTHOR_NAME="${CORRECT_NAME}"
///     export GIT_AUTHOR_EMAIL="${CORRECT_EMAIL}"
/// fi
/// ' --tag-name-filter cat
/// ~~~
///
/// Then the remote branch must be overwritten:
/// ```{.sh}
/// git push --force origin my_branch_name
/// ```
///
/// If that branch is being used on other machines, the usual `git pull origin
/// my_branch_name` **won't work** to properly fetch the updates. Run the
/// following commands instead.
/// ```{.sh}
/// git checkout main
/// git branch -D my_branch_name
/// git pull
/// git checkout my_branch_name
/// ```
///
/// <a name="development-rules"></a>
/// # Development rules
/// Here are a few steps to remember before you start to code.
///
/// 1. Always keep your branch up-to-date with the <kbd>main</kbd> branch by
/// running `git pull origin main`.
/// 2. Check periodically for recent relevant changes by refering to the <a
/// href="https://gitlab.com/lifex/lifex/-/blob/main/CHANGELOG.md">changelog</a>.
/// 3. Always [document](http://www.doxygen.nl/) the code you write. Don't be
/// rough: a wrong documentation is worse than no documentation at all. Also,
/// provide the proper references.
///
/// Run `make doc` from your build folder to generate the documentation to the
/// `doc` subdirectory (requires <kbd>Doxygen</kbd> and <kbd>graphviz</kbd>).
/// Please check that `make doc` runs without warnings or errors and open
/// `/path/to/lifex/build/doc/html/index.html` with a web browser to
/// check that the documentation you wrote is typeset and displayed correctly.
/// 4. Always compile and run your model in debug mode before opening a
/// merge request or an issue.
/// 5. Always run and/or update the testsuite to ensure that your modifications
/// did not break any previous functionality and are backward-compatible.
/// 6. Questions? Check the documentation first. Can't find the answer? Before
/// you assume, @b ask.
///
/// <a name="archived-branches"></a>
/// # Archived branches
/// Stall branches, @a i.e. those that are not ready to merge and show no
/// activity within 4 months, will be archived and deleted.
///
/// Archived branches can be fetched and inspected with:
/// ~~~{.sh}
/// # List archived branches in the remote.
/// git ls-remote origin refs/archive/*
///
/// # List archived branches available locally.
/// git for-each-ref origin refs/archive/*
///
/// # Fetch archived branches from the remote.
/// git fetch origin refs/archive/*:refs/archive/*
///
/// # Checkout an archived branch available locally.
/// git checkout refs/archive/archived_branch_name
/// ~~~
///
/// <a name="coding-conventions"></a>
/// # Coding conventions
/// <a name="lifex-coding-rules"></a>
/// ## @lifex coding rules
/// 1. Include directives should be added only where strictly needed (either
/// `.hpp` or `.cpp` files).
/// 2. @b Never use raw pointers (like `int *`) or the keyword `new`.
/// For memory handling use `std::unique_ptr` (with `std::make_unique`)
/// or `std::shared_ptr` (with `std::make_shared`), instead.
/// 3. Function inputs should be declared as const references,
/// @a e.g. `fun(const type& ...)`, unless needed otherwise.
/// 4. Polymorphic base classes should @b always provide a virtual destructor.
/// 5. The @ref lifex::Core "Core" class is implemented to provide a common interface
/// for parameter handling and to some utilities such as <kbd>MPI</kbd>
/// communicator, rank and size, a
/// @ref pcout "parallel standard output object", a
/// @ref pcerr "parallel error output object" and a
/// @ref timer_output "timer output". Additionally, the
/// @ref lifex::CoreModel "CoreModel" class provides a common interface
/// for parameter handling. If your class needs any of such
/// functionalities it should publicly inherit from @ref lifex::Core "Core"
/// or @ref lifex::CoreModel "CoreModel".
/// 6. All output should be redirected to the
/// @ref lifex::Core::prm_output_directory folder.
/// 7. Every class inheriting from @ref lifex::CoreModel "CoreModel"
/// @b must override the following methods:
/// ~~~{.cpp}
/// public:
///   virtual void
///   declare_parameters(ParamHandler &params) const override;
///
///   virtual void
///   parse_parameters(ParamHandler &params) override;
///
///   // Not mandatory: the abstract implementation does nothing by default.
///   virtual void
///   run() override;
/// ~~~
/// If the class depends on other core models whose parameters have to be
/// exposed externally, such dependencies should be declared as class members
/// (or smart pointers, if they are not default constructable).
/// 8. The `declare_parameters()` method should set the verbosity of
/// parameters and sections
/// (as explained in @ref lifex::VerbosityParam and
/// @ref lifex::Core::prm_verbosity_param), @a e.g.:
/// ~~~{.cpp}
/// void
/// MyClass::declare_parameters(ParamHandler &params) const
/// {
///   params.enter_subsection_path(prm_subsection_path);
///   {
///     params.enter_subsection("Very important parameters");
///     {
///       // ...
///     }
///     params.leave_subsection();
///
///     params.set_verbosity(VerbosityParam::Standard);
///     {
///       params.enter_subsection("Less important parameters");
///       {
///         // ...
///       }
///       params.leave_subsection();
///     }
///     params.reset_verbosity();
///
///     params.set_verbosity(VerbosityParam::Full);
///     {
///       params.enter_subsection("Advanced usage parameters");
///       {
///         // ...
///       }
///       params.leave_subsection();
///     }
///     params.reset_verbosity();
///   }
///   params.leave_subsection_path();
/// }
/// ~~~
/// 9. The `parse_parameters()` method should parse all parameters declared
/// above, as well as to verify the correctness and meaningfulness of the input
/// parameters (possibly throwing proper exceptions):
/// ~~~{.cpp}
/// void
/// MyClass::parse_parameters(ParamHandler &params) const
/// {
///   params.parse();
///   params.enter_subsection_path(prm_subsection_path);
///   {
///     // ...
///     AssertThrow(!prm_boundary_ids.empty(),
///                 ExcMessage("List of boundary IDs cannot be left empty.");
///
///     // ...
///     AssertThrow(prm_boundary_ids.size() == prm_bcs_type.size(),
///                 ExcMessage("The list of boundary IDs and of BC types "
///                            "must have the same length.");
///
///     // ...
///   }
///   params.leave_subsection_path();
/// }
/// ~~~
/// 10. Class members storing the value of parameters read from file should be
/// prefixed with `prm_` and enclosed in a <kbd>Doxygen</kbd> group named
/// `Parameters read from file.`.
/// 11. <kbd>Doxygen</kbd> comments should start with a `@` rather than with a `\` (@a e.g. `@param`).
/// 12. Always start the `%main()` function with @lifex initialization:
///    ~~~{.cpp}
///    lifex::lifex_init lifex_initializer(argc, argv, 1);
///    ~~~
/// 13. Always start relevant methods within your classes with
///    ~~~{.cpp}
///    TimerOutput::Scope timer_section(timer_output,
///                                     prm_subsection_path +
///                                      " / What is done here");
///    ~~~
///    This will automaticaly generate timing reports for the code you write.
///
/// <a name="install-clang-format"></a>
/// ## Install <kbd>clang-format</kbd>
/// <kbd>clang-format</kbd> is required to automatically indent and properly
/// format the source files. The corresponding style file
/// is <kbd>/path/to/lifex/.clang-format</kbd>.
///
/// In order to install <kbd>clang-format-10.0.0</kbd>, run the
/// `/path/to/lifex/scripts/format/download_clang-format` script.
///
/// This will download a statically-linked binary of
/// <kbd>clang-format-10.0.0</kbd> and install it under
/// <kbd>/path/to/lifex/scripts/format/clang-10.0.0/bin</kbd>.
///
/// It is recommended to add the previous folder to your <kbd>PATH</kbd>
/// environment variable, @a e.g. by adding the following line to your
/// <kbd>${HOME}/.bashrc</kbd> file (or equivalent).
/// ~~~{.sh}
/// export PATH="/lifex/dir/scripts/format/clang-format-10.0.0/bin:${PATH}"
/// ~~~
///
/// > **Note**: in case of binary incompatibility with your system, use the
/// > `/path/to/lifex/scripts/format/compile_clang-format` script instead.
///
/// <a name="lifex-coding-style"></a>
/// ## @lifex coding style
/// 1. @lifex follows the [deal.II coding
/// conventions](https://www.dealii.org/current/doxygen/deal.II/CodingConventions.html):
/// please read them carefully before coding.
/// 2. Run `/path/to/lifex/scripts/format/static_analysis` from @lifex source
/// directory before @b any commit to perform a static analysis of the code,
/// relying upon the <kbd>cppcheck</kbd> and <kbd>cpplint</kbd> tools.
/// 3. Run `/path/to/lifex/scripts/format/indent_all` from @lifex source
/// directory before @b any commit to indent all files (or
/// `/path/to/lifex/scripts/format/indent` to process only new files or files
/// that have changed since last merge commit). Two equivalent build targets
/// `make indent_all` and `make indent` are available.
/// 4. In order to make sure that the indentation is correct for all of your
/// commits, you might want to setup a pre-commit hook. One way to do so is
/// to run `cp ./scripts/format/pre-commit .git/hooks/pre-commit && chmod u+x
/// .git/hooks/pre-commit` from your <kbd>/path/to/lifex</kbd> directory.
///
/// > **Note**: the automatic indentation script makes use of the
/// > <kbd>clang-format</kbd> package.
///
/// When a merge request is open, the copyright notice and its formatting
/// are also checked. Please run
/// `/path/to/lifex/scripts/format/update_copyright` from @lifex source
/// directory to update the copyright notice.
///
/// <a name="configure-your-editor"></a>
/// # Configure your editor
/// <a name="use-with-clang-format"></a>
/// ## Use with <kbd>clang-format</kbd>
/// Different plugins and extensions are available to configure your favorite
/// editor for formatting the code you write using <kbd>clang-format</kbd>:
/// -
/// [<kbd>Emacs</kbd>](https://clang.llvm.org/docs/ClangFormat.html#emacs-integration):
///   1. Download
/// [<kbd>clang-format.el</kbd>](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/clang-format.el).
///   2. Add the following lines to your <kbd>~/.emacs</kbd> file (toggle
///   comments on one of the two `define-key` lines depending on your needs):
///   ~~~{.lisp}
///   ;; clang-format.
///   (load "/path/to/clang-format.el")
///   (require 'cc-mode)
///
///   ;; Uncomment the following line to format the current selection.
///   ;; (define-key c-mode-base-map (kbd "<tab>") 'clang-format-region)
///
///   ;; Uncomment the following line to format the current file.
///   (define-key c-mode-base-map (kbd "<tab>") 'clang-format-buffer)
///   ~~~
///   3. Use <kbd>&lt;TAB&gt;</kbd> to format the current selection or file.
/// -
/// [<kbd>Vim</kbd>](https://clang.llvm.org/docs/ClangFormat.html#vim-integration)
/// ([<kbd>clang-format.py</kbd>](https://llvm.org/svn/llvm-project/cfe/trunk/tools/clang-format/clang-format.el))
/// - [<kbd>Atom</kbd>](https://atom.io/packages/clang-format)
/// - [<kbd>Sublime
/// Text</kbd>](https://packagecontrol.io/packages/Clang%20Format):
///   1. Install <kbd>Clang Format</kbd> via <kbd>Package Control</kbd>.
///   2. Change user setting of the package (<kbd>Preferences -> Package
///   Settings
///   ->
/// Clang Format -> Settings - User</kbd>) as follows:
///   ~~~{.js}
///   {
///     "binary": "/path/to/clang-format",
///     "style": "File",
///     "format_on_save": true
///   }
///   ~~~
///   3. Use <kbd>&lt;CTRL+ALT+A&gt;</kbd> to format the current selection or
/// <kbd>&lt;CTRL+S&gt;</kbd> to save and format the current file.
/// - [<kbd>Eclipse</kbd>](https://github.com/wangzw/CppStyle)
///
/// <a name="syntax-highlight-for-parameter-files"></a>
/// ## Syntax highlight for parameter files
/// Different plugins and extensions are available to configure your favorite
/// editor for indenting or highlighting @dealii parameter (<kbd>.prm</kbd>)
/// files, which are used throughout @lifex:
/// - [<kbd>Emacs</kbd>](https://github.com/drwells/prm-mode):
///   1. Download
/// [<kbd>prm-mode.el</kbd>](https://github.com/drwells/prm-mode/blob/master/prm-mode.el).
///   2. Add the following lines to your <kbd>~/.emacs</kbd> file:
///   ~~~{.lisp}
///   ;; prm-mode.
///   (require 'prm-mode "/path/to/prm-mode.el" nil)
///   (add-to-list 'auto-mode-alist '("/*.\.prm$" . prm-mode))
///   ~~~
/// - [<kbd>Vim</kbd>](https://github.com/xywei/vim-dealii-prm)
/// - <kbd>Atom</kbd>: [syntax
/// highlighter](https://atom.io/packages/language-dealii-prm), [tree
/// view](https://atom.io/packages/dealii-prm-tree)
/// - [<kbd>Sublime Text</kbd>](https://gitlab.com/ifumagalli/utilsublime)
///
