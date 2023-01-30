# <img alt="lifex" width="150" src="https://gitlab.com/lifex/lifex/-/raw/main/doc/logo/lifex.png" /> reference list
This folder contains a list of publications that are cited in the **<kbd>life<sup>x</sup></kbd>** documentation.

The content of the <kbd>references.bib</kbd> file is used to generate the
[references](https://lifex.gitlab.io/lifex/references.html) page.

## How to generate a reference list with <kbd>LaTeX</kbd>
A <kbd>references.pdf</kbd> file containing the reference list
can be generated in the <kbd>latex/</kbd> directory using a
[<kbd>LaTeX</kbd>](https://www.latex-project.org/) distribution:
```sh
cd /path/to/lifex/doc/references/
make latex
```

## How to generate a reference list with <kbd>JabRef</kbd>
A <kbd>Doxygen</kbd>-formatted reference page <kbd>references.hpp</kbd>
can be generated in the current directory using
[<kbd>JabRef</kbd>](https://www.jabref.org/):

```sh
cd /path/to/lifex/doc/references/
make jabref
```

The output will be processed by <kbd>Doxygen</kbd> when generating
the documentation from the build directory (as described in the
[coding guidelines](https://lifex.gitlab.io/lifex/coding-guidelines.html#development-rules)):

```sh
cd /path/to/lifex/build/
make doc
```
