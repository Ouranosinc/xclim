.. highlight:: console

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Implement Features, Indices or Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xclim's structure makes it easy to create and register new user-defined indices and indicators.
For the general implementation of indices and their wrapping into indicators, refer to
:ref:`notebooks/extendxclim:Extending xclim`  and  :ref:`notebooks/customize:Customizing and controlling xclim`.

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

General to-do list for implementing a new Indicator:

1. Implement the indice

    * Indices are function wrapped with :py:func:`~xclim.core.units.declare_units`
    * Their input arguments should have type annotations, as documented in :py:class:`~xclim.core.utils.InputKind`
    * Their docstring should follow the scheme explained in :ref:`notebooks/extendxclim:Defining new indices`.
    * They should set the units on their outputs, but no other metadata fields.
    * Their code should be found in the most relevant ``xclim/indices/_*.py``  file. Functions are explicitly added to the ``__all__`` at the top of the file.

2. Add unit tests

    * Indices are best tested with made up, idealized data to explicitly test the edge cases. Many pytest fixtures are available to help this data generation.
    * Tests should be added as one or more functions in ``xclim/testing/tests/test_indices.py``, see other tests for inspiration.

3. Add the indicator

    * See :ref:`notebooks/extendxclim:Defining new indicators` for more info and look at the other indicators for inspiration.
    * They are added in the most relevant ``xclim/indicators/{realm}/_*.py`` file.
    * Indicator are instances of subclasses of :py:class:`xclim.core.indicator.Indicator`.
      They should use a class declared within the ``{realm}`` folder, creating a dummy one if needed. They are explicitly added to the file's ``__all__``.

4. Add unit tests

    * Indicators are best tested with real data, also looking at missing value propagation and metadata formatting.
      In addition to the ``atmos_ds`` fixture, only datasets that can be accessed with :py:func:`xclim.testing.open_dataset` should be used.
    * Tests are added in the most relevant ``xclim/testing/tests/test_{variable}.py`` file.

5. Add French translations

    xclim comes with an internationalization module and all "official" indicators
    (those in ``xclim.atmos.indicators``) must have a french translation added to ``xclim/data/fr.json``.
    This part can be done by the core team after you open a Pull Request.

General notes for implementing new bias-adjustment methods:

* Method are implemented as classes in ``xclim/sdba/adjustment.py``.
* If the algorithm gets complicated and would generate many dask tasks, it should be implemented as functions wrapped
  by :py:func:`~xclim.sdba.map_blocks` or :py:func:`~xclim.sdba.map_groups` in ``xclim/sdba/_adjustment.py``.
* xclim doesn't implement monolithic multi-parameter methods, but rather smaller modular functions to construct post-processing workflows.

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/Ouranosinc/xclim/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xclim could always use more documentation, whether as part of the official xclim docs, in docstrings, or even on the
web in blog posts, articles, and such.

To reference documents (article, presentation, thesis, etc) in the documentation or in a docstring, xclim uses
`sphinxcontrib-bibtex`_. Metadata of the documents is stored as BibTeX entries in the ``docs/references.bib`` file.
To properly generate internal reference links, we suggest using the following roles:

- For references cited in the `References` section of function docstrings, use ``:cite:cts:`label```.
- For in-text references with first author and year, use ``:cite:t:`label```.
- For reference citations in parentheses, use ``:cite:p:`label```.

Multiple references can be added to a single role using commas (e.g. ``:cite:cts:`label1,label2,label3```).
For more information see: `sphinxcontrib-bibtex`_.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at: https://github.com/Ouranosinc/xclim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* The Xclim development team welcomes you and is always on hand to help. :)

Get Started!
------------

Ready to contribute? Here's how to set up `xclim` for local development.

1. Fork the `xclim` repo on GitHub.

2. Clone your fork locally::

    $ git clone git@github.com:{my_github_username}/xclim.git
    $ cd xclim/

3. Create a development environment. We recommend using ``conda``::

    $ conda create -n xclim python=3.8 --file=environment.yml
    $ pip install -e .[dev]

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally!

5. Before committing your changes, we ask that you install ``pre-commit`` in your development environment. Pre-commit runs git hooks that ensure that your code resembles that of the project and catches and corrects any small errors or inconsistencies when you ``git commit``::

    # To install the necessary pre-commit hooks:
    $ pre-commit install
    # To run pre-commit hooks manually:
    $ pre-commit run --all-files

  Instead of ``pre-commit``, you could also verify your changes manually with `black`, `flake8`, `flake8-rst-docstrings`, `pydocstyle`, and `yamllint`::

    $ black --check --target-version py38 xclim xclim/testing/tests
    $ black --check --target-version py38 --include "\.ipynb$" docs
    $ flake8 xclim xclim/testing/tests
    $ pydocstyle --config=setup.cfg xclim xclim
    $ yamllint --config-file .yamllint.yaml xclim

6. When unit/doc tests are added or notebooks updated, use ``pytest`` to run them. Alternatively, one can use ``tox`` to run all testing suites as would github do when the PR is submitted and new commits are pushed::

    $ pytest --nbval docs/notebooks  # for notebooks, exclusively.
    $ pytest --no-cov --rootdir xclim/testing/tests/ --xdoctest xclim --ignore=xclim/testing/tests/  # for doctests, exclusively.
    $ pytest  # for all unit tests, excluding doctests and notebooks.
    $ tox  # run all testing suites

7. Docs should also be tested to ensure that the documentation will build correctly on ReadTheDocs. This can be performed in a number of ways::

    # To run in a contained virtualenv environment
    $ tox -e docs
    # or, alternatively, to build the docs directly
    $ make docs

8. After clearing the previous checks, commit your changes and push your branch to GitHub::

    $ git add *

    $ git commit -m "Your detailed description of your changes."

If installed, `pre-commit` will run checks at this point:

* If no errors are found, changes will be committed.
* If errors are found, modifications will be made and warnings will be raised if intervention is needed.
* After adding changes, simply `git commit` again::

    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, please follow these guidelines:

1. Open an *issue* on our `GitHub repository`_ with your issue that you'd like to fix or feature that you'd like to implement.
2. Perform the changes, commit and push them either to new a branch within Ouranosinc/xclim or to your personal fork of xclim.

.. warning::
     Try to keep your contributions within the scope of the issue that you are addressing.
     While it might be tempting to fix other aspects of the library as it comes up, it's better to
     simply to flag the problems in case others are already working on it.

     Consider adding a "**# TODO:**" comment if the need arises.

3. Pull requests should raise test coverage for the xclim library. Code coverage is an indicator of how extensively tested the library is.
   If you are adding a new set of functions, they **must be tested** and **coverage percentage should not significantly decrease.**
4. If the pull request adds functionality, your functions should include docstring explanations.
   So long as the docstrings are syntactically correct, sphinx-autodoc will be able to automatically parse the information.
   Please ensure that the docstrings and documentation adhere to the following standards (badly formed docstrings will fail build tests):

   * `numpydoc`_
   * `reStructuredText (ReST)`_

.. note::
    If you aren't accustomed to writing documentation in reStructuredText (`.rst`), we encourage you to spend a few minutes going over the
    incredibly well-summarized `reStructuredText Primer`_ from the sphinx-doc maintainer community.

5. The pull request should work for Python 3.8, 3.9, and 3.10 as well as raise test coverage.
   Pull requests are also checked for documentation build status and for `PEP8`_ compliance.

   The build statuses and build errors for pull requests can be found at: https://github.com/Ouranosinc/xclim/actions

.. warning::
    PEP8, black, pytest (with xdoctest) and pydocstyle (for numpy docstrings) conventions are strongly enforced.
    Ensure that your changes pass all tests prior to pushing your final commits to your branch.
    Code formatting errors are treated as build errors and will block your pull request from being accepted.

6. The version changes (HISTORY.rst) should briefly describe changes introduced in the Pull request. Changes should be organized by type
   (ie: `New indicators`, `New features and enhancements`, `Breaking changes`, `Bug fixes`, `Internal changes`) and the GitHub Pull Request,
   GitHub Issue. Your name and/or GitHub handle should also be listed among the contributors to this version. This can be done as follows::

     Contributors to this version: John Jacob Jingleheimer Schmidt (:user:`username`).

     Internal changes
     ^^^^^^^^^^^^^^^^
     * Updated the contribution guidelines. (:issue:`868`, :pull:`869`).

   If this is your first contribution to Ouranosinc/xclim, we ask that you also add your name to the `AUTHORS.rst <https://github.com/Ouranosinc/xclim/blob/master/AUTHORS.rst>`_,
   under *Contributors* as well as to the `.zenodo.json <https://github.com/Ouranosinc/xclim/blob/master/.zenodo.json>`_, at the end of the *creators* block.

Tips
----

To run a subset of tests, we suggest a few approaches. For running only a test file::

    $ pytest xclim/testing/tests/test_xclim.py

To skip all slow tests::

    $ pytest -m "not slow"

To run all conventions tests at once::

    $ pre-commit run --all-files

Versioning
----------

In order to update and release the library to PyPI, it's good to use a semantic versioning scheme.
The method we use is as follows::

  major.minor.patch-release

**Major** releases denote major changes resulting in a stable API;

**Minor** is to be used when adding a module, process or set of components;

**Patch** should be used for bug fixes and optimizations;

**Release** is a keyword used to specify the degree of production readiness (`beta` [, and optionally, `gamma`]). *Only versions built from the main development branch will ever have this tag!*

  An increment to the Major or Minor will reset the Release to `beta`. When a build is promoted above `beta` (ie: release-ready), it's a good idea to push this version towards PyPi.

Deploying
---------

A reminder for the maintainers on how to prepare the library for a tagged version.

Make sure all your changes are committed (**including an entry in HISTORY.rst**).
Then run::

    $ bump2version <option>  # possible options: major / minor / patch / release

These commands will increment the version and create a commit with an autogenerated message.

For PyPI releases/stable versions, ensure that the last version bumping command run is `$ bump2version release` to remove the `-dev`.
These changes can now be merged to the main development branch::

    $ git push

With this performed, we can tag a version that will act as the GitHub-provided stable source archive.
Be sure to only tag from the `main` branch when all changes from PRs have been merged! Commands needed are::

    $ git tag v1.2.3-XYZ
    $ git push --tags

.. note::
    Starting from October, 2021, all tags pushed to GitHub will trigger a build and publish a package to TestPyPI by default. TestPyPI is a testing ground that is not indexed or easily available to `pip`.
    The test package can be found at `xclim on TestPyPI`_.

Packaging
---------

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate),
maintainers should update the pip-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

The Automated Approach
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to package `xclim` is to "publish" a version on GitHuh. GitHub CI Actions are presently configured to build the library and publish the packages on PyPI automatically.

When publishing on GitHub, maintainers will need to generate the release notes for the current version, replacing the ``:issue:``, ``:pull:``, and ``:user:`` tags. The `xclim` CLI offers a helper function for performing this action::

    # For Markdown format (needed when publishing a new version on GitHub):
    $ xclim release_notes -m
    # For ReStructuredText format (offered for convenience):
    $ xclim release_notes -r

When publishing to GitHub, you will still need to replace subsection headers in the Markdown (`^^^^` -> `###`) and the history published should not extend past the changes for the current version. This behaviour may eventually change.

.. warning::
    Be warned that a published package version on PyPI can never be overwritten. Be sure to verify that the package published at https://test.pypi.org/project/xclim/ matches expectations before publishing a version on GitHub.

The Manual Approach
~~~~~~~~~~~~~~~~~~~

The manual approach to library packaging for general support (pip wheels) requires that the `flit <https://flit.pypa.io/en/stable/index.html>`_ library is installed.

From the command line on your Linux distribution, simply run the following from the clone's main dev branch::

    # To build the packages (sources and wheel)
    $ flit build

    # To upload to PyPI
    $ flit publish

The new version based off of the version checked out will now be available via `pip` (`$ pip install xclim`).

Releasing on conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~

Initial Release
^^^^^^^^^^^^^^^

In order to prepare an initial release on conda-forge, we *strongly* suggest consulting the following links:
 * https://conda-forge.org/docs/maintainer/adding_pkgs.html
 * https://github.com/conda-forge/staged-recipes

Subsequent releases
^^^^^^^^^^^^^^^^^^^

If the conda-forge feedstock recipe is built from PyPI, then when a new release is published on PyPI, `regro-cf-autotick-bot` will open Pull Requests automatically on the conda-forge feedstock.
It is up to the conda-forge feedstock maintainers to verify that the package is building properly before merging the Pull Request to the main branch.

Before updating the main conda-forge recipe, we *strongly* suggest performing the following checks:
 * Ensure that dependencies and dependency versions correspond with those of the tagged version, with open or pinned versions for the `host` requirements.
 * If possible, configure tests within the conda-forge build CI (e.g. `imports: xclim`, `commands: pytest xclim`)

.. _`numpydoc`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _`reStructuredText (ReST)`: https://www.jetbrains.com/help/pycharm/using-docstrings-to-specify-types.html
.. _`reStructuredText Primer`: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _`GitHub Repository`: https://github.com/Ouranosinc/xclim
.. _`PEP8`: https://peps.python.org/pep-0008/
.. _`sphinxcontrib-bibtex`: https://sphinxcontrib-bibtex.readthedocs.io
.. _`xclim on TestPyPI`: https://test.pypi.org/project/xclim/
