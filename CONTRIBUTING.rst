.. highlight:: console

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Implement Features, Indices or Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`xclim`'s structure makes it easy to create and register new user-defined indices and indicators.
For the general implementation of indices and their wrapping into indicators, refer to :ref:`notebooks/extendxclim:Extending xclim` and :ref:`notebooks/customize:Customizing and controlling xclim`.

Look through the GitHub issues for features; Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

General to-do list for implementing a new Indicator:

#. Implement the indice

    * Indices are function wrapped with :py:func:`~xclim.core.units.declare_units`
    * Their input arguments should have type annotations, as documented in :py:class:`~xclim.core.utils.InputKind`
    * Their docstring should follow the scheme explained in :ref:`notebooks/extendxclim:Defining new indices`.
    * They should set the units on their outputs, but no other metadata fields.
    * Their code should be found in the most relevant ``xclim/indices/_*.py``  file. Functions are explicitly added to the ``__all__`` at the top of the file.

#. Add unit tests

    * Indices are best tested with made up, idealized data to explicitly test the edge cases. Many pytest fixtures are available to help this data generation.
    * Tests should be added as one or more functions in ``tests/test_indices.py``, see other tests for inspiration.

#. Add the indicator

    * See :ref:`notebooks/extendxclim:Defining new indicators` for more info and look at the other indicators for inspiration.
    * They are added in the most relevant ``xclim/indicators/{realm}/_*.py`` file.
    * Indicator are instances of subclasses of :py:class:`xclim.core.indicator.Indicator`.
      They should use a class declared within the ``{realm}`` folder, creating a dummy one if needed. They are explicitly added to the file's ``__all__``.

#. Add unit tests

    * Indicators are best tested with real data, also looking at missing value propagation and metadata formatting.
      In addition to the ``atmosds`` fixture, only datasets that can be accessed with :py:func:`xclim.testing.open_dataset` should be used.
      For convenience, this special function is accessible as the ``open_dataset`` pytest fixture.
    * Tests are added in the most relevant ``tests/test_{variable}.py`` file.

#. Add French translations

    xclim comes with an internationalization module and all "official" indicators
    (those in ``xclim.atmos.indicators``) must have a french translation added to ``xclim/data/fr.json``.
    This part can be done by the core team after you open a Pull Request.

.. note::
    If you are adding new translations to the library (for languages other than French), please begin by opening a discussion on the `xclim Discussions page`_ to coordinate the scope and implementation of these translations.

General notes for implementing new bias-adjustment methods:

* Method are implemented as classes in ``xclim/sdba/adjustment.py``.
* If the algorithm gets complicated and would generate many dask tasks, it should be implemented as functions wrapped by :py:func:`~xclim.sdba.map_blocks` or :py:func:`~xclim.sdba.map_groups` in ``xclim/sdba/_adjustment.py``.
* xclim doesn't implement monolithic multi-parameter methods, but rather smaller modular functions to construct post-processing workflows.
* If you are working on numba-accelerated function that use ``@guvectorize``, consider disabling caching during the development phase and reactivating it once all changes are ready for review. This is done by commenting ``cache=True`` in the decorator.

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

xclim could always use more documentation, whether as part of the official xclim docs, in docstrings, or even on the web in blog posts, articles, and such.

To reference documents (article, presentation, thesis, etc) in the documentation or in a docstring, xclim uses `sphinxcontrib-bibtex`_.
Metadata of the documents is stored as BibTeX entries in the ``docs/references.bib`` file.
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
* The xclim development team welcomes you and is always on hand to help. :)

Get Started!
------------

Ready to contribute? Here's how to set up `xclim` for local development.

#. Fork the `xclim` repo on GitHub.

#. Clone your fork locally::

    $ git clone git@github.com:{my_github_username}/xclim.git
    $ cd xclim/

#. Create a development environment. We recommend using ``conda``::

    $ conda create -n xclim python=3.10 --file=environment.yml
    $ python -m pip install -e ".[dev]"

#. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally!

#. Before committing your changes, we ask that you install ``pre-commit`` in your development environment. Pre-commit runs git hooks that ensure that your code resembles that of the project and catches and corrects any small errors or inconsistencies when you ``git commit``::

    # To install the necessary pre-commit hooks:
    $ pre-commit install
    # To run pre-commit hooks manually:
    $ pre-commit run --all-files

   Instead of ``pre-commit``, you can also verify your changes using the `Make` recipe for code linting checks::

    $ make lint

   Or, alternatively, you can check individual hooks manually with `black`, `isort`, `ruff`, `flake8`, `flake8-rst-docstrings`, `nbqa`, `blackdoc`, and `yamllint`::

	$ black --check xclim tests
	$ isort --check xclim tests
	$ ruff xclim tests
	$ flake8 --config=.flake8 xclim tests
	$ nbqa black --check docs
	$ nbqa isort --check docs
	$ blackdoc --check --exclude=xclim/indices/__init__.py xclim
	$ blackdoc --check docs
	$ yamllint --config-file=.yamllint.yaml xclim

#. When features or bug fixes have been contributed, unit tests and doctests have been added, or notebooks have been updated, use ``$ pytest`` to test them::

    $ pytest --no-cov --nbval --dist=loadscope --rootdir=tests/ docs/notebooks --ignore=docs/notebooks/example.ipynb  # for notebooks, exclusively.
    $ pytest --no-cov --rootdir=tests/ --xdoctest xclim  # for doctests, exclusively.
    $ pytest  # for all unit tests, excluding doctests and notebooks.
    $ pytest -m "not slow"  # for all unit tests, excluding doctests, notebooks, and "slow" marked tests.

   Alternatively, one can use ``$ tox`` to run very specific testing configurations, as GitHub Workflows would do when a Pull Request is submitted and new commits are pushed::

    $ tox -e py39  # run tests on Python 3.9
    $ tox -e py310-upstream-doctest  # run tests on Python 3.10, including doctests, with upstream dependencies
    $ tox -e py311 -- -m "not slow  # run tests on Python 3.11, excluding "slow" marked tests
    $ tox -e py312-numba -- -m "not slow  # run tests on Python 3.12, installing upstream `numba`, excluding "slow" marked tests
    $ tox -e notebooks_doctests  # run tests using the base Python on doctests and evaluate all notebooks
    $ tox -e offline  # run tests using the base Python, excluding tests requiring internet access

    $ tox -m test  # run all builds listed above

   .. warning::
    Starting from `xclim` v0.46.0, when running tests with `tox`, any `pytest` markers passed to `pyXX` builds (e.g. `-m "not slow"`) must be passed to `tox` directly. This can be done as follows::

        $ tox -e py38 -- -m "not slow"

    The exceptions to this rule are:
      `notebooks_doctests`: this configuration does not pass test  markers to its `pytest` call.
      `offline`: this configuration runs by default with the `-m "not requires_internet"` test marker. Be aware that running `tox` and manually setting a `pytest` marker will override this default.

   .. note::
    `xclim` tests are organized to support the `pytest-xdist`_ plugin for distributed testing across workers or CPUs.
    In order to benefit from multiple processes, add the flag `--numprocesses=auto` or `-n auto` to your `pytest` calls.

    When running tests via `tox`, `numprocesses` is set to the number of logical cores available (`numprocesses=logical`), with a maximum amount of `8`.

#. Docs should also be tested to ensure that the documentation will build correctly on ReadTheDocs. This can be performed in a number of ways::

    # To run in a contained virtualenv environment
    $ tox -e docs
    # or, alternatively, to build the docs directly
    $ make docs

   .. note::

    When building the documentation, the default behaviour is to evaluate notebooks ('`nbsphinx_execute = "auto"`'), rather than simply parse the content ('`nbsphinx_execute = "never"`').
    Due to their complexity, this is a very computationally demanding task and should only be performed when necessary (i.e.: when the notebooks have been modified).

    In order to speed up documentation builds, setting a value for the environment variable "`SKIP_NOTEBOOKS`" (e.g. "`$ export SKIP_NOTEBOOKS=1`") will prevent the notebooks from being evaluated on all subsequent "`$ tox -e docs`" or "`$ make docs`" invocations.

#. After clearing the previous checks, commit your changes and push your branch to GitHub::

    $ git add *
    $ git commit -m "Your detailed description of your changes."

   If installed, `pre-commit` will run checks at this point:

   * If no errors are found, changes will be committed.
   * If errors are found, modifications will be made and warnings will be raised if intervention is needed.
   * After addressing errors and effecting changes, simply `git commit` again::

        $ git push origin name-of-your-bugfix-or-feature

#. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, please follow these guidelines:

#. Open an *issue* on our `GitHub repository`_ with your issue that you'd like to fix or feature that you'd like to implement.

#. Perform the changes, commit and push them either to new a branch within `Ouranosinc/xclim` or to your personal fork of xclim.

   .. warning::
    Try to keep your contributions within the scope of the issue that you are addressing.
    While it might be tempting to fix other aspects of the library as it comes up, it's better to simply to flag the problems in case others are already working on it.

    Consider adding a "**# TODO:**" or "**# FIXME:**" comment if the need arises.

#. Pull requests should raise test coverage for the xclim library. Code coverage is an indicator of how extensively tested the library is.

   If you are adding a new set of functions, they **must be tested** and **coverage percentage should not significantly decrease.**

#. If the pull request adds functionality, your functions should include docstring explanations. So long as the docstrings are syntactically correct, sphinx-autodoc will be able to automatically parse the information. Please ensure that the docstrings and documentation adhere to the following standards (badly formed docstrings will fail build tests):

   * `numpydoc`_
   * `reStructuredText (ReST)`_

   .. note::
    If you aren't accustomed to writing documentation in reStructuredText (`.rst`), we encourage you to spend a few minutes going over the
    incredibly well-summarized `reStructuredText Primer`_ from the sphinx-doc maintainer community.

#. The pull request should work for Python 3.9, 3.10, 3.11, and 3.12 as well as raise test coverage.
   Pull requests are also checked for documentation build status and for `PEP8`_ compliance.

   The build statuses and build errors for pull requests can be found at: https://github.com/Ouranosinc/xclim/actions

   .. warning::
    PEP8, black, pytest (with xdoctest) and pydocstyle (for numpy docstrings) conventions are strongly enforced.
    Ensure that your changes pass all tests prior to pushing your final commits to your branch.
    Code formatting errors are treated as build errors and will block your pull request from being accepted.

#. The version changes (CHANGES.rst) should briefly describe changes introduced in the Pull request. Changes should be organized by type (ie: `New indicators`, `New features and enhancements`, `Breaking changes`, `Bug fixes`, `Internal changes`) and the GitHub Pull Request, GitHub Issue. Your name and/or GitHub handle should also be listed among the contributors to this version. This can be done as follows::

     Contributors to this version: John Jacob Jingleheimer Schmidt (:user:`username`).

     Internal changes
     ^^^^^^^^^^^^^^^^
     * Updated the contribution guidelines. (:issue:`868`, :pull:`869`).

   If this is your first contribution to `Ouranosinc/xclim`, we ask that you also add your name to the `AUTHORS.rst <https://github.com/Ouranosinc/xclim/blob/main/AUTHORS.rst>`_, under *Contributors* as well as to the `.zenodo.json <https://github.com/Ouranosinc/xclim/blob/main/.zenodo.json>`_, at the end of the *creators* block.

Updating Testing Data
~~~~~~~~~~~~~~~~~~~~~

If your code changes require changes to the testing data of `xclim` (i.e.: modifications to existing datasets or new datasets), these changes must be made via a Pull Request at the `xclim-testdata repository`_.

`xclim` allows for developers to test specific branches/versions of `xclim-testdata` via the `XCLIM_TESTDATA_BRANCH` environment variable, either through export, e.g.::

    $ export XCLIM_TESTDATA_BRANCH="my_new_branch_of_testing_data"

    $ pytest
    # or, alternatively:
    $ tox

or by setting the variable at runtime::

    $ env XCLIM_TESTDATA_BRANCH="my_new_branch_of_testing_data" pytest
    # or, alternatively:
    $ env XCLIM_TESTDATA_BRANCH="my_new_branch_of_testing_data" tox

This will ensure that tests load the testing data from this branch before running.

If you anticipate not having internet access, we suggest prefetching the testing data from `xclim-testdata repository`_ and storing it in your local cache. This can be done by running the following console command::

    $ xclim prefetch_testing_data

If your development branch relies on a specific branch of `Ouranosinc/xclim-testdata`, you can specify this using environment variables::

    $ export XCLIM_TESTDATA_BRANCH="my_new_branch_of_testing_data"
    $ xclim prefetch_testing_data

or, alternatively, with the `--branch` option::

    $ xclim prefetch_testing_data --branch my_new_branch_of_testing_data

If you wish to test a specific branch using GitHub CI, this can be set in `.github/workflows/main.yml`:

.. code-block:: yaml

    env:
      XCLIM_TESTDATA_BRANCH: my_new_branch_of_testing_data

.. warning::
    In order for a Pull Request to be allowed to merge to main development branch, this variable must match the latest tagged commit name on `xclim-testdata repository`_.
    We suggest merging changed testing data first, tagging a new version of `xclim-testdata`, then re-running tests on your Pull Request at `Ouranosinc/xclim` with the newest tag.

Running Tests in Offline Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`xclim` testing is designed with the assumption that the machine running the tests has internet access. Many calls to `xclim` functions will attempt to download data or verify checksums from the `Ouranosinc/xclim-testdata` repository.
This can be problematic for developers working on features where internet access is not reliably available.

If you wish to ensure that your feature or bugfix can be developed without internet access, `xclim` leverages the `pytest-socket`_ plugin so that testing can be run in "offline" mode by invoking pytest with the following options::

    $ pytest --disable-socket --allow-unix-socket -m "not requires_internet"

or, alternatively, using `tox` ::

    $ tox -e offline

These options will disable all network calls and skip tests marked with the `requires_internet` marker.
The `--allow-unix-socket` option is required to allow the `pytest-xdist`_ plugin to function properly.

Tips
----

To run a subset of tests, we suggest a few approaches. For running only a test file::

    $ pytest tests/test_xclim.py

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

**Release** is a keyword used to specify the degree of production readiness (`dev` [, and optionally, `release`]). *Only versions built from the main development branch will ever have this marker!*

**Build** is a keyword used to specify the build number. *Only versions built from the main development branch will ever have this number!*

An increment to the Major or Minor will reset the Release to `beta`. When a build is promoted above `beta` (ie: the release/stable version), it's a good idea to push this version towards PyPI.

Packaging and Deployment
------------------------

This section serves as a reminder for the maintainers on how to prepare the library for a tagged version and how to deploy packages to TestPyPI and PyPI.

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the pip-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

From a new branch (e.g. `prepare-v123`), open a Pull Request and make sure all your changes to support a new version are committed (**update the entry for newest version in CHANGES.rst**), Then run::

    $ bump-my-version bump <option>  # possible options: major / minor / patch / release / build

These commands will increment the version and create a commit with an autogenerated message.

For PyPI releases/stable versions, ensure that the last version bumping command run is `$ bump-my-version bump release` to remove the `-dev`. These changes can now be merged to the `prepare-v123` branch::

    $ git push origin prepare-v123

With this performed, we can tag a version that will act as the GitHub-provided stable source archive. **Be sure to only tag from the `main` branch when all changes from PRs have been merged!** The commands needed are::

    $ git tag v1.2.3
    $ git push --tags

.. note::
    Starting from October, 2021, all tags pushed to GitHub will trigger a build and publish a package to TestPyPI by default. TestPyPI is a testing ground that is not indexed or easily available to `pip`. The test package can be found at `xclim on TestPyPI`_.

The Automated Approach
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to package `xclim` is to "publish" a version on GitHub. GitHub CI Actions are presently configured to build the library and publish the packages on PyPI automatically.

When publishing on GitHub, maintainers will need to generate the release notes for the current version, replacing the ``:issue:``, ``:pull:``, and ``:user:`` tags.
The `xclim` CLI offers a helper function for performing this action::

    # For Markdown format (needed when publishing a new version on GitHub):
    $ xclim release_notes -m
    # For ReStructuredText format (offered for convenience):
    $ xclim release_notes -r

.. note::
    The changelog should not extend past those entries relevant for the current version.

.. warning::
    A published version on PyPI can never be overwritten. Be sure to verify that the package published at https://test.pypi.org/project/xclim/ matches expectations before publishing a version on GitHub.

The Manual Approach
~~~~~~~~~~~~~~~~~~~

The manual approach to library packaging for general support (pip wheels) requires that the `flit`_ library is installed.

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

.. _`GitHub Repository`: https://github.com/Ouranosinc/xclim
.. _`PEP8`: https://peps.python.org/pep-0008/
.. _`flit`: https://flit.pypa.io/en/stable/index.html
.. _`numpydoc`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _`pytest-socket`: https://github.com/miketheman/pytest-socket
.. _`pytest-xdist`: https://pytest-xdist.readthedocs.io/en/latest/
.. _`reStructuredText (ReST)`: https://www.jetbrains.com/help/pycharm/using-docstrings-to-specify-types.html
.. _`reStructuredText Primer`: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _`sphinxcontrib-bibtex`: https://sphinxcontrib-bibtex.readthedocs.io
.. _`xclim on TestPyPI`: https://test.pypi.org/project/xclim/
.. _`xclim Discussions page`: https://github.com/Ouranosinc/xclim/discussions
.. _`xclim-testdata repository`: https://github.com/Ouranosinc/xclim-testdata
