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

xclim's structure makes it easy to create and register new user-defined indices and indicators. Refer to the :ref:`Customizing and controlling xclim` page for more information.

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

.. warning::
     If you plan to implement new indicators into xclim, be aware that metadata translations
     for all official xclim languages (for now only French) must be provided, or else the tests
     will fail and the PR will not be mergeable. See :ref:`Internationalization` for more details.
     Don't hesitate to ask for help in your PR for this task!

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/Ouranosinc/xclim/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xclim could always use more documentation, whether as part of the
official xclim docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/Ouranosinc/xclim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `xclim` for local development.

1. Fork the `xclim` repo on GitHub.

2. Clone your fork locally::

    $ git clone git@github.com:Ouranosinc/xclim.git
    $ cd xclim/

3. Create a development environment. Assuming you have `virtualenvwrapper` installed, this is how you set up your fork for local development::

    # For virtualenv environments (ensure that you have the necessary system libraries).
    $ mkvirtualenv xclim
    $ pip install -e .[dev]

    # For Anaconda/Miniconda environments:
    $ conda create -n xclim python=3.7 --file=environment.yml
    $ pip install -e .[dev]

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally!

5. When you're done making changes, check that you verify your changes with `black`, `pydocstyle`, and run the tests, including testing other available Python versions with `tox`::

    $ black --check --target-version py37 xclim xclim/testing/tests
    $ black --check --target-version py37 --include "\.ipynb$" docs
    $ flake8 xclim xclim/testing/tests
    $ pytest --nbval docs/notebooks
    $ pytest --rootdir=xclim/testing/tests --xdoctest xclim
    $ pydocstyle --convention=numpy --match='(?!test_).*\.py' xclim
    $ tox

6. Before committing your changes, we ask that you install `pre-commit` in your development environment. `Pre-commit` runs git hooks that ensure that your code resembles that of the project and catches and corrects any small errors or inconsistencies when you `git commit`::

    # To install the necessary pre-commit hooks:
    $ pre-commit install
    # To run pre-commit hooks manually:
    $ pre-commit run --all-files

7. Commit your changes and push your branch to GitHub::

    $ git add *

    $ git commit -m "Your detailed description of your changes."
    # If installed, `pre-commit` will run checks at this point:
    # If no errors are found, changes will be committed.
    # If errors are found, modifications will be made and warnings will be raised if intervention is needed. After changes, simply `git commit` again.

    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

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

5. The pull request should work for Python 3.7, 3.8, and 3.9 as well as raise test coverage.
   Pull requests are also checked for documentation build status and for `PEP8`_ compliance.

   The build statuses and build errors for pull requests can be found at:
    https://github.com/Ouranosinc/xclim/actions

.. warning::
    PEP8, black, pytest (with xdoctest) and pydocstyle (for numpy docstrings) conventions are strongly enforced.
    Ensure that your changes pass all tests prior to pushing your final commits to your branch.
    Code formatting errors are treated as build errors and will block your pull request from being accepted.

6. The version changes (HISTORY.rst) should briefly describe changes introduced in the Pull request. Changes should be organized by type
   (ie: `New Indicators`, `New features and enhancements`, `Breaking changes`, `Bug fixes`, `Internal changes`) and the GitHub Pull Request,
   GitHub Issue. Your name and/or GitHub handle should also be listed among the contributors to this version. This can be done as follows::

     Contributors to this version: John Jacob Jingleheimer Schmidt (:user:`username`).

     Internal changes
     ~~~~~~~~~~~~~~~~
     * Updated the contribution guidelines. (:issue:`868`, :pull:`869`).

   If this is your first contribution to Ouranosinc/xclim, we ask that you also add your name to the `AUTHORS.rst <https://github.com/Ouranosinc/xclim/blob/master/AUTHORS.rst>`_, under *Contributors*.

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

    $ bumpversion <option> # possible options: major / minor / patch / release
    $ git add *
    $ git commit -m "Bumped version to v1.2.3-XYZ"

For PyPI releases/stable versions, ensure that the last version bumping command run is `$ bumpversion release`.
These changes can now be merged to the main development branch::

    $ git push

With this performed, we can tag a version that will act as the GitHub-provided stable source archive::

    $ git tag v1.2.3-XYZ
    $ git push --tags

.. note::
    Starting from October, 2021, all tags pushed to GitHub will trigger a build and publish a package to TestPyPI by default. TestPyPI is a testing ground that is not indexed or easily available to `pip`.
    The test package can be found at: https://test.pypi.org/project/xclim/

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

The manual approach to library packaging for general support (pip wheels) requires the following packages installed:
 * setuptools
 * wheel
 * twine

From the command line on your Linux distribution, simply run the following from the clone's main dev branch::

    # To build the packages (sources and wheel)
    $ python setup.py sdist bdist_wheel

    # To upload to PyPI
    $ twine upload dist/*

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

.. _`numpydoc`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`reStructuredText (ReST)`: https://www.jetbrains.com/help/pycharm/using-docstrings-to-specify-types.html
.. _`GitHub Repository`: https://github.com/Ouranosinc/xclim
.. _`PEP8`: https://www.python.org/dev/peps/pep-0008/
