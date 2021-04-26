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

3. Install your local copy into a virtualenv. Assuming you have `virtualenvwrapper` installed, this is how you set up your fork for local development::

    # For virtualenv environments:
    $ mkvirtualenv xclim

    # For Anaconda/Miniconda environments:
    $ conda create -n xclim python=3.6 --file=environment.yml

    $ cd xclim/
    $ pip install -e .[dev]

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally!

5. When you're done making changes, check that you verify your changes with `black`, `pydocstyle`, and run the tests, including testing other available Python versions with `tox`::

    # For virtualenv environments:
    $ pip install black pytest nbval xdoctest pydocstyle tox

    # For Anaconda/Miniconda environments:
    $ conda install -c conda-forge black pytest pydocstyle xdoctest tox

    $ black --check --target-version py37 xclim tests
    $ flake8 xclim tests
    $ pytest --nbval docs/notebooks
    $ pytest --root-dir xclim/testing/tests/ --xdoctest xclim
    $ pydocstyle --convention=numpy --match="(?!test_).*\.py" xclim
    $ tox

6. Before committing your changes, we ask that you install `pre-commit` in your dev environment. `Pre-commit` runs git hooks that ensure that your code resembles that of the project and catches and corrects any small errors or inconsistencies when you `git commit`::

    # For virtualenv environments:
    $ pip install pre-commit

    # For Anaconda/Miniconda environments:
    $ conda install -c conda-forge pre-commit

    # To install the necessary pre-commit hooks:
    $ pre-commit install

7. Commit your changes and push your branch to GitHub::

    $ git add *

    $ git commit -m "Your detailed description of your changes."
    # `pre-commit` will run checks at this point:
    # if no errors are found, changes will be committed.
    # if errors are found, modifications will be made. Simply `git commit` again.

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
   Please ensure that the docstrings adhere to one of the following standards (badly formed docstrings will fail build tests):

   * `numpydoc`_
   * `reStructuredText (ReST)`_

5. The pull request should work for Python 3.7, 3.8, and 3.9 as well as raise test coverage.
   Pull requests are also checked for documentation build status and for `PEP8`_ compliance.

   The build statuses and build errors for pull requests can be found at:
    https://github.com/Ouranosinc/xclim/actions

.. warning::
    PEP8, Black, pytest (with xdoctest) and pydocstyle (for numpy docstrings) conventions are strongly enforced.
    Ensure that your changes pass all tests prior to pushing your final commits to your branch.
    Code formatting errors are treated as build errors and will block your pull request from being accepted.

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

**Release** is a keyword used to specify the degree of production readiness (`beta` [, and optionally, `gamma`])

  An increment to the Major or Minor will reset the Release to `beta`. When a build is promoted above `beta` (ie: release-ready), it's a good idea to push this version towards PyPi.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (**including an entry in HISTORY.rst**).
Then run::

$ bumpversion <option> # possible options: major / minor / patch / release
$ git push
$ git push --tags

Packaging
---------

When test coverage and stability is adequate, maintainers should update the pip-installable package (wheel) on PyPI.
In order to do this, you will need the following libraries installed:

* twine
* setuptools
* wheel

.. TODO::

    Finish the packaging documentation

.. _`numpydoc`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`reStructuredText (ReST)`: https://www.jetbrains.com/help/pycharm/using-docstrings-to-specify-types.html
.. _`GitHub Repository`: https://github.com/Ouranosinc/xclim
.. _`PEP8`: https://www.python.org/dev/peps/pep-0008/
