======================================
xclim: Climate services library |logo|
======================================

+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |conda| |versions|                           |
+----------------------------+-----------------------------------------------------+
| Documentation and Support  | |docs| |gitter|                                     |
+----------------------------+-----------------------------------------------------+
| Open Source                | |license| |fair| |zenodo| |pyOpenSci| |joss|        |
+----------------------------+-----------------------------------------------------+
| Coding Standards           | |black| |pre-commit| |security| |fossa|             |
+----------------------------+-----------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                        |
+----------------------------+-----------------------------------------------------+

`xclim` is an operational Python library for climate services, providing numerous climate-related indicator tools
with an extensible framework for constructing custom climate indicators, statistical downscaling and bias
adjustment of climate model simulations, as well as climate model ensemble analysis tools.

`xclim` is built using `xarray`_ and can seamlessly benefit from the parallelization handling provided by `dask`_.
Its objective is to make it as simple as possible for users to perform typical climate services data treatment workflows.
Leveraging xarray and dask, users can easily bias-adjust climate simulations over large spatial domains or compute indices from large climate datasets.

For example, the following would compute monthly mean temperature from daily mean temperature:

.. code-block:: python

    import xclim
    import xarray as xr

    ds = xr.open_dataset(filename)
    tg = xclim.atmos.tg_mean(ds.tas, freq="MS")

For applications where metadata and missing values are important to get right, xclim provides a class for each index
that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output.
This also provides a mechanism for users to customize the indices to their own specifications and preferences.
`xclim` currently provides over 150 indices related to mean, minimum and maximum daily temperature, daily precipitation,
streamflow and sea ice concentration, numerous bias-adjustment algorithms, as well as a dedicated module for ensemble analysis.

.. _xarray: https://docs.xarray.dev/
.. _dask: https://docs.dask.org/

Quick Install
-------------
`xclim` can be installed from PyPI:

.. code-block:: shell

    $ pip install xclim

or from Anaconda (conda-forge):

.. code-block:: shell

    $ conda install -c conda-forge xclim

Documentation
-------------
The official documentation is at https://xclim.readthedocs.io/

How to make the most of xclim: `Basic Usage Examples`_ and `In-Depth Examples`_.

.. _Basic Usage Examples: https://xclim.readthedocs.io/en/stable/notebooks/usage.html
.. _In-Depth Examples: https://xclim.readthedocs.io/en/stable/notebooks/index.html

Conventions
-----------
In order to provide a coherent interface, `xclim` tries to follow different sets of conventions. In particular, input data should follow the `CF conventions`_ whenever possible for variable attributes. Variable names are usually the ones used in `CMIP6`_, when they exist.

However, xclim will *always* assume the temporal coordinate is named "time". If your data uses another name (for example: "T"), you can rename the variable with:

.. code-block:: python

    ds = ds.rename(T="time")

.. _CF Conventions: http://cfconventions.org/
.. _CMIP6: https://clipc-services.ceda.ac.uk/dreq/mipVars.html

Contributing to xclim
---------------------
`xclim` is in active development and is being used in production by climate services specialists around the world.

* If you're interested in participating in the development of `xclim` by suggesting new features, new indices or report bugs, please leave us a message on the `issue tracker`_. There is also a chat room on gitter (|gitter|).

* If you would like to contribute code or documentation (which is greatly appreciated!), check out the `Contributing Guidelines`_ before you begin!

.. _issue tracker: https://github.com/Ouranosinc/xclim/issues
.. _Contributing Guidelines: https://github.com/Ouranosinc/xclim/blob/master/CONTRIBUTING.rst

How to cite this library
------------------------
If you wish to cite `xclim` in a research publication, we kindly ask that you refer to our article published in The Journal of Open Source Software (`JOSS`_): https://doi.org/10.21105/joss.05415

To cite a specific version of `xclim`, the bibliographical reference information can be found through `Zenodo`_

.. _JOSS: https://joss.theoj.org/
.. _Zenodo: https://doi.org/10.5281/zenodo.2795043

License
-------
This is free software: you can redistribute it and/or modify it under the terms of the `Apache License 2.0`_. A copy of this license is provided in the code repository (`LICENSE`_).

.. _Apache License 2.0: https://opensource.org/license/apache-2-0/
.. _LICENSE: https://github.com/Ouranosinc/xclim/blob/master/LICENSE

Credits
-------
`xclim` development is funded through Ouranos_, Environment and Climate Change Canada (ECCC_), the `Fonds vert`_ and the Fonds d'électrification et de changements climatiques (FECC_), the Canadian Foundation for Innovation (CFI_), and the Fonds de recherche du Québec (FRQ_).

This package was created with Cookiecutter_ and the `audreyfeldroy/cookiecutter-pypackage`_ project template.

.. _audreyfeldroy/cookiecutter-pypackage: https://github.com/audreyfeldroy/cookiecutter-pypackage/
.. _CFI: https://www.innovation.ca/
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter/
.. _ECCC: https://www.canada.ca/en/environment-climate-change.html
.. _FECC: https://www.environnement.gouv.qc.ca/ministere/fonds-electrification-changements-climatiques/index.htm
.. _Fonds vert: https://www.environnement.gouv.qc.ca/ministere/fonds-vert/index.htm
.. _FRQ: https://frq.gouv.qc.ca/
.. _Ouranos: https://www.ouranos.ca/

.. |pypi| image:: https://img.shields.io/pypi/v/xclim.svg
        :target: https://pypi.python.org/pypi/xclim
        :alt: Python Package Index Build

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/xclim.svg
        :target: https://anaconda.org/conda-forge/xclim
        :alt: Conda-forge Build Version

.. |gitter| image:: https://badges.gitter.im/Ouranosinc/xclim.svg
        :target: https://gitter.im/Ouranosinc/xclim?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
        :alt: Gitter Chat

.. |build| image:: https://github.com/Ouranosinc/xclim/workflows/xclim/badge.svg
        :target: https://github.com/Ouranosinc/xclim/actions
        :alt: Build Status

.. |coveralls| image:: https://coveralls.io/repos/github/Ouranosinc/xclim/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/xclim
        :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/xclim/badge
        :target: https://xclim.readthedocs.io/en/latest
        :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/142608764.svg
        :target: https://zenodo.org/badge/latestdoi/142608764
        :alt: DOI

.. |pyOpenSci| image:: https://tinyurl.com/y22nb8up
        :target: https://github.com/pyOpenSci/software-review/issues/73
        :alt: pyOpenSci

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.05415/status.svg
        :target: https://doi.org/10.21105/joss.05415
        :alt: JOSS

.. |license| image:: https://img.shields.io/github/license/Ouranosinc/xclim.svg
        :target: https://github.com/Ouranosinc/xclim/blob/master/LICENSE
        :alt: License

.. |security| image:: https://bestpractices.coreinfrastructure.org/projects/6041/badge
        :target: https://bestpractices.coreinfrastructure.org/projects/6041
        :alt: Open Source Security Foundation

.. |fair| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow
        :target: https://fair-software.eu
        :alt: FAIR Software Compliance

.. |fossa| image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2FOuranosinc%2Fxclim.svg?type=shield
        :target: https://app.fossa.com/projects/git%2Bgithub.com%2FOuranosinc%2Fxclim?ref=badge_shield
        :alt: FOSSA

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xclim/master/docs/logos/xclim-logo-small.png
        :target: https://github.com/Ouranosinc/xclim
        :alt: Xclim

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/xclim/master.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/xclim/master
        :alt: pre-commit.ci status

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/xclim.svg
        :target: https://pypi.python.org/pypi/xclim
        :alt: Supported Python Versions
