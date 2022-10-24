==========================================
xclim: Climate services library |logo|
==========================================

+----------------+----------------------------+-----------------------------------+-----------------------------------------------------+
| Downloads      | Documentation and Support  | Open Source                       | Development Status                                  |
+----------------+----------------------------+-----------------------------------+-----------------------------------------------------+
| |pypi| |conda| | |docs| |gitter|            | |license| |fair| |fossa| |zenodo| | |build| |pre-commit| |coveralls| |black| |security| |
+----------------+----------------------------+-----------------------------------+-----------------------------------------------------+


`xclim` is an operational Python library for climate services, providing numerous climate-related indicator tools
with an extensible framework for constructing custom climate indicators, statistical downscaling and bias
adjustment of climate model simulations, as well as climate model ensemble analysis tools.

xclim is built using `xarray`_ and can seamlessly benefit from the parallelization handling provided by `dask`_.
Its objective is to make it as simple as possible for users to perform typical climate services data treatment workflows.
Leveraging xarray and dask, users can easily bias-adjust climate simulations over large spatial domains or compute indices from large climate datasets.

For example, the following would compute monthly mean temperature from daily mean temperature:

.. code-block:: python

    import xclim
    import xarray as xr

    ds = xr.open_dataset(filename)
    tg = xclim.atmos.tg_mean(ds.tas, freq="YS")

For applications where metadata and missing values are important to get right, xclim provides a class for each index
that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output.
This also provides a mechanism for users to customize the indices to their own specifications and preferences.
xclim currently provides over 150 indices related to mean, minimum and maximum daily temperature, daily precipitation,
streamflow and sea ice concentration, numerous bias-adjustment algorithms, as well as a dedicated module for ensemble analysis.

.. _xarray: https://docs.xarray.dev/
.. _dask: https://docs.dask.org/

Documentation
-------------
The official documentation is at https://xclim.readthedocs.io/

Contributing
------------
xclim is in active development and is being used in production by climate services specialists around the world.

* If you're interested in participating in the development of xclim by suggesting new features, new indices or report bugs, please leave us a message on the `issue tracker`_. There is also a chat room on gitter (|gitter|).

* If you would like to contribute code or documentation (which is greatly appreciated!), check out the `Contributing Guidelines`_ before you begin!

.. _issue tracker: https://github.com/Ouranosinc/xclim/issues
.. _Contributing Guidelines: https://github.com/Ouranosinc/xclim/blob/master/.github/CONTRIBUTING.rst

How to cite this library
------------------------
If you wish to cite xclim in a research publication, we kindly ask that you use the bibliographical reference information available through `Zenodo`_

.. _Zenodo: https://doi.org/10.5281/zenodo.2795043

Credits
-------

xclim development is funded through Ouranos_, Environment and Climate Change Canada (ECCC_), the `Fonds vert`_ and the Fonds d’électrification et de changements climatiques (FECC_), the Canadian Foundation for Innovation (CFI_), and the Fonds de recherche du Québec (FRQ_).

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _audreyr/cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage/
.. _CFI: https://www.innovation.ca/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter/
.. _ECCC: https://www.canada.ca/en/environment-climate-change.html
.. _FECC: https://www.environnement.gouv.qc.ca/ministere/fonds-electrification-changements-climatiques/index.htm
.. _Fonds vert: https://www.environnement.gouv.qc.ca/ministere/fonds-vert/index.htm
.. _FRQ: https://frq.gouv.qc.ca/
.. _Ouranos: https://ouranos.ca/

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
        :target: https://github.com/python/black
        :alt: Python Black

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xclim/master/_static/_images/xclim-logo-small.png
        :target: https://github.com/Ouranosinc/xclim
        :alt: Xclim

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/xclim/master.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/xclim/master
        :alt: pre-commit.ci status
