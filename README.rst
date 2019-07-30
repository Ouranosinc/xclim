======
xclim
======

.. image:: _static/_images/xclim-logo.png
        :align: center
        :target: _static/_images/xclim-logo.png
        :alt: xclim

.. image:: https://img.shields.io/pypi/v/xclim.svg
        :align: center
        :target: https://pypi.python.org/pypi/xclim
        :alt: Python Package Index Build

.. image:: https://img.shields.io/conda/vn/conda-forge/xclim.svg
        :align: center
        :target: https://anaconda.org/conda-forge/xclim
        :alt: Conda-forge Build Version

----

|build| |coveralls| |codefactor| |zenodo| |black|

* Documentation: |docs|
* Chat with us: |gitter|

``xclim`` is a library of functions to compute climate indices. It is built using xarray and can benefit from the parallelization handling provided by dask. Its objective is to make it as simple as possible for users to compute indices from large climate datasets and for scientists to write new indices with very little boilerplate.

For example, the following would compute monthly mean temperature from daily mean temperature:

.. code-block:: python

  import xclim
  import xarray as xr
  ds = xr.open_dataset(filename)
  tg = xclim.icclim.TG(ds.tas, freq='YS')

For applications where meta-data and missing values are important to get right, ``xclim`` also provides a class for each index that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output. This provides a mechanism for users to customize the indices to their own specifications and preferences.

``xclim`` is still in active development at the moment, but is close to being production ready. We're are currently nearing a release candidate (as of Q2 2019). If you're interested in participating to the development, please leave us a message on the issue tracker.

* Free software: |license|

Credits
-------

This work is made possible thanks to the contributions of the Canadian Center for Climate Services.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


.. |gitter| image:: https://badges.gitter.im/Ouranosinc/xclim.svg
        :target: https://gitter.im/Ouranosinc/xclim?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
        :alt: Gitter Chat


.. |build| image:: https://img.shields.io/travis/Ouranosinc/xclim.svg
        :target: https://travis-ci.org/Ouranosinc/xclim
        :alt: Build Status

.. |coveralls| image:: https://coveralls.io/repos/github/Ouranosinc/xclim/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/xclim
        :alt: Coveralls

.. |codefactor| image:: https://www.codefactor.io/repository/github/ouranosinc/xclim/badge
        :target: https://www.codefactor.io/repository/github/ouranosinc/xclim
        :alt: CodeFactor

.. |docs| image:: https://readthedocs.org/projects/xclim/badge
        :target: https://xclim.readthedocs.io/en/latest
        :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/142608764.svg
        :target: https://zenodo.org/badge/latestdoi/142608764
        :alt: DOI

.. |license| image:: https://img.shields.io/github/license/Ouranosinc/xclim.svg
        :target: https://github.com/Ouranosinc/xclim/blob/master/LICENSE
        :alt: License

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/python/black
        :alt: Python Black
