======
xclim
======


.. image:: _static/_images/xclim-logo.png
        :align: center
        :target: _static/_images/xclim-logo.png
        :alt: xclim

.. image:: https://img.shields.io/pypi/v/xclim.svg
        :target: https://pypi.python.org/pypi/xclim
        :alt: Python Package Index Build

.. image:: https://img.shields.io/travis/Ouranosinc/xclim.svg
        :target: https://travis-ci.org/Ouranosinc/xclim
        :alt: Build Status

.. image:: https://coveralls.io/repos/github/Ouranosinc/xclim/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/xclim
        :alt: Coveralls

.. image:: https://www.codefactor.io/repository/github/ouranosinc/xclim/badge
        :target: https://www.codefactor.io/repository/github/ouranosinc/xclim
        :alt: CodeFactor

.. image:: https://www.codefactor.io/repository/github/ouranosinc/xclim/badge
   :target: https://www.codefactor.io/repository/github/ouranosinc/xclim
   :alt: CodeFactor

.. image:: https://readthedocs.org/projects/xclim/badge/?version=latest
        :target: https://xclim.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/Ouranosinc/xclim.svg
        :target: https://github.com/bird-house/birdhouse-docs/blob/master/LICENSE
        :alt: License

``xclim`` is a library of functions computing climate indices It is based on xarray and can benefit from the parallelization provided by dask. It's objective is to make it as simple as possible for users to compute indices from large climate datasets, and for scientists to write new indices with little to no boilerplate.

For example, the following would compute seasonal mean temperature from daily mean temperature:

.. code-block:: python

  import xclim
  import xarray as xr
  ds = xr.open_dataset(filename)
  tg = xclim.icclim.TG(ds, freq='QS-DEC')


* Free software: Apache Software License 2.0
* Documentation: https://xclim.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
