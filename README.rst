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

.. image:: https://readthedocs.org/projects/xclim/badge
        :target: https://xclim.readthedocs.io/en/latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/Ouranosinc/xclim.svg
        :target: https://github.com/bird-house/birdhouse-docs/blob/master/LICENSE
        :alt: License

``xclim`` is a library of functions computing climate indices. It is based on xarray and can benefit from the parallelization provided by dask. It's objective is to make it as simple as possible for users to compute indices from large climate datasets, and for scientists to write new indices with very little boilerplate.

For example, the following would compute monthly mean temperature from daily mean temperature:

.. code-block:: python

  import xclim
  import xarray as xr
  ds = xr.open_dataset(filename)
  tg = xclim.icclim.TG(ds.tas, freq='YS')

For applications where meta-data and missing values are important to get right, ``xclim`` also provides a class for each index that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output. This provides a mechanism for users to customize the indices to their own specifications and preferences.  

``xclim`` is still in active development at the moment, but is close to be production ready. We're are at a beta release (as of Q1 2019). If you're interested in participating to the development, please leave us a message on the issue tracker.


* Free software: Apache Software License 2.0
* Documentation: https://xclim.readthedocs.io.


Credits
-------

This work is made possible by the Canadian Center for Climate Services. 

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
