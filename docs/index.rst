xclim Official Documentation
============================

``xclim`` is a library of functions to compute climate indices from observations or model simulations. It is built using `xarray`_ and can benefit from the parallelization handling provided by `dask`_. Its objective is to make it as simple as possible for users to compute indices from large climate datasets and for scientists to write new indices with very little boilerplate.

For applications where meta-data and missing values are important to get right, xclim provides a class for each index that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output. This also provides a mechanism for users to customize the indices to their own specifications and preferences.

xclim currently provides over 50 indices related to mean, minimum and maximum daily temperature, daily precipitation, streamflow and sea ice concentration.

.. _xarray: https://docs.xarray.dev/
.. _dask: https://www.dask.org/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   notebooks/usage
   notebooks/index
   indicators
   indices
   checks
   notebooks/units
   internationalization
   notebooks/cli
   sdba
   analogues
   contributing
   authors
   history
   references

.. toctree::
   :maxdepth: 2
   :caption: User API

   api

.. toctree::
   :maxdepth: 1
   :caption: All Modules

   modules

.. only:: html

    Indices and tables
    ==================
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
