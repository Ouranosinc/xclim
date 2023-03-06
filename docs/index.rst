xclim Official Documentation
============================

`xclim` is an operational Python library for climate services, providing numerous climate-related indicator tools
with an extensible framework for constructing custom climate indicators, statistical downscaling and bias
adjustment of climate model simulations, as well as climate model ensemble analysis tools.

xclim is built using `xarray`_ and can seamlessly benefit from the parallelization handling provided by `dask`_.
Its objective is to make it as simple as possible for users to perform typical climate services data treatment workflows.
Leveraging xarray and dask, users can easily bias-adjust climate simulations over large spatial domains or compute indices from large climate datasets.

.. _xarray: https://docs.xarray.dev/
.. _dask: https://docs.dask.org/

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   About <readme>
   installation
   Why xclim? <explanation>
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

.. toctree::
   :titlesonly:

   authors
   changes
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
