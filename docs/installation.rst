============
Installation
============

Stable release
--------------

To install `xclim` via `pip`, run this command in your terminal:

.. code-block:: shell

    python -m pip install xclim

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io/
.. _Python installation guide: https://docs.python-guide.org/starting/installation/

Anaconda release
----------------

For ease of installation across operating systems, we also offer an Anaconda Python package hosted on conda-forge.
This version tends to be updated at around the same frequency as the PyPI-hosted library, but can lag by a few days at times.

`xclim` can be installed from conda-forge with the following:

.. code-block:: shell

    conda install -c conda-forge xclim

.. _extra-dependencies:

Extra Dependencies
------------------

Speedups and Helper Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To improve performance of `xclim`, we highly recommend you also install `flox`_ (see: :doc:`flox API <flox:api>`).
This package seamlessly integrates into `xarray` and significantly improves the performance of the grouping and resampling algorithms, especially when using `dask` on large datasets.

For grid subsetting, we also recommend using the tools found in `clisops`_ (see: :doc:`clisops.core.subset API <clisops:api>`) for spatial manipulation of geospatial data. `clisops` began as a component of `xclim` and is designed to work alongside `xclim` and the `Pangeo`_ stack (`xarray`, `dask`, `jupyter`). In order to install `clisops`, the `GDAL`_ system libraries must be available.

On Debian/Ubuntu, `GDAL` can be installed via `apt`:

.. code-block:: shell

    sudo apt-get install libgdal-dev

If on Anaconda Python, `GDAL` will be installed if needed as a `clisops` dependency.

Both of these libraries are available on PyPI and conda-forge:

.. code-block:: shell

    python -m pip install flox clisops

Or, alternatively:

.. code-block:: shell

    conda install -c conda-forge flox clisops

.. _GDAL: https://gdal.org/download.html#binaries
.. _Pangeo: https://pangeo.io/

Upstream Dependencies
^^^^^^^^^^^^^^^^^^^^^

`xclim` is regularly tested against the main development branches of a handful of key base libraries (`cftime`, `flox`, `pint`, `xarray`).
For convenience, these libraries can be installed alongside `xclim` using the following `pip`-install command:

.. code-block:: shell

    python -m pip install -r requirements_upstream.txt

Or, alternatively:

.. code-block:: shell

    make upstream

.. _flox: https://github.com/xarray-contrib/flox
.. _clisops: https://github.com/roocs/clisops

From Sources
------------

.. warning::

    While `xclim` strives to be compatible with latest releases and development versions of upstream libraries, many of the required base libraries (`numpy`, `scipy`, `numba`, etc.) may lag by several months before supporting the latest minor releases of Python.

    In order to ensure that installation of `xclim` doesn't fail, we suggest installing the `Cython` module before installing `xclim` in order to compile necessary libraries from their source packages, if required.

The sources for xclim can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: shell

    git clone git@github.com:Ouranosinc/xclim.git

Or download the `tarball`_:

.. code-block:: shell

    curl -OL https://github.com/Ouranosinc/xclim/tarball/main

Once you have extracted a copy of the source, you can install it with `pip`_:

.. code-block:: shell

    python -m pip install -e ".[all]"

Alternatively, you can also install a local development copy via `flit`_:

.. code-block:: shell

    flit install [--symlink] xclim

.. _Github repo: https://github.com/Ouranosinc/xclim
.. _tarball: https://github.com/Ouranosinc/xclim/tarball/main
.. _flit: https://flit.pypa.io/en/stable

Creating a Conda Environment
----------------------------

To create a conda environment including `xclim`'s dependencies and several optional libraries (notably: `clisops`, `eigen`, `sbck`, and `flox`) and development dependencies, run the following command from within your cloned repo:

.. code-block:: console

    conda env create -n my_xclim_env --file=environment.yml
    conda activate my_xclim_env
    (my_xclim_env) python -m pip install --no-deps -e  .
