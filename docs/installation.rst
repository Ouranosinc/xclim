.. highlight:: shell

============
Installation
============

Stable release
--------------

To install `xclim` via `pip`, run this command in your terminal:

.. code-block:: shell

    $ pip install xclim

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io/
.. _Python installation guide: https://docs.python-guide.org/starting/installation/

Anaconda release
----------------

For ease of installation across operating systems, we also offer an Anaconda Python package hosted on conda-forge.
This version tends to be updated at around the same frequency as the PyPI-hosted library, but can lag by a few days at times.

`xclim` can be installed from conda-forge wth the following:

.. code-block:: shell

    $ conda install -c conda-forge xclim

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

    $ sudo apt-get install libgdal-dev

If on Anaconda Python, `GDAL` will be installed if needed as a `clisops` dependency.

Both of these libraries are available on PyPI and conda-forge:

.. code-block:: shell

    $ pip install flox clisops
    # Or, alternatively:
    $ conda install -c conda-forge flox clisops

.. _GDAL: https://gdal.org/download.html#binaries
.. _Pangeo: https://pangeo.io/

Upstream Dependencies
^^^^^^^^^^^^^^^^^^^^^

`xclim` is regularly tested against the main development branches of a handful of key base libraries (`cftime`, `flox`, `pint`, `xarray`).
For convenience, these libraries can be installed alongside `xclim` using the following `pip`-install command:

.. code-block:: shell

    $ pip install -r requirements_upstream.txt

Or, alternatively:

.. code-block:: shell

    $ make upstream

.. _flox: https://github.com/xarray-contrib/flox
.. _clisops: https://github.com/roocs/clisops

Experimental SDBA Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`xclim` also offers support for a handful of experimental adjustment methods to extend :doc:`xclim.sdba <sdba>`, available only if some additional libraries are installed. These libraries are completely optional.

One experimental library is `SBCK`_. `SBCK` is available from PyPI but has one complex dependency: `Eigen3`_.
As `SBCK` is compiled at installation time, a **C++** compiler (`GCC`, `Clang`, `MSVC`, etc.) must also be available.

On Debian/Ubuntu, `Eigen3` can be installed via `apt`:

.. code-block:: shell

    $ sudo apt-get install libeigen3-dev

Eigen3 is also available on conda-forge, so, if already using Anaconda, one can do:

.. code-block:: shell

    $ conda install -c conda-forge eigen

Afterwards, `SBCK` can be installed from PyPI using `pip`:

.. code-block:: shell

    $ pip install SBCK

Another experimental function :py:indicator:`xclim.sdba.property.first_eof` makes use of the `eofs`_ library, which is available on both PyPI and conda-forge:

.. code-block:: shell

    $ pip install eofs
    # or alternatively,
    $ conda install -c conda-forge eofs

.. _eofs: https://ajdawson.github.io/eofs/
.. _SBCK: https://github.com/yrobink/SBCK
.. _Eigen3: https://eigen.tuxfamily.org/index.php

From sources
------------

.. warning::
    For Python3.11+ users: Many of the required scientific libraries do not currently have wheels that support the latest
    python. In order to ensure that installation of xclim doesn't fail, we suggest installing the `Cython` module
    before installing xclim in order to compile necessary libraries from source packages.

The sources for xclim can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: shell

    $ git clone git@github.com:Ouranosinc/xclim.git

Or download the `tarball`_:

.. code-block:: shell

    $ curl -OL https://github.com/Ouranosinc/xclim/tarball/master

Once you have extracted a copy of the source, you can install it with pip:

.. code-block:: shell

    $ pip install -e ".[dev]"

Alternatively, you can also install a local development copy via `flit`_:

.. code-block:: shell

    $ flit install [--symlink] xclim

.. _Github repo: https://github.com/Ouranosinc/xclim
.. _tarball: https://github.com/Ouranosinc/xclim/tarball/master
.. _flit: https://flit.pypa.io/en/stable

Creating a Conda environment
----------------------------

To create a conda environment including `xclim`'s dependencies and several optional libraries (notably: `clisops`, `eigen`, `eofs`, and `flox`) and development dependencies, run the following command from within your cloned repo:

.. code-block:: console

    $ conda env create -n my_xclim_env python=3.8 --file=environment.yml
    $ conda activate my_xclim_env
    (my_xclim_env) $ pip install -e .
