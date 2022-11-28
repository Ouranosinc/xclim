.. highlight:: shell

============
Installation
============

Stable release
--------------
To install xclim via pip, run this command in your terminal:

.. code-block:: console

    $ pip install xclim

This is the preferred method to install xclim, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io/
.. _Python installation guide: https://docs.python-guide.org/starting/installation/
.. _OSGeo4W installer: https://trac.osgeo.org/osgeo4w/

Anaconda release
----------------
For ease of installation across operating systems, we also offer an Anaconda Python package hosted on conda-forge.
This version tends to be updated at around the same frequency as the pip library, but can lag by a few days at times.

To install the xclim Anaconda binary, run this command in your terminal:

.. code-block:: console

    $ conda install -c conda-forge xclim

.. _extra-dependencies:

Extra dependencies
------------------
To improve performance of xclim, we highly recommend you also install `flox`_ (see: :doc:`flox API <flox:api>`).
This package integrates into xarray and significantly improves the performance of the grouping and resampling algorithms, especially when using `dask` on large datasets.

We also recommend using the subsetting tools in `clisops`_ (see: :doc:`clisops.core.subset API <clisops:api>`) for spatial manipulation of geospatial data.

`xclim` is regularly tested against the main development branches of a handful of key base libraries (xarray, cftime, flox, pint).
For convenience, these libraries can be installed alongside `xclim` using the following pip-installable recipe:

.. code-block::

    $ pip install -r requirements_upstream.txt
    # Or, alternatively:
    $ make upstream

.. _flox: https://github.com/xarray-contrib/flox
.. _clisops: https://github.com/roocs/clisops

Another optional library is `SBCK`_, which provides experimental adjustment methods to extend :doc:`xclim.sdba <sdba>`.
It can't be installed directly from pip or conda and has one complex dependency : `Eigen`_.
Please refer to Eigen's and SBCK's docs for the recommended installation instructions.
However, Eigen is available on conda, so one can do:

.. code-block::

   $ conda install -c conda-forge eigen pybind11
   $ pip install "git+https://github.com/Ouranosinc/SBCK.git@easier-install#egg=sbck&subdirectory=python"

The last line will install SBCK>=v0.4.0 from Ouranos' fork which implements a small fix that allows this
direct installation from pip within a virtual environment.

Finally, the function :py:indicator:`xclim.sdba.property.first_eof` makes use of `eofs`_, another optional dependency, which is available on both pip and conda.

.. _SBCK: https://github.com/yrobink/SBCK
.. _Eigen: https://eigen.tuxfamily.org/index.php
.. _eofs: https://ajdawson.github.io/eofs/

From sources
------------
.. Warning::
    For Python3.10+ users: Many of the required scientific libraries do not currently have wheels that support the latest
    python. In order to ensure that installation of xclim doesn't fail, we suggest installing the `Cython` module
    before installing xclim in order to compile necessary libraries from source packages.

The sources for xclim can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:Ouranosinc/xclim.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OL https://github.com/Ouranosinc/xclim/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

Alternatively, you can also install a local development copy via pip:

.. code-block:: console

    $ pip install -e .[dev]

.. _Github repo: https://github.com/Ouranosinc/xclim
.. _tarball: https://github.com/Ouranosinc/xclim/tarball/master

Creating a Conda environment
----------------------------

To create a conda development environment including all xclim dependencies, enter the following command from within your cloned repo:

.. code-block:: console

    $ conda create -n my_xclim_env python=3.8 --file=environment.yml
    $ conda activate my_xclim_env
    (my_xclim_env) $ pip install ".[dev]"
