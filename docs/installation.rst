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

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------
.. Warning::
    For Python3.8 users: Many of the required scientific libraries do not currently have wheels that support the latest
    python. In order to ensure that installation of xclim doesn't fail, we suggest installation the `Cython` module before
    installing xclim.

The sources for xclim can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/Ouranosinc/xclim

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/Ouranosinc/xclim/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

Alternatively, you can also install a local copy via pip:

.. code-block:: console

    $ pip install .

.. _Github repo: https://github.com/Ouranosinc/xclim
.. _tarball: https://github.com/Ouranosinc/xclim/tarball/master


Creating a Conda environment
----------------------------

To create a conda development environment including all xclim dependencies, enter the following command from within your cloned repo:

.. code-block:: console

    $ conda create -n my_xclim_env python=3.6 --file=requirements_dev.txt
    $ conda activate my_xclim_env
    $ python setup.py install
