.. highlight:: shell

============
Installation
============


Stable release
--------------


To install xclim, run this command in your terminal:

.. code-block:: console

    $ pip install xclim

This is the preferred method to install xclim, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

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


.. _Github repo: https://github.com/Ouranosinc/xclim
.. _tarball: https://github.com/Ouranosinc/xclim/tarball/master


Creating a Conda environment
----------------------------

To create a conda environment including all xclim dependencies, enter the following command, where :

.. code-block:: console

    $ conda create -n my_xclim_env python=3.6 --file=requirements.txt --file=requirements_dev.txt
