==========================================
xclim: Climate indices computations |logo|
==========================================

|license| |build| |pypi| |conda| |coveralls| |codefactor| |zenodo| |black| |docs| |gitter|

----

``xclim`` is a library of functions to compute climate indices from observations or model simulations. It is built using `xarray`_ and can benefit from the parallelization handling provided by `dask`_. Its objective is to make it as simple as possible for users to compute indices from large climate datasets and for scientists to write new indices with very little boilerplate.

For example, the following would compute monthly mean temperature from daily mean temperature:

.. code-block:: python

  import xclim
  import xarray as xr
  ds = xr.open_dataset(filename)
  tg = xclim.icclim.TG(ds.tas, freq='YS')

For applications where meta-data and missing values are important to get right, xclim provides a class for each index that validates inputs, checks for missing values, converts units and assigns metadata attributes to the output. This also provides a mechanism for users to customize the indices to their own specifications and preferences.

xclim currently provides over 50 indices related to mean, minimum and maximum daily temperature, daily precipitation, streamflow and sea ice concentration.

.. _xarray: http://xarray.pydata.org/
.. _dask: https://dask.org/


Documentation
-------------
The official documentation is at https://xclim.readthedocs.io/


Contributing
------------
xclim is in active development and it's being used in production by climate services specialists. If you're interested in participating to the development, want to  suggest features, new indices or report bugs, please leave us a message on the `issue tracker <https://github.com/Ouranosinc/xclim/issues>`_. There is also a chat room on gitter (|gitter|).


How to cite this library
------------------------
If you wish to cite `xclim` in a research publication, we kindly ask that you use the bibliographical reference information available through `Zenodo`_

.. _Zenodo: https://doi.org/10.5281/zenodo.2795043

Credits
-------

This work is made possible thanks to the contribution of the Canadian Center for Climate Services.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


.. |pypi| image:: https://img.shields.io/pypi/v/xclim.svg
        :target: https://pypi.python.org/pypi/xclim
        :alt: Python Package Index Build

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/xclim.svg
        :target: https://anaconda.org/conda-forge/xclim
        :alt: Conda-forge Build Version

.. |gitter| image:: https://badges.gitter.im/Ouranosinc/xclim.svg
        :target: https://gitter.im/Ouranosinc/xclim?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
        :alt: Gitter Chat

.. |build| image:: https://github.com/Ouranosinc/xclim/workflows/xclim/badge.svg
        :target: https://github.com/Ouranosinc/xclim/actions
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

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xclim/master/_static/_images/xclim-logo-small.png
        :target: https://github.com/Ouranosinc/xclim
