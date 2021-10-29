# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
===============
Indices library
===============

This module contains climate indices functions operating on `xarray.DataArray`. Most of these
functions operate on daily time series, but might accept other sampling frequencies as well. All
functions perform units checks to make sure that inputs have the expected dimensions (for example
have units of temperature, whether it is celsius, kelvin or fahrenheit), and set the `units`
attribute of the output DataArray.

The `calendar`, `fwi`, `generic`, `run_length` and `utils` submodule provide helpers to simplify
the implementation of the indices.

.. note::

    Indices functions do not perform missing value checks, and do not set CF-Convention attributes
    (long_name, standard_name, description, cell_methods, etc). These functionalities are provided by
    :class:`xclim.indicators.Indicator` instances found in the :mod:`xclim.indicators.atmos`,
    :mod:`xclim.indicators.land` and :mod:`xclim.indicators.seaIce` modules, documented in :ref:`Climate Indicators`.

"""
from ._agro import *
from ._anuclim import *
from ._conversion import *
from ._hydrology import *
from ._multivariate import *
from ._simple import *
from ._threshold import *
from .fwi import drought_code, fire_season, fire_weather_indexes

"""
Notes for docstring
-------------------

The docstrings adhere to the `NumPy`_ style convention and is meant as a way to store CF-Convention metadata as
well as information relevant to third party libraries (such as a WPS server).

The first line of the docstring (the short summary), will be assigned to the output's `long_name` attribute. The
`long_name` attribute is defined by the NetCDF User Guide to contain a long descriptive name which may, for example,
be used for labeling plots

The second paragraph will be considered as the "*abstract*", or the CF global "*comment*" (miscellaneous information
about the data or methods used to produce it).

The third and fourth sections are the **Parameters** and **Returns** sections describing the input and output values
respectively.

The following example shows the structure of an indice definition:

.. code-block:: python

   @declare_units(var1="[units dimension]", thresh="[units dimension]")
   def indice_name(var1: xr.DataArray, thresh: str = "0 degC", freq: str = "YS"):
       \"\"\"
       The first line is the title

       The second paragraph is the abstract.

       Parameters
       ----------
       var1 : xarray.DataArray
         Description of variable (no need to specify units, the signature and decorator carry this information).
         <var1> is a short name like "tas", "pr" or "sfcWind".
       thresh : str
         Description of the threshold (no need to specify units or the default again).
         Parameters required to run the computation must always have a working default value.
       freq : str
         Resampling frequency. (the signature carries the default value information)

       Returns
       -------
       <var_name> : xarray.DataArray, [output units dimension]
         Output's <long_name>
         Specifying <var_name> is optional.
       \"\"\"
       <body of the function>
       # Don't forget to explicitly handle the units!

The next sections would be **Notes** and **References**:

.. code-block:: python

    Notes
    -----
    This is where the mathematical equation is described.
    At the end of the description, convention suggests
    to add a reference [example]_:

        .. math::

            3987^12 + 4365^12 = 4472^12

    References
    ----------
    .. [example] Smith, T.J. and Huard, D. (2018). "CF Docstrings:
        A manifesto on conventions and the metaphysical nature
        of ontological python documentation." Climate Aesthetics,
        vol. 1, pp. 121-155.

Indice descriptions
===================
.. _`NumPy`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
"""

# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/
