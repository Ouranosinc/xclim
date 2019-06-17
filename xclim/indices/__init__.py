# -*- coding: utf-8 -*-

"""
Indices library
===============

This module describes climate indicator functions. Functions are listed in alphabetical order and describe the raw
computation performed over xarray.DataArrays. DataArrays should carry unit information to allow for any needed
unit conversions. The output's attributes (CF-Convention) are not modified. Validation checks and output attributes
are handled by indicator classes described in files named by the physical variable (temperature, precip, streamflow).

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

.. code-block:: python

   Parameters
   ----------
   <standard_name> : xarray.DataArray
     <Long_name> of variable [acceptable units].
   threshold : string
     Description of the threshold / units.
     e.g. The 10th percentile of historical temperature [K].
   freq : str, optional
     Resampling frequency.

   Returns
   -------
   xarray.DataArray
     Output's <long_name> [units]

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

from ._simple import *
from ._threshold import *
from ._multivariate import *

# TODO: Define a unit conversion system for temperature [K, C, F] and precipitation [mm h-1, Kg m-2 s-1] metrics
# TODO: Move utility functions to another file.
# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/
