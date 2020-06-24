=============
Health checks
=============
The :class:`Indicator` class performs a number of sanity checks on inputs to make sure valid data is fed to indices
computations and output values are properly masked in case input values are missing or invalid.


Missing values
==============
The following functions can be used to mask missing values in :class:`Indicator` outputs.

.. autofunction:: xclim.core.missing.missing_any
   :noindex:

.. autofunction:: xclim.core.missing.missing_pct
   :noindex:
.. autofunction:: xclim.core.missing.missing_wmo
   :noindex:

.. autofunction:: xclim.core.missing.at_least_n_valid
   :noindex:
