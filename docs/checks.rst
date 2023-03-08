=============
Health Checks
=============
The :class:`Indicator` class performs a number of sanity checks on inputs to make sure valid data is fed to indices
computations (:py:mod:`~xclim.core.cfchecks` for checks on the metadata and :py:mod:`~xclim.core.datachecks` for checks on the coordinates).
Output values are properly masked in case input values are missing or invalid (:py:mod:`~xclim.core.missing`).
Finally, a user can use functions of :py:mod:`~xclim.core.dataflags` to explore potential issues with its data (extreme values, suspicious runs, etc).

.. automodule:: xclim.core.cfchecks
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.core.datachecks
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.core.missing
   :noindex:

.. note::

    Corresponding stand-alone functions are also exposed to run the same missing value checks independent from indicator calculations.

.. autofunction:: xclim.core.missing.missing_any
   :noindex:

.. autofunction:: xclim.core.missing.at_least_n_valid
   :noindex:

.. autofunction:: xclim.core.missing.missing_pct
   :noindex:

.. autofunction:: xclim.core.missing.missing_wmo
   :noindex:

.. autofunction:: xclim.core.missing.missing_from_context
   :noindex:

.. automodule:: xclim.core.dataflags
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
