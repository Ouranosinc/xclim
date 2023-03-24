===============
Climate Indices
===============

.. note::

    Climate `Indices` serve as the driving mechanisms behind `Indicators` and should be used in cases where
    default settings for an Indicator may need to be tweaked, metadata completeness is not required, or a user
    wishes to design a virtual module from existing indices (see: :ref:`notebooks/extendxclim:Defining new indicators`).

    For higher-level and general purpose use, the xclim developers suggest using the :ref:`indicators:Climate Indicators`.

Indices Library
---------------

Climate indices functions are designed to operate on :py:class:`xarray.DataArray` objects.
Most of these functions operate on daily time series, but in some cases might accept other sampling
frequencies as well. All functions perform units checks to make sure that inputs have the expected dimensions
(e.g. handling for units of temperature, whether they are Celsius, kelvin or Fahrenheit), and set the `units`
attribute of the output `DataArray`.

The :py:mod:`xclim.indices.generic`, :py:mod:`xclim.indices.helpers`, :py:mod:`xclim.indices.run_length`, and
:py:mod:`xclim.indices.stats` submodules provide helper functions to simplify the implementation of indices
while functions under :py:mod:`xclim.core.calendar` can aid with challenges arising from variable calendar
types.

.. warning::

    Indices functions do not perform missing value checks, and usually do not set CF-Convention attributes
    (long_name, standard_name, description, cell_methods, etc.). These functionalities are provided by
    :py:class:`xclim.core.indicator.Indicator` instances found in the :py:mod:`xclim.indicators.atmos`,
    :py:mod:`xclim.indicators.land` and :mod:`xclim.indicators.seaIce` modules.

.. automodule:: xclim.indices
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

Indices submodules
------------------

.. automodule:: xclim.indices.generic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.helpers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.run_length
   :members:
   :undoc-members:
   :show-inheritance:

Fire indices submodule
^^^^^^^^^^^^^^^^^^^^^^
Indices related to fire and fire weather. Currently, submodules exist for calculating indices from the Canadian Forest Fire Weather Index System and the McArthur Forest Fire Danger (Mark 5) System. All fire indices can be accessed from the :py:mod:`xclim.indices` module.

.. automodule:: xclim.indices.fire._cffwis
   :members: fire_weather_ufunc, fire_season, overwintering_drought_code, drought_code, cffwis_indices
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.fire._ffdi
   :members:
   :undoc-members:
   :show-inheritance:

.. only:: html

    Fire indices footnotes
    ~~~~~~~~~~~~~~~~~~~~~~

    .. _ffdi-footnotes:

    McArthur Forest Fire Danger Indices methods
    *******************************************

.. bibliography::
   :labelprefix: FFDI-
   :keyprefix: ffdi-

.. only:: html

    .. _fwi-footnotes:

    Canadian Forest Fire Weather Index System codes
    ***********************************************

.. bibliography::
   :labelprefix: CODE-
   :keyprefix: code-

.. only:: html

    .. note::

       Matlab code of the GFWED obtained through personal communication.

    Fire season determination methods
    *********************************

.. bibliography::
   :labelprefix: FIRE-
   :keyprefix: fire-

.. only:: html

    Drought Code overwintering background
    *************************************

.. bibliography::
   :labelprefix: DROUGHT-
   :keyprefix: drought-
