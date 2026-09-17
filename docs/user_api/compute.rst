=================
Compute Functions
=================

.. note::

    Index-like compute functions (formerly "Indices") serve as the scientific logic behind `Indicators`. End users should usually
    not have to use these functions directly, unless creating a new :py:class:`~xclim.core.collection.IndicatorCollection`.
    (see: :ref:`notebooks/extendxclim:Defining new indicators`).

    Otherwise, we suggest using the indicators listed in :ref:`indicators:Climate Indicators`.

Compute functions are designed to operate on :py:class:`xarray.DataArray` objects.
Most of these functions operate on daily time series, but they usually don't check this.
All functions perform units checks to make sure that inputs have the expected dimensions
(e.g. handling for units of temperature, whether they are Celsius, kelvin or Fahrenheit), and set the `units`
attribute of the output `DataArray`.

Helper submodules
-----------------
The :py:mod:`xclim.compute.generic`, :py:mod:`xclim.compute.helpers`, :py:mod:`xclim.compute.run_length`, and
:py:mod:`xclim.compute.stats` submodules provide helper functions to simplify the implementation of index-like compute functions.
Finally, :py:mod:`xclim.compute.clix` gives an interface to the "generic" functions with vocabulary and signatures as close as possible to the definitions of `clix-meta`_.

.. automodule:: xclim.compute.generic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.compute.helpers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.compute.run_length
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.compute.stats
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.compute.clix
   :members:

Function Library
----------------
When an indicator can't be simply implemented only using a :py:mod:`xclim.compute.generic` function, then a custom compute function
is implemented here.

.. automodule:: xclim.compute
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: cffwis_indices, drought_code, duff_moisture_code, fire_season, griffiths_drought_factor, keetch_byram_drought_index, mcarthur_forest_fire_danger_index

Fire Indices
^^^^^^^^^^^^
Compute functions related to fire and fire weather. Currently, submodules exist for calculating indices from the Canadian Forest Fire Weather Index System and the McArthur Forest Fire Danger (Mark 5) System.

.. automodule:: xclim.compute.fire._cffwis
   :members: fire_weather_ufunc, fire_season, overwintering_drought_code, drought_code, cffwis_indices
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.compute.fire._ffdi
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

       MATLAB code of the GFWED obtained through personal communication, reimplemented in Python.

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

.. _clix-meta: https://github.com/clix-meta/clix-meta
