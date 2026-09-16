===
API
===

Indicators
==========

.. toctree::
   :maxdepth: 1

   api_indicators

.. automodule:: xclim.core.collection
   :members:

Compute functions
=================

.. note::

    Index-like compute functions (formerly "Indices") serve as the scientific logic behind `Indicators`. End users should usually
    not have to use these functions directly, unless creating a new :py:class:`~xclim.core.collection.IndicatorCollection`.
    (see: :ref:`notebooks/extendxclim:Defining new indicators`).

    Otherwise, we suggest using the :ref:`indicators:Climate Indicators`.

Compute functions are designed to operate on :py:class:`xarray.DataArray` objects.
Most of these functions operate on daily time series, but they usually don't check this.
All functions perform units checks to make sure that inputs have the expected dimensions
(e.g. handling for units of temperature, whether they are Celsius, kelvin or Fahrenheit), and set the `units`
attribute of the output `DataArray`.

Helper submodules
-----------------
The :py:mod:`xclim.compute.generic`, :py:mod:`xclim.compute.helpers`, :py:mod:`xclim.compute.run_length`, and
:py:mod:`xclim.compute.stats` submodules provide helper functions to simplify the implementation of index-like compute functions
while functions under :py:mod:`xclim.core.calendar` can aid with challenges arising from variable calendar
types.

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

Function Library
----------------
When an indicator can't be simply implemented only using a :py:mod:`xclim.compute.generic` function, then a custom compute function
is implemented here.

.. automodule:: xclim.compute
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

Fire indices submodule
^^^^^^^^^^^^^^^^^^^^^^
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


Health Checks
=============

See: :ref:`checks:Health Checks`

Translation Tools
=================

See: :ref:`internationalization:Internationalization`

Ensembles Module
================

.. automodule:: xclim.ensembles
   :members: create_ensemble, ensemble_mean_std_max_min, ensemble_percentiles

.. automodule:: xclim.ensembles._reduce

.. Use of autofunction is so that paths do not include private modules.
.. autofunction:: xclim.ensembles.kkz_reduce_ensemble

.. autofunction:: xclim.ensembles.kmeans_reduce_ensemble

.. autofunction:: xclim.ensembles.plot_rsqprofile

.. automodule:: xclim.ensembles._robustness

.. autofunction:: xclim.ensembles.robustness_fractions

.. autofunction:: xclim.ensembles.robustness_categories

.. autofunction:: xclim.ensembles.robustness_coefficient

.. automodule:: xclim.ensembles._partitioning

.. autofunction:: xclim.ensembles.hawkins_sutton

.. autofunction:: xclim.ensembles.lafferty_sriver

Units Handling Submodule
========================

.. automodule:: xclim.core.units
   :members:
   :undoc-members:
   :show-inheritance:

.. _spatial-analogues-api:

Spatial Analogues Module
========================

.. autoclass:: xclim.analog.spatial_analogs

.. autofunction:: xclim.analog.friedman_rafsky

.. autofunction:: xclim.analog.kldiv

.. autofunction:: xclim.analog.kolmogorov_smirnov

.. autofunction:: xclim.analog.nearest_neighbor

.. autofunction:: xclim.analog.seuclidean

.. autofunction:: xclim.analog.szekely_rizzo

.. autofunction:: xclim.analog.zech_aslan

.. autofunction:: xclim.analog.mahalanobis

Other Utilities
===============

.. automodule:: xclim.core.calendar
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.formatting
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.options
   :members: set_options

.. automodule:: xclim.core.utils
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Modules for xclim Developers
============================

.. automodule:: xclim.core.indicator

.. autoclass:: xclim.core.indicator.Parameter
   :members: injected, json, update

.. autoclass:: xclim.core.indicator.Output
   :members:
   :show-inheritance:

.. autoclass:: xclim.core.indicator.Indicator
   :members:
   :inherited-members:
   :special-members: __init__

.. autoclass:: ReducingIndicator
   :members:

.. autoclass:: IndexingIndicator
   :members:

.. autoclass:: ResamplingIndicator
   :members:

.. autoclass:: ResamplingIndicatorWithIndexing
   :members:

.. autoclass:: Hourly
   :members:

.. autoclass:: Daily
   :members:


Bootstrapping Algorithms for Indicators Submodule
-------------------------------------------------

.. automodule:: xclim.core.bootstrapping
   :members:
   :show-inheritance:

.. _`spatial-analogues-developer-api`:

Spatial Analogues Helpers
-------------------------

.. autofunction:: xclim.analog.metric

.. autofunction:: xclim.analog.standardize

Testing Module
--------------

.. automodule:: xclim.testing.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.testing.helpers
   :members:
   :undoc-members:
   :show-inheritance:
