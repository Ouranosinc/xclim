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
   :noindex:

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
   :noindex:

.. automodule:: xclim.compute.helpers
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.compute.run_length
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.compute.stats
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Function Library
----------------
When an indicator can't be simply implemented only using a :py:mod:`xclim.compute.generic` function, then a custom compute function
is implemented here.


.. automodule:: xclim.compute
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Fire indices submodule
^^^^^^^^^^^^^^^^^^^^^^
Compute functions related to fire and fire weather. Currently, submodules exist for calculating indices from the Canadian Forest Fire Weather Index System and the McArthur Forest Fire Danger (Mark 5) System.

.. automodule:: xclim.compute.fire._cffwis
   :members: fire_weather_ufunc, fire_season, overwintering_drought_code, drought_code, cffwis_indices
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.compute.fire._ffdi
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

.. automodule:: xclim.ensembles._reduce
   :noindex:

.. Use of autofunction is so that paths do not include private modules.
.. autofunction:: xclim.ensembles.kkz_reduce_ensemble
   :noindex:

.. autofunction:: xclim.ensembles.kmeans_reduce_ensemble
   :noindex:

.. autofunction:: xclim.ensembles.plot_rsqprofile
   :noindex:

.. automodule:: xclim.ensembles._robustness
   :noindex:

.. autofunction:: xclim.ensembles.robustness_fractions
   :noindex:

.. autofunction:: xclim.ensembles.robustness_categories
   :noindex:

.. autofunction:: xclim.ensembles.robustness_coefficient
   :noindex:

.. automodule:: xclim.ensembles._partitioning
    :noindex:

.. autofunction:: xclim.ensembles.hawkins_sutton
    :noindex:

.. autofunction:: xclim.ensembles.lafferty_sriver
    :noindex:

Units Handling Submodule
========================

.. automodule:: xclim.core.units
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _spatial-analogues-api:

Spatial Analogues Module
========================

.. autoclass:: xclim.analog.spatial_analogs
   :noindex:

.. autofunction:: xclim.analog.friedman_rafsky
   :noindex:

.. autofunction:: xclim.analog.kldiv
   :noindex:

.. autofunction:: xclim.analog.kolmogorov_smirnov
   :noindex:

.. autofunction:: xclim.analog.nearest_neighbor
   :noindex:

.. autofunction:: xclim.analog.seuclidean
   :noindex:

.. autofunction:: xclim.analog.szekely_rizzo
   :noindex:

.. autofunction:: xclim.analog.zech_aslan
   :noindex:

.. autofunction:: xclim.analog.mahalanobis
   :noindex:

Other Utilities
===============

.. automodule:: xclim.core.calendar
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.core.formatting
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.core.options
   :members: set_options
   :noindex:

.. automodule:: xclim.core.utils
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   :noindex:

Modules for xclim Developers
============================

.. automodule:: xclim.core.indicator
   :noindex:

.. autoclass:: xclim.core.indicator.Parameter
   :members: injected, json, update
   :noindex:

.. autoclass:: xclim.core.indicator.Output
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: xclim.core.indicator.Indicator
   :members:
   :inherited-members:
   :noindex:

.. autoclass:: ReducingIndicator
   :members:
   :noindex:

.. autoclass:: IndexingIndicator
   :members:
   :noindex:

.. autoclass:: ResamplingIndicator
   :members:
   :noindex:

.. autoclass:: ResamplingIndicatorWithIndexing
   :members:
   :noindex:

.. autoclass:: Hourly
   :members:
   :noindex:

.. autoclass:: Daily
   :members:
   :noindex:


Bootstrapping Algorithms for Indicators Submodule
-------------------------------------------------

.. automodule:: xclim.core.bootstrapping
   :members:
   :show-inheritance:
   :noindex:

.. _`spatial-analogues-developer-api`:

Spatial Analogues Helpers
-------------------------

.. autofunction:: xclim.analog.metric
   :noindex:

.. autofunction:: xclim.analog.standardize
   :noindex:

Testing Module
--------------

.. automodule:: xclim.testing.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.testing.helpers
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
