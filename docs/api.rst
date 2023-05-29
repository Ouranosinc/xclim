===
API
===

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: none

Indicators
==========

Indicators are the main tool xclim provides to compute climate indices. In contrast
to the function defined in `xclim.indices`, Indicators add a layer of health checks
and metadata handling. Indicator objects are split into realms : atmos, land and seaIce.

Virtual modules are also inserted here. A normal installation of xclim comes with three virtual modules:

 - :py:mod:`xclim.indicators.cf`, Indicators defined in `cf-index-meta`.
 - :py:mod:`xclim.indicators.icclim`, Indicators defined by ECAD, as found in  python package Icclim.
 - :py:mod:`xclim.indicators.anuclim`, Indicators of the Australian National University's Fenner School of Environment and Society.

Climate Indicators API
----------------------

.. automodule:: xclim.indicators.atmos
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: xclim.indicators.land
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: xclim.indicators.seaIce
   :members:
   :undoc-members:
   :imported-members:

Virtual Indicator Submodules
----------------------------

.. automodule:: xclim.indicators.cf
   :members:
   :imported-members:
   :undoc-members:

.. automodule:: xclim.indicators.icclim
   :members:
   :imported-members:
   :undoc-members:

.. automodule:: xclim.indicators.anuclim
   :members:
   :imported-members:
   :undoc-members:

Indices
=======

See: :ref:`indices:Climate Indices`

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

.. autofunction:: xclim.ensembles.change_significance
   :noindex:

.. autofunction:: xclim.ensembles.robustness_coefficient
   :noindex:

Units Handling Submodule
========================

.. automodule:: xclim.core.units
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _sdba-user-api:

SDBA Module
===========

.. automodule:: xclim.sdba.adjustment
   :members:
   :exclude-members: BaseAdjustment
   :special-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.sdba.processing
   :members:
   :noindex:

.. automodule:: xclim.sdba.detrending
   :members:
   :show-inheritance:
   :exclude-members: BaseDetrend
   :noindex:

.. automodule:: xclim.sdba.utils
   :members:
   :noindex:

.. autoclass:: xclim.sdba.base.Grouper
   :members:
   :class-doc-from: init
   :noindex:

.. automodule:: xclim.sdba.nbutils
   :members:
   :noindex:

.. automodule:: xclim.sdba.loess
   :members:
   :noindex:

.. automodule:: xclim.sdba.properties
   :members:
   :exclude-members: StatisticalProperty
   :noindex:

.. automodule:: xclim.sdba.measures
   :members:
   :exclude-members: StatisticalMeasure
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

Subset Module
=============

.. warning::
    The `xclim.subset` module was removed in `xclim==0.40`. Subsetting is now offered via `clisops.core.subset`.
    The subsetting functions offered by `clisops` are available at the following link: :doc:`CLISOPS core subsetting API <clisops:api>`

.. note::
    For more information about `clisops` refer to their documentation here:
    :doc:`CLISOPS documentation <clisops:readme>`

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

Indicator Tools
---------------

.. automodule:: xclim.core.indicator
   :members:
   :member-order: bysource
   :show-inheritance:
   :noindex:

Bootstrapping Algorithms for Indicators Submodule
-------------------------------------------------

.. automodule:: xclim.core.bootstrapping
   :members:
   :show-inheritance:
   :noindex:

.. _`sdba-developer-api`:

SDBA Utilities
--------------

.. automodule:: xclim.sdba.base
   :members:
   :show-inheritance:
   :exclude-members: Grouper
   :noindex:

.. autoclass:: xclim.sdba.detrending.BaseDetrend
   :members:
   :noindex:

.. autoclass:: xclim.sdba.adjustment.TrainAdjust
   :members:
   :noindex:

.. autoclass:: xclim.sdba.adjustment.Adjust
   :members:
   :noindex:

.. autofunction:: xclim.sdba.properties.StatisticalProperty
   :noindex:

.. autofunction:: xclim.sdba.measures.StatisticalMeasure
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
