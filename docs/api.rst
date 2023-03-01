===
API
===

.. note::

    The API of the ``cfchecks``, ``datachecks``, ``missing`` and ``dataflags`` modules are located under :ref:`checks:Health Checks`.

    The API for the translation tools can be found within :ref:`internationalization:Internationalization`.

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: none

Indicators
==========

.. toctree::

  indicators_api

Indices
=======

.. toctree::
   :maxdepth: 3

   indices

Ensembles module
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

Indicator Tools
===============

.. automodule:: xclim.core.indicator
   :members:
   :member-order: bysource
   :show-inheritance:
   :noindex:

Unit Handling module
====================

.. automodule:: xclim.core.units
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _sdba-user-api:

SDBA module
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

SDBA Developer tools
--------------------

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

Other xclim modules
===================

Spatial Analogs module
----------------------

See :ref:`analogues:Spatial analogues`.

Testing module
--------------

.. automodule:: xclim.testing
    :members:
    :noindex:

.. automodule:: xclim.testing.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Subset module
-------------
.. warning::
    The `xclim.subset` module was removed in `xclim==0.40`. Subsetting is now offered via `clisops.core.subset`.
    The subsetting functions offered by `clisops` are available at the following link:

:doc:`CLISOPS API <clisops:api>`

.. note::
    For more information about `clisops` refer to their documentation here:
    :doc:`CLISOPS documentation <clisops:readme>`
