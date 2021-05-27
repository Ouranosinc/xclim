===
API
===

Indicators
==========

.. toctree::

  indicators_api


Indices
=======

.. toctree::

   indices

Ensembles module
================

.. automodule:: xclim.ensembles
   :members: create_ensemble, ensemble_mean_std_max_min, ensemble_percentiles

.. automodule:: xclim.ensembles._reduce

.. Use of autofunction is so that paths do not include private modules.
.. autofunction:: xclim.ensembles.kkz_reduce_ensemble
.. autofunction:: xclim.ensembles.kmeans_reduce_ensemble
.. autofunction:: xclim.ensembles.plot_rsqprofile

.. automodule:: xclim.ensembles._robustness

.. autofunction:: xclim.ensembles.change_significance
.. autofunction:: xclim.ensembles.robustness_coefficient

Indicator tools
===============

.. automodule:: xclim.core.indicator
   :members:
   :member-order: bysource
   :show-inheritance:


Unit handling module
====================

.. automodule:: xclim.core.units
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Downscaling and Bias Adjustment
===========================================

.. toctree::

     sdba_api


Other utilities
===============

.. automodule:: xclim.core.calendar
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.cfchecks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.datachecks
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


Other xclim modules
===================

Spatial Analogs module
----------------------

.. automodule:: xclim.analog
   :members:

Testing module
--------------

.. automodule:: xclim.testing
    :members:

Subset module
-------------
.. warning::
    Subsetting is now offered via `clisops.core.subset`. The subsetting functions offered by `clisops`
    are available at the following link:

:doc:`CLISOPS API <clisops:api>`

.. note::
    For more information about `clisops` refer to their documentation here:
    :doc:`CLISOPS documentation <clisops:readme>`
