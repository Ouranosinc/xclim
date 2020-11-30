===
API
===

Indices
=======

.. toctree::

   indices

Indices submodules
------------------


.. automodule:: xclim.indices.generic
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.run_length
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


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
   :noindex:
   :members:
   :member-order: bysource
   :show-inheritance:


Unit handling module
====================

.. automodule:: xclim.core.units
   :noindex:
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
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.checks
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.formatting
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.core.utils
   :noindex:
   :members:
   :undoc-members:
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
    Subsetting is now offered via `clisops`. The functions offered by clisops
    will be described here once the subsetting functions API is made available.
    For now, refer to their documentation here:
    :doc:`clisops subset examples <clisops:notebooks/subset>`
