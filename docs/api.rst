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
   :members:
   :imported-members:
   :exclude-members: ensemble_robustness, kmeans_reduce_ensemble, kkz_reduce_ensemble, plot_rsqprofile
   :show-inheritance:


Ensemble reduction
------------------

.. automodule:: xclim.ensembles._reduce

.. autofunction:: xclim.ensembles.kkz_reduce_ensemble
.. autofunction:: kmeans_reduce_ensemble
.. autofunction:: plot_rsqprofile

Ensemble robustness
-------------------

.. autofunction:: xclim.ensembles.ensemble_robustness

.. automodule:: xclim.ensembles._robustness
   :members: knutti_sedlacek, tebaldi_et_al


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

spatial analogs module
----------------------

.. automodule:: xclim.analog
   :members:

testing module
--------------

.. automodule:: xclim.testing
    :members:

subset module
-------------
.. warning::
    Subsetting is now offered via `clisops`. The functions offered by clisops
    will be described here once the subsetting functions API is made available.
    For now, refer to their documentation here:
    :doc:`clisops subset examples <clisops:notebooks/subset>`
