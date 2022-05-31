===
API
===

The API of the statistical downscaling and bias adjustment module (sdba) is documented :ref:`on this page <sdba:SDBA User API>`. The API of the ``cfchecks``, ``datachecks``, ``missing`` and ``dataflags`` modules are in :ref:`checks:Health Checks`. Finally, the API of the translating tools is on the :ref:`internationalization:Internationalization` page.


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
    :noindeX:

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
    Subsetting is now offered via `clisops.core.subset`. The subsetting functions offered by `clisops`
    are available at the following link:

:doc:`CLISOPS API <clisops:api>`

.. note::
    For more information about `clisops` refer to their documentation here:
    :doc:`CLISOPS documentation <clisops:readme>`
