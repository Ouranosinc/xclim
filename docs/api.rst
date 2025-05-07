===
API
===

Indicators
==========

.. toctree::
   :maxdepth: 1

   api_indicators

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

.. _sdba-user-api:

SDBA Module
===========

.. warning::

    The `xclim.sdba` module was split from the library in `xclim==0.57` in order to facilitate development and effective maintenance of the SDBA utilities. This functionality is now available in the `xsdba` package. While the package aims to maintain compatibility with `xclim`, some algorithms have been slightly modified.

    For convenience, the `xclim.sdba` module will still available exposing the functionality of the `xsdba` package. This may change in the future.

.. note::

    For more information, the documentation is available at the following link: :doc:`xsdba API <xsdba:api>`

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

Subset Module
=============

.. warning::

    The `xclim.subset` module was removed in `xclim==0.40`. Subsetting is now offered via `clisops.core.subset`.
    The subsetting functions offered by `clisops` are available at the following link: :doc:`CLISOPS core subsetting API <clisops:api>`

.. note::

    For more information about `clisops`, please refer to the documentation at the following link: :doc:`CLISOPS documentation <clisops:readme>`

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

.. warning::

    The `xclim.sdba` module was split from the library in `xclim==0.57` in order to facilitate development and effective maintenance of the SDBA utilities. This functionality is now available in the `xsdba` package. While the package aims to maintain compatibility with `xclim`, some algorithms have been slightly modified.

    For convenience, the `xclim.sdba` module will still available exposing the functionality of the `xsdba` package. This may change in the future.

.. note::

    For more information about `xsdba`, please refer to the documentation at the following link: :doc:`xsdba API <xsdba:api>`

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
