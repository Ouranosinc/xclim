==========================
Indicators and collections
==========================

Indicators are the main tool xclim provides. In contrast to the functions defined in :py:mod:`xclim.compute`, Indicators add a layer of health checks and metadata handling and return :py:class:`xarray.Dataset` objects by default. Indicator objects are split into submodules according to their "realm" : atmos, land and seaIce, with two additional submodules : generic (for indicators that don't apply to a specific variable) and convert (for non-resampling indicators that transform between variables).


.. toctree::
   :maxdepth: 1

   indicator_list

Control indicator behaviour
===========================

.. autoclass:: xclim.core.options.set_options


.. automodule:: xclim.core.indicator

.. autoclass:: xclim.core.indicator.Indicator
   :members:
   :inherited-members:
   :special-members: __init__

.. autoclass:: xclim.core.indicator.Parameter
   :members: injected, json, update

.. autoclass:: xclim.core.indicator.Output
   :members:

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

.. automodule:: xclim.core.collection
   :members:
