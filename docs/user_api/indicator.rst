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

.. Explicit definition with py:data is because sphinx has trouble autodocumenting "constants" unless they are referred to by their exact path. And the whole submodules is actually from _indicator.py
.. py:data:: xclim.core.indicator.registry

   Registry of indicator instances.

   This case-insensitive dictionary holds all the indicators defined in xclim, mapped by their identifier.
   Indicators are added here by default, but it can be avoided by passing `register=False` to the constructor.
   Indicators here can be referred to in the `base` field of the YAML definitions when creating a
   :py:class:`~xclim.core.collection.IndicatorCollection`.

.. py:data:: xclim.core.indicator.base_registry

   Registry of standard base indicator classes.

   This dictionary holds some useful base indicator classes to construct new indicators.
   It is mostly useful when defining indicators within a :py:class:`~xclim.core.collection.IndicatorCollection`,
   keys of this dictionary can be values of the `base` field of the YAML definitions.
