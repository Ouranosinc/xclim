Climate Indicators API
----------------------

Indicators are the main tool xclim provides. In contrast
to the functions defined in `xclim.compute`, Indicators add a layer of health checks
and metadata handling and return `Dataset` by default. Indicator objects are split into submodules according to their
"realm" : atmos, land and seaIce, with two additional submodules : generic (for
indicator that don't apply to a specific variable) and convert (for non-resampling
indicators that transform between variables).

Three :py:class:`IndicatorCollection` that come with xclim are also added here.

 - :py:data:`xclim.indicators.cf`, Indicators defined in `cf-index-meta`.
 - :py:data:`xclim.indicators.icclim`, Indicators defined by ECAD, as found in  python package Icclim.
 - :py:data:`xclim.indicators.anuclim`, Indicators of the Australian National University's Fenner School of Environment and Society.

Climate Indicator Submodules
----------------------------

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

.. automodule:: xclim.indicators.generic
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: xclim.indicators.convert
   :members:
   :undoc-members:
   :imported-members:


Built-in Indicator Collections
------------------------------

.. automodule:: xclim.indicators
   :members: cf, icclim, anuclim
