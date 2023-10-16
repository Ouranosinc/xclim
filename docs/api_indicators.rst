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
