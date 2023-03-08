===============
Climate Indices
===============

.. note::

    Climate `Indices` serve as the driving mechanisms behind `Indicators` and should be used in cases where default settings for an Indicator may need to be tweaked, metadata completeness is not required, or a user wishes to design a virtual module from existing indices (e.g. see :ref:`notebooks/extendxclim:Defining new indicators`).

    For higher-level and general purpose use, the xclim developers suggest using the :ref:`indicators:Climate Indicators`.


.. automodule:: xclim.indices
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

Indices submodules
------------------

.. automodule:: xclim.indices.generic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.helpers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.run_length
   :members:
   :undoc-members:
   :show-inheritance:

Fire indices submodule
^^^^^^^^^^^^^^^^^^^^^^
Indices related to fire and fire weather. Currently, submodules exist for calculating indices from the Canadian Forest Fire Weather Index System and the McArthur Forest Fire Danger (Mark 5) System. All fire indices can be accessed from the :py:mod:`xclim.indices` module.

.. automodule:: xclim.indices.fire._cffwis
   :members: fire_weather_ufunc, fire_season, overwintering_drought_code, drought_code, cffwis_indices
   :undoc-members:
   :show-inheritance:

.. automodule:: xclim.indices.fire._ffdi
   :members:
   :undoc-members:
   :show-inheritance:

.. only:: html

    Fire indices footnotes
    ~~~~~~~~~~~~~~~~~~~~~~

    .. _ffdi-footnotes:

    McArthur Forest Fire Danger Indices methods
    *******************************************

.. bibliography::
   :labelprefix: FFDI-
   :keyprefix: ffdi-

.. only:: html

    .. _fwi-footnotes:

    Canadian Forest Fire Weather Index System codes
    ***********************************************

.. bibliography::
   :labelprefix: CODE-
   :keyprefix: code-

.. only:: html

    .. note::

       Matlab code of the GFWED obtained through personal communication.

    Fire season determination methods
    *********************************

.. bibliography::
   :labelprefix: FIRE-
   :keyprefix: fire-

.. only:: html

    Drought Code overwintering background
    *************************************

.. bibliography::
   :labelprefix: DROUGHT-
   :keyprefix: drought-
