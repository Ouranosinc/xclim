==========================================
Bias adjustment and downscaling algorithms
==========================================

`xarray` data structures allow for relatively straightforward implementations of simple bias-adjustment and downscaling algorithms documented in :ref:`Adjustment Methods <sdba:SDBA User API>`.
Each algorithm is split into `train` and `adjust` components. The `train` function will compare two DataArrays `x` and `y`, and create a dataset storing the *transfer* information allowing to go from `x` to `y`.
This dataset, stored in the adjustment object, can then be used by the `adjust` method to apply this information to `x`. `x` could be the same `DataArray` used for training, or another `DataArray` with similar characteristics.

For example, given a daily time series of observations `ref`, a model simulation over the observational period `hist` and a model simulation over a future period `sim`, we would apply a bias-adjustment method such as *detrended quantile mapping* (DQM) as::

  from xclim import sdba

  dqm = sdba.adjustment.DetrendedQuantileMapping.train(ref, hist)
  scen = dqm.adjust(sim)

Most method can either be applied additively by multiplication. Also, most methods can be applied independently on different time groupings (monthly, seasonally) or according to the day of the year and a rolling window width.

When transfer factors are applied in adjustment, they can be interpolated according to the time grouping.
This helps avoid discontinuities in adjustment factors at the beginning of each season or month and is computationally cheaper than computing adjustment factors for each day of the year.
(Currently only implemented for monthly grouping)

Application in multivariate settings
====================================

When applying univariate adjustment methods to multiple variables, some strategies are recommended to avoid introducing unrealistic artifacts in adjusted outputs.

Minimum and maximum temperature
-------------------------------

When adjusting both minimum and maximum temperature, adjustment factors sometimes yield minimum temperatures larger than the maximum temperature on the same day, which of course, is nonsensical.
One way to avoid this is to first adjust maximum temperature using an additive adjustment, then adjust the diurnal temperature range (DTR) using a multiplicative adjustment, and then determine minimum temperature by subtracting DTR from the maximum temperature :cite:p:`thrasher_technical_2012,agbazo_characterizing_2020`.

Relative and specific humidity
------------------------------

When adjusting both relative and specific humidity, we want to preserve the relationship between both.
To do this, :cite:t:`grenier_two_2018` suggests to first adjust the relative humidity using a multiplicative factor, ensure values are within 0-100%, then apply an additive adjustment factor to the surface pressure before estimating the specific humidity from thermodynamic relationships.

Radiation and precipitation
---------------------------

In theory, short wave radiation should be capped when precipitation is not zero, but there is as of yet no mechanism proposed to do that, see :cite:t:`hoffmann_meteorologically_2012`.

SDBA User API
=============

.. automodule:: xclim.sdba.adjustment
   :members:
   :exclude-members: BaseAdjustment
   :special-members:
   :show-inheritance:
   :noindex:

.. automodule:: xclim.sdba.processing
   :members:
   :noindex:

.. automodule:: xclim.sdba.detrending
   :members:
   :show-inheritance:
   :exclude-members: BaseDetrend
   :noindex:

.. automodule:: xclim.sdba.utils
   :members:
   :noindex:

.. autoclass:: xclim.sdba.base.Grouper
   :members:
   :class-doc-from: init
   :noindex:

.. automodule:: xclim.sdba.nbutils
   :members:
   :noindex:

.. automodule:: xclim.sdba.loess
   :members:
   :noindex:

.. automodule:: xclim.sdba.properties
   :members:
   :exclude-members: register_statistical_properties
   :noindex:

.. automodule:: xclim.sdba.measures
   :members:
   :exclude-members: check_same_units_and_convert
   :noindex:

Developer tools
===============

.. automodule:: xclim.sdba.base
   :members:
   :show-inheritance:
   :exclude-members: Grouper
   :noindex:

.. autoclass:: xclim.sdba.detrending.BaseDetrend
   :members:
   :noindex:

.. autoclass:: xclim.sdba.adjustment.TrainAdjust
   :members:
   :noindex:

.. autoclass:: xclim.sdba.adjustment.Adjust
   :members:
   :noindex:

.. autofunction:: xclim.sdba.properties.register_statistical_properties
   :noindex:

.. autofunction:: xclim.sdba.measures.check_same_units_and_convert
   :noindex:

.. only:: html or text

    .. _sdba-footnotes:

    SDBA Footnotes
    ==============

    .. bibliography::
       :style: xcstyle
       :labelprefix: SDBA-
       :keyprefix: sdba-

.. [RRJF2021] Roy, P., Rondeau-Genesse, G., Jalbert, J., Fournier, É. 2021. Climate Scenarios of Extreme Precipitation Using a Combination of Parametric and Non-Parametric Bias Correction Methods. Submitted to Climate Services, April 2021.
