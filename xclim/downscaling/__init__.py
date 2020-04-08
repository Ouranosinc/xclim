# -*- coding: utf-8 -*-
"""
===============
Bias Correction
===============

The `xclim.downscaling` submodule provides bias-correction and downscaling algorithms.
Correction algorithms are available through correction object,following the `train` - `predict` scheme.
Given time series of observations (obs), historical simulations (hist) and future simulations (fut),
any bias-correction method would be applied by first estimating the correction factors from the historical and
observations series, and then applying these factors to the future series::

  Corr = Correction(group="time.month", interp="linear")
  Corr.train(obs, hist)
  fut_bc = Corr.predict(fut)
  Corr.ds.cf  # Correction factors.

The `group` argument allows correction factors to be estimated independently for different periods, either months,
seasons or day of the year. The `interp` argument then allows for interpolation between these correction factors to
avoid discontinuities in the bias-corrected series.

The same interpolation principle is also used for quantiles. Indeed, for methods extracting correction factors by
quantile, interpolation is also done between quantiles. This can help reduce discontinuities in the corrected time
series, and possibly reduce the number of quantile bins used.

Modular approach
================

This module adopts a modular approach instead of implementing published and named methods directly.
A generic bias correction process is to layed out like:

- preprocessing on `obs`, `hist` and `fut` (using methods in `xclim.downscaling.processing` or `xclim.downscaling.detrending`)
- creating the correction object `Corr = Correction(**kwargs)` (from `xclim.downscaling.correction`)
- training `Corr.train(obs, sim)`
- prediction `fut_bc = Corr.predict(fut)`
- post-processing on `fut_bc` (for example: re-trending)

The train-predict approach allows to inspect the trained correction object. The correction information is stored in
the underlying `Corr.ds` dataset and always has a `cf` variable with the correction factors. Its layout and the
other available variables vary between the different algorithm, refer to :ref:`Bias-Correction algorithms`.

Grouping
========

For basic time period grouping (months, day of year, season), passing a string to the methods needing it is sufficient.
Most methods acting on grouped data also accept a `window` int argument to pad the groups with data from adjacent ones.
Units of `window` are the sampling frequency of the main grouping dimension (usually `time`). For more complex grouping,
one can pass a :py:class:`xclim.downscaling.base.Grouper` directly.
"""
import xarray
from xarray.tests import LooseVersion


"""
TODO: ISIMIP ? Used for precip freq adjustment in biasCorrection.R

Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash;
The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013
"""
if LooseVersion(xarray.__version__) <= "0.15.1":
    raise ImportError("Update xarray to master to use the downscaling package.")
