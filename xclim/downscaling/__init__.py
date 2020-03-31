# -*- coding: utf-8 -*-
"""
This submodule provides bias-correction and downscaling algorithms. Each method has a `train` and `predict` method.
Given time series of observations (obs), historical simulations (hist) and future simulations (fut),
any bias-correction method would be applied by first estimating the correction factors from the historical and
observations series, and then applying these factors to the future series::

  cf = train(hist, obs, group="time.month")
  fut_bc = predict(fut, cf, interp="linear")

The `group` argument allows correction factors to be estimated independently for different periods, either months,
seasons or day of the year. The `interp` argument then allows for interpolation between these correction factors to
avoid discontinuities in the bias-corrected series.

The same interpolation principle is also used for quantiles. Indeed, for methods extracting correction factors by
quantile, interpolation is also done between quantiles. This can help reduce discontinuities in the corrected time
series, and possibly reduce the number of quantile bins used.

"""
import xarray
from xarray.tests import LooseVersion


"""
TODO: ISIMIP ? Used for precip freq ajdustment in biasCorrection.R

Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash; The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013


Pipeline logic
==============

A pipeline is made out of these building blocks

- preprocessing on x_train
- preprocessing on y_train
- training `train(x_train, y_train) -> t_out`
- preprocessing on x_predict
- prediction `predict(x_predict, t_out) -> y_predict`
- post-processing on y_predict

Preprocessing methods include:

- detrending
- adding jitter to null values

Post-processing methods include

- retrending

Could the pre/post processing could be done with the `with` statement ?

---


What the use calls is a "Pipeline".
Given a detrending, a grouping and a mapping object, a typical pipeline would call:

- detrend.fit() + detrend.detrend() on each of obs, sim and fut
- grouping.group() on obs and sim
- mapping.fit() on obs+sim
- grouping.add_group_axis() on fut
- mapping.predict() on fut
- detrend.retrend() on fut

The whole module assumes the bias-correction is made along coord "time", for each group in coord "group", independent of any other coordinates.

The Detrending objects will detrend/retrend DataArrays along "time".

The Grouping objects will group elements of a DataArray along time and give back either a DataArrayGroupBy or a DataArray object with a time coord on which mapping should be done.
Once the action called, the new dimension is called "group" and has a "group_name" attribute with the real group name (the one a simple groupby(dim) would have given.)
Also, the add_group_axis() method, adds a coord that says where is each element along the time coord in the group. Ex : If the grouping is monthly, the 15th of April  group = 4, the 30th group = 4.5.

The Mapping objects perform a fit from obs and sim, the reference observation and the simulated climate for the same reference period, the fit is performed point-wise and group-wise, along "time".
Then, the predict() method takes in fut, the simulated climate for the projection period, that must have a "time" and a "group" coord, as given by the add_group_axis() grouping method.
It returns the corrected fut timeseries,
"""
if LooseVersion(xarray.__version__) <= "0.15.1":
    raise ImportError("Update xarray to master to use the downscaling package.")
