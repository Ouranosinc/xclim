# -*- coding: utf-8 -*-
"""
Downscaling and bias-correction
===============================

This submodule provides tools for bias-correction and downscaling process on
climate data using xarray. Once complete and usable, chances are that this
submodule will move into its own package, so don't get attached too much.

Available methods:
- Scaling
- Local intensity scaling (LOCI)
- Empirical quantile mapping
- Quantile delta mapping


TODO: ISIMIP ? Used for precip freq ajdustment in biasCorrection.R
Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash; The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013


Logic:

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
import xarray
from xarray.tests import LooseVersion

if LooseVersion(xarray.__version__) <= "0.15.1":
    raise ImportError("Update xarray to master to use the downscaling package.")
