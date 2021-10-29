# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
===========================================
Statistical Downscaling and Bias Adjustment
===========================================

The `xclim.sdba` submodule provides bias-adjustment methods and will eventually provide statistical downscaling algorithms.
Almost all adjustment algorithms conform to the `train` - `adjust` scheme, formalized within `TrainAdjust` classes.
Given a reference time series (ref), historical simulations (hist) and simulations to be adjusted (sim),
any bias-adjustment method would be applied by first estimating the adjustment factors between the historical simulation and the observations series, and then applying these factors to `sim`, which could be a future simulation::

  # Create the adjustment object by training it with reference and model data, plus certains arguments
  Adj = Adjustment.train(ref, hist, group="time.month")
  # Get a scenario by applying the adjustment to a simulated timeseries.
  scen = Adj.adjust(sim, interp="linear")
  Adj.ds.af  # adjustment factors.

The `group` argument allows adjustment factors to be estimated independently for different periods: the full
time series,  months, seasons or day of the year. The `interp` argument then allows for interpolation between these
adjustment factors to avoid discontinuities in the bias-adjusted series (only applicable for monthly grouping).

.. warning::
    If grouping according to the day of the year is needed, the :py:mod:`xclim.core.calendar` submodule contains useful tools to manage the
    different calendars that the input data can have. By default, if 2 different calendars are passed, the adjustment
    factors will always be interpolated to the largest range of day of the years but this can lead to strange values
    and we recommend converting the data beforehand to a common calendar.

The same interpolation principle is also used for quantiles. Indeed, for methods extracting adjustment factors by
quantile, interpolation is also done between quantiles. This can help reduce discontinuities in the adjusted time
series, and possibly reduce the number of quantile bins used.

Modular approach
================

This module adopts a modular approach instead of implementing published and named methods directly.
A generic bias adjustment process is laid out as follows:

- preprocessing on `ref`, `hist` and `sim` (using methods in `xclim.sdba.processing` or `xclim.sdba.detrending`)
- creating and training the adjustment object `Adj = Adjustment.train(obs, sim, **kwargs)` (from `xclim.sdba.adjustment`)
- adjustment `scen = Adj.adjust(sim, **kwargs)`
- post-processing on `scen` (for example: re-trending)

The train-adjust approach allows to inspect the trained adjustment object. The training information is stored in
the underlying `Adj.ds` dataset and always has a `af` variable with the adjustment factors. Its layout and the
other available variables vary between the different algorithm, refer to :ref:`bias-adjustment-algos`.

Parameters needed by the training and the adjustment are saved to the `Adj.ds` dataset as a  `adj_params` attribute.
Other parameters, those only needed by the adjustment are passed in the `adjust` call and written to the history attribute
in the output scenario dataarray.

Grouping
========

For basic time period grouping (months, day of year, season), passing a string to the methods needing it is sufficient.
Most methods acting on grouped data also accept a `window` int argument to pad the groups with data from adjacent ones.
Units of `window` are the sampling frequency of the main grouping dimension (usually `time`). For more complex grouping,
one can pass a :py:class:`xclim.sdba.base.Grouper` directly.

Notes for developers
====================
To be scalable and performant, the sdba module makes use of the special decorators :py:func`xclim.sdba.base.map_blocks`
and :py:func:`xclim.sdba.base.map_groups`. However, they have the inconvenient that functions wrapped by them are unable
to manage xarray attributes (including units) correctly and their signatures are sometime wrong and often unclear. For
this reason, the module is often divided in two parts : the (decorated) compute functions in a "private" file (ex: ``_adjustment.py``)
and the user-facing functions or objects in corresponding public file (ex: ``adjustment.py``). See the `sdba-advanced`
notebook for more info on the reasons for this move.

Other restrictions : `map_blocks` will remove any "auxiliary" coordinates before calling the wrapped function and will add them back on exit.
"""
from . import detrending, processing, utils
from .adjustment import *
from .base import (
    Grouper,
    construct_moving_yearly_window,
    stack_variables,
    unpack_moving_yearly_window,
    unstack_variables,
)

# TODO: ISIMIP ? Used for precip freq adjustment in biasCorrection.R
# Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash;
# The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013
