"""
Generic Index Submodule
=======================

A generic index function is a function that computes a resampling indicator without a specific application or input
variable. The functions defined here are the building blocks for most xclim indicators.

A generic index function should take in one or multiple variable in the form of :py:class:`xarray.DataArray`,
as its first arguments. Almost all functions here should also take a `freq` argument, defining the resampling period.
A specific vocabulary and annotations are used in this submodule to define arguments as clearly as possible.
The vocabulary is strongly inspired from `clix-meta <https://github.com/clix-meta/clix-meta/>`_.

- ``data: xr.DataArray`` : The first(s) arguments of all index function. When multiple variables are required,
    an integer suffix is added.
- ``statistic: Reducer`` : The name of a time-reducing operation, usually a built-in numpy/xarray method or a member of
    :py:data:`~xclim.indices.reducers.XCLIM_OPS`.
- ``condition: Condition`` : The string or symbol of a binary comparison operator. Should usually be a key or valid of
    :py:data:`~xclim.indices.helpers.BINARY_OPS`.
- ``thresh: Quantified`` : A threshold for thresholded index. Usually a string with a value and units (``" 0 °C"``),
    index functions should also accept non-temporal DataArrays and pint Quantity objects.
- ``freq: Freq`` : A frequency string referring to a pandas
    `date offset object <https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects>`_.
    Xclim only officially supports the frequency strings that xarray's implementation of CFtime supports,
    so the ones completely independent of a specific calendar.
- ``**indexer`` : Time selection arguments as implemented by :py:func:`~xclim.core.calendar.select_time`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from xclim.core import Condition, DayOfYearStr, Freq, Quantified, Reducer, TimeRange
from xclim.core.bootstrapping import percentile_bootstrap
from xclim.core.calendar import (
    doy_to_days_since,
    get_calendar,
    percentile_doy,
    resample_doy,
    select_time,
)
from xclim.core.units import (
    convert_units_to,
    declare_relative_units,
    is_temporal_rate,
    pint2cfattrs,
    str2pint,
    to_agg_units,
    units2pint,
)
from xclim.indices import run_length as rl
from xclim.indices.helpers import compare, resample_map, spell_mask
from xclim.indices.reducers import XCLIM_OPS


def statistics(
    data: xr.DataArray, statistic: Reducer, freq: Freq, out_units: str | None = None, **indexer
) -> xr.DataArray:
    r"""
    Calculate a statistic over the data for each requested period.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    statistic : {"min", "max", "mean", "std", "var", 'count', 'sum', 'integral', 'doymax', 'doymin'} or Callable
        Reducing operation. It can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    out_units : str, optional
        Output units to assign (no conversion tried).
        Only necessary if `statistic` is function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        {statistic} of the data.
    """
    data = select_time(data, **indexer)
    if isinstance(statistic, str):
        # Get function for xclim-implemented statistics
        statistic = XCLIM_OPS.get(statistic, statistic)
    if statistic == "sum" and is_temporal_rate(data):
        statistic = "integral"
    if isinstance(statistic, str):
        out = getattr(data.resample(time=freq), statistic.replace("integral", "sum"))(dim="time", keep_attrs=True)
    else:
        with xr.set_options(keep_attrs=True):
            out = resample_map(data, "time", freq, statistic)

    if out_units is not None:
        return out.assign_attrs(units=out_units)

    return to_agg_units(out, data, statistic)


def running_statistics(
    data: xr.DataArray,
    window: int,
    window_statistic: Reducer,
    statistic: Reducer,
    freq: Freq,
    window_center: bool = True,
    out_units=None,
    **indexer,
) -> xr.DataArray:
    r"""
    Calculate a running statistic over the data and then another statistic for each requested period.

    This is an extension of :py:func:`statistics`, with a rolling aggregation done before the indexing and resampling.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    window : int
        Size of the rolling window.
    window_statistic : {"min", "max", "mean", "std", "var", "count", "sum", "integral"}
        Operation to apply to the rolling window.
    statistic : {"min", "max", "mean", "std", "var", "count", "sum", "integral", "doymax", "doymin"} or Callable
        Reducing operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Resampling is done after the running statistic.
    window_center : bool
        If True, the window is centered on the date. If False, the window is right-aligned.
    out_units : str, optional
        Output units to assign.
        Only necessary if `statistic` is a function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Time selection is done after applying the running statistic.

    Returns
    -------
    xr.DataArray
        {statistic} of the {window}-day {window_statistic} of the data.
    """
    if window_statistic == "sum" and is_temporal_rate(data):
        window_statistic = "integral"
    rolled = getattr(
        data.rolling(time=window, center=window_center),
        window_statistic.replace("integral", "sum"),
    )()
    rolled = to_agg_units(rolled, data, window_statistic)
    return statistics(rolled, statistic=statistic, freq=freq, out_units=out_units, **indexer)


@declare_relative_units(thresh="<data>")
def thresholded_statistics(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    statistic: Reducer,
    freq: Freq,
    constrain: Sequence[str] | None = None,
    out_units=None,
    **indexer,
) -> xr.DataArray:
    """
    Calculate a statistic over data that fulfills a threshold condition for each requested periods.

    This is a thresolded extension of :py:func:`statistics`.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Comparison is done as ``data {condition} thresh``.
    thresh : Quantified
        Threshold, should have the same dimensionality as ``data``.
    statistic :  {"min", "max", "mean", "std", "var", "count", "sum", "integral", "doymin", "doymax"} or Callable
        Reducing operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    constrain : sequence of str, optional
        Allowed conditions, to be used when creating a more specific indicator from this function.
    out_units : str, optional
        Output units to assign.
        Only necessary if `statistic` is a function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        {statistic} of data where it is {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")

    cond = compare(data, condition, thresh, constrain)
    return statistics(data.where(cond), statistic, freq, out_units=out_units, **indexer)


@declare_relative_units(thresh="<data>")
def thresholded_running_statistics(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    window: int,
    window_statistic: Reducer,
    statistic: Reducer,
    freq: Freq,
    window_center: bool = True,
    constrain=None,
    out_units=None,
    **indexer,
) -> xr.DataArray:
    """
    Calculate a running statistic of the data for which some condition is met, then compute a resampling statistic.

    This is an extension of :py:func:`running_statistics` with a threshold.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Comparison is done as ``data {condition} thresh``.
    thresh : Quantified
        Threshold, should have the same dimensionality as ``data``.
    window : int
        Size of the rolling window.
    window_statistic : {"min", "max", "mean", "std", "var", "count", "sum", "integral"}
        Operation to apply to the rolling window.
    statistic : {"min", "max", "mean", "std", "var", "count", "sum", "integral", "doymax", "doymin"} or Callable
        Reducing operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Resampling is done after the running statistic.
    window_center : bool
        If True, the window is centered on the date. If False, the window is right-aligned.
    constrain : sequence of str, optional
        Allowed conditions, to be used when creating a more specific indicator from this function.
    out_units : str, optional
        Output units to assign.
        Only necessary if `statistic` is a function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Time selection is done after the running statistic.

    Returns
    -------
    xr.DataArray
        {statistic} of the {window}-day {window_statistic} of the data, where it is {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")
    cond = compare(data, condition, thresh, constrain)
    return running_statistics(
        data.where(cond),
        window=window,
        window_center=window_center,
        window_statistic=window_statistic,
        statistic=statistic,
        freq=freq,
        out_units=out_units,
        **indexer,
    )


@declare_relative_units(thresh="<data>")
def count_occurrences(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    freq: Freq,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Count number of timesteps where the data fulfills a thresholded condition.

    The output has a temporal dimensionality, it is the total duration of moments where
    the condition is fulfilled, considering all variables as interval variables.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    condition : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
        Logical comparison operator. Comparison is done as ``data {condition} thresh``.
    thresh : Quantified
        Threshold value. Should have the same dimensionality as data.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    constrain : sequence of str, optional
        Allowed conditions, to be used when creating a more specific indicator from this function.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        Number of timesteps where data {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")
    cond = compare(data, condition, thresh, constrain) * 1
    out = cond.resample(time=freq).sum(dim="time")
    return to_agg_units(out, data, "count")


@declare_relative_units(low_bound="<data>", high_bound="<data>")
def count_domain_occurrences(
    data: xr.DataArray,
    low_bound: Quantified,
    high_bound: Quantified,
    freq: Freq,
    low_condition: Literal[">", ">=", "gt", "ge"] = ">",
    high_condition: Literal["<", "<=", "lt", "le"] = "<=",
    **indexer,
) -> xr.DataArray:
    """
    Count number of timesteps where the data is within two bounds.

    The output has a temporal dimensionality, it is the total duration of moments where
    the condition is fulfilled, considering all variables as interval variables.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    low_bound : Quantified
        Minimum value.
    high_bound : Quantified
        Maximum value.
    freq : str
        Resampling frequency defining the periods defined in :ref:`timeseries.resampling`.
    low_condition : {'>', '>=', 'gt', 'ge'}
        The comparison operator to use on the lower bound. Default is ">" which means
        equality does not fulfill the condition.
    high_condition : {'<', '<=', 'lt', 'le'}
        The comparison operator to use on the higher bound. Default is "<=" which means
        equality does fulfill the condition.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        {The number of days where value is within [low, high] for each period.
    """
    low = convert_units_to(low_bound, data, context="infer")
    high = convert_units_to(high_bound, data, context="infer")
    cond = (
        compare(data, low_condition, low, constrain=(">", ">="))
        & compare(data, high_condition, high, constrain=("<", "<="))
    ) * 1
    cond = select_time(cond, **indexer)
    out = cond.resample(time=freq).sum(dim="time")
    return to_agg_units(out, data, "count")


@declare_relative_units(thresh1="<data1>", thresh2="<data2>")
def bivariate_count_occurrences(
    data1: xr.DataArray,
    data2: xr.DataArray,
    condition1: Condition,
    condition2: Condition,
    thresh1: Quantified,
    thresh2: Quantified,
    freq: Freq,
    var_reducer: Literal["all", "any"] = "all",
    constrain1: Sequence[str] | None = None,
    constrain2: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Count number of timesteps where two variables fulfill thresholded conditions.

    The output has a temporal dimensionality, it is the total duration of moments where
    the condition is fulfilled, considering all variables as interval variables.

    Parameters
    ----------
    data1 : xr.DataArray
        An array.
    data2 : xr.DataArray
        An array.
    condition1 : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator for data variable 1.
    condition2 : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator for data variable 2.
        If None, ``condition1`` is used.
    thresh1 : Quantified
        Threshold for data variable 1.
    thresh2 : Quantified
        Threshold for data variable 2.
        If None, ``thresh1`` is used.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    var_reducer : {"all", "any"}
        The condition must either be fulfilled on *all* or *any* variables
        for the timestep to be considered an occurrence.
    constrain1 : sequence of str, optional
        Allowed comparison operators for variable 1, None to allow all.
    constrain2 : sequence of str, optional
        Allowed comparison operators for variable 2, None to allow all.
        If ``condition2`` is None, ``constrain1`` is used and ``constrain2`` is ignored.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray,  [time]
        Number of timesteps where data1 is {condition1} {thresh1} and data2 is {condition2} {thresh2}.

    Notes
    -----
    Sampling length is derived from `data1`.
    """
    if thresh2 is None:
        thresh1 = thresh1
    thresh1 = convert_units_to(thresh1, data1, context="infer")
    thresh2 = convert_units_to(thresh2, data2, context="infer")

    if condition2 is None:
        condition2 = condition1
        constrain2 = constrain1
    cond1 = compare(data1, condition1, thresh1, constrain1)
    cond2 = compare(data2, condition2, thresh2, constrain2)

    if var_reducer == "all":
        cond = cond1 & cond2
    elif var_reducer == "any":
        cond = cond1 | cond2
    else:
        raise ValueError(f"Unsupported value for var_reducer: {var_reducer}")

    cond = select_time(cond, **indexer)
    out = cond.resample(time=freq).sum()
    return to_agg_units(out, data1, "count", dim="time")


def count_percentile_occurrences(
    data: xr.DataArray,
    percentile: float,
    condition: Condition,
    reference_period: TimeRange,
    freq: Freq,
    window: int = 5,
    bootstrap: bool = False,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Count how many times an annually varying percentile-based thresholded condition is fulfilled.

    For each day-of-year, the percentile is computed over the reference period with a doy window.
    Then the number of timesteps where this threshold fulfills the condition is counted
    for each requested period. The output has a temporal dimensionality, it is the total duration of moments where
    the condition is fulfilled, considering all variables as interval variables.

    Parameters
    ----------
    data : xr.DataArray
        An array. Should have a daily step.
    percentile : float
        The percentile to compute on the reference period, between 0 and 100.
    condition :  {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Computed as  ``data[i] {condition} climatology[doy(i)]``.
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates should be given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.
    window : int
        The number of days on each side of the given day-of-year to include in the climatology.
    bootstrap : bool
        Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
        Bootstrapping is only useful when the percentiles are computed on a part of the studied sample (like here).
        This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
        the rest of the time series
        Note that bootstrapping is computationally expensive.
    constrain : sequence of str, optional
        Allowed conditions. None to allow them all.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Subsetting is not done one the data used to compute the climatology, only on the data against
        which the condition is checked.

    Returns
    -------
    xr.DataArray
        Number of timesteps where data is {condition} the {percentile}th percentile computed over {reference_period}.
    """
    clim = percentile_doy(data.sel(time=slice(*reference_period)), window=window, per=percentile)
    data = select_time(data, **indexer)

    @percentile_bootstrap
    def _count_percentile_occurrences(data, per, freq, bootstrap, op):
        thresh = resample_doy(clim, data)
        cond = compare(data, op, thresh, constrain)
        out = cond.resample(time=freq).sum()
        return to_agg_units(out, data, "count", dim="time")

    return _count_percentile_occurrences(data, clim, freq, bootstrap, condition)


@declare_relative_units(thresh="<data>")
def count_thresholded_percentile_occurrences(
    data: xr.DataArray,
    data_condition: Condition,
    thresh: Quantified,
    percentile: float,
    condition: Condition,
    reference_period: TimeRange,
    freq: Freq,
    window: int = 5,
    bootstrap: bool = False,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Count how many times an annually-varying percentile condition is fulfilled over data fulfilling another condition.

    The data is first filtered to keep only the timesteps where the thresholded ``data_condition`` is fulfilled.
    Then, for each day-of-year, the percentile is computed over the reference period with a doy window.
    Then the number of timesteps where this threshold fulfills the condition is counted
    for each requested period. The output has a temporal dimensionality, it is the total duration of moments where
    the condition is fulfilled, considering all variables as interval variables.

    Parameters
    ----------
    data : xr.DataArray
        An array. Should have a daily step.
    data_condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator to filter data with threshold.
    thresh : Quantified
        Threshold for the ``data_condition``.
    percentile : float
        The percentile to compute on the reference period, between 0 and 100.
    condition :  {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator to find occurrences. Computed as  ``data[i] {condition} climatology[doy(i)]``.
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates should be given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.
    window : int
        The number of days on each side of the given day-of-year to include in the climatology.
    bootstrap : bool
        Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
        Bootstrapping is only useful when the percentiles are computed on a part of the studied sample (like here).
        This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
        the rest of the time series
        Note that bootstrapping is computationally expensive.
    constrain : sequence of str, optional
        Allowed conditions. None to allow them all.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Subsetting is not done one the data used to compute the climatology, only on the data against
        which the condition is checked.

    Returns
    -------
    xr.DataArray
        Number of timesteps where data is {condition} the {percentile}th percentile computed over {reference_period}.
        Only data {data_condition} {thresh} is considered.
    """
    thresh = convert_units_to(thresh, data, context="infer")
    data = data.where(compare(data, data_condition, thresh, constrain))
    return count_percentile_occurrences(
        data,
        percentile,
        condition=condition,
        freq=freq,
        reference_period=reference_period,
        window=window,
        constrain=constrain,
        bootstrap=bootstrap,
        **indexer,
    )


def _spell_length_statistics(
    data: xr.DataArray | Sequence[xr.DataArray],
    window: int,
    window_statistic: Reducer,
    condition: Condition,
    thresh: float | xr.DataArray | Sequence[xr.DataArray] | Sequence[float],
    statistic: Reducer | Sequence[Reducer],
    freq: Freq,
    min_gap: int = 1,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    if isinstance(statistic, str):
        statistic = [statistic]
    is_in_spell = spell_mask(data, window, window_statistic, condition, thresh, min_gap=min_gap).astype(np.float32)
    is_in_spell = select_time(is_in_spell, **indexer)

    outs = []
    for sr in statistic:
        out = rl.resample_and_rl(
            is_in_spell,
            resample_before_rl,
            rl.rle_statistics,
            reducer=sr,
            # The code above already ensured only spell of the minimum length are selected
            window=1,
            freq=freq,
        )

        if sr == "count":
            outs.append(out.assign_attrs(units=""))
        else:
            dd = data if isinstance(data, xr.DataArray) else data[0]
            # All other cases are statistics of the number of timesteps
            # Get a DataArray with units of counting timesteps
            ts_units = to_agg_units(dd, dd, "count")
            outs.append(to_agg_units(out, ts_units, sr))
    if len(outs) == 1:
        return outs[0]
    return tuple(outs)


@declare_relative_units(thresh="<data>")
def spell_length_statistics(
    data: xr.DataArray,
    window: int,
    window_statistic: Literal["min", "max", "sum", "mean"],
    condition: Condition,
    thresh: Quantified,
    statistic: Literal["max", "sum", "count"] | Sequence[Literal["max", "sum", "count"]],
    freq: Freq,
    min_gap: int = 1,
    constrain: Sequence[str] | None = None,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    r"""
    Statistics of spells lengths.

    A spell is when a running statistic (`window_statistic`) over a (minimum) number (`window`) of consecutive timesteps
    respects a condition (`condition` `thresh`). This returns a statistic over the spell lengths.
    Two consecutive spells are merged into a single one.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    window : int
        Minimum length of a spell.
    window_statistic : {'min', 'max', 'sum', 'mean', 'integral'}
        Reduction along the window length to compute running statistic.
        Note that this does not matter when `window` is 1, in which case any occurrence
        of ``data {condition} thresh`` is considered a valid "spell".
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Computed as ``rolling_stat {condition} thresh``.
    thresh : Quantified
        Threshold to test against.
    statistic : {'max', 'sum', 'count'} or sequence of str
        Statistic on the spell lengths. If a list, multiple statistics are computed.
    freq : str
        Resampling frequency.
    min_gap : int
        The shortest possible gap between two spells. Spells closer than this are merged by assigning
        the gap steps to the merged spell.
    constrain : sequence of str, optional
        Allowed conditions. None to allow them all.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Indexing is done after finding the days part of a spell, but before taking the spell statistics.

    Returns
    -------
    xr.DataArray or sequence of xr.DataArray
        {statistic} of spell lengths. A spell is when the {window}-day {window_statistic} is {condition} {thresh}.

    See Also
    --------
    xclim.indices.helpers.spell_mask : The lower level functions that finds spells.
    bivariate_spell_length_statistics : The bivariate version of this function.

    Examples
    --------
    >>> spell_length_statistics(
    ...     tas,
    ...     window=7,
    ...     window_statistic="min",
    ...     condition=">",
    ...     thresh="35 °C",
    ...     statistic="sum",
    ...     freq="YS",
    ... )

    Here, a day is part of a spell if it is in any seven (7) day period where the minimum temperature is over 35°C.
    We then return the annual sum of the spell lengths, so the total number of days in such spells.
    >>> from xclim.core.units import rate2amount
    >>> pram = rate2amount(pr, out_units="mm")
    >>> spell_length_statistics(
    ...     pram,
    ...     thresh="20 mm",
    ...     window=5,
    ...     op=">=",
    ...     win_reducer="sum",
    ...     spell_reducer="max",
    ...     freq="YS",
    ... )

    Here, a day is part of a spell if it is in any five (5) day period where the total accumulated precipitation
    reaches or exceeds 20 mm. We then return the length of the longest of such spells.
    """
    thresh = convert_units_to(thresh, data, context="infer")
    return _spell_length_statistics(
        data,
        window,
        window_statistic,
        condition,
        thresh,
        statistic,
        freq,
        min_gap=min_gap,
        resample_before_rl=resample_before_rl,
        **indexer,
    )


@declare_relative_units(thresh1="<data1>", thresh2="<data2>")
def bivariate_spell_length_statistics(
    data1: xr.DataArray,
    data2: xr.DataArray,
    window: int,
    window_statistic: Literal["min", "max", "sum", "mean"],
    condition: Condition,
    thresh1: Quantified,
    thresh2: Quantified,
    statistic: Literal["max", "sum", "count"] | Sequence[Literal["max", "sum", "count"]],
    freq: Freq,
    min_gap: int = 1,
    constrain: Sequence[str] | None = None,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    r"""
    Statistics of bivariate spells lengths.

    A spell is when a running statistic (`window_statistic`) over a (minimum) number (`window`) of consecutive timesteps
    respects a condition (data ``condition`` ``thresh``). Then this returns a statistic over the spell lengths.
    Two consecutive spells are merged into a single one.

    Parameters
    ----------
    data1 : xr.DataArray
        Input data.
    data2 : xr.DataArray
        Input data.
    window : int
        Minimum length of a spell.
    window_statistic : {'min', 'max', 'sum', 'mean', 'integral'}
        Reduction along the window length to compute running statistic.
        Note that this does not matter when `window` is 1, in which case any occurrence
        of ``data {condition} thresh`` is considered a valid "spell".
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Computed as ``rolling_stat {condition} thresh``.
    thresh1 : Quantified
        Threshold to test against for data1.
    thresh2 : Quantified
        Threshold to test against for data2.
    statistic : {'max', 'sum', 'count'} or sequence of str
        Statistic on the spell lengths. If a list, multiple statistics are computed.
    freq : str
        Resampling frequency.
    min_gap : int
        The shortest possible gap between two spells. Spells closer than this are merged by assigning
        the gap steps to the merged spell.
    constrain : sequence of str, optional
        Allowed conditions. None to allow them all.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Indexing is done after finding the days part of a spell, but before taking the spell statistics.

    Returns
    -------
    xr.DataArray or sequence of xr.DataArray
        {statistic} of spell lengths. A spell is when the {window}-day {window_statistic}
        of data1 is {condition} {thresh1} and the one of data2 is {condition} {thresh2}.

    See Also
    --------
    spell_length_statistics : The univariate version.
    xclim.indices.helpers.spell_mask : The lower level functions that finds spells.
    """
    thresh1 = convert_units_to(thresh1, data1, context="infer")
    thresh2 = convert_units_to(thresh2, data2, context="infer")
    return _spell_length_statistics(
        [data1, data2],
        window,
        window_statistic,
        condition,
        [thresh1, thresh2],
        statistic,
        freq,
        min_gap=min_gap,
        resample_before_rl=resample_before_rl,
        **indexer,
    )


@declare_relative_units(thresh="<data>")
def season(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    window: int,
    aspect: Literal["start", "end", "length"] | Sequence[Literal["start", "end", "length"]],
    freq: Freq,
    mid_date: DayOfYearStr | None = None,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    r"""
    Season.

    A season starts when a variable fulfills some condition for a consecutive run of ``window`` days. It stops
    when the inverse condition is fulfilled for ``window`` days. Seasons with "gaps" where the condition is not met
    for fewer than ``window`` days are thus allowed. Additionally, a middle date can serve as a latest start date
    and earliest end date.

    Parameters
    ----------
    data : xr.DataArray
        Variable.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Comparison operation. Computed as ``data {condition} thresh``.
    thresh : Quantified
        Threshold for the condition.
    window : int
        Minimum number of days that the condition must be met / not met for the start / end of the season.
    aspect : {'start', 'end', 'length'}, or a list of those
        Which season aspect(s) to return. If a list, this function returns a tuple in the same order as this argument.
    freq : str
        Resampling frequency.
    mid_date : DayOfYearStr, optional
        An optional middle date. The start must happen before and the end after for the season to be valid.
    constrain : Sequence of strings, optional
        A list of acceptable comparison operators. Optional, but indicators wrapping this function should inject it.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray, [dimensionless] or [time]
        {aspect} of the season. The season starts with {window} consecutibe days {condition} {thresh} and ends
        when the inverse condition is fulfilled for as much consecutive days.

    See Also
    --------
    xclim.indices.run_length.season_start : The function that finds the start of the season.
    xclim.indices.run_length.season_length : The function that finds the length of the season.
    xclim.indices.run_length.season_end : The function that finds the end of the season.

    Examples
    --------
    >>> season(tas, thresh="0 °C", window=5, op=">", stat="start", freq="YS")

    Returns the start of the "frost-free" season. The season starts with 5 consecutive days with mean temperature
    above 0°C and ends with as many days under or equal to 0°C, and end does not need to be found for a
    start to be valid.

    >>> season(
    ...     pr,
    ...     thresh="2 mm/d",
    ...     window=7,
    ...     op="<=",
    ...     mid_date="08-01",
    ...     stat="length",
    ...     freq="YS",
    ... )

    Returns the length of the "dry" season. The season starts with 7 consecutive days with precipitation under or
    equal to 2 mm/d and ends with as many days above 2 mm/d. If no start is found before the first of august,
    the season is invalid. If a start is found but no end, the end is set to the last day of the period
    (December 31st if the dataset is complete).
    """
    thresh = convert_units_to(thresh, data, context="infer")
    cond = compare(data, condition, thresh, constrain=constrain)
    cond = select_time(cond, **indexer)
    func = {"start": rl.season_start, "end": rl.season_end, "length": rl.season_length}
    map_kwargs = {"window": window, "mid_date": mid_date}

    if aspect in ["start", "end"]:
        map_kwargs["coord"] = "dayofyear"
    out = resample_map(cond, "time", freq, func[aspect], map_kwargs=map_kwargs)
    if aspect == "length":
        return to_agg_units(out, data, "count")
    # else, a date
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(data))
    return out


def season_length_from_boundaries(season_start: xr.DataArray, season_end: xr.DataArray) -> xr.DataArray:
    """
    Season length using pre-computed boundaries.

    Parameters
    ----------
    season_start : xr.DataArray
        Day of year where the season starts.
    season_end : xr.DataArray
        Day of year where the season ends.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Length of the season.

    Notes
    -----
    If `season_start` and `season_end` are computed with different resampling frequencies, the time
    of `season_start` are selected to write the output.  This is only useful when season start and end were computed
    at an annual frequency but with different anchor months. Otherwise, functions in ``xclim.indices.run_length``
    will be appropriate. `season_start` and `season_end` should be annual indicators with the same length. `season_end`
    should be in the same year as `season_start` or one year later.
    """
    if (
        season_start.time.size == season_end.time.size
        or 0 <= (season_end.time[0] - season_start.time[0]).astype("timedelta64[s]") < 365 * 24 * 60 * 60
    ) is False:
        raise ValueError(
            "`season_start` and `season_end` should have the same length, and `season_end`'s"
            "times coordinates should start with the time coordinates of `season_start`, "
            "or after, within a year."
        )

    freq_start = xr.infer_freq(season_start.time)
    freq_end = xr.infer_freq(season_end.time)
    if (freq_start.startswith("Y") and freq_end.startswith("Y")) is False:
        raise ValueError(
            "`season_start` and `season_end` should both be annual indicators, but the following frequencies"
            "were inferred: {freq_start} and {freq_end}."
        )
    days_since_start = doy_to_days_since(season_start)
    days_since_end = doy_to_days_since(season_end)
    days_since_end["time"] = days_since_start.time
    doy_start = season_start.time.dt.dayofyear
    doy_end = season_end.time.dt.dayofyear
    # days_since we computed with the respective time arrays of season_start and season_end,
    # but now we will express the season_length using the times of season_start
    doy_end["time"] = doy_start.time
    out = (days_since_end + doy_end - doy_start) - days_since_start
    out.attrs.update(units="days")
    return out


@declare_relative_units(data2="<data1>")
def difference_statistics(
    data1: xr.DataArray, data2: xr.DataArray, statistic: Reducer, freq: Freq, absolute: bool = False, **indexer
) -> xr.DataArray:
    """
    Calculate a statistic over the difference between two variables.

    The difference is taken as ``data2 - data1``.

    Parameters
    ----------
    data1 : xr.DataArray
        The lowest variable (ex: tasmin)).
    data2 : xr.DataArray
        The highest variable (ex: tasmax).
    statistic : {'max', 'min', 'mean', 'sum'}
        The statistic to compute over the difference between the two variables.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    absolute : bool
        If True, the statistic is computed over the absolute difference.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray, [difference of data1]
        {statistic} of the difference between data2 and data1.
    """
    data2 = convert_units_to(data2, data1, context="infer")

    dtr = data2 - data1
    if absolute:
        dtr = abs(dtr)
    u = str2pint(data1.units)
    dtr.attrs.update(pint2cfattrs(u, is_difference=True))

    return statistics(dtr, statistic=statistic, freq=freq, **indexer)


@declare_relative_units(data2="<data1>")
def extreme_range(data1: xr.DataArray, data2: xr.DataArray, freq: Freq, **indexer) -> xr.DataArray:
    """
    Calculate the extreme's range.

    The maximum of data2 minus the minimum of data1, for each period.

    Parameters
    ----------
    data1 : xr.DataArray
        The lowest data.
    data2 : xr.DataArray
        The highest data.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        The DataArray for the extreme temperature range.
    """
    data2 = convert_units_to(data2, data1, context="infer")
    data2 = select_time(data2, **indexer)
    data1 = select_time(data1, **indexer)

    out = data2.resample(time=freq).max() - data1.resample(time=freq).min()

    u = str2pint(data1.units)
    out.attrs.update(pint2cfattrs(u, is_difference=True))
    return out


def interday_difference_statistics(
    data1: xr.DataArray, data2: xr.DataArray, statistic: Reducer, freq: Freq, absolute: bool = True, **indexer
) -> xr.DataArray:
    """
    Calculate a statistic of the day-to-day difference of the difference between two variables.

    The difference is taken as ``data2 - data1``, then it is differentiated along the time dimension before
    calculating the resampling statistic.

    Parameters
    ----------
    data1 : xr.DataArray
        The lowest data.
    data2 : xr.DataArray
        The highest data.
    statistic : {'max', 'min', 'mean', 'sum'}
        Resampling statistic.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    absolute : bool
        If True, the statistic is computed over the absolute value of the differentiated difference.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.
        Subsetting is done after differentiating along time.

    Returns
    -------
    xr.DataArray, [difference of low_data]
        {statistic} of the day-to-day difference of the difference between data2 and data1.
    """
    data2 = convert_units_to(data2, data1, context="infer")
    vdtr = abs((data2 - data1).diff(dim="time"))
    u = str2pint(data1.units)
    vdtr.attrs.update(pint2cfattrs(u, is_difference=True))
    return statistics(vdtr, statistic=statistic, freq=freq, **indexer)


def percentile(data: xr.DataArray, percentile: float, freq: Freq, **indexer):
    """
    Calculate the percentile statistic for each requested period.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    percentile : float
        A percentile (0, 100).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray, [same as data]
        {percentile}th percentile of the data.
    """
    q = percentile / 100
    data = select_time(data, **indexer)
    out = data.resample(time=freq).quantile(q).drop_vars("quantile")
    out.attrs["units"] = data.attrs["units"]
    return out


@declare_relative_units(thresh="<data>")
def thresholded_percentile(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    percentile: float,
    freq: Freq,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Calculate a percentile of the data for which some condition is met, for each requested period.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator. Calculated as ``data {condition} thresh``.
    thresh : Quantified
        Threshold.
    percentile : float
        A percentile (0, 100).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    constrain : sequence of str, optional
        Optionally allowed conditions. Default: None.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        {percentile}th percentile of the data where it is {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")
    cond = compare(data, condition, thresh, constrain)
    return percentile(data.where(cond), percentile, freq, **indexer)


def statistics_between_dates(
    data: xr.DataArray,
    start: xr.DataArray | DayOfYearStr,
    end: xr.DataArray | DayOfYearStr,
    statistic: Reducer,
    freq: str | None = None,
) -> xr.DataArray:
    """
    Calculate a statistic for each requested period but only considering timesteps with a time-varying range.

    This is similar to using :py:func:`statistics` with an indexer but for cases where the start and end bounds of the
    period of interest are changing along time, i.e, at least one of them is given as a DataArray.
    ``start`` and ``end`` must have aligneable time coordinates and be at the target frequency ``freq``.

    Usually, ``start`` and/or ``end`` will be the output of other indicators like :py:func:`season`
    or :py:func:`day_threshold_reached`.

    Parameters
    ----------
    data : xr.DataArray
        Data.
    start : xr.DataArray or DayOfYearStr
        Start dates (as day-of-year) for the statistic computation. The start date is included in the statistic.
    end : xr.DataArray or DayOfYearStr
        End (as day-of-year) dates for the statistic computation. The end date is not included in the statistic.
    statistic : {'min', 'max', 'sum', 'mean', 'std'}
        Statistic to compute over the selected period.
    freq : str, optional
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Default (None) tries to infer the frequency from ``start`` and ``end``.

    Returns
    -------
    xr.DataArray, [time]
        {statistic} of data over a time-varying period.
    """

    def _get_days(_bound, _group, _base_time):
        """Get bound in number of days since base_time. Bound can be a days_since array or a DayOfYearStr."""
        if isinstance(_bound, str):
            b_i = rl.index_of_date(_group.time, _bound, max_idxs=1)
            if not b_i.size > 0:
                return None
            return (_group.time.isel(time=b_i[0]) - _group.time.isel(time=0)).dt.days
        if _base_time in _bound.time:
            return _bound.sel(time=_base_time)
        return None

    if freq is None:
        frequencies = []
        for bound in [start, end]:
            try:
                frequencies.append(xr.infer_freq(bound.time))
            except AttributeError:
                frequencies.append(None)

        good_freq = set(frequencies) - {None}

        if len(good_freq) != 1:
            raise ValueError(
                f"Non-inferrable resampling frequency or inconsistent frequencies. Got start, end = {frequencies}."
                " Please consider providing `freq` manually or fixing the frequencies of start and end."
            )
        freq = good_freq.pop()

    cal = data.time.dt.calendar
    if not isinstance(start, str):
        start = start.convert_calendar(cal)
        start.attrs["calendar"] = cal
        start = doy_to_days_since(start)
    if not isinstance(end, str):
        end = end.convert_calendar(cal)
        end.attrs["calendar"] = cal
        end = doy_to_days_since(end)

    if isinstance(statistic, str):
        # Get function for xclim-implemented statistics
        statistic = XCLIM_OPS.get(statistic, statistic)
    if statistic == "sum" and is_temporal_rate(data):
        statistic = "integral"

    out = []
    for base_time, indexes in data.resample(time=freq).groups.items():
        # get group slice
        group = data.isel(time=indexes)

        start_d = _get_days(start, group, base_time)
        end_d = _get_days(end, group, base_time)

        # convert bounds for this group
        if start_d is not None and end_d is not None:
            days = (group.time - base_time).dt.days
            days = days.where(days >= 0)

            masked = group.where((days >= start_d) & (days <= end_d - 1))

            if isinstance(statistic, str):
                res = getattr(masked, statistic.replace("integral", "sum"))(dim="time", keep_attrs=True)
            else:
                with xr.set_options(keep_attrs=True):
                    res = statistic(masked, dim="time")

            res = xr.where(((start_d > end_d) | (start_d.isnull()) | (end_d.isnull())), np.nan, res)
            # Re-add the time dimension with the period's base time.
            res = res.expand_dims(time=[base_time])
            out.append(res)
        else:
            # Get an array with the good shape, put nans and add the new time.
            res = (group.isel(time=0) * np.nan).expand_dims(time=[base_time])
            out.append(res)
            continue

    out = xr.concat(out, dim="time")
    return to_agg_units(out, data, statistic)


@declare_relative_units(thresh="<data>")
def integrated_difference(
    data: xr.DataArray, condition: Condition, thresh: Quantified, freq: Freq, **indexer
) -> xr.DataArray:
    """
    Integrate difference of data below/above a given value threshold.

    If ``condition`` is ">", then the difference is taken as ``data - thresh``. The inverse
    is done for "<". Values below zero are removed from the integral. "Integral" means summed
    difference are multiplied by the timestep length.

    Parameters
    ----------
    data : xr.DataArray
        Data.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    thresh : Quantified
        The value threshold.
    freq : str, optional
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray, [data][time]
        Integral of the differences when {data} {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")

    if condition in ["<", "<=", "lt", "le"]:
        diff = (thresh - data).clip(0)
    elif condition in [">", ">=", "gt", "ge"]:
        diff = (data - thresh).clip(0)
    else:
        raise NotImplementedError(f"Condition not supported: '{condition}'.")

    diff.attrs.update(pint2cfattrs(units2pint(data.attrs["units"]), is_difference=True))
    return statistics(diff, statistic="integral", freq=freq, **indexer)


@declare_relative_units(thresh="<data>")
def day_threshold_reached(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    freq: Freq,
    date: DayOfYearStr | None = None,
    which: Literal["first", "last"] = "first",
    window: int = 1,
    constrain: Sequence[str] | None = None,
    **indexer,
) -> xr.DataArray:
    r"""
    First or last day of values fulfilling a condition.

    Returns first or last day of period where values meet a given condition for a minimum number of consecutive days,
    limited to a starting or ending calendar date.

    Parameters
    ----------
    data : xr.DataArray
        Data.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator.
    thresh : str
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    date : str or None
        Date of the year after which to look for the first event, or before which to look for the last event.
        Should have the format '%m-%d'. None means there is no limit.
    which : {'first', 'last'}
        Whether to look for the first or the last event.
    window : int
        Minimum number of days with values above thresh needed for evaluation. Default: 1.
    constrain : sequence of str, optional
        Optionally allowed conditions.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Day-of-year of the {which} time where data {condition} {thresh}.
    """
    thresh = convert_units_to(thresh, data, context="infer")

    cond = compare(data, condition, thresh, constrain=constrain)

    if which == "first":
        func = rl.first_run_after_date
    elif which == "last":
        func = rl.last_run_before_date
    else:
        raise ValueError(f"'which' must be 'first' or 'last'. Got {which}.")

    cond = select_time(cond, **indexer)
    out: xr.DataArray = resample_map(
        cond,
        "time",
        freq,
        func,
        map_kwargs={"window": window, "date": date, "dim": "time", "coord": "dayofyear"},
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(data))
    return out


@declare_relative_units(thresh="<data>")
def thresholded_events(
    data: xr.DataArray,
    condition: Condition,
    thresh: Quantified,
    window: int,
    condition_stop: Condition | None = None,
    thresh_stop: Quantified | None = None,
    window_stop: int | None = None,
    freq: str | None = None,
) -> xr.Dataset:
    r"""
    Find thresholded events.

    Finds all events along the time dimension.
    An event starts if the start condition is fulfilled for a given number of consecutive time steps.
    It ends when the end condition is fulfilled for a given number of consecutive time steps.

    Conditions are simple comparison of the data with a threshold: ``cond = data {condition} thresh``.
    The end conditions defaults to the negation of the start condition.

    The resulting ``event`` dimension always has its maximal possible size : ``data.size / (window + window_stop)``.

    Parameters
    ----------
    data : xr.DataArray
        Variable.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator for the start condition.
    thresh : Quantified
        Threshold defining the event.
    window : int
        Number of time steps where the event condition must be true to start an event.
    condition_stop : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}, optional
        Logical comparison operator for the end condition. Defaults to the opposite of `condition`.
    thresh_stop : Quantified, optional
        Threshold defining the end of an event. Defaults to `thresh`.
    window_stop : int, optional
        Number of time steps where the end condition must be true to end an event. Defaults to ``window``.
    freq : str, optional
        A frequency to divide the data into periods. If absent, the output has not time dimension.
        If given, the events are searched within in each resample period independently.

    Returns
    -------
    xr.Dataset
        Same shape as the data except the time dimension is replaced by an "event" dimension
        or it is resampled if ``freq`` is given.

        The dataset contains the following variables:
            event_length: The number of time steps in each event including gaps shorter than ``window_stop``
            event_effective_length: The number of time steps of even event where the start condition is true.
            event_sum: The sum within each event, only considering the steps where start condition is true.
            event_start: The datetime of the start of the run.
    """
    thresh = convert_units_to(thresh, data)

    # Start and end conditions
    da_start = compare(data, condition, thresh)
    if thresh_stop is None and condition_stop is None:
        da_stop = ~da_start
    else:
        thresh_stop = convert_units_to(thresh_stop or thresh, data)
        if condition_stop is not None:
            da_stop = compare(data, condition_stop, thresh_stop)
        else:
            da_stop = ~compare(data, condition, thresh_stop)

    return rl.find_events(da_start, window, da_stop, window_stop or window, data, freq)
