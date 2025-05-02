"""
Generic Indices Submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Sequence

import cftime
import numpy as np
import xarray as xr
from pint import Quantity

from xclim.core import DayOfYearStr, Quantified
from xclim.core.calendar import _MONTH_ABBREVIATIONS, doy_to_days_since, get_calendar, select_time
from xclim.core.units import (
    convert_units_to,
    declare_relative_units,
    infer_context,
    pint2cfattrs,
    pint2cfunits,
    str2pint,
    to_agg_units,
    units2pint,
)
from xclim.indices import run_length as rl
from xclim.indices.helpers import resample_map

__all__ = [
    "aggregate_between_dates",
    "binary_ops",
    "bivariate_count_occurrences",
    "bivariate_spell_length_statistics",
    "compare",
    "count_level_crossings",
    "count_occurrences",
    "cumulative_difference",
    "default_freq",
    "detrend",
    "diurnal_temperature_range",
    "domain_count",
    "doymax",
    "doymin",
    "extreme_temperature_range",
    "first_day_threshold_reached",
    "first_occurrence",
    "get_daily_events",
    "get_op",
    "get_zones",
    "interday_diurnal_temperature_range",
    "last_occurrence",
    "season",
    "select_resample_op",
    "select_rolling_resample_op",
    "spell_length",
    "spell_length_statistics",
    "spell_mask",
    "statistics",
    "temperature_sum",
    "threshold_count",
    "thresholded_statistics",
]

binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


def select_resample_op(
    da: xr.DataArray, op: str | Callable, freq: str = "YS", out_units=None, **indexer
) -> xr.DataArray:
    r"""
    Apply operation over each period that is part of the index selection.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'integral', 'argmax', 'argmin'} or func
        Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    out_units : str, optional
        Output units to assign.
        Only necessary if `op` is function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array.
        For example, use season='DJF' to select winter values, month=1 to select January, or month=[6,7,8]
        to select summer months. If not indexer is given, all values are considered.

    Returns
    -------
    xr.DataArray
        The maximum value for each period.
    """
    da = select_time(da, **indexer)
    if isinstance(op, str):
        op = _xclim_ops.get(op, op)
    if isinstance(op, str):
        out = getattr(da.resample(time=freq), op.replace("integral", "sum"))(dim="time", keep_attrs=True)
    else:
        with xr.set_options(keep_attrs=True):
            out = resample_map(da, "time", freq, op)
        op = op.__name__
    if out_units is not None:
        return out.assign_attrs(units=out_units)

    if op in ["std", "var"]:
        out.attrs.update(pint2cfattrs(units2pint(out.attrs["units"]), is_difference=True))

    return to_agg_units(out, da, op)


def select_rolling_resample_op(
    da: xr.DataArray,
    op: str,
    window: int,
    window_center: bool = True,
    window_op: str = "mean",
    freq: str = "YS",
    out_units=None,
    **indexer,
) -> xr.DataArray:
    r"""
    Apply operation over each period that is part of the index selection, using a rolling window before the operation.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'integral', 'argmax', 'argmin'} or func
        Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    window : int
        Size of the rolling window (centered).
    window_center : bool
        If True, the window is centered on the date. If False, the window is right-aligned.
    window_op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'integral'}
        Operation to apply to the rolling window. Default: 'mean'.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Applied after the rolling window.
    out_units : str, optional
        Output units to assign.
        Only necessary if `op` is function not supported by :py:func:`xclim.core.units.to_agg_units`.
    **indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
        month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
        considered.

    Returns
    -------
    xr.DataArray
        The array for which the operation has been applied over each period.
    """
    rolled = getattr(
        da.rolling(time=window, center=window_center),
        window_op.replace("integral", "sum"),
    )()
    rolled = to_agg_units(rolled, da, window_op)
    return select_resample_op(rolled, op=op, freq=freq, out_units=out_units, **indexer)


def doymax(da: xr.DataArray) -> xr.DataArray:
    """
    Return the day of year of the maximum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.

    Returns
    -------
    xr.DataArray
        The day of year of the maximum value.
    """
    i = da.argmax(dim="time")
    out = da.time.dt.dayofyear.isel(time=i, drop=True)
    return to_agg_units(out, da, "doymax")


def doymin(da: xr.DataArray) -> xr.DataArray:
    """
    Return the day of year of the minimum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.

    Returns
    -------
    xr.DataArray
        The day of year of the minimum value.
    """
    i = da.argmin(dim="time")
    out = da.time.dt.dayofyear.isel(time=i, drop=True)
    return to_agg_units(out, da, "doymin")


_xclim_ops = {"doymin": doymin, "doymax": doymax}


def default_freq(**indexer) -> str:
    r"""
    Return the default frequency.

    Parameters
    ----------
    **indexer : {dim: indexer, }
        The indexer to use to compute the frequency.

    Returns
    -------
    str
        The default frequency.
    """
    freq = "YS-JAN"
    if indexer:
        group, value = indexer.popitem()
        if group == "season":
            month = 12  # The "season" scheme is based on YS-DEC
        elif group == "month":
            month = np.take(value, 0)
        elif group == "doy_bounds":
            month = cftime.num2date(value[0] - 1, "days since 2004-01-01").month
        elif group == "date_bounds":
            month = int(value[0][:2])
        else:
            raise ValueError(f"Unknown group `{group}`.")
        freq = "YS-" + _MONTH_ABBREVIATIONS[month]
    return freq


def get_op(op: str, constrain: Sequence[str] | None = None) -> Callable:
    """
    Get python's comparing function according to its name of representation and validate allowed usage.

    Accepted op string are keys and values of xclim.indices.generic.binary_ops.

    Parameters
    ----------
    op : str
        Operator.
    constrain : sequence of str, optional
        A tuple of allowed operators.

    Returns
    -------
    Callable
        The operator function.
    """
    if op == "gteq":
        warnings.warn(f"`{op}` is being renamed `ge` for compatibility.")
        op = "ge"
    if op == "lteq":
        warnings.warn(f"`{op}` is being renamed `le` for compatibility.")
        op = "le"

    if op in binary_ops:
        binary_op = binary_ops[op]
    elif op in binary_ops.values():
        binary_op = op
    else:
        raise ValueError(f"Operation `{op}` not recognized.")

    constraints = []
    if isinstance(constrain, list | tuple | set):
        constraints.extend([binary_ops[c] for c in constrain])
        constraints.extend(constrain)
    elif isinstance(constrain, str):
        constraints.extend([binary_ops[constrain], constrain])

    if constrain:
        if op not in constraints:
            raise ValueError(f"Operation `{op}` not permitted for indice.")

    return getattr(operator, f"__{binary_op}__")


def compare(
    left: xr.DataArray,
    op: str,
    right: float | int | np.ndarray | xr.DataArray,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Compare a DataArray to a threshold using given operator.

    Parameters
    ----------
    left : xr.DataArray
        A DatArray being evaluated against `right`.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    right : float, int, np.ndarray, or xr.DataArray
        A value or array-like being evaluated against left`.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        Boolean mask of the comparison.
    """
    return get_op(op, constrain)(left, right)


def threshold_count(
    da: xr.DataArray,
    op: str,
    threshold: float | int | xr.DataArray,
    freq: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Count number of days where value is above or below threshold.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
        Logical operator. e.g. arr > thresh.
    threshold : Union[float, int]
        Threshold value.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        The number of days meeting the constraints for each period.
    """
    if constrain is None:
        constrain = (">", "<", ">=", "<=")

    c = compare(da, op, threshold, constrain) * 1
    return c.resample(time=freq).sum(dim="time")


def domain_count(
    da: xr.DataArray,
    low: float | int | xr.DataArray,
    high: float | int | xr.DataArray,
    freq: str,
) -> xr.DataArray:
    """
    Count number of days where value is within low and high thresholds.

    A value is counted if it is larger than `low`, and smaller or equal to `high`, i.e. in `]low, high]`.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    low : scalar or DataArray
        Minimum threshold value.
    high : scalar or DataArray
        Maximum threshold value.
    freq : str
        Resampling frequency defining the periods defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The number of days where value is within [low, high] for each period.
    """
    c = compare(da, ">", low) * compare(da, "<=", high) * 1
    return c.resample(time=freq).sum(dim="time")


def get_daily_events(
    da: xr.DataArray,
    threshold: float | int | xr.DataArray,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Return a 0/1 mask when a condition is True or False.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    threshold : float
        Threshold value.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        The mask array of daily events.

    Notes
    -----
    The function returns:

    - ``1`` where operator(da, da_value) is ``True``
    - ``0`` where operator(da, da_value) is ``False``
    - ``nan`` where da is ``nan``
    """
    events = compare(da, op, threshold, constrain) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


def spell_mask(
    data: xr.DataArray | Sequence[xr.DataArray],
    window: int,
    win_reducer: str,
    op: str,
    thresh: float | Sequence[float],
    min_gap: int = 1,
    weights: Sequence[float] = None,
    var_reducer: str = "all",
) -> xr.DataArray:
    """
    Compute the boolean mask of data points that are part of a spell as defined by a rolling statistic.

    A day is part of a spell (True in the mask) if it is contained in any period that fulfills the condition.

    Parameters
    ----------
    data : DataArray or sequence of DataArray
        The input data. Can be a list, in which case the condition is checked on all variables.
        See var_reducer for the latter case.
    window : int
        The length of the rolling window in which to compute statistics.
    win_reducer : {'min', 'max', 'sum', 'mean'}
        The statistics to compute on the rolling window.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        The comparison operator to use when finding spells.
    thresh : float or sequence of floats
        The threshold to compare the rolling statistics against, as ``{window_stats} {op} {threshold}``.
        If data is a list, this must be a list of the same length with a threshold for each variable.
        This function does not handle units and can't accept Quantified objects.
    min_gap : int
        The shortest possible gap between two spells.
        Spells closer than this are merged by assigning the gap steps to the merged spell.
    weights : sequence of floats
        A list of weights of the same length as the window.
        Only supported if `win_reducer` is `"mean"`.
    var_reducer : {'all', 'any'}
        If the data is a list, the condition must either be fulfilled on *all*
        or *any* variables for the period to be considered a spell.

    Returns
    -------
    xr.DataArray
        Same shape as ``data``, but boolean.
        If ``data`` was a list, this is a DataArray of the same shape as the alignment of all variables.
    """
    # Checks
    if not isinstance(data, xr.DataArray):
        # thus a sequence
        if np.isscalar(thresh) or len(data) != len(thresh):
            raise ValueError("When ``data`` is given as a list, ``threshold`` must be a sequence of the same length.")
        data = xr.concat(data, "variable")
        if isinstance(thresh[0], xr.DataArray):
            thresh = xr.concat(thresh, "variable")
        else:
            thresh = xr.DataArray(thresh, dims=("variable",))
    if weights is not None:
        if win_reducer != "mean":
            raise ValueError(f"Argument 'weights' is only supported if 'win_reducer' is 'mean'. Got :  {win_reducer}")
        elif len(weights) != window:
            raise ValueError(f"Weights have a different length ({len(weights)}) than the window ({window}).")
        weights = xr.DataArray(weights, dims=("window",))

    if window == 1:  # Fast path
        is_in_spell = compare(data, op, thresh)
        if not np.isscalar(thresh):
            is_in_spell = getattr(is_in_spell, var_reducer)("variable")
    elif (win_reducer == "min" and op in [">", ">=", "ge", "gt"]) or (
        win_reducer == "max" and op in ["`<", "<=", "le", "lt"]
    ):
        # Fast path for specific cases, this yields a smaller dask graph (rolling twice is expensive!)
        # For these two cases, a day can't be part of a spell if it doesn't respect the condition itself
        mask = compare(data, op, thresh)
        if not np.isscalar(thresh):
            mask = getattr(mask, var_reducer)("variable")
        # We need to filter out the spells shorter than "window"
        # find sequences of consecutive respected constraints
        cs_s = rl._cumsum_reset(mask)
        # end of these sequences
        cs_s = cs_s.where(mask.shift({"time": -1}, fill_value=0) == 0)
        # propagate these end of sequences
        # the `.where(mask>0, 0)` acts a stopper
        is_in_spell = cs_s.where(cs_s >= window).where(mask > 0, 0).bfill("time") > 0
    else:
        data_pad = data.pad(time=(0, window))
        # The spell-wise value to test
        # For example "window_reducer='sum'",
        # we want the sum over the minimum spell length (window) to be above the thresh
        if weights is not None:
            spell_value = data_pad.rolling(time=window).construct("window").dot(weights)
        else:
            spell_value = getattr(data_pad.rolling(time=window), win_reducer)()
        # True at the end of a spell respecting the condition
        mask = compare(spell_value, op, thresh)
        if not np.isscalar(thresh):
            mask = getattr(mask, var_reducer)("variable")
        # True for all days part of a spell that respected the condition (shift because of the two rollings)
        is_in_spell = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1), fill_value=False)
        # Cut back to the original size
        is_in_spell = is_in_spell.isel(time=slice(0, data.time.size))

    if min_gap > 1:
        is_in_spell = rl.runs_with_holes(is_in_spell, 1, ~is_in_spell, min_gap).astype(bool)

    return is_in_spell


def _spell_length_statistics(
    data: xr.DataArray | Sequence[xr.DataArray],
    thresh: float | xr.DataArray | Sequence[xr.DataArray] | Sequence[float],
    window: int,
    win_reducer: str,
    op: str,
    spell_reducer: str | Sequence[str],
    freq: str,
    min_gap: int = 1,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    if isinstance(spell_reducer, str):
        spell_reducer = [spell_reducer]
    is_in_spell = spell_mask(data, window, win_reducer, op, thresh, min_gap=min_gap).astype(np.float32)
    is_in_spell = select_time(is_in_spell, **indexer)

    outs = []
    for sr in spell_reducer:
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
            # All other cases are statistics of the number of timesteps
            outs.append(
                to_agg_units(
                    out,
                    data if isinstance(data, xr.DataArray) else data[0],
                    "count",
                )
            )
    if len(outs) == 1:
        return outs[0]
    return tuple(outs)


@declare_relative_units(threshold="<data>")
def spell_length_statistics(
    data: xr.DataArray,
    threshold: Quantified,
    window: int,
    win_reducer: str,
    op: str,
    spell_reducer: str,
    freq: str,
    min_gap: int = 1,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    r"""
    Generate statistic on spells lengths.

    A spell is when a statistic (`win_reducer`) over a minimum number (`window`) of consecutive timesteps
    respects a condition (`op` `thresh`). This returns a statistic over the spells count or lengths.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold to test against.
    window : int
        Minimum length of a spell.
    win_reducer : {'min', 'max', 'sum', 'mean'}
        Reduction along the spell length to compute the spell value.
        Note that this does not matter when `window` is 1.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. Ex: spell_value > thresh.
    spell_reducer : {'max', 'sum', 'count'} or sequence thereof
        Statistic on the spell lengths. If a list, multiple statistics are computed.
    freq : str
        Resampling frequency.
    min_gap : int
        The shortest possible gap between two spells. Spells closer than this are merged by assigning
        the gap steps to the merged spell.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.
    **indexer : {dim: indexer, }, optional
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the days part of a spell, but before taking the spell statistics.

    Returns
    -------
    xr.DataArray or sequence of xr.DataArray
        The length of the longest of such spells.

    See Also
    --------
    spell_mask : The lower level functions that finds spells.
    bivariate_spell_length_statistics : The bivariate version of this function.

    Examples
    --------
    >>> spell_length_statistics(
    ...     tas,
    ...     threshold="35 °C",
    ...     window=7,
    ...     op=">",
    ...     win_reducer="min",
    ...     spell_reducer="sum",
    ...     freq="YS",
    ... )

    Here, a day is part of a spell if it is in any seven (7) day period where the minimum temperature is over 35°C.
    We then return the annual sum of the spell lengths, so the total number of days in such spells.
    >>> from xclim.core.units import rate2amount
    >>> pram = rate2amount(pr, out_units="mm")
    >>> spell_length_statistics(
    ...     pram,
    ...     threshold="20 mm",
    ...     window=5,
    ...     op=">=",
    ...     win_reducer="sum",
    ...     spell_reducer="max",
    ...     freq="YS",
    ... )

    Here, a day is part of a spell if it is in any five (5) day period where the total accumulated precipitation
    reaches or exceeds 20 mm. We then return the length of the longest of such spells.
    """
    thresh = convert_units_to(threshold, data, context="infer")
    return _spell_length_statistics(
        data,
        thresh,
        window,
        win_reducer,
        op,
        spell_reducer,
        freq,
        min_gap=min_gap,
        resample_before_rl=resample_before_rl,
        **indexer,
    )


@declare_relative_units(threshold1="<data1>", threshold2="<data2>")
def bivariate_spell_length_statistics(
    data1: xr.DataArray,
    threshold1: Quantified,
    data2: xr.DataArray,
    threshold2: Quantified,
    window: int,
    win_reducer: str,
    op: str,
    spell_reducer: str,
    freq: str,
    min_gap: int = 1,
    resample_before_rl: bool = True,
    **indexer,
) -> xr.DataArray | Sequence[xr.DataArray]:
    r"""
    Generate statistic on spells lengths based on two variables.

    A spell is when a statistic (`win_reducer`) over a minimum number (`window`) of consecutive timesteps
    respects a condition (`op` `thresh`). This returns a statistic over the spells count or lengths.
    In this bivariate version, conditions on both variables must be fulfilled.

    Parameters
    ----------
    data1 : xr.DataArray
        First input data.
    threshold1 : Quantified
        Threshold to test against data1.
    data2 : xr.DataArray
        Second input data.
    threshold2 : Quantified
        Threshold to test against data2.
    window : int
        Minimum length of a spell.
    win_reducer : {'min', 'max', 'sum', 'mean'}
        Reduction along the spell length to compute the spell value.
        Note that this does not matter when `window` is 1.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. Ex: spell_value > thresh.
    spell_reducer : {'max', 'sum', 'count'} or sequence thereof
        Statistic on the spell lengths. If a list, multiple statistics are computed.
    freq : str
        Resampling frequency.
    min_gap : int
        The shortest possible gap between two spells. Spells closer than this are merged by assigning
        the gap steps to the merged spell.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.
    **indexer : {dim: indexer, }, optional
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the days part of a spell, but before taking the spell statistics.

    Returns
    -------
    xr.DataArray or sequence of xr.DataArray
        The length of the longest of such spells.

    See Also
    --------
    spell_length_statistics : The univariate version.
    spell_mask : The lower level functions that finds spells.
    """
    thresh1 = convert_units_to(threshold1, data1, context="infer")
    thresh2 = convert_units_to(threshold2, data2, context="infer")
    return _spell_length_statistics(
        [data1, data2],
        [thresh1, thresh2],
        window,
        win_reducer,
        op,
        spell_reducer,
        freq,
        min_gap,
        resample_before_rl,
        **indexer,
    )


@declare_relative_units(thresh="<data>")
def season(
    data: xr.DataArray,
    thresh: Quantified,
    window: int,
    op: str,
    stat: str,
    freq: str,
    mid_date: DayOfYearStr | None = None,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    r"""
    Season.

    A season starts when a variable respects some condition for a consecutive run of `N` days. It stops
    when the condition is inverted for `N` days. Runs where the condition is not met for fewer than `N` days
    are thus allowed. Additionally, a middle date can serve as a maximal start date and minimum end date.

    Parameters
    ----------
    data : xr.DataArray
        Variable.
    thresh : Quantified
        Threshold on which to base evaluation.
    window : int
        Minimum number of days that the condition must be met / not met for the start / end of the season.
    op : str
        Comparison operation.
    stat : {'start', 'end', 'length'}
        Which season facet to return.
    freq : str
        Resampling frequency.
    mid_date : DayOfYearStr, optional
        An optional middle date. The start must happen before and the end after for the season to be valid.
    constrain : Sequence of strings, optional
        A list of acceptable comparison operators. Optional, but indicators wrapping this function should inject it.

    Returns
    -------
    xr.DataArray, [dimensionless] or [time]
        Depends on 'stat'. If 'start' or 'end', this is the day of year of the season's start or end.
        If 'length', this is the length of the season.

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
    cond = compare(data, op, thresh, constrain=constrain)
    func = {"start": rl.season_start, "end": rl.season_end, "length": rl.season_length}
    map_kwargs = {"window": window, "mid_date": mid_date}

    if stat in ["start", "end"]:
        map_kwargs["coord"] = "dayofyear"
    out = resample_map(cond, "time", freq, func[stat], map_kwargs=map_kwargs)
    if stat == "length":
        return to_agg_units(out, data, "count")
    # else, a date
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(data))
    return out


# CF-INDEX-META Indices


@declare_relative_units(threshold="<low_data>")
def count_level_crossings(
    low_data: xr.DataArray,
    high_data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    *,
    op_low: str = "<",
    op_high: str = ">=",
) -> xr.DataArray:
    """
    Calculate the number of times low_data is below threshold while high_data is above threshold.

    First, the threshold is transformed to the same standard_name and units as the input data,
    then the thresholding is performed, and finally, the number of occurrences is counted.

    Parameters
    ----------
    low_data : xr.DataArray
        Variable that must be under the threshold.
    high_data : xr.DataArray
        Variable that must be above the threshold.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op_low : {"<", "<=", "lt", "le"}
        Comparison operator for low_data. Default: "<".
    op_high : {">", ">=", "gt", "ge"}
        Comparison operator for high_data. Default: ">=".

    Returns
    -------
    xr.DataArray
        The DataArray of level crossing events.
    """
    # Convert units to low_data
    high_data = convert_units_to(high_data, low_data)
    threshold = convert_units_to(threshold, low_data)

    lower = compare(low_data, op_low, threshold, constrain=("<", "<="))
    higher = compare(high_data, op_high, threshold, constrain=(">", ">="))

    out = (lower & higher).resample(time=freq).sum()
    return to_agg_units(out, low_data, "count", dim="time")


@declare_relative_units(threshold="<data>")
def count_occurrences(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Calculate the number of times some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data;
    Then the thresholding is performed as condition(data, threshold),
    i.e. if condition is `<`, then this counts the number of times `data < threshold`;
    Finally, count the number of occurrences when condition is met.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        The DataArray of counted occurrences.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = cond.resample(time=freq).sum()
    return to_agg_units(out, data, "count", dim="time")


@declare_relative_units(threshold_var1="<data_var1>", threshold_var2="<data_var2>")
def bivariate_count_occurrences(
    *,
    data_var1: xr.DataArray,
    data_var2: xr.DataArray,
    threshold_var1: Quantified,
    threshold_var2: Quantified,
    freq: str,
    op_var1: str,
    op_var2: str,
    var_reducer: str,
    constrain_var1: Sequence[str] | None = None,
    constrain_var2: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Calculate the number of times some conditions are met for two variables.

    First, the thresholds are transformed to the same standard_name and units as their corresponding input data;
    Then the thresholding is performed as condition(data, threshold) for each variable,
    i.e. if condition is `<`, then this counts the number of times `data < threshold`;
    Then the conditions are combined according to `var_reducer`;
    Finally, the number of occurrences where conditions are met for "all" or "any" events are counted.

    Parameters
    ----------
    data_var1 : xr.DataArray
        An array.
    data_var2 : xr.DataArray
        An array.
    threshold_var1 : Quantified
        Threshold for data variable 1.
    threshold_var2 : Quantified
        Threshold for data variable 2.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op_var1 : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator for data variable 1. e.g. arr > thresh.
    op_var2 : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator for data variable 2. e.g. arr > thresh.
    var_reducer : {"all", "any"}
        The condition must either be fulfilled on *all* or *any* variables
        for the period to be considered an occurrence.
    constrain_var1 : sequence of str, optional
        Optionally allowed comparison operators for variable 1.
    constrain_var2 : sequence of str, optional
        Optionally allowed comparison operators for variable 2.

    Returns
    -------
    xr.DataArray
        The DataArray of counted occurrences.

    Notes
    -----
    Sampling and variable units are derived from `data_var1`.
    """
    threshold_var1 = convert_units_to(threshold_var1, data_var1)
    threshold_var2 = convert_units_to(threshold_var2, data_var2)

    cond_var1 = compare(data_var1, op_var1, threshold_var1, constrain_var1)
    cond_var2 = compare(data_var2, op_var2, threshold_var2, constrain_var2)

    if var_reducer == "all":
        cond = cond_var1 & cond_var2
    elif var_reducer == "any":
        cond = cond_var1 | cond_var2
    else:
        raise ValueError(f"Unsupported value for var_reducer: {var_reducer}")

    out = cond.resample(time=freq).sum()

    return to_agg_units(out, data_var1, "count", dim="time")


def diurnal_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, reducer: str, freq: str) -> xr.DataArray:
    """
    Calculate the diurnal temperature range and reduce according to a statistic.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The DataArray of the diurnal temperature range.
    """
    high_data = convert_units_to(high_data, low_data)

    dtr = high_data - low_data
    out = getattr(dtr.resample(time=freq), reducer)()

    u = str2pint(low_data.units)
    out.attrs.update(pint2cfattrs(u, is_difference=True))
    return out


@declare_relative_units(threshold="<data>")
def first_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Calculate the first time some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, locate the first occurrence when condition is met.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        The DataArray of times of first occurrences.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = resample_map(
        cond,
        "time",
        freq,
        rl.first_run,
        map_kwargs=dict(window=1, dim="time", coord="dayofyear"),
    )
    out.attrs["units"] = ""
    return out


@declare_relative_units(threshold="<data>")
def last_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Calculate the last time some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, locate the last occurrence when condition is met.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        The DataArray of times of last occurrences.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = resample_map(
        cond,
        "time",
        freq,
        rl.last_run,
        map_kwargs=dict(window=1, dim="time", coord="dayofyear"),
    )
    out.attrs["units"] = ""
    return out


@declare_relative_units(threshold="<data>")
def spell_length(data: xr.DataArray, threshold: Quantified, reducer: str, freq: str, op: str) -> xr.DataArray:
    """
    Calculate statistics on lengths of spells.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Then the spells are determined, and finally the statistics according to the specified reducer are calculated.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.

    Returns
    -------
    xr.DataArray
        The DataArray of spell lengths.
    """
    threshold = convert_units_to(
        threshold,
        data,
        context=infer_context(standard_name=data.attrs.get("standard_name")),
    )

    cond = compare(data, op, threshold)

    out = resample_map(
        cond,
        "time",
        freq,
        rl.rle_statistics,
        map_kwargs=dict(reducer=reducer, window=1, dim="time"),
    )
    return to_agg_units(out, data, "count")


def statistics(data: xr.DataArray, reducer: str, freq: str) -> xr.DataArray:
    """
    Calculate a simple statistic of the data.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The DataArray for the given statistic.
    """
    out = getattr(data.resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


@declare_relative_units(threshold="<data>")
def thresholded_statistics(
    data: xr.DataArray,
    op: str,
    threshold: Quantified,
    reducer: str,
    freq: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Calculate a simple statistic of the data for which some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, the statistic is calculated for those data values that fulfill the condition.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    threshold : Quantified
        Threshold.
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    constrain : sequence of str, optional
        Optionally allowed conditions. Default: None.

    Returns
    -------
    xr.DataArray
        The DataArray for the given thresholded statistic.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = getattr(data.where(cond).resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


@declare_relative_units(threshold="<data>")
def temperature_sum(data: xr.DataArray, op: str, threshold: Quantified, freq: str) -> xr.DataArray:
    """
    Calculate the temperature sum above/below a threshold.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, the sum is calculated for those data values that fulfill the condition after subtraction of the threshold
    value. If the sum is for values below the threshold the result is multiplied by -1.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr > thresh.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The DataArray for the sum of temperatures above or below a threshold.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain=("<", "<=", ">", ">="))
    direction = -1 if op in ["<", "<=", "lt", "le"] else 1

    out = (data - threshold).where(cond).resample(time=freq).sum()
    out = direction * out
    out.attrs["units_metadata"] = "temperature: difference"
    return to_agg_units(out, data, "integral")


def interday_diurnal_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the average absolute day-to-day difference in diurnal temperature range.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The DataArray for the average absolute day-to-day difference in diurnal temperature range.
    """
    high_data = convert_units_to(high_data, low_data)

    vdtr = abs((high_data - low_data).diff(dim="time"))
    out = vdtr.resample(time=freq).mean(dim="time")

    out.attrs["units"] = low_data.attrs["units"]
    out.attrs["units_metadata"] = "temperature: difference"
    return out


def extreme_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the extreme daily temperature range.

    The maximum of daily maximum temperature minus the minimum of daily minimum temperature.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        The DataArray for the extreme temperature range.
    """
    high_data = convert_units_to(high_data, low_data)

    out = high_data.resample(time=freq).max() - low_data.resample(time=freq).min()

    out.attrs["units"] = low_data.attrs["units"]
    out.attrs["units_metadata"] = "temperature: difference"
    return out


def aggregate_between_dates(
    data: xr.DataArray,
    start: xr.DataArray | DayOfYearStr,
    end: xr.DataArray | DayOfYearStr,
    op: str = "sum",
    freq: str | None = None,
) -> xr.DataArray:
    """
    Aggregate the data over a period between start and end dates and apply the operator on the aggregated data.

    Parameters
    ----------
    data : xr.DataArray
        Data to aggregate between start and end dates.
    start : xr.DataArray or DayOfYearStr
        Start dates (as day-of-year) for the aggregation periods.
    end : xr.DataArray or DayOfYearStr
        End (as day-of-year) dates for the aggregation periods.
    op : {'min', 'max', 'sum', 'mean', 'std'}
        Operator.
    freq : str, optional
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`. Default: `None`.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Aggregated data between the start and end dates. If the end date is before the start date, returns np.nan.
        If there is no start and/or end date, returns np.nan.
    """

    def _get_days(_bound, _group, _base_time):
        """Get bound in number of days since base_time. Bound can be a days_since array or a DayOfYearStr."""
        if isinstance(_bound, str):
            b_i = rl.index_of_date(_group.time, _bound, max_idxs=1)  # noqa
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
                " Please consider providing `freq` manually."
            )
        freq = good_freq.pop()

    cal = get_calendar(data, dim="time")

    if not isinstance(start, str):
        start = start.convert_calendar(cal)
        start.attrs["calendar"] = cal
        start = doy_to_days_since(start)
    if not isinstance(end, str):
        end = end.convert_calendar(cal)
        end.attrs["calendar"] = cal
        end = doy_to_days_since(end)

    out = []
    for base_time, indexes in data.resample(time=freq).groups.items():
        # get group slice
        group = data.isel(time=indexes)

        start_d = _get_days(start, group, base_time)
        end_d = _get_days(end, group, base_time)

        # convert bounds for this group
        if start_d is not None and end_d is not None:
            days = (group.time - base_time).dt.days
            days[days < 0] = np.nan

            masked = group.where((days >= start_d) & (days <= end_d - 1))
            res = getattr(masked, op)(dim="time", skipna=True)
            res = xr.where(((start_d > end_d) | (start_d.isnull()) | (end_d.isnull())), np.nan, res)
            # Re-add the time dimension with the period's base time.
            res = res.expand_dims(time=[base_time])
            out.append(res)
        else:
            # Get an array with the good shape, put nans and add the new time.
            res = (group.isel(time=0) * np.nan).expand_dims(time=[base_time])
            out.append(res)
            continue

    return xr.concat(out, dim="time")


@declare_relative_units(threshold="<data>")
def cumulative_difference(data: xr.DataArray, threshold: Quantified, op: str, freq: str | None = None) -> xr.DataArray:
    """
    Calculate the cumulative difference below/above a given value threshold.

    Parameters
    ----------
    data : xr.DataArray
        Data for which to determine the cumulative difference.
    threshold : Quantified
        The value threshold.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr > thresh.
    freq : str, optional
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        If `None`, no resampling is performed. Default: `None`.

    Returns
    -------
    xr.DataArray
        The DataArray for the cumulative difference between values and a given threshold.
    """
    threshold = convert_units_to(threshold, data)

    if op in ["<", "<=", "lt", "le"]:
        diff = (threshold - data).clip(0)
    elif op in [">", ">=", "gt", "ge"]:
        diff = (data - threshold).clip(0)
    else:
        raise NotImplementedError(f"Condition not supported: '{op}'.")

    if freq is not None:
        diff = diff.resample(time=freq).sum(dim="time")

    diff.attrs.update(pint2cfattrs(units2pint(data.attrs["units"]), is_difference=True))
    # return diff
    return to_agg_units(diff, data, op="integral")


@declare_relative_units(threshold="<data>")
def first_day_threshold_reached(
    data: xr.DataArray,
    *,
    threshold: Quantified,
    op: str,
    after_date: DayOfYearStr,
    window: int = 1,
    freq: str = "YS",
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    r"""
    First day of values exceeding threshold.

    Returns first day of period where values reach or exceed a threshold over a given number of days,
    limited to a starting calendar date.

    Parameters
    ----------
    data : xr.DataArray
        Dataset being evaluated.
    threshold : str
        Threshold on which to base evaluation.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    after_date : str
        Date of the year after which to look for the first event. Should have the format '%m-%d'.
    window : int
        Minimum number of days with values above threshold needed for evaluation. Default: 1.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Default: "YS".
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Day of the year when value reaches or exceeds a threshold over a given number of days for the first time.
        If there is no such day, returns np.nan.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain=constrain)

    out: xr.DataArray = resample_map(
        cond,
        "time",
        freq,
        rl.first_run_after_date,
        map_kwargs=dict(window=window, date=after_date, dim="time", coord="dayofyear"),
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(data))
    return out


def _get_zone_bins(
    zone_min: Quantity,
    zone_max: Quantity,
    zone_step: Quantity,
):
    """
    Bin boundary values as defined by zone parameters.

    Parameters
    ----------
    zone_min : Quantity
        Left boundary of the first zone.
    zone_max : Quantity
        Right boundary of the last zone.
    zone_step: Quantity
        Size of zones.

    Returns
    -------
    xr.DataArray, [units of `zone_step`]
        Array of values corresponding to each zone: [zone_min, zone_min+step, ..., zone_max].
    """
    units = pint2cfunits(str2pint(zone_step))
    mn, mx, step = (convert_units_to(str2pint(z), units) for z in [zone_min, zone_max, zone_step])
    bins = np.arange(mn, mx + step, step)
    if (mx - mn) % step != 0:
        warnings.warn("`zone_max` - `zone_min` is not an integer multiple of `zone_step`. Last zone will be smaller.")
        bins[-1] = mx
    return xr.DataArray(bins, attrs={"units": units})


def get_zones(
    da: xr.DataArray,
    zone_min: Quantity | None = None,
    zone_max: Quantity | None = None,
    zone_step: Quantity | None = None,
    bins: xr.DataArray | list[Quantity] | None = None,
    exclude_boundary_zones: bool = True,
    close_last_zone_right_boundary: bool = True,
) -> xr.DataArray:
    r"""
    Divide data into zones and attribute a zone coordinate to each input value.

    Divide values into zones corresponding to bins of width zone_step beginning at zone_min and ending at zone_max.
    Bins are inclusive on the left values and exclusive on the right values.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    zone_min : Quantity, optional
        Left boundary of the first zone.
    zone_max : Quantity, optional
        Right boundary of the last zone.
    zone_step : Quantity, optional
        Size of zones.
    bins : xr.DataArray or list of Quantity, optional
        Zones to be used, either as a DataArray with appropriate units or a list of Quantity.
    exclude_boundary_zones : bool
        Determines whether a zone value is attributed for values in ]`-np.inf`,
        `zone_min`[ and [`zone_max`, `np.inf`\ [.
    close_last_zone_right_boundary : bool
        Determines if the right boundary of the last zone is closed.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Zone index for each value in `da`. Zones are returned as an integer range, starting from `0`.
    """
    # Check compatibility of arguments
    zone_params = np.array([zone_min, zone_max, zone_step])
    if bins is None:
        if (zone_params == [None] * len(zone_params)).any():
            raise ValueError(
                "`bins` is `None` as well as some or all of [`zone_min`, `zone_max`, `zone_step`]. "
                "Expected defined parameters in one of these cases."
            )
    elif set(zone_params) != {None}:
        warnings.warn("Expected either `bins` or [`zone_min`, `zone_max`, `zone_step`], got both. `bins` will be used.")

    # Get zone bins (if necessary)
    bins = bins if bins is not None else _get_zone_bins(zone_min, zone_max, zone_step)
    if isinstance(bins, list):
        bins = sorted([convert_units_to(b, da) for b in bins])
    else:
        bins = convert_units_to(bins, da)

    def _get_zone(_da):
        return np.digitize(_da, bins) - 1

    zones = xr.apply_ufunc(_get_zone, da, dask="parallelized")

    if close_last_zone_right_boundary:
        zones = zones.where(da != bins[-1], _get_zone(bins[-2]))
    if exclude_boundary_zones:
        zones = zones.where((zones != _get_zone(bins[0] - 1)) & (zones != _get_zone(bins[-1])))

    return zones


def detrend(ds: xr.DataArray | xr.Dataset, dim="time", deg=1) -> xr.DataArray | xr.Dataset:
    """
    Detrend data along a given dimension computing a polynomial trend of a given order.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
      The data to detrend. If a Dataset, detrending is done on all data variables.
    dim : str
      Dimension along which to compute the trend.
    deg : int
      Degree of the polynomial to fit.

    Returns
    -------
    xr.Dataset or xr.DataArray
      Same as `ds`, but with its trend removed (subtracted).
    """
    if isinstance(ds, xr.Dataset):
        return ds.map(detrend, keep_attrs=False, dim=dim, deg=deg)
    # is a DataArray
    # detrend along a single dimension
    coeff = ds.polyfit(dim=dim, deg=deg)
    trend = xr.polyval(ds[dim], coeff.polyfit_coefficients)
    with xr.set_options(keep_attrs=True):
        return ds - trend


@declare_relative_units(thresh="<data>")
def thresholded_events(
    data: xr.DataArray,
    thresh: Quantified,
    op: str,
    window: int,
    thresh_stop: Quantified | None = None,
    op_stop: str | None = None,
    window_stop: int = 1,
    freq: str | None = None,
) -> xr.Dataset:
    r"""
    Thresholded events.

    Finds all events along the time dimension.
    An event starts if the start condition is fulfilled for a given number of consecutive time steps.
    It ends when the end condition is fulfilled for a given number of consecutive time steps.

    Conditions are simple comparison of the data with a threshold: ``cond = data op thresh``.
    The end conditions defaults to the negation of the start condition.

    The resulting ``event`` dimension always has its maximal possible size : ``data.size / (window + window_stop)``.

    Parameters
    ----------
    data : xr.DataArray
        Variable.
    thresh : Quantified
        Threshold defining the event.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator defining the event, e.g. arr > thresh.
    window : int
        Number of time steps where the event condition must be true to start an event.
    thresh_stop : Quantified, optional
        Threshold defining the end of an event. Defaults to `thresh`.
    op_stop : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}, optional
        Logical operator for the end of an event. Defaults to the opposite of `op`.
    window_stop : int, optional
        Number of time steps where the end condition must be true to end an event. Defaults to `1`.
    freq : str, optional
        A frequency to divide the data into periods. If absent, the output has not time dimension.
        If given, the events are searched within in each resample period independently.

    Returns
    -------
    xr.Dataset, same shape as the data except the time dimension is replaced by an "event" dimension.
        The dataset contains the following variables:
            event_length: The number of time steps in each event
            event_effective_length: The number of time steps of even event where the start condition is true.
            event_sum: The sum within each event, only considering the steps where start condition is true.
            event_start: The datetime of the start of the run.
    """
    thresh = convert_units_to(thresh, data)

    # Start and end conditions
    da_start = compare(data, op, thresh)
    if thresh_stop is None and op_stop is None:
        da_stop = ~da_start
    else:
        thresh_stop = convert_units_to(thresh_stop or thresh, data)
        if op_stop is not None:
            da_stop = compare(data, op_stop, thresh_stop)
        else:
            da_stop = ~compare(data, op, thresh_stop)

    return rl.find_events(da_start, window, da_stop, window_stop, data, freq)
