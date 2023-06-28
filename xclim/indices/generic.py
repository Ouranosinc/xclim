"""
Generic Indices Submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Sequence

import cftime
import numpy as np
import xarray as xr
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS  # noqa

from xclim.core.calendar import (
    convert_calendar,
    doy_to_days_since,
    get_calendar,
    select_time,
)
from xclim.core.units import (
    convert_units_to,
    infer_context,
    pint2cfunits,
    str2pint,
    to_agg_units,
)
from xclim.core.utils import DayOfYearStr, Quantified, Quantity

from . import run_length as rl

__all__ = [
    "aggregate_between_dates",
    "compare",
    "count_level_crossings",
    "count_occurrences",
    "cumulative_difference",
    "default_freq",
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
    "select_resample_op",
    "spell_length",
    "statistics",
    "temperature_sum",
    "threshold_count",
    "thresholded_statistics",
]

binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


def select_resample_op(
    da: xr.DataArray, op: str, freq: str = "YS", **indexer
) -> xr.DataArray:
    """Apply operation over each period that is part of the index selection.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'argmax', 'argmin'} or func
        Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
        month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
        considered.

    Returns
    -------
    xr.DataArray
        The maximum value for each period.
    """
    da = select_time(da, **indexer)
    r = da.resample(time=freq)
    if isinstance(op, str):
        return getattr(r, op)(dim="time", keep_attrs=True)

    return r.map(op)


def doymax(da: xr.DataArray) -> xr.DataArray:
    """Return the day of year of the maximum value."""
    i = da.argmax(dim="time")
    out = da.time.dt.dayofyear.isel(time=i, drop=True)
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(da))
    return out


def doymin(da: xr.DataArray) -> xr.DataArray:
    """Return the day of year of the minimum value."""
    i = da.argmin(dim="time")
    out = da.time.dt.dayofyear.isel(time=i, drop=True)
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(da))
    return out


def default_freq(**indexer) -> str:
    """Return the default frequency."""
    freq = "AS-JAN"
    if indexer:
        group, value = indexer.popitem()
        if group == "season":
            month = 12  # The "season" scheme is based on AS-DEC
        elif group == "month":
            month = np.take(value, 0)
        elif group == "doy_bounds":
            month = cftime.num2date(value[0] - 1, "days since 2004-01-01").month
        elif group == "date_bounds":
            month = int(value[0][:2])
        freq = "AS-" + _MONTH_ABBREVIATIONS[month]
    return freq


def get_op(op: str, constrain: Sequence[str] | None = None) -> Callable:
    """Get python's comparing function according to its name of representation and validate allowed usage.

    Accepted op string are keys and values of xclim.indices.generic.binary_ops.

    Parameters
    ----------
    op : str
        Operator.
    constrain : sequence of str, optional
        A tuple of allowed operators.
    """
    if op == "gteq":
        warnings.warn(f"`{op}` is being renamed `ge` for compatibility.")
        op = "ge"
    if op == "lteq":
        warnings.warn(f"`{op}` is being renamed `le` for compatibility.")
        op = "le"

    if op in binary_ops.keys():
        binary_op = binary_ops[op]
    elif op in binary_ops.values():
        binary_op = op
    else:
        raise ValueError(f"Operation `{op}` not recognized.")

    constraints = list()
    if isinstance(constrain, (list, tuple, set)):
        constraints.extend([binary_ops[c] for c in constrain])
        constraints.extend(constrain)
    elif isinstance(constrain, str):
        constraints.extend([binary_ops[constrain], constrain])

    if constrain:
        if op not in constraints:
            raise ValueError(f"Operation `{op}` not permitted for indice.")

    return xr.core.ops.get_op(binary_op)  # noqa


def compare(
    left: xr.DataArray,
    op: str,
    right: float | int | np.ndarray | xr.DataArray,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """Compare a dataArray to a threshold using given operator.

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
    """Count number of days where value is above or below threshold.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
        Logical operator. e.g. arr > thresh.
    threshold : Union[float, int]
        Threshold value.
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
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
    """Count number of days where value is within low and high thresholds.

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
        Resampling frequency defining the periods defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

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
    """Return a 0/1 mask when a condition is True or False.

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

    Notes
    -----
    The function returns:

    - ``1`` where operator(da, da_value) is ``True``
    - ``0`` where operator(da, da_value) is ``False``
    - ``nan`` where da is ``nan``

    Returns
    -------
    xr.DataArray
    """
    events = compare(da, op, threshold, constrain) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


# CF-INDEX-META Indices


def count_level_crossings(
    low_data: xr.DataArray,
    high_data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    *,
    op_low: str = "<",
    op_high: str = ">=",
) -> xr.DataArray:
    """Calculate the number of times low_data is below threshold while high_data is above threshold.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    op_low : {"<", "<=", "lt", "le"}
        Comparison operator for low_data. Default: "<".
    op_high : {">", ">=", "gt", "ge"}
        Comparison operator for high_data. Default: ">=".

    Returns
    -------
    xr.DataArray
    """
    # Convert units to low_data
    high_data = convert_units_to(high_data, low_data)
    threshold = convert_units_to(threshold, low_data)

    lower = compare(low_data, op_low, threshold, constrain=("<", "<="))
    higher = compare(high_data, op_high, threshold, constrain=(">", ">="))

    out = (lower & higher).resample(time=freq).sum()
    return to_agg_units(out, low_data, "count", dim="time")


def count_occurrences(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """Calculate the number of times some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold),
    i.e. if condition is `<`, then this counts the number of times `data < threshold`.
    Finally, count the number of occurrences when condition is met.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = cond.resample(time=freq).sum()
    return to_agg_units(out, data, "count", dim="time")


def diurnal_temperature_range(
    low_data: xr.DataArray, high_data: xr.DataArray, reducer: str, freq: str
) -> xr.DataArray:
    """Calculate the diurnal temperature range and reduce according to a statistic.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq: str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    dtr = high_data - low_data
    out = getattr(dtr.resample(time=freq), reducer)()

    u = str2pint(low_data.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


def first_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """Calculate the first time some condition is met.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = cond.resample(time=freq).map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


def last_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    freq: str,
    op: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """Calculate the last time some condition is met.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = cond.resample(time=freq).map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


def spell_length(
    data: xr.DataArray, threshold: Quantified, reducer: str, freq: str, op: str
) -> xr.DataArray:
    """Calculate statistics on lengths of spells.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(
        threshold,
        data,
        context=infer_context(standard_name=data.attrs.get("standard_name")),
    )

    cond = compare(data, op, threshold)

    out = cond.resample(time=freq).map(
        rl.rle_statistics,
        reducer=reducer,
        window=1,
        dim="time",
    )
    return to_agg_units(out, data, "count")


def statistics(data: xr.DataArray, reducer: str, freq: str) -> xr.DataArray:
    """Calculate a simple statistic of the data.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    reducer : {'max', 'min', 'mean', 'sum'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
    """
    out = getattr(data.resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


def thresholded_statistics(
    data: xr.DataArray,
    op: str,
    threshold: Quantified,
    reducer: str,
    freq: str,
    constrain: Sequence[str] | None = None,
) -> xr.DataArray:
    """Calculate a simple statistic of the data for which some condition is met.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
    constrain : sequence of str, optional
        Optionally allowed conditions. Default: None.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain)

    out = getattr(data.where(cond).resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


def temperature_sum(
    data: xr.DataArray, op: str, threshold: Quantified, freq: str
) -> xr.DataArray:
    """Calculate the temperature sum above/below a threshold.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain=("<", "<=", ">", ">="))
    direction = -1 if op in ["<", "<=", "lt", "le"] else 1

    out = (data - threshold).where(cond).resample(time=freq).sum()
    out = direction * out
    return to_agg_units(out, data, "delta_prod")


def interday_diurnal_temperature_range(
    low_data: xr.DataArray, high_data: xr.DataArray, freq: str
) -> xr.DataArray:
    """Calculate the average absolute day-to-day difference in diurnal temperature range.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.


    Returns
    -------
    xr.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    vdtr = abs((high_data - low_data).diff(dim="time"))
    out = vdtr.resample(time=freq).mean(dim="time")

    u = str2pint(low_data.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


def extreme_temperature_range(
    low_data: xr.DataArray, high_data: xr.DataArray, freq: str
) -> xr.DataArray:
    """Calculate the extreme temperature range as the maximum of daily maximum temperature minus the minimum of daily minimum temperature.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    freq : str
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    out = high_data.resample(time=freq).max() - low_data.resample(time=freq).min()

    u = str2pint(low_data.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


def aggregate_between_dates(
    data: xr.DataArray,
    start: xr.DataArray | DayOfYearStr,
    end: xr.DataArray | DayOfYearStr,
    op: str = "sum",
    freq: str | None = None,
) -> xr.DataArray:
    """Aggregate the data over a period between start and end dates and apply the operator on the aggregated data.

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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
        Default: `None`.

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
        start = convert_calendar(start, cal)
        start.attrs["calendar"] = cal
        start = doy_to_days_since(start)
    if not isinstance(end, str):
        end = convert_calendar(end, cal)
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
            res = xr.where(
                ((start_d > end_d) | (start_d.isnull()) | (end_d.isnull())), np.nan, res
            )
            # Re-add the time dimension with the period's base time.
            res = res.expand_dims(time=[base_time])
            out.append(res)
        else:
            # Get an array with the good shape, put nans and add the new time.
            res = (group.isel(time=0) * np.nan).expand_dims(time=[base_time])
            out.append(res)
            continue

    return xr.concat(out, dim="time")


def cumulative_difference(
    data: xr.DataArray, threshold: Quantified, op: str, freq: str | None = None
) -> xr.DataArray:
    """Calculate the cumulative difference below/above a given value threshold.

    Parameters
    ----------
    data : xr.DataArray
        Data for which to determine the cumulative difference.
    threshold : Quantified
        The value threshold.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr > thresh.
    freq : str, optional
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
        If `None`, no resampling is performed. Default: `None`.

    Returns
    -------
    xr.DataArray
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

    return to_agg_units(diff, data, op="delta_prod")


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
    r"""First day of values exceeding threshold.

    Returns first day of period where values reach or exceed a threshold over a given number of days,
    limited to a starting calendar date.

    Parameters
    ----------
    data : xarray.DataArray
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
        Resampling frequency defining the periods as defined in
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
        Default: "YS".
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Day of the year when value reaches or exceeds a threshold over a given number of days for the first time.
        If there is no such day, returns np.nan.
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, op, threshold, constrain=constrain)

    out = cond.resample(time=freq).map(
        rl.first_run_after_date,
        window=window,
        date=after_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(data))
    return out


def _get_zone_bins(
    zone_min: Quantity,
    zone_max: Quantity,
    zone_step: Quantity,
):
    """Bin boundary values as defined by zone parameters.

    Parameters
    ----------
    zone_min : Quantity
        Left boundary of the first zone
    zone_max : Quantity
        Right boundary of the last zone
    zone_step: Quantity
        Size of zones

    Returns
    -------
    xarray.DataArray, [units of `zone_step`]
        Array of values corresponding to each zone: [zone_min, zone_min+step, ..., zone_max]
    """
    units = pint2cfunits(str2pint(zone_step))
    mn, mx, step = (
        convert_units_to(str2pint(z), units) for z in [zone_min, zone_max, zone_step]
    )
    bins = np.arange(mn, mx + step, step)
    if (mx - mn) % step != 0:
        warnings.warn(
            "`zone_max` - `zone_min` is not an integer multiple of `zone_step`. Last zone will be smaller."
        )
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
    r"""Divide data into zones and attribute a zone coordinate to each input value.

    Divide values into zones corresponding to bins of width zone_step beginning at zone_min and ending at zone_max.
    Bins are inclusive on the left values and exclusive on the right values.

    Parameters
    ----------
    da : xarray.DataArray
        Input data
    zone_min : Quantity | None
        Left boundary of the first zone
    zone_max : Quantity | None
        Right boundary of the last zone
    zone_step: Quantity | None
        Size of zones
    bins : xr.DataArray | list[Quantity] | None
        Zones to be used, either as a DataArray with appropriate units or a list of Quantity
    exclude_boundary_zones : Bool
        Determines whether a zone value is attributed for values in ]`-np.inf`, `zone_min`[ and [`zone_max`, `np.inf`\ [.
    close_last_zone_right_boundary : Bool
        Determines if the right boundary of the last zone is closed.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Zone index for each value in `da`. Zones are returned as an integer range, starting from `0`
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
        warnings.warn(
            "Expected either `bins` or [`zone_min`, `zone_max`, `zone_step`], got both. "
            "`bins` will be used."
        )

    # Get zone bins (if necessary)
    bins = bins or _get_zone_bins(zone_min, zone_max, zone_step)
    if isinstance(bins, list):
        bins = sorted([convert_units_to(b, da) for b in bins])
    else:
        bins = convert_units_to(bins, da)

    def _get_zone(da):
        return np.digitize(da, bins) - 1

    zones = xr.apply_ufunc(_get_zone, da, dask="parallelized")

    if close_last_zone_right_boundary:
        zones = zones.where(da != bins[-1], _get_zone(bins[-2]))
    if exclude_boundary_zones:
        zones = zones.where(
            (zones != _get_zone(bins[0] - 1)) & (zones != _get_zone(bins[-1]))
        )

    return zones
