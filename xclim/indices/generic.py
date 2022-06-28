# noqa: D205,D400
"""
Generic indices submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import cftime
import numpy as np
import xarray
import xarray as xr
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS

from xclim.core.calendar import (
    DayOfYearStr,
    convert_calendar,
    doy_to_days_since,
    get_calendar,
)
from xclim.core.calendar import select_time as _select_time
from xclim.core.units import (
    convert_units_to,
    declare_units,
    pint2cfunits,
    str2pint,
    to_agg_units,
)

from . import run_length as rl

__all__ = [
    "aggregate_between_dates",
    "compare",
    "count_level_crossings",
    "count_occurrences",
    "default_freq",
    "degree_days",
    "diurnal_temperature_range",
    "domain_count",
    "doymax",
    "doymin",
    "get_daily_events",
    "get_op",
    "interday_diurnal_temperature_range",
    "last_occurrence",
    "select_resample_op",
    "statistics",
    "temperature_sum",
    "threshold_count",
    "thresholded_statistics",
]

binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


def select_time(
    da: xr.DataArray | xr.Dataset,
    drop: bool = False,
    season: str | Sequence[str] = None,
    month: int | Sequence[int] = None,
    doy_bounds: tuple[int, int] = None,
    date_bounds: tuple[str, str] = None,
) -> xr.DataArray | xr.Dataset:
    """Select entries according to a time period."""
    warnings.warn(
        "'select_time()' has moved from `xclim.indices.generic` to `xclim.core.calendar`. "
        "Please update your scripts accordingly.",
        DeprecationWarning,
    )
    return _select_time(
        da,
        drop=drop,
        season=season,
        month=month,
        doy_bounds=doy_bounds,
        date_bounds=date_bounds,
    )


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
    xarray.DataArray
      The maximum value for each period.
    """
    da = _select_time(da, **indexer)
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


def get_op(op: str):
    """Get python's comparing function according to its name of representation.

    Accepted op string are keys and values of xclim.indices.generic.binary_ops.
    """
    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError(f"Operation `{op}` not recognized.")
    return xr.core.ops.get_op(op)  # noqa


def compare(da: xr.DataArray, op: str, thresh: float | int) -> xr.DataArray:
    """Compare a dataArray to a threshold using given operator.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    op : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
      Logical operator {>, <, >=, <=, gt, lt, ge, le }. e.g. arr > thresh.
    thresh : Union[float, int]
      Threshold value.

    Returns
    -------
    xr.DataArray
        Boolean mask of the comparison.
    """
    return get_op(op)(da, thresh)


def threshold_count(
    da: xr.DataArray, op: str, thresh: float | int | xr.DataArray, freq: str
) -> xr.DataArray:
    """Count number of days where value is above or below threshold.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    op : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
      Logical operator {>, <, >=, <=, gt, lt, ge, le }. e.g. arr > thresh.
    thresh : Union[float, int]
      Threshold value.
    freq : str
      Resampling frequency defining the periods as defined in
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
      The number of days meeting the constraints for each period.
    """
    c = compare(da, op, thresh) * 1
    return c.resample(time=freq).sum(dim="time")


def domain_count(da: xr.DataArray, low: float, high: float, freq: str) -> xr.DataArray:
    """Count number of days where value is within low and high thresholds.

    A value is counted if it is larger than `low`, and smaller or equal to `high`, i.e. in `]low, high]`.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    low : float
      Minimum threshold value.
    high : float
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


def get_daily_events(da: xr.DataArray, da_value: float, operator: str) -> xr.DataArray:
    """Return a 0/1 mask when a condition is True or False.

    Parameters
    ----------
    da : xr.DataArray
    da_value : float
    operator : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
      Logical operator {>, <, >=, <=, gt, lt, ge, le}. e.g. arr > thresh.

    Notes
    -----
    the function returns::
        - 1 where operator(da, da_value) is True
        - 0 where operator(da, da_value) is False
        - nan where da is nan

    Returns
    -------
    xr.DataArray
    """
    func = getattr(da, "_binary_op")(get_op(operator))
    events = func(da, da_value) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


# CF-INDEX-META Indices


def count_level_crossings(
    low_data: xr.DataArray, high_data: xr.DataArray, threshold: str, freq: str
) -> xr.DataArray:
    """Calculate the number of times low_data is below threshold while high_data is above threshold.

    First, the threshold is transformed to the same standard_name and units as the input data,
    then the thresholding is performed, and finally, the number of occurrences is counted.

    Parameters
    ----------
    low_data: xr.DataArray
      Variable that must be under the threshold.
    high_data: xr.DataArray
      Variable that must be above the threshold.
    threshold: str
      Quantity.
    freq: str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    # Convert units to low_data
    high_data = convert_units_to(high_data, low_data)
    threshold = convert_units_to(threshold, low_data)

    lower = compare(low_data, "<", threshold)
    higher = compare(high_data, ">=", threshold)

    out = (lower & higher).resample(time=freq).sum()
    return to_agg_units(out, low_data, "count", dim="time")


def count_occurrences(
    data: xr.DataArray, threshold: str, condition: str, freq: str
) -> xr.DataArray:
    """Calculate the number of times some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold),
    i.e. if condition is `<`, then this counts the number of times `data < threshold`.
    Finally, count the number of occurrences when condition is met.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity.
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)

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
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    dtr = high_data - low_data
    out = getattr(dtr.resample(time=freq), reducer)()

    u = str2pint(low_data.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


def first_occurrence(
    data: xr.DataArray, threshold: str, condition: str, freq: str
) -> xr.DataArray:
    """Calculate the first time some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, locate the first occurrence when condition is met.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)

    out = cond.resample(time=freq).map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


def last_occurrence(
    data: xr.DataArray, threshold: str, condition: str, freq: str
) -> xr.DataArray:
    """Calculate the last time some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, locate the last occurrence when condition is met.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)

    out = cond.resample(time=freq).map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


def spell_length(
    data: xr.DataArray, threshold: str, condition: str, reducer: str, freq: str
) -> xr.DataArray:
    """Calculate statistics on lengths of spells.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Then the spells are determined, and finally the statistics according to the specified reducer are calculated.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity.
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    reducer : {'max', 'min', 'mean', 'sum'}
      Reducer.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)

    out = cond.resample(time=freq).map(
        rl.rle_statistics,
        reducer=reducer,
        dim="time",
    )
    return to_agg_units(out, data, "count")


def statistics(data: xr.DataArray, reducer: str, freq: str) -> xr.DataArray:
    """Calculate a simple statistic of the data.

    Parameters
    ----------
    data : xr.DataArray
    reducer : {'max', 'min', 'mean', 'sum'}
      Reducer.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    out = getattr(data.resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


def thresholded_statistics(
    data: xr.DataArray, threshold: str, condition: str, reducer: str, freq: str
) -> xr.DataArray:
    """Calculate a simple statistic of the data for which some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, the statistic is calculated for those data values that fulfill the condition.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity.
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    reducer : {'max', 'min', 'mean', 'sum'}
      Reducer.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)

    out = getattr(data.where(cond).resample(time=freq), reducer)()
    out.attrs["units"] = data.attrs["units"]
    return out


def temperature_sum(
    data: xr.DataArray, threshold: str, condition: str, freq: str
) -> xr.DataArray:
    """Calculate the temperature sum above/below a threshold.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the thresholding is performed as condition(data, threshold), i.e. if condition is <, data < threshold.
    Finally, the sum is calculated for those data values that fulfill the condition after subtraction of the threshold value.
    If the sum is for values below the threshold the result is multiplied by -1.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    threshold = convert_units_to(threshold, data)

    cond = compare(data, condition, threshold)
    direction = -1 if "<" in condition else 1

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
    freq: str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
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
    freq: str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    out = (high_data - low_data).resample(time=freq).mean()

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
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Aggregated data between the start and end dates. If the end date is before the start date, returns np.nan.
      If there is no start and/or end date, returns np.nan.
    """

    def _get_days(_bound, _group, _base_time):
        """Get bound in number of days since base_time. Bound can be a days_since array or a DayOfYearStr."""
        if isinstance(_bound, str):
            b_i = rl.index_of_date(_group.time, _bound, max_idxs=1)  # noqa
            if not len(b_i):
                return None
            return (_group.time.isel(time=b_i[0]) - _group.time.isel(time=0)).dt.days
        if _base_time in _bound.time:
            return _bound.sel(time=_base_time)
        return None

    if freq is None:
        frequencies = []
        for i, bound in enumerate([start, end], start=1):
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

    out = list()
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

    out = xr.concat(out, dim="time")
    return out


@declare_units(tas="[temperature]")
def degree_days(tas: xr.DataArray, thresh: str, condition: str) -> xr.DataArray:
    """Calculate the degree days below/above the temperature threshold.

    Parameters
    ----------
    tas : xr.DataArray
      Mean daily temperature.
    thresh : str
      The temperature threshold.
    condition : {"<", ">"}
      Operator.

    Returns
    -------
    xarray.DataArray
    """
    thresh = convert_units_to(thresh, tas)

    if "<" in condition:
        out = (thresh - tas).clip(0)
    elif ">" in condition:
        out = (tas - thresh).clip(0)
    else:
        raise NotImplementedError(f"Condition not supported: '{condition}'.")

    out = to_agg_units(out, tas, op="delta_prod")
    return out
