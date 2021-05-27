# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Generic indices submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""
from typing import Union

import numpy as np
import xarray as xr

from xclim.core.calendar import get_calendar
from xclim.core.units import convert_units_to, pint2cfunits, str2pint, to_agg_units

from . import run_length as rl

# __all__ = [
#     "select_time",
#     "select_resample_op",
#     "doymax",
#     "doymin",
#     "default_freq",
#     "threshold_count",
#     "get_daily_events",
#     "daily_downsampler",
# ]


binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


def select_time(da: xr.DataArray, **indexer):
    """Select entries according to a time period.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xr.DataArray
      Selected input values.
    """
    if not indexer:
        selected = da
    else:
        key, val = indexer.popitem()
        time_att = getattr(da.time.dt, key)
        selected = da.sel(time=time_att.isin(val)).dropna(dim="time")

    return selected


def select_resample_op(da: xr.DataArray, op: str, freq: str = "YS", **indexer):
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
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xarray.DataArray
      The maximum value for each period.
    """
    da = select_time(da, **indexer)
    r = da.resample(time=freq, keep_attrs=True)
    if isinstance(op, str):
        return getattr(r, op)(dim="time", keep_attrs=True)

    return r.map(op)


def doymax(da: xr.DataArray) -> xr.DataArray:
    """Return the day of year of the maximum value."""
    i = da.argmax(dim="time")
    out = da.time.dt.dayofyear[i]
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(da))
    return out


def doymin(da: xr.DataArray) -> xr.DataArray:
    """Return the day of year of the minimum value."""
    i = da.argmin(dim="time")
    out = da.time.dt.dayofyear[i]
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(da))
    return out


def default_freq(**indexer) -> str:
    """Return the default frequency."""
    freq = "AS-JAN"
    if indexer:
        group, value = indexer.popitem()
        if "DJF" in value:
            freq = "AS-DEC"
        if group == "month" and sorted(value) != value:
            raise NotImplementedError

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
    return xr.core.ops.get_op(op)


def compare(da: xr.DataArray, op: str, thresh: Union[float, int]) -> xr.DataArray:
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
    da: xr.DataArray, op: str, thresh: Union[float, int], freq: str
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
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
      The number of days where value is within [low, high] for each period.
    """
    c = compare(da, ">", low) * compare(da, "<=", high) * 1
    return c.resample(time=freq).sum(dim="time")


def get_daily_events(da: xr.DataArray, da_value: float, operator: str) -> xr.DataArray:
    r"""Return a 0/1 mask when a condition is True or False.

    the function returns 1 where operator(da, da_value) is True
                         0 where operator(da, da_value) is False
                         nan where da is nan

    Parameters
    ----------
    da : xr.DataArray
    da_value : float
    operator : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
      Logical operator {>, <, >=, <=, gt, lt, ge, le}. e.g. arr > thresh.

    Returns
    -------
    xr.DataArray
    """
    func = getattr(da, "_binary_op")(get_op(operator))
    events = func(da, da_value) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


def daily_downsampler(da: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    r"""Daily climate data downsampler.

    Parameters
    ----------
    da : xr.DataArray
    freq : str

    Returns
    -------
    xr.DataArray

    Note
    ----

        Usage Example

            grouper = daily_downsampler(da_std, freq='YS')
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')
    """
    # generate tags from da.time and freq
    if isinstance(da.time.values[0], np.datetime64):
        years = [f"{y:04d}" for y in da.time.dt.year.values]
        months = [f"{m:02d}" for m in da.time.dt.month.values]
    else:
        # cannot use year, month, season attributes, not available for all calendars ...
        years = [f"{v.year:04d}" for v in da.time.values]
        months = [f"{v.month:02d}" for v in da.time.values]
    seasons = [
        "DJF DJF MAM MAM MAM JJA JJA JJA SON SON SON DJF".split()[int(m) - 1]
        for m in months
    ]

    n_t = da.time.size
    if freq == "YS":
        # year start frequency
        l_tags = years
    elif freq == "MS":
        # month start frequency
        l_tags = [years[i] + months[i] for i in range(n_t)]
    elif freq == "QS-DEC":
        # DJF, MAM, JJA, SON seasons
        # construct tags from list of season+year, increasing year for December
        ys = []
        for i in range(n_t):
            m = months[i]
            s = seasons[i]
            y = years[i]
            if m == "12":
                y = str(int(y) + 1)
            ys.append(y + s)
        l_tags = ys
    else:
        raise RuntimeError(f"Frequency `{freq}` not implemented.")

    # add tags to buffer DataArray
    buffer = da.copy()
    buffer.coords["tags"] = ("time", l_tags)

    # return groupby according to tags
    return buffer.groupby("tags")


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
    freq: str
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
    low_data: xr.DataArray, high_data: xr.DataArray, freq: str
) -> xr.DataArray:
    """Calculate the average diurnal temperature range.

    Parameters
    ----------
    low_data : xr.DataArray
      Lowest daily temperature (tasmin).
    high_data : xr.DataArray
      Highest daily temperature (tasmax).
    freq: str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
    """
    high_data = convert_units_to(high_data, low_data)

    dtr = high_data - low_data
    out = dtr.resample(time=freq).mean()

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
    reducer : {'maximum', 'minimum', 'mean', 'sum'}
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
    reducer : {'maximum', 'minimum', 'mean', 'sum'}
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
    Finally, the statistic is calculated for those data values that fulfil the condition.

    Parameters
    ----------
    data : xr.DataArray
    threshold : str
      Quantity.
    condition : {">", "<", ">=", "<=", "==", "!="}
      Operator
    reducer : {'maximum', 'minimum', 'mean', 'sum'}
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
    Finally, the sum is calculated for those data values that fulfil the condition after subtraction of the threshold value.
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
      Lowest daily temperature (tasmin).
    high_data : xr.DataArray
      Highest daily temperature (tasmax).
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
      Lowest daily temperature (tasmin).
    high_data : xr.DataArray
      Highest daily temperature (tasmax).
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
