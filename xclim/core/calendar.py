# -*- coding: utf-8 -*-
"""
Calendar handling utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""
from datetime import timedelta

import numpy as np
import xarray as xr
from xarray.coding.cftime_offsets import MonthBegin
from xarray.coding.cftime_offsets import MonthEnd
from xarray.coding.cftime_offsets import QuarterBegin
from xarray.coding.cftime_offsets import QuarterEnd
from xarray.coding.cftime_offsets import to_offset
from xarray.coding.cftime_offsets import YearBegin
from xarray.coding.cftime_offsets import YearEnd
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.resample import DataArrayResample


# Maximum day of year in each calendar.
calendars = {
    "standard": 366,
    "gregorian": 366,
    "proleptic_gregorian": 366,
    "julian": 366,
    "no_leap": 365,
    "365_day": 365,
    "all_leap": 366,
    "366_day": 366,
    "uniform30day": 360,
    "360_day": 360,
}


def percentile_doy(
    arr: xr.DataArray, window: int = 5, per: float = 0.1
) -> xr.DataArray:
    """Percentile value for each day of the year

    Return the climatological percentile over a moving window around each day of the year.

    Parameters
    ----------
    arr : xr.DataArray
      Input data.
    window : int
      Number of days around each day of the year to include in the calculation.
    per : float
      Percentile between [0,1]

    Returns
    -------
    xr.DataArray
      The percentiles indexed by the day of the year.
    """
    # TODO: Support percentile array, store percentile in coordinates.
    #  This is supported by DataArray.quantile, but not by groupby.reduce.
    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    # Create empty percentile array
    g = rr.groupby("time.dayofyear")

    p = g.reduce(np.nanpercentile, dim=("time", "window"), q=per * 100)

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = adjust_doy_calendar(p.sel(dayofyear=(p.dayofyear < 366)), arr)

    p.attrs.update(arr.attrs.copy())
    return p


def infer_doy_max(arr: xr.DataArray) -> int:
    """Return the largest doy allowed by calendar.

    Parameters
    ----------
    arr : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    int
      The largest day of the year found in calendar.
    """
    cal = arr.time.encoding.get("calendar", None)
    if cal in calendars:
        doy_max = calendars[cal]
    else:
        # If source is an array with no calendar information and whose length is not at least of full year,
        # then this inference could be wrong (
        doy_max = arr.time.dt.dayofyear.max().data
        if len(arr.time) < 360:
            raise ValueError(
                "Cannot infer the calendar from a series less than a year long."
            )
        if doy_max not in [360, 365, 366]:
            raise ValueError(f"The target array's calendar `{cal}` is not recognized.")

    return doy_max


def _interpolate_doy_calendar(source: xr.DataArray, doy_max: int) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
      Array with `dayofyear` coordinates.
    doy_max : int
      Largest day of the year allowed by calendar.

    Returns
    -------
    xr.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    if "dayofyear" not in source.coords.keys():
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Interpolation of source to target dayofyear range
    doy_max_source = int(source.dayofyear.max())

    # Interpolate to fill na values
    tmp = source.interpolate_na(dim="dayofyear")

    # Interpolate to target dayofyear range
    tmp.coords["dayofyear"] = np.linspace(start=1, stop=doy_max, num=doy_max_source)

    return tmp.interp(dayofyear=range(1, doy_max + 1))


def adjust_doy_calendar(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another calendar.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
      Array with `dayofyear` coordinate.
    target : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    doy_max_source = source.dayofyear.max()

    doy_max = infer_doy_max(target)
    if doy_max_source == doy_max:
        return source

    return _interpolate_doy_calendar(source, doy_max)


def resample_doy(doy: xr.DataArray, arr: xr.DataArray) -> xr.DataArray:
    """Create a temporal DataArray where each day takes the value defined by the day-of-year.

    Parameters
    ----------
    doy : xr.DataArray
      Array with `dayofyear` coordinate.
    arr : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      An array with the same `time` dimension as `arr` whose values are filled according to the day-of-year value in
      `doy`.
    """
    if "dayofyear" not in doy.coords:
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Adjust calendar
    adoy = adjust_doy_calendar(doy, arr)

    # Create array with arr shape and coords
    out = xr.full_like(arr, np.nan)

    # Fill with values from `doy`
    d = out.time.dt.dayofyear.values
    out.data = adoy.sel(dayofyear=d)

    return out


def cftime_start_time(date, freq):
    """
    Get the cftime.datetime for the start of a period. As we are not supplying
    actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used
    on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    datetime : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    cftime.datetime
        The starting datetime of the period inferred from date and freq.
    """

    freq = to_offset(freq)
    if isinstance(freq, (YearBegin, QuarterBegin, MonthBegin)):
        raise ValueError("Invalid frequency: " + freq.rule_code())
    if isinstance(freq, YearEnd):
        month = freq.month
        return date - YearEnd(n=1, month=month) + timedelta(days=1)
    if isinstance(freq, QuarterEnd):
        month = freq.month
        return date - QuarterEnd(n=1, month=month) + timedelta(days=1)
    if isinstance(freq, MonthEnd):
        return date - MonthEnd(n=1) + timedelta(days=1)
    return date


def cftime_end_time(date, freq):
    """
    Get the cftime.datetime for the end of a period. As we are not supplying
    actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used
    on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    datetime : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    cftime.datetime
        The ending datetime of the period inferred from date and freq.
    """
    freq = to_offset(freq)
    if isinstance(freq, (YearBegin, QuarterBegin, MonthBegin)):
        raise ValueError("Invalid frequency: " + freq.rule_code())
    if isinstance(freq, YearEnd):
        mod_freq = YearBegin(n=freq.n, month=freq.month)
    elif isinstance(freq, QuarterEnd):
        mod_freq = QuarterBegin(n=freq.n, month=freq.month)
    elif isinstance(freq, MonthEnd):
        mod_freq = MonthBegin(n=freq.n)
    else:
        mod_freq = freq
    return cftime_start_time(date + mod_freq, freq) - timedelta(microseconds=1)


def cfindex_start_time(cfindex, freq):
    """
    Get the start of a period for a pseudo-period index. As we are using
    datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function
    cannot be used on greater-than-day freq that start at the beginning of a
    month, e.g., 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    CFTimeIndex
        The starting datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_start_time(date, freq) for date in cfindex])


def cfindex_end_time(cfindex, freq):
    """
    Get the start of a period for a pseudo-period index. As we are using
    datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function
    cannot be used on greater-than-day freq that start at the beginning of a
    month, e.g., 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    CFTimeIndex
        The ending datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_end_time(date, freq) for date in cfindex])


def time_bnds(group, freq):
    """
    Find the time bounds for a pseudo-period index. As we are using datetime
    indices to stand in for period indices, assumptions regarding the period
    are made based on the given freq. IMPORTANT NOTE: this function cannot be
    used on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    group : CFTimeIndex or DataArrayResample
        Object which contains CFTimeIndex as a proxy representation for
        CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', or '3T'

    Returns
    _______
    start_time : cftime.datetime
        The start time of the period inferred from datetime and freq.

    Examples
    --------
    >>> index = xr.cftime_range(start='2000-01-01', periods=3,
                                freq='2QS', calendar='360_day')
    >>> time_bnds(index, '2Q')
    ((cftime.Datetime360Day(2000, 1, 1, 0, 0, 0, 0, 1, 1),
      cftime.Datetime360Day(2000, 3, 30, 23, 59, 59, 999999, 0, 91)),
     (cftime.Datetime360Day(2000, 7, 1, 0, 0, 0, 0, 6, 181),
      cftime.Datetime360Day(2000, 9, 30, 23, 59, 59, 999999, 5, 271)),
     (cftime.Datetime360Day(2001, 1, 1, 0, 0, 0, 0, 4, 1),
      cftime.Datetime360Day(2001, 3, 30, 23, 59, 59, 999999, 3, 91)))
    """
    if isinstance(group, CFTimeIndex):
        cfindex = group
    elif isinstance(group, DataArrayResample):
        if isinstance(group._full_index, CFTimeIndex):
            cfindex = group._full_index
        else:
            raise TypeError(
                "Index must be a CFTimeIndex, but got an instance of {}".format(
                    type(group).__name__
                )
            )
    else:
        raise TypeError(
            "Index must be a CFTimeIndex, but got an instance of {}".format(
                type(group).__name__
            )
        )

    return tuple(
        zip(cfindex_start_time(cfindex, freq), cfindex_end_time(cfindex, freq))
    )
