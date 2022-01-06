# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Calendar handling utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""
import datetime as pydt
import re
from typing import Optional, Sequence, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import (
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
    to_offset,
)
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.resample import DataArrayResample

from .formatting import update_xclim_history
from .utils import (
    DayOfYearStr,
    calc_perc,
    convert_calendar,
    date_range,
    datetime_classes,
    days_in_year,
    get_calendar,
    uniform_calendars,
)

# Some xclim.core.utils functions made accessible here for backwards compatibility reasons.
__all__ = [
    "adjust_doy_calendar",
    "cfindex_end_time",
    "cfindex_start_time",
    "cftime_end_time",
    "cftime_start_time",
    "climatological_mean_doy",
    "compare_offsets",
    "datetime_to_decimal_year",
    "days_since_to_doy",
    "doy_to_days_since",
    "ensure_cftime_array",
    "interp_calendar",
    "max_doy",
    "parse_offset",
    "percentile_doy",
    "resample_doy",
    "time_bnds",
    "within_bnds_doy",
    "convert_calendar",
    "date_range",
    "datetime_classes",
    "days_in_year",
    "get_calendar",
    "uniform_calendars",
]


# Maximum day of year in each calendar.
max_doy = {
    "default": 366,
    "standard": 366,
    "gregorian": 366,
    "proleptic_gregorian": 366,
    "julian": 366,
    "noleap": 365,
    "365_day": 365,
    "all_leap": 366,
    "366_day": 366,
    "360_day": 360,
}


def interp_calendar(
    source: Union[xr.DataArray, xr.Dataset],
    target: xr.DataArray,
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """Interpolates a DataArray/Dataset to another calendar based on decimal year measure.

    Each timestamp in source and target are first converted to their decimal year equivalent
    then source is interpolated on the target coordinate. The decimal year is the number of
    years since 0001-01-01 AD.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar or 2000.16301 in a 'noleap' calendar.

    This method should be used with daily data or coarser. Sub-daily result will have a modified day cycle.

    Parameters
    ----------
    source: Union[xr.DataArray, xr.Dataset]
      The source data to interpolate, must have a time coordinate of a valid dtype (np.datetime64 or cftime objects)
    target: xr.DataArray
      The target time coordinate of a valid dtype (np.datetime64 or cftime objects)
    dim : str
      The time coordinate name.

    Return
    ------
    Union[xr.DataArray, xr.Dataset]
      The source interpolated on the decimal years of target,
    """
    cal_src = get_calendar(source, dim=dim)
    cal_tgt = get_calendar(target, dim=dim)

    out = source.copy()
    out[dim] = datetime_to_decimal_year(source[dim], calendar=cal_src).drop_vars(dim)
    target_idx = datetime_to_decimal_year(target, calendar=cal_tgt)
    out = out.interp(time=target_idx)
    out[dim] = target
    return out


def ensure_cftime_array(time: Sequence) -> np.ndarray:
    """Convert an input 1D array to a numpy array of cftime objects.

    Python's datetime are converted to cftime.DatetimeGregorian ("standard" calendar).

    Raises ValueError when unable to cast the input.
    """
    if isinstance(time, xr.DataArray):
        time = time.indexes["time"]
    elif isinstance(time, np.ndarray):
        time = pd.DatetimeIndex(time)
    if isinstance(time, xr.CFTimeIndex):
        return time.values
    if isinstance(time[0], cftime.datetime):
        return time
    if isinstance(time[0], pydt.datetime):
        return np.array(
            [cftime.DatetimeGregorian(*ele.timetuple()[:6]) for ele in time]
        )
    raise ValueError("Unable to cast array to cftime dtype")


def datetime_to_decimal_year(times: xr.DataArray, calendar: str = "") -> xr.DataArray:
    """Convert a datetime xr.DataArray to decimal years according to its calendar or the given one.

    Decimal years are the number of years since 0001-01-01 00:00:00 AD.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar, 2000.16301 in a "noleap" or 2000.16806 in a "360_day".
    """
    calendar = calendar or get_calendar(times)
    if calendar == "default":
        calendar = "standard"

    def _make_index(time) -> xr.DataArray:
        year = int(time.dt.year[0])
        doys = cftime.date2num(
            ensure_cftime_array(time), f"days since {year:04d}-01-01", calendar=calendar
        )
        return xr.DataArray(
            year + doys / days_in_year(year, calendar),
            dims=time.dims,
            coords=time.coords,
            name="time",
        )

    return times.groupby("time.year").map(_make_index)


@update_xclim_history
def percentile_doy(
    arr: xr.DataArray,
    window: int = 5,
    per: Union[float, Sequence[float]] = 10.0,
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
) -> xr.DataArray:
    """Percentile value for each day of the year.

    Return the climatological percentile over a moving window around each day of the year.
    Different quantile estimators can be used by specifying `alpha` and `beta` according to specifications given by [HyndmanFan]_. The default definition corresponds to method 8, which meets multiple desirable statistical properties for sample quantiles. Note that `numpy.percentile` corresponds to method 7, with alpha and beta set to 1.

    Parameters
    ----------
    arr : xr.DataArray
      Input data, a daily frequency (or coarser) is required.
    window : int
      Number of time-steps around each day of the year to include in the calculation.
    per : float or sequence of floats
      Percentile(s) between [0, 100]
    alpha: float
        Plotting position parameter.
    beta: float
        Plotting position parameter.

    Returns
    -------
    xr.DataArray
      The percentiles indexed by the day of the year.
      For calendars with 366 days, percentiles of doys 1-365 are interpolated to the 1-366 range.

    References
    ----------
    .. [HyndmanFan] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. The American Statistician, 50(4), 361-365.
    """

    # Ensure arr sampling frequency is daily or coarser
    # but cowardly escape the non-inferrable case.
    if compare_offsets(xr.infer_freq(arr.time) or "D", "<", "D"):
        raise ValueError("input data should have daily or coarser frequency")

    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    ind = pd.MultiIndex.from_arrays(
        (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
        names=("year", "dayofyear"),
    )
    rrr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))

    if rrr.chunks is not None and len(rrr.chunks[rrr.get_axis_num("stack_dim")]) > 1:
        rrr = rrr.chunk(dict(stack_dim=-1))

    if np.isscalar(per):
        per = [per]

    p = xr.apply_ufunc(
        calc_perc,
        rrr,
        input_core_dims=[["stack_dim"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(percentiles=per, alpha=alpha, beta=beta),
        dask="parallelized",
        output_dtypes=[rrr.dtype],
        dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(per)}),
    )
    p = p.assign_coords(percentiles=xr.DataArray(per, dims=("percentiles",)))

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = adjust_doy_calendar(p.sel(dayofyear=(p.dayofyear < 366)), arr)

    p.attrs.update(arr.attrs.copy())

    # Saving percentile attributes
    n = len(arr.time)
    p.attrs["climatology_bounds"] = (
        arr.time[0 :: n - 1].dt.strftime("%Y-%m-%d").values.tolist()
    )
    p.attrs["window"] = window
    p.attrs["alpha"] = alpha
    p.attrs["beta"] = beta
    return p.rename("per")


def compare_offsets(freqA: str, op: str, freqB: str) -> bool:  # noqa
    """Compare offsets string based on their approximate length, according to a given operator.

    Offset are compared based on their length approximated for a period starting
    after 1970-01-01 00:00:00. If the offsets are from the same category (same first letter),
    only the multiplicator prefix is compared (QS-DEC == QS-JAN, MS < 2MS).
    "Business" offsets are not implemented.

    Parameters
    ----------
    freqA: str
      RHS Date offset string ('YS', '1D', 'QS-DEC', ...)
    op : {'<', '<=', '==', '>', '>=', '!='}
      Operator to use.
    freqB: str
      LHS Date offset string ('YS', '1D', 'QS-DEC', ...)

    Returns
    -------
    bool
      freqA op freqB
    """
    from xclim.indices.generic import get_op

    # Get multiplicator and base frequency
    t_a, b_a, _, _ = parse_offset(freqA)
    t_b, b_b, _, _ = parse_offset(freqB)

    if b_a != b_b:
        # Different base freq, compare length of first period after beginning of time.
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqA)
        t_a = (t[1] - t[0]).total_seconds()
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqB)
        t_b = (t[1] - t[0]).total_seconds()
    # else Same base freq, compare multiplicator only.

    return get_op(op)(t_a, t_b)


def parse_offset(freq: str) -> Sequence[str]:
    """Parse an offset string.

    Parse a frequency offset and, if needed, convert to cftime-compatible components.

    Parameters
    ----------
    freq : str
      Frequency offset.

    Returns
    -------
    multiplicator (int), offset base (str), is start anchored (bool), anchor (str or None)
      "[n]W" is always replaced with "[7n]D", as xarray doesn't support "W" for cftime indexes.
      "Y" is always replaced with "A".
    """
    patt = r"(\d*)(\w)(S)?(?:-(\w{2,3}))?"
    mult, base, start, anchor = re.search(patt, freq).groups()
    mult = int(mult or "1")
    base = base.replace("Y", "A")
    if base == "W":
        mult = 7 * mult
        base = "D"
        anchor = ""
    return mult, base, start == "S", anchor


def _interpolate_doy_calendar(source: xr.DataArray, doy_max: int) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another.

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


def adjust_doy_calendar(
    source: xr.DataArray, target: Union[xr.DataArray, xr.Dataset]
) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another calendar.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
      Array with `dayofyear` coordinate.
    target : xr.DataArray or xr.Dataset
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    doy_max_source = source.dayofyear.max()

    doy_max = max_doy[get_calendar(target)]
    if doy_max_source == doy_max:
        return source

    return _interpolate_doy_calendar(source, doy_max)


def resample_doy(
    doy: xr.DataArray, arr: Union[xr.DataArray, xr.Dataset]
) -> xr.DataArray:
    """Create a temporal DataArray where each day takes the value defined by the day-of-year.

    Parameters
    ----------
    doy : xr.DataArray
      Array with `dayofyear` coordinate.
    arr : xr.DataArray or xr.Dataset
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      An array with the same dimensions as `doy`, except for `dayofyear`, which is
      replaced by the `time` dimension of `arr`. Values are filled according to the
      day of year value in `doy`.
    """
    if "dayofyear" not in doy.coords:
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Adjust calendar
    adoy = adjust_doy_calendar(doy, arr)

    out = adoy.rename(dayofyear="time").reindex(time=arr.time.dt.dayofyear)
    out["time"] = arr.time

    return out


def cftime_start_time(date: cftime.datetime, freq: str) -> cftime.datetime:
    """Get the cftime.datetime for the start of a period.

    As we are not supplying actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used on greater-than-day freq that start at the
    beginning of a month, e.g. 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    ----------
    date : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    -------
    cftime.datetime
        The starting datetime of the period inferred from date and freq.
    """
    freq = to_offset(freq)
    if isinstance(freq, (YearBegin, QuarterBegin, MonthBegin)):
        raise ValueError("Invalid frequency: " + freq.rule_code())
    if isinstance(freq, YearEnd):
        month = freq.month
        return date - YearEnd(n=1, month=month) + pydt.timedelta(days=1)
    if isinstance(freq, QuarterEnd):
        month = freq.month
        return date - QuarterEnd(n=1, month=month) + pydt.timedelta(days=1)
    if isinstance(freq, MonthEnd):
        return date - MonthEnd(n=1) + pydt.timedelta(days=1)
    return date


def cftime_end_time(date: cftime.datetime, freq: str) -> cftime.datetime:
    """Get the cftime.datetime for the end of a period.

    As we are not supplying actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used on greater-than-day freq that start at the
    beginning of a month, e.g. 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    ----------
    date : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    -------
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
    return cftime_start_time(date + mod_freq, freq) - pydt.timedelta(microseconds=1)


def cfindex_start_time(cfindex: CFTimeIndex, freq: str) -> CFTimeIndex:
    """
    Get the start of a period for a pseudo-period index.

    As we are using datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function cannot be used on greater-than-day
    freq that start at the beginning of a month, e.g. 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    ----------
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    -------
    CFTimeIndex
        The starting datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_start_time(date, freq) for date in cfindex])  # noqa


def cfindex_end_time(cfindex: CFTimeIndex, freq: str) -> CFTimeIndex:
    """
    Get the end of a period for a pseudo-period index.

    As we are using datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function cannot be used on greater-than-day
    freq that start at the beginning of a month, e.g. 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    ----------
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    -------
    CFTimeIndex
        The ending datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_end_time(date, freq) for date in cfindex])  # noqa


def time_bnds(group, freq: str) -> Sequence[Tuple[cftime.datetime, cftime.datetime]]:
    """
    Find the time bounds for a pseudo-period index.

    As we are using datetime indices to stand in for period indices, assumptions regarding the period
    are made based on the given freq. IMPORTANT NOTE: this function cannot be used on greater-than-day freq
    that start at the beginning of a month, e.g. 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    ----------
    group : CFTimeIndex or DataArrayResample
        Object which contains CFTimeIndex as a proxy representation for
        CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', or '3T'

    Returns
    -------
    Sequence[(cftime.datetime, cftime.datetime)]
        The start and end times of the period inferred from datetime and freq.

    Examples
    --------
    >>> from xarray import cftime_range
    >>> from xclim.core.calendar import time_bnds
    >>> index = cftime_range(start='2000-01-01', periods=3, freq='2QS', calendar='360_day')
    >>> out = time_bnds(index, '2Q')
    >>> for bnds in out:
    ...     print(bnds[0].strftime("%Y-%m-%dT%H:%M:%S"), ' -', bnds[1].strftime("%Y-%m-%dT%H:%M:%S"))
    2000-01-01T00:00:00  - 2000-03-30T23:59:59
    2000-07-01T00:00:00  - 2000-09-30T23:59:59
    2001-01-01T00:00:00  - 2001-03-30T23:59:59
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


def climatological_mean_doy(
    arr: xr.DataArray, window: int = 5
) -> Tuple[xr.DataArray, xr.DataArray]:
    """The climatological mean and standard deviation for each day of the year.

    Parameters
    ----------
    arr : xarray.DataArray
      Input array.
    window : int
      Window size in days.

    Returns
    -------
    xarray.DataArray, xarray.DataArray
      Mean and standard deviation.
    """
    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    # Create empty percentile array
    g = rr.groupby("time.dayofyear")

    m = g.mean(["time", "window"])
    s = g.std(["time", "window"])

    return m, s


def within_bnds_doy(
    arr: xr.DataArray, *, low: xr.DataArray, high: xr.DataArray
) -> xr.DataArray:
    """Return whether or not array values are within bounds for each day of the year.

    Parameters
    ----------
    arr : xarray.DataArray
      Input array.
    low : xarray.DataArray
      Low bound with dayofyear coordinate.
    high : xarray.DataArray
      High bound with dayofyear coordinate.

    Returns
    -------
    xarray.DataArray
    """
    low = resample_doy(low, arr)
    high = resample_doy(high, arr)
    return (low < arr) * (arr < high)


def _doy_days_since_doys(
    base: xr.DataArray, start: Optional[DayOfYearStr] = None
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Common calculation for doy to days since and inverse conversions.

    Parameters
    ----------
    base: xr.DataArray
      1D time coordinate.
    start: DayOfYearStr, optional
      A date to compute the offset relative to. If note given, start_doy is the same as base_doy.

    Returns
    -------
    base_doy : xr.DataArray
      Day of year for each element in base.
    start_doy : xr.DataArray
      Day of year of the "start" date.
      The year used is the one the start date would take as a doy for the corresponding base element.
    doy_max : xr.DataArray
      Number of days (maximum doy) for the year of each value in base.
    """
    calendar = get_calendar(base)

    base_doy = base.dt.dayofyear

    doy_max = xr.apply_ufunc(
        lambda y: days_in_year(y, calendar), base.dt.year, vectorize=True
    )

    if start is not None:
        mm, dd = map(int, start.split("-"))
        starts = xr.apply_ufunc(
            lambda y: datetime_classes[calendar](y, mm, dd),
            base.dt.year,
            vectorize=True,
        )
        start_doy = starts.dt.dayofyear
        start_doy = start_doy.where(start_doy >= base_doy, start_doy + doy_max)
    else:
        start_doy = base_doy

    return base_doy, start_doy, doy_max


def doy_to_days_since(
    da: xr.DataArray,
    start: Optional[DayOfYearStr] = None,
    calendar: Optional[str] = None,
) -> xr.DataArray:
    """Convert day-of-year data to days since a given date

    This is useful for computing meaningful statistics on doy data.

    Parameters
    ----------
    da: xr.DataArray
      Array of "day-of-year", usually int dtype, must have a `time` dimension.
      Sampling frequency should be finer or similar to yearly and coarser then daily.
    start: date of year str, optional
      A date in "MM-DD" format, the base day of the new array.
      If None (default), the `time` axis is used.
      Passing `start` only makes sense if `da` has a yearly sampling frequency.
    calendar: str, optional
      The calendar to use when computing the new interval.
      If None (default), the calendar attribute of the data or of its `time` axis is used.
      All time coordinates of `da` must exist in this calendar.
      No check is done to ensure doy values exist in this calendar.

    Returns
    -------
    xr.DataArray
      Same shape as `da`, int dtype, day-of-year data translated to a number of days since a given date.
      If start is not None, there might be negative values.

    Notes
    -----
    The time coordinates of `da` are considered as the START of the period. For example, a doy value of
    350 with a timestamp of '2020-12-31' is understood as '2021-12-16' (the 350th day of 2021).
    Passing `start=None`, will use the time coordinate as the base, so in this case the converted value
    will be 350 "days since time coordinate".

    Examples
    --------
    >>> from xarray import DataArray
    >>> time = date_range('2020-07-01', '2021-07-01', freq='AS-JUL')
    >>> da = DataArray([190, 2], dims=('time',), coords={'time': time})  # July 8th 2020 and Jan 2nd 2022
    >>> doy_to_days_since(da, start='10-02').values  # Convert to days since Oct. 2nd, of the data's year.
    array([-86, 92])
    """
    base_calendar = get_calendar(da)
    calendar = calendar or da.attrs.get("calendar", base_calendar)
    dac = convert_calendar(da, calendar)

    base_doy, start_doy, doy_max = _doy_days_since_doys(dac.time, start)

    # 2cases:
    # val is a day in the same year as its index : da - offset
    # val is a day in the next year : da + doy_max - offset
    out = xr.where(dac > base_doy, dac, dac + doy_max) - start_doy
    out.attrs.update(da.attrs)
    if start is not None:
        out.attrs.update(units=f"days after {start}")
    else:
        starts = np.unique(out.time.dt.strftime("%m-%d"))
        if len(starts) == 1:
            out.attrs.update(units=f"days after {starts[0]}")
        else:
            out.attrs.update(units="days after time coordinate")

    out.attrs.pop("is_dayofyear", None)
    out.attrs.update(calendar=calendar)
    return convert_calendar(out, base_calendar).rename(da.name)


def days_since_to_doy(
    da: xr.DataArray,
    start: Optional[DayOfYearStr] = None,
    calendar: Optional[str] = None,
) -> xr.DataArray:
    """Reverse the conversion made by :py:func:`doy_to_days_since`.

    Converts data given in days since a specific date to day-of-year.

    Parameters
    ----------
    da: xr.DataArray
      The result of :py:func:`doy_to_days_since`.
    start: DateOfYearStr, optional
      `da` is considered as days since that start date (in the year of the time index).
      If None (default), it is read from the attributes.
    calendar: str, optional
      Calendar the "days since" were computed in.
      If None (default), it is read from the attributes.

    Returns
    -------
    xr.DataArray
      Same shape as `da`, values as `day of year`.

    Examples
    --------
    >>> from xarray import DataArray
    >>> time = date_range('2020-07-01', '2021-07-01', freq='AS-JUL')
    >>> da = DataArray(
            [-86, 92], dims=('time',), coords={'time': time}, attrs={'units': 'days since 10-02'}
        )
    >>> days_since_to_doy(da).values
    array([190, 2])
    """
    if start is None:
        unitstr = da.attrs.get("units", "  time coordinate").split(" ", maxsplit=2)[-1]
        if unitstr != "time coordinate":
            start = unitstr

    base_calendar = get_calendar(da)
    calendar = calendar or da.attrs.get("calendar", base_calendar)

    dac = convert_calendar(da, calendar)

    _, start_doy, doy_max = _doy_days_since_doys(dac.time, start)

    # 2cases:
    # val is a day in the same year as its index : da + offset
    # val is a day in the next year : da + offset - doy_max
    out = dac + start_doy
    out = xr.where(out > doy_max, out - doy_max, out)

    out.attrs.update(
        {k: v for k, v in da.attrs.items() if k not in ["units", "calendar"]}
    )
    out.attrs.update(calendar=calendar, is_dayofyear=1)
    return convert_calendar(out, base_calendar).rename(da.name)
