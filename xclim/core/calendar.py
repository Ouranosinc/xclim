# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Calendar handling utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""
import datetime as pydt
import re
from typing import Any, Optional, Sequence, Tuple, Union

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

from xclim.core.formatting import update_xclim_history
from xclim.core.utils import DayOfYearStr, calc_perc

# cftime and datetime classes to use for each calendar name
datetime_classes = {"default": pydt.datetime, **cftime._cftime.DATE_TYPES}  # noqa

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


def get_calendar(obj: Any, dim: str = "time") -> str:
    """Return the calendar of an object.

    Parameters
    ----------
    obj : Any
      An object defining some date.
      If `obj` is an array/dataset with a datetime coordinate, use `dim` to specify its name.
      Values must have either a datetime64 dtype or a cftime dtype.
      `obj` can also be a python datetime.datetime, a cftime object or a pandas Timestamp
      or an iterable of those, in which case the calendar is inferred from the first value.
    dim : str
      Name of the coordinate to check (if `obj` is a DataArray or Dataset).

    Raises
    ------
    ValueError
      If no calendar could be inferred.

    Returns
    -------
    str
      The cftime calendar name or "default" when the data is using numpy's or python's datetime types.
    """
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        if obj[dim].dtype == "O":
            obj = obj[dim].where(obj[dim].notnull(), drop=True)[0].item()
        elif "datetime64" in obj[dim].dtype.name:
            return "default"

    obj = np.take(
        obj, 0
    )  # Take zeroth element, overcome cases when arrays or lists are passed.
    if isinstance(obj, pydt.datetime):  # Also covers pandas Timestamp
        return "default"
    if isinstance(obj, cftime.datetime):
        return obj.calendar

    raise ValueError(f"Calendar could not be inferred from object of type {type(obj)}.")


def convert_calendar(
    source: Union[xr.DataArray, xr.Dataset],
    target: Union[xr.DataArray, str],
    align_on: Optional[str] = None,
    missing: Optional[Any] = None,
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert a DataArray/Dataset to another calendar using the specified method.

    Only converts the individual timestamps, does not modify any data except in dropping invalid/surplus dates or inserting missing dates.

    If the source and target calendars are either no_leap, all_leap or a standard type, only the type of the time array is modified.
    When converting to a leap year from a non-leap year, the 29th of February is removed from the array.
    In the other direction and if `target` is a string, the 29th of February will be missing in the output,
    unless `missing` is specified, in which case that value is inserted.

    For conversions involving `360_day` calendars, see Notes.

    This method is safe to use with sub-daily data as it doesn't touch the time part of the timestamps.

    Parameters
    ----------
    source : xr.DataArray
      Input array/dataset with a time coordinate of a valid dtype (datetime64 or a cftime.datetime).
    target : Union[xr.DataArray, str]
      Either a calendar name or the 1D time coordinate to convert to.
      If an array is provided, the output will be reindexed using it and in that case, days in `target`
      that are missing in the converted `source` are filled by `missing` (which defaults to NaN).
    align_on : {None, 'date', 'year'}
      Must be specified when either source or target is a `360_day` calendar, ignored otherwise. See Notes.
    missing : Optional[any]
      A value to use for filling in dates in the target that were missing in the source.
      If `target` is a string, default (None) is not to fill values. If it is an array, default is to fill with NaN.
    dim : str
      Name of the time coordinate.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
      Copy of source with the time coordinate converted to the target calendar.
      If `target` is given as an array, the output is reindexed to it, with fill value `missing`.
      If `target` was a string and `missing` was None (default), invalid dates in the new calendar are dropped, but missing dates are not inserted.
      If `target` was a string and `missing` was given, then start, end and frequency of the new time axis are inferred and
      the output is reindexed to that a new array.

    Notes
    -----
    If one of the source or target calendars is `360_day`, `align_on` must be specified and two options are offered.

    "year"
      The dates are translated according to their rank in the year (dayofyear), ignoring their original month and day information,
      meaning that the missing/surplus days are added/removed at regular intervals.

      From a `360_day` to a standard calendar, the output will be missing the following dates (day of year in parenthesis):
        To a leap year:
          January 31st (31), March 31st (91), June 1st (153), July 31st (213), September 31st (275) and November 30th (335).
        To a non-leap year:
          February 6th (36), April 19th (109), July 2nd (183), September 12th (255), November 25th (329).

      From standard calendar to a '360_day', the following dates in the source array will be dropped:
        From a leap year:
          January 31st (31), April 1st (92), June 1st (153), August 1st (214), September 31st (275), December 1st (336)
        From a non-leap year:
          February 6th (37), April 20th (110), July 2nd (183), September 13th (256), November 25th (329)

      This option is best used on daily and subdaily data.

    "date"
      The month/day information is conserved and invalid dates are dropped from the output. This means that when converting from
      a `360_day` to a standard calendar, all 31st (Jan, March, May, July, August, October and December) will be missing as there is no equivalent
      dates in the `360_day` and the 29th (on non-leap years) and 30th of February will be dropped as there are no equivalent dates in
      a standard calendar.

      This option is best used with data on a frequency coarser than daily.
    """
    cal_src = get_calendar(source, dim=dim)

    if isinstance(target, str):
        cal_tgt = target
    else:
        cal_tgt = get_calendar(target, dim=dim)

    if cal_src == cal_tgt:
        return source

    if (cal_src == "360_day" or cal_tgt == "360_day") and align_on is None:
        raise ValueError(
            "Argument `align_on` must be specified with either 'date' or "
            "'year' when converting to or from a '360_day' calendar."
        )
    if cal_src != "360_day" and cal_tgt != "360_day":
        align_on = None

    out = source.copy()
    # TODO Maybe the 5-6 days to remove could be given by the user?
    if align_on == "year":

        def _yearly_interp_doy(time):
            # Returns the nearest day in the target calendar of the corresponding "decimal year" in the source calendar
            yr = int(time.dt.year[0])
            return np.round(
                days_in_year(yr, cal_tgt)
                * time.dt.dayofyear
                / days_in_year(yr, cal_src)
            ).astype(int)

        new_doy = source.time.groupby(f"{dim}.year").map(_yearly_interp_doy)

        # Convert the source datetimes, but override the doy with our new doys
        out[dim] = xr.DataArray(
            [
                _convert_datetime(datetime, new_doy=doy, calendar=cal_tgt)
                for datetime, doy in zip(source[dim].indexes[dim], new_doy)
            ],
            dims=(dim,),
            name=dim,
        )
        # Remove duplicate timestamps, happens when reducing the number of days
        out = out.isel({dim: np.unique(out[dim], return_index=True)[1]})
    else:
        time_idx = source[dim].indexes[dim]
        out[dim] = xr.DataArray(
            [_convert_datetime(time, calendar=cal_tgt) for time in time_idx],
            dims=(dim,),
            name=dim,
        )
        # Remove NaN that where put on invalid dates in target calendar
        out = out.where(out[dim].notnull(), drop=True)

    if isinstance(target, str) and missing is not None:
        target = date_range_like(source[dim], cal_tgt)

    if isinstance(target, xr.DataArray):
        out = out.reindex({dim: target}, fill_value=missing or np.nan)

    # Copy attrs but change remove `calendar` is still present.
    out[dim].attrs.update(source[dim].attrs)
    out[dim].attrs.pop("calendar", None)
    return out


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


def date_range(
    *args, calendar: str = "default", **kwargs
) -> Union[pd.DatetimeIndex, CFTimeIndex]:
    """Wrap pd.date_range (if calendar == 'default') or xr.cftime_range (otherwise)."""
    if calendar == "default":
        return pd.date_range(*args, **kwargs)
    return xr.cftime_range(*args, calendar=calendar, **kwargs)


def date_range_like(source: xr.DataArray, calendar: str) -> xr.DataArray:
    """Generate a datetime array with the same frequency, start and end as another one, but in a different calendar.

    Parameters
    ----------
    source : xr.DataArray
      1D datetime coordinate DataArray
    calendar : str
      New calendar name.

    Raises
    ------
    ValueError
      If the source's frequency was not found.

    Returns
    -------
    xr.DataArray
      1D datetime coordinate with the same start, end and frequency as the source, but in the new calendar.
        The start date is assumed to exist in the target calendar.
        If the end date doesn't exist, the code tries 1 and 2 calendar days before.
        Exception when the source is in 360_day and the end of the range is the 30th of a 31-days month,
        then the 31st is appended to the range.
    """
    freq = xr.infer_freq(source)
    if freq is None:
        raise ValueError(
            "`date_range_like` was unable to generate a range as the source frequency was not inferrable."
        )

    src_cal = get_calendar(source)
    if src_cal == calendar:
        return source

    index = source.indexes[source.dims[0]]
    end_src = index[-1]
    end = _convert_datetime(end_src, calendar=calendar)
    if end is np.nan:  # Day is invalid, happens at the end of months.
        end = _convert_datetime(end_src.replace(day=end_src.day - 1), calendar=calendar)
        if end is np.nan:  # Still invalid : 360_day to non-leap february.
            end = _convert_datetime(
                end_src.replace(day=end_src.day - 2), calendar=calendar
            )
    if src_cal == "360_day" and end_src.day == 30 and end.daysinmonth == 31:
        # For the specific case of daily data from 360_day source, the last day is expected to be "missing"
        end = end.replace(day=31)

    return xr.DataArray(
        date_range(
            _convert_datetime(index[0], calendar=calendar),
            end,
            freq=freq,
            calendar=calendar,
        ),
        dims=source.dims,
        name=source.dims[0],
    )


def _convert_datetime(
    datetime: Union[pydt.datetime, cftime.datetime],
    new_doy: Optional[Union[float, int]] = None,
    calendar: str = "default",
) -> Union[cftime.datetime, pydt.datetime, float]:
    """Convert a datetime object to another calendar.

    Nanosecond information are lost as cftime.datetime doesn't support them.

    Parameters
    ----------
    datetime: Union[datetime.datetime, cftime.datetime]
      A datetime object to convert.
    new_doy:  Optional[Union[float, int]]
      Allows for redefining the day of year (thus ignoring month and day information from the source datetime).
    calendar: str
      The target calendar

    Returns
    -------
    Union[cftime.datetime, datetime.datetime, np.nan]
      A datetime object of the target calendar with the same year, month, day and time
      as the source (month and day according to `new_doy` if given).
      If the month and day doesn't exist in the target calendar, returns np.nan. (Ex. 02-29 in "noleap")
    """
    if new_doy is not None:
        new_date = cftime.num2date(
            new_doy - 1,
            f"days since {datetime.year}-01-01",
            calendar=calendar if calendar != "default" else "standard",
        )
    else:
        new_date = datetime
    try:
        return datetime_classes[calendar](
            datetime.year,
            new_date.month,
            new_date.day,
            datetime.hour,
            datetime.minute,
            datetime.second,
            datetime.microsecond,
        )
    except ValueError:
        return np.nan


def ensure_cftime_array(
    time: Sequence,
) -> Union[CFTimeIndex, np.ndarray]:
    """Convert an input 1D array to an array of cftime objects.

    Python's datetime are converted to cftime.DatetimeGregorian.

    Raises ValueError when unable to cast the input.
    """
    if isinstance(time, xr.DataArray):
        time = time.indexes["time"]
    elif isinstance(time, np.ndarray):
        time = pd.DatetimeIndex(time)
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


def days_in_year(year: int, calendar: str = "default") -> int:
    """Return the number of days in the input year according to the input calendar."""
    return (
        (datetime_classes[calendar](year + 1, 1, 1) - pydt.timedelta(days=1))
        .timetuple()
        .tm_yday
    )


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

    if b_a == b_b:
        # Same base freq, compare mulitplicator only.
        t_a = int(t_a or "1")
        t_b = int(t_b or "1")
    else:
        # Different base freq, compare length of first period after beginning of time.
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqA)
        t_a = (t[1] - t[0]).total_seconds()
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqB)
        t_b = (t[1] - t[0]).total_seconds()

    return get_op(op)(t_a, t_b)


def parse_offset(freq: str) -> Sequence[str]:
    """Parse an offset string.

    Returns: multiplicator, offset base, start stamp (or None), anchor (or None)
    """
    patt = r"(\d*)(\w)(S)?(?:-(\w{2,3}))?"
    return re.search(patt, freq.replace("Y", "A")).groups()


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
