"""
Calendar Handling Utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""

from __future__ import annotations

import datetime as pydt
from collections.abc import Sequence
from typing import Any, TypeVar

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import to_cftime_datetime
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes
from xarray.core.resample import DataArrayResample, DatasetResample

from xclim.core.utils import DayOfYearStr, uses_dask

from .formatting import update_xclim_history

__all__ = [
    "DayOfYearStr",
    "adjust_doy_calendar",
    "build_climatology_bounds",
    "climatological_mean_doy",
    "common_calendar",
    "compare_offsets",
    "construct_offset",
    "convert_calendar",
    "convert_doy",
    "date_range",
    "date_range_like",
    "datetime_to_decimal_year",
    "days_in_year",
    "days_since_to_doy",
    "doy_from_string",
    "doy_to_days_since",
    "ensure_cftime_array",
    "get_calendar",
    "interp_calendar",
    "is_offset_divisor",
    "max_doy",
    "parse_offset",
    "percentile_doy",
    "resample_doy",
    "select_time",
    "stack_periods",
    "time_bnds",
    "uniform_calendars",
    "unstack_periods",
    "within_bnds_doy",
    "yearly_interpolated_doy",
    "yearly_random_doy",
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

# Some xclim.core.utils functions made accessible here for backwards compatibility reasons.
datetime_classes = {"default": pydt.datetime, **cftime._cftime.DATE_TYPES}  # noqa

# Names of calendars that have the same number of days for all years
uniform_calendars = ("noleap", "all_leap", "365_day", "366_day", "360_day")


DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)


def days_in_year(year: int, calendar: str = "default") -> int:
    """Return the number of days in the input year according to the input calendar."""
    return (
        (datetime_classes[calendar](year + 1, 1, 1) - pydt.timedelta(days=1))
        .timetuple()
        .tm_yday
    )


def doy_from_string(doy: DayOfYearStr, year: int, calendar: str) -> int:
    """Return the day-of-year corresponding to a "MM-DD" string for a given year and calendar."""
    MM, DD = doy.split("-")
    return datetime_classes[calendar](year, int(MM), int(DD)).timetuple().tm_yday


def date_range(
    *args, calendar: str = "default", **kwargs
) -> pd.DatetimeIndex | CFTimeIndex:
    """Wrap a Pandas date_range object.

    Uses pd.date_range (if calendar == 'default') or xr.cftime_range (otherwise).
    """
    if calendar == "default":
        return pd.date_range(*args, **kwargs)
    return xr.cftime_range(*args, calendar=calendar, **kwargs)


def yearly_interpolated_doy(
    time: pd.DatetimeIndex | CFTimeIndex, source_calendar: str, target_calendar: str
):
    """Return the nearest day in the target calendar of the corresponding "decimal year" in the source calendar."""
    yr = int(time.dt.year[0])
    return np.round(
        days_in_year(yr, target_calendar)
        * time.dt.dayofyear
        / days_in_year(yr, source_calendar)
    ).astype(int)


def yearly_random_doy(
    time: pd.DatetimeIndex | CFTimeIndex,
    rng: np.random.Generator,
    source_calendar: str,
    target_calendar: str,
):
    """Return a day of year in the new calendar.

    Removes Feb 29th and five other days chosen randomly within five sections of 72 days.
    """
    yr = int(time.dt.year[0])
    new_doy = np.arange(360) + 1
    rm_idx = rng.integers(0, 72, 5) + (np.arange(5) * 72)
    if source_calendar == "360_day":
        for idx in rm_idx:
            new_doy[idx + 1 :] = new_doy[idx + 1 :] + 1
        if days_in_year(yr, target_calendar) == 366:
            new_doy[new_doy >= 60] = new_doy[new_doy >= 60] + 1
    elif target_calendar == "360_day":
        new_doy = np.insert(new_doy, rm_idx - np.arange(5), -1)
        if days_in_year(yr, source_calendar) == 366:
            new_doy = np.insert(new_doy, 60, -1)
    return new_doy[time.dt.dayofyear - 1]


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
      Will always return "standard" instead of "gregorian", following CF conventions 1.9.
    """
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        if obj[dim].dtype == "O":
            obj = obj[dim].where(obj[dim].notnull(), drop=True)[0].item()
        elif "datetime64" in obj[dim].dtype.name:
            return "default"
    elif isinstance(obj, xr.CFTimeIndex):
        obj = obj.values[0]
    else:
        obj = np.take(obj, 0)
        # Take zeroth element, overcome cases when arrays or lists are passed.
    if isinstance(obj, pydt.datetime):  # Also covers pandas Timestamp
        return "default"
    if isinstance(obj, cftime.datetime):
        if obj.calendar == "gregorian":
            return "standard"
        return obj.calendar

    raise ValueError(f"Calendar could not be inferred from object of type {type(obj)}.")


def common_calendar(calendars: Sequence[str], join="outer") -> str:
    """Return a calendar common to all calendars from a list.

    Uses the hierarchy: 360_day < noleap < standard < all_leap.
    Returns "default" only if all calendars are "default."

    Parameters
    ----------
    calendars: Sequence of string
      List of calendar names.
    join : {'inner', 'outer'}
      The criterion for the common calendar.

      - 'outer': the common calendar is the smallest calendar (in number of days by year) that will include all the
                 dates of the other calendars.
                 When converting the data to this calendar, no timeseries will lose elements, but some
                 might be missing (gaps or NaNs in the series).
      - 'inner': the common calendar is the smallest calendar of the list.
                 When converting the data to this calendar, no timeseries will have missing elements (no gaps or NaNs),
                 but some might be dropped.

    Examples
    --------
    >>> common_calendar(["360_day", "noleap", "default"], join="outer")
    'standard'
    >>> common_calendar(["360_day", "noleap", "default"], join="inner")
    '360_day'
    """
    if all(cal == "default" for cal in calendars):
        return "default"

    trans = {
        "proleptic_gregorian": "standard",
        "gregorian": "standard",
        "default": "standard",
        "366_day": "all_leap",
        "365_day": "noleap",
        "julian": "standard",
    }
    ranks = {"360_day": 0, "noleap": 1, "standard": 2, "all_leap": 3}
    calendars = sorted([trans.get(cal, cal) for cal in calendars], key=ranks.get)

    if join == "outer":
        return calendars[-1]
    if join == "inner":
        return calendars[0]
    raise NotImplementedError(f"Unknown join criterion `{join}`.")


def _convert_doy_date(doy: int, year: int, src, tgt):
    fracpart = doy - int(doy)
    date = src(year, 1, 1) + pydt.timedelta(days=int(doy - 1))
    try:
        same_date = tgt(date.year, date.month, date.day)
    except ValueError:
        return np.nan
    else:
        if tgt is pydt.datetime:
            return float(same_date.timetuple().tm_yday) + fracpart
        return float(same_date.dayofyr) + fracpart


def convert_doy(
    source: xr.DataArray,
    target_cal: str,
    source_cal: str | None = None,
    align_on: str = "year",
    missing: Any = np.nan,
    dim: str = "time",
) -> xr.DataArray:
    """Convert the calendar of day of year (doy) data.

    Parameters
    ----------
    source : xr.DataArray
      Day of year data (range [1, 366], max depending on the calendar).
    target_cal : str
      Name of the calendar to convert to.
    source_cal : str, optional
      Calendar the doys are in. If not given, uses the "calendar" attribute of `source` or,
      if absent, the calendar of its `dim` axis.
    align_on : {'date', 'year'}
      If 'year' (default), the doy is seen as a "percentage" of the year and is simply rescaled unto the new doy range.
      This always result in floating point data, changing the decimal part of the value.
      if 'date', the doy is seen as a specific date. See notes. This never changes the decimal part of the value.
    missing : Any
      If `align_on` is "date" and the new doy doesn't exist in the new calendar, this value is used.
    dim : str
      Name of the temporal dimension.
    """
    source_cal = source_cal or source.attrs.get("calendar", get_calendar(source[dim]))
    is_calyear = xr.infer_freq(source[dim]) in ("YS-JAN", "Y-DEC", "YE-DEC")

    if is_calyear:  # Fast path
        year_of_the_doy = source[dim].dt.year
    else:  # Doy might refer to a date from the year after the timestamp.
        year_of_the_doy = source[dim].dt.year + 1 * (source < source[dim].dt.dayofyear)

    if align_on == "year":
        if source_cal in ["noleap", "all_leap", "360_day"]:
            max_doy_src = max_doy[source_cal]
        else:
            max_doy_src = xr.apply_ufunc(
                days_in_year,
                year_of_the_doy,
                vectorize=True,
                dask="parallelized",
                kwargs={"calendar": source_cal},
            )
        if target_cal in ["noleap", "all_leap", "360_day"]:
            max_doy_tgt = max_doy[target_cal]
        else:
            max_doy_tgt = xr.apply_ufunc(
                days_in_year,
                year_of_the_doy,
                vectorize=True,
                dask="parallelized",
                kwargs={"calendar": target_cal},
            )
        new_doy = source.copy(data=source * max_doy_tgt / max_doy_src)
    elif align_on == "date":
        new_doy = xr.apply_ufunc(
            _convert_doy_date,
            source,
            year_of_the_doy,
            vectorize=True,
            dask="parallelized",
            kwargs={
                "src": datetime_classes[source_cal],
                "tgt": datetime_classes[target_cal],
            },
        )
    else:
        raise NotImplementedError('"align_on" must be one of "date" or "year".')
    return new_doy.assign_attrs(is_dayofyear=np.int32(1), calendar=target_cal)


def convert_calendar(
    source: xr.DataArray | xr.Dataset,
    target: xr.DataArray | str,
    align_on: str | None = None,
    missing: Any | None = None,
    doy: bool | str = False,
    dim: str = "time",
) -> DataType:
    """Convert a DataArray/Dataset to another calendar using the specified method.

    By default, only converts the individual timestamps, does not modify any data except in dropping invalid/surplus dates or inserting missing dates.

    If the source and target calendars are either no_leap, all_leap or a standard type, only the type of the time array is modified.
    When converting to a leap year from a non-leap year, the 29th of February is removed from the array.
    In the other direction and if `target` is a string, the 29th of February will be missing in the output,
    unless `missing` is specified, in which case that value is inserted.

    For conversions involving `360_day` calendars, see Notes.

    This method is safe to use with sub-daily data as it doesn't touch the time part of the timestamps.

    Parameters
    ----------
    source : xr.DataArray or xr.Dataset
        Input array/dataset with a time coordinate of a valid dtype (datetime64 or a cftime.datetime).
    target : xr.DataArray or str
        Either a calendar name or the 1D time coordinate to convert to.
        If an array is provided, the output will be reindexed using it and in that case, days in `target`
        that are missing in the converted `source` are filled by `missing` (which defaults to NaN).
    align_on : {None, 'date', 'year', 'random'}
        Must be specified when either source or target is a `360_day` calendar, ignored otherwise. See Notes.
    missing : Any, optional
        A value to use for filling in dates in the target that were missing in the source.
        If `target` is a string, default (None) is not to fill values. If it is an array, default is to fill with NaN.
    doy: bool or {'year', 'date'}
        If not False, variables flagged as "dayofyear" (with a `is_dayofyear==1` attribute) are converted to the new calendar too.
        Can be a string, which will be passed as the `align_on` argument of :py:func:`convert_doy`.
        If True, `year` is passed.
    dim : str
        Name of the time coordinate.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Copy of source with the time coordinate converted to the target calendar.
        If `target` is given as an array, the output is reindexed to it, with fill value `missing`.
        If `target` was a string and `missing` was None (default), invalid dates in the new calendar are dropped,
        but missing dates are not inserted.
        If `target` was a string and `missing` was given, then start, end and frequency of the new time axis are
        inferred and the output is reindexed to that a new array.

    Notes
    -----
    If one of the source or target calendars is `360_day`, `align_on` must be specified and two options are offered.

    "year"
      The dates are translated according to their rank in the year (dayofyear), ignoring their original month and day information,
      meaning that the missing/surplus days are added/removed at regular intervals.

      From a `360_day` to a standard calendar, the output will be missing the following dates (day of year in parentheses):
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
      The month/day information is conserved and invalid dates are dropped from the output. This means that when
      converting from a `360_day` to a standard calendar, all 31st (Jan, March, May, July, August, October and December)
      will be missing as there is no equivalent dates in the `360_day` and the 29th (on non-leap years) and 30th of
      February will be dropped as there are no equivalent dates in a standard calendar.

      This option is best used with data on a frequency coarser than daily.

    "random"
      Similar to "year", each day of year of the source is mapped to another day of year of the target. However, instead
      of having always the same missing days according the source and target years, here 5 days are chosen randomly, one
      for each fifth of the year. However, February 29th is always missing when converting to a leap year, or its value
      is dropped when converting from a leap year. This is similar to method used in the
      :cite:t:`pierce_statistical_2014` dataset.

      This option is best used on daily data.

    References
    ----------
    :cite:cts:`pierce_statistical_2014`

    Examples
    --------
    This method does not try to fill the missing dates other than with a constant value, passed with `missing`.
    In order to fill the missing dates with interpolation, one can simply use xarray's method:

    >>> tas_nl = convert_calendar(tas, "noleap")  # For the example
    >>> with_missing = convert_calendar(tas_nl, "standard", missing=np.NaN)
    >>> out = with_missing.interpolate_na("time", method="linear")

    Here, if Nans existed in the source data, they will be interpolated too. If that is,
    for some reason, not wanted, the workaround is to do:

    >>> mask = convert_calendar(tas_nl, "standard").notnull()
    >>> out2 = out.where(mask)
    """
    cal_src = get_calendar(source, dim=dim)

    if isinstance(target, str):
        cal_tgt = target
    else:
        cal_tgt = get_calendar(target, dim=dim)

    if cal_src == cal_tgt:
        return source

    if (cal_src == "360_day" or cal_tgt == "360_day") and align_on not in [
        "year",
        "date",
        "random",
    ]:
        raise ValueError(
            "Argument `align_on` must be specified with either 'date', 'year' or "
            "'random' when converting to or from a '360_day' calendar."
        )
    if cal_src != "360_day" and cal_tgt != "360_day":
        align_on = None

    if doy:
        doy_align_on = "year" if doy is True else doy
        if isinstance(source, xr.DataArray) and source.attrs.get("is_dayofyear") == 1:
            out = convert_doy(source, cal_tgt, align_on=doy_align_on)
        else:
            out = source.map(
                lambda da: (
                    da
                    if da.attrs.get("is_dayofyear") != 1
                    else convert_doy(da, cal_tgt, align_on=doy_align_on)
                )
            )
    else:
        out = source.copy()

    # TODO Maybe the 5-6 days to remove could be given by the user?
    if align_on in ["year", "random"]:
        if align_on == "year":
            new_doy = source.time.groupby(f"{dim}.year").map(
                yearly_interpolated_doy,
                source_calendar=cal_src,
                target_calendar=cal_tgt,
            )
        else:  # align_on == "random"
            new_doy = source.time.groupby(f"{dim}.year").map(
                yearly_random_doy,
                rng=np.random.default_rng(),
                source_calendar=cal_src,
                target_calendar=cal_tgt,
            )

        # Convert the source datetimes, but override the doy with our new doys
        out[dim] = xr.DataArray(
            [
                _convert_datetime(datetime, new_doy=doy, calendar=cal_tgt)
                for datetime, doy in zip(source[dim].indexes[dim], new_doy)
            ],
            dims=(dim,),
            name=dim,
        )
        # Remove NaN that where put on invalid dates in target calendar
        out = out.where(out[dim].notnull(), drop=True)
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
        out = out.reindex(
            {dim: target}, fill_value=missing if missing is not None else np.nan
        )

    # Copy attrs but change remove `calendar` is still present.
    out[dim].attrs.update(source[dim].attrs)
    out[dim].attrs.pop("calendar", None)

    return out


def interp_calendar(
    source: xr.DataArray | xr.Dataset,
    target: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Interpolates a DataArray/Dataset to another calendar based on decimal year measure.

    Each timestamp in source and target are first converted to their decimal year equivalent
    then source is interpolated on the target coordinate. The decimal year is the number of
    years since 0001-01-01 AD.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar or 2000.16301 in a 'noleap' calendar.

    This method should be used with daily data or coarser. Sub-daily result will have a modified day cycle.

    Parameters
    ----------
    source : xr.DataArray or xr.Dataset
        The source data to interpolate, must have a time coordinate of a valid dtype (np.datetime64 or cftime objects)
    target : xr.DataArray
        The target time coordinate of a valid dtype (np.datetime64 or cftime objects)
    dim : str
        The time coordinate name.

    Return
    ------
    xr.DataArray or xr.Dataset
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


def ensure_cftime_array(time: Sequence) -> np.ndarray | Sequence[cftime.datetime]:
    """Convert an input 1D array to a numpy array of cftime objects.

    Python's datetime are converted to cftime.DatetimeGregorian ("standard" calendar).

    Parameters
    ----------
    time : sequence
        A 1D array of datetime-like objects.

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError: When unable to cast the input.
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

    Parameters
    ----------
    times : xr.DataArray
    calendar : str

    Returns
    -------
    xr.DataArray
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
    per: float | Sequence[float] = 10.0,
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    copy: bool = True,
) -> xr.DataArray:
    """Percentile value for each day of the year.

    Return the climatological percentile over a moving window around each day of the year. Different quantile estimators
    can be used by specifying `alpha` and `beta` according to specifications given by :cite:t:`hyndman_sample_1996`.
    The default definition corresponds to method 8, which meets multiple desirable statistical properties for sample
    quantiles. Note that `numpy.percentile` corresponds to method 7, with alpha and beta set to 1.

    Parameters
    ----------
    arr : xr.DataArray
      Input data, a daily frequency (or coarser) is required.
    window : int
      Number of time-steps around each day of the year to include in the calculation.
    per : float or sequence of floats
      Percentile(s) between [0, 100]
    alpha : float
        Plotting position parameter.
    beta : float
        Plotting position parameter.
    copy : bool
        If True (default) the input array will be deep-copied. It's a necessary step
        to keep the data integrity, but it can be costly.
        If False, no copy is made of the input array. It will be mutated and rendered
        unusable but performances may significantly improve.
        Put this flag to False only if you understand the consequences.

    Returns
    -------
    xr.DataArray
        The percentiles indexed by the day of the year.
        For calendars with 366 days, percentiles of doys 1-365 are interpolated to the 1-366 range.

    References
    ----------
    :cite:cts:`hyndman_sample_1996`
    """
    from .utils import calc_perc  # pylint: disable=import-outside-toplevel

    # Ensure arr sampling frequency is daily or coarser
    # but cowardly escape the non-inferrable case.
    if compare_offsets(xr.infer_freq(arr.time) or "D", "<", "D"):
        raise ValueError("input data should have daily or coarser frequency")

    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    crd = xr.Coordinates.from_pandas_multiindex(
        pd.MultiIndex.from_arrays(
            (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
            names=("year", "dayofyear"),
        ),
        "time",
    )
    rr = rr.drop_vars("time").assign_coords(crd)
    rrr = rr.unstack("time").stack(stack_dim=("year", "window"))

    if rrr.chunks is not None and len(rrr.chunks[rrr.get_axis_num("stack_dim")]) > 1:
        # Preserve chunk size
        time_chunks_count = len(arr.chunks[arr.get_axis_num("time")])
        doy_chunk_size = np.ceil(len(rrr.dayofyear) / (window * time_chunks_count))
        rrr = rrr.chunk(dict(stack_dim=-1, dayofyear=doy_chunk_size))

    if np.isscalar(per):
        per = [per]

    p = xr.apply_ufunc(
        calc_perc,
        rrr,
        input_core_dims=[["stack_dim"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(percentiles=per, alpha=alpha, beta=beta, copy=copy),
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
    p.attrs["climatology_bounds"] = build_climatology_bounds(arr)
    p.attrs["window"] = window
    p.attrs["alpha"] = alpha
    p.attrs["beta"] = beta
    return p.rename("per")


def build_climatology_bounds(da: xr.DataArray) -> list[str]:
    """Build the climatology_bounds property with the start and end dates of input data.

    Parameters
    ----------
    da : xr.DataArray
        The input data.
        Must have a time dimension.
    """
    n = len(da.time)
    return da.time[0 :: n - 1].dt.strftime("%Y-%m-%d").values.tolist()


def compare_offsets(freqA: str, op: str, freqB: str) -> bool:  # noqa
    """Compare offsets string based on their approximate length, according to a given operator.

    Offset are compared based on their length approximated for a period starting
    after 1970-01-01 00:00:00. If the offsets are from the same category (same first letter),
    only the multiplier prefix is compared (QS-DEC == QS-JAN, MS < 2MS).
    "Business" offsets are not implemented.

    Parameters
    ----------
    freqA : str
        RHS Date offset string ('YS', '1D', 'QS-DEC', ...)
    op : {'<', '<=', '==', '>', '>=', '!='}
        Operator to use.
    freqB : str
        LHS Date offset string ('YS', '1D', 'QS-DEC', ...)

    Returns
    -------
    bool
        freqA op freqB
    """
    from ..indices.generic import get_op  # pylint: disable=import-outside-toplevel

    # Get multiplier and base frequency
    t_a, b_a, _, _ = parse_offset(freqA)
    t_b, b_b, _, _ = parse_offset(freqB)

    if b_a != b_b:
        # Different base freq, compare length of first period after beginning of time.
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqA)
        t_a = (t[1] - t[0]).total_seconds()
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqB)
        t_b = (t[1] - t[0]).total_seconds()
    # else Same base freq, compare multiplier only.

    return get_op(op)(t_a, t_b)


def parse_offset(freq: str) -> tuple[int, str, bool, str | None]:
    """Parse an offset string.

    Parse a frequency offset and, if needed, convert to cftime-compatible components.

    Parameters
    ----------
    freq : str
      Frequency offset.

    Returns
    -------
    multiplier : int
        Multiplier of the base frequency. "[n]W" is always replaced with "[7n]D",
        as xarray doesn't support "W" for cftime indexes.
    offset_base : str
        Base frequency.
    is_start_anchored : bool
        Whether coordinates of this frequency should correspond to the beginning of the period (`True`)
        or its end (`False`). Can only be False when base is Y, Q or M; in other words, xclim assumes frequencies finer
        than monthly are all start-anchored.
    anchor : str, optional
        Anchor date for bases Y or Q. As xarray doesn't support "W",
        neither does xclim (anchor information is lost when given).

    """
    # Useful to raise on invalid freqs, convert Y to A and get default anchor (A, Q)
    offset = pd.tseries.frequencies.to_offset(freq)
    base, *anchor = offset.name.split("-")
    anchor = anchor[0] if len(anchor) > 0 else None
    start = ("S" in base) or (base[0] not in "AYQM")
    if base.endswith("S") or base.endswith("E"):
        base = base[:-1]
    mult = offset.n
    if base == "W":
        mult = 7 * mult
        base = "D"
        anchor = None
    return mult, base, start, anchor


def construct_offset(mult: int, base: str, start_anchored: bool, anchor: str | None):
    """Reconstruct an offset string from its parts.

    Parameters
    ----------
    mult : int
        The period multiplier (>= 1).
    base : str
        The base period string (one char).
    start_anchored : bool
        If True and base in [Y, Q, M], adds the "S" flag, False add "E".
    anchor : str, optional
        The month anchor of the offset. Defaults to JAN for bases YS and QS and to DEC for bases YE and QE.

    Returns
    -------
    str
      An offset string, conformant to pandas-like naming conventions.

    Notes
    -----
    This provides the mirror opposite functionality of :py:func:`parse_offset`.
    """
    start = ("S" if start_anchored else "E") if base in "YAQM" else ""
    if anchor is None and base in "AQY":
        anchor = "JAN" if start_anchored else "DEC"
    return (
        f"{mult if mult > 1 else ''}{base}{start}{'-' if anchor else ''}{anchor or ''}"
    )


def is_offset_divisor(divisor: str, offset: str):
    """Check that divisor is a divisor of offset.

    A frequency is a "divisor" of another if a whole number of periods of the
    former fit within a single period of the latter.

    Parameters
    ----------
    divisor : str
        The divisor frequency.
    offset: str
        The large frequency.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_offset_divisor("QS-Jan", "YS")
    True
    >>> is_offset_divisor("QS-DEC", "YS-JUL")
    False
    >>> is_offset_divisor("D", "M")
    True
    """
    if compare_offsets(divisor, ">", offset):
        return False
    # Reconstruct offsets anchored at the start of the period
    # to have comparable quantities, also get "offset" objects
    mA, bA, _sA, aA = parse_offset(divisor)
    offAs = pd.tseries.frequencies.to_offset(construct_offset(mA, bA, True, aA))

    mB, bB, _sB, aB = parse_offset(offset)
    offBs = pd.tseries.frequencies.to_offset(construct_offset(mB, bB, True, aB))
    tB = pd.date_range("1970-01-01T00:00:00", freq=offBs, periods=13)

    if bA in ["W", "D", "h", "min", "s", "ms", "us", "ms"] or bB in [
        "W",
        "D",
        "h",
        "min",
        "s",
        "ms",
        "us",
        "ms",
    ]:
        # Simple length comparison is sufficient for submonthly freqs
        # In case one of bA or bB is > W, we test many to be sure.
        tA = pd.date_range("1970-01-01T00:00:00", freq=offAs, periods=13)
        return np.all(
            (np.diff(tB)[:, np.newaxis] / np.diff(tA)[np.newaxis, :]) % 1 == 0
        )

    # else, we test alignment with some real dates
    # If both fall on offAs, then is means divisor is aligned with offset at those dates
    # if N=13 is True, then it is always True
    # As divisor <= offset, this means divisor is a "divisor" of offset.
    return all(offAs.is_on_offset(d) for d in tB)


def _interpolate_doy_calendar(
    source: xr.DataArray, doy_max: int, doy_min: int = 1
) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
        Array with `dayofyear` coordinates.
    doy_max : int
        The largest day of the year allowed by calendar.
    doy_min : int
        The smallest day of the year in the output.
        This parameter is necessary when the target time series does not span over a full year (e.g. JJA season).
        Default is 1.

    Returns
    -------
    xr.DataArray
        Interpolated source array over coordinates spanning the target `dayofyear` range.
    """
    if "dayofyear" not in source.coords.keys():
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Interpolate to fill na values
    da = source
    if uses_dask(source):
        # interpolate_na cannot run on chunked dayofyear.
        da = source.chunk(dict(dayofyear=-1))
    filled_na = da.interpolate_na(dim="dayofyear")

    # Interpolate to target dayofyear range
    filled_na.coords["dayofyear"] = np.linspace(
        start=doy_min, stop=doy_max, num=len(filled_na.coords["dayofyear"])
    )

    return filled_na.interp(dayofyear=range(doy_min, doy_max + 1))


def adjust_doy_calendar(
    source: xr.DataArray, target: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another calendar.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1 to 365).

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
    max_target_doy = int(target.time.dt.dayofyear.max())
    min_target_doy = int(target.time.dt.dayofyear.min())

    def has_same_calendar():
        # case of full year (doys between 1 and 360|365|366)
        return source.dayofyear.max() == max_doy[get_calendar(target)]

    def has_similar_doys():
        # case of partial year (e.g. JJA, doys between 152|153 and 243|244)
        return (
            source.dayofyear.min == min_target_doy
            and source.dayofyear.max == max_target_doy
        )

    if has_same_calendar() or has_similar_doys():
        return source
    return _interpolate_doy_calendar(source, max_target_doy, min_target_doy)


def resample_doy(doy: xr.DataArray, arr: xr.DataArray | xr.Dataset) -> xr.DataArray:
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


def time_bnds(  # noqa: C901
    time: (
        xr.DataArray
        | xr.Dataset
        | CFTimeIndex
        | pd.DatetimeIndex
        | DataArrayResample
        | DatasetResample
    ),
    freq: str | None = None,
    precision: str | None = None,
):
    """Find the time bounds for a datetime index.

    As we are using datetime indices to stand in for period indices, assumptions regarding the period
    are made based on the given freq.

    Parameters
    ----------
    time : DataArray, Dataset, CFTimeIndex, DatetimeIndex, DataArrayResample or DatasetResample
        Object which contains a time index as a proxy representation for a period index.
    freq : str, optional
        String specifying the frequency/offset such as 'MS', '2D', or '3min'
        If not given, it is inferred from the time index, which means that index must
        have at least three elements.
    precision : str, optional
        A timedelta representation that :py:class:`pandas.Timedelta` understands.
        The time bounds will be correct up to that precision. If not given,
        1 ms ("1U") is used for CFtime indexes and 1 ns ("1N") for numpy datetime64 indexes.

    Returns
    -------
    DataArray
        The time bounds: start and end times of the periods inferred from the time index and a frequency.
        It has the original time index along it's `time` coordinate and a new `bnds` coordinate.
        The dtype and calendar of the array are the same as the index.

    Notes
    -----
    xclim assumes that indexes for greater-than-day frequencies are "floored" down to a daily resolution.
    For example, the coordinate "2000-01-31 00:00:00" with a "ME" frequency is assumed to mean a period
    going from "2000-01-01 00:00:00" to "2000-01-31 23:59:59.999999".

    Similarly, it assumes that daily and finer frequencies yield indexes pointing to the period's start.
    So "2000-01-31 00:00:00" with a "3h" frequency, means a period going from "2000-01-31 00:00:00" to
    "2000-01-31 02:59:59.999999".
    """
    if isinstance(time, (xr.DataArray, xr.Dataset)):
        time = time.indexes[time.name]
    elif isinstance(time, (DataArrayResample, DatasetResample)):
        for grouper in time.groupers:
            if "time" in grouper.dims:
                datetime = grouper.unique_coord.data
                freq = freq or grouper.grouper.freq
                if datetime.dtype == "O":
                    time = xr.CFTimeIndex(datetime)
                else:
                    time = pd.DatetimeIndex(datetime)
                break

        else:
            raise ValueError(
                'Got object resampled along another dimension than "time".'
            )

    if freq is None and hasattr(time, "freq"):
        freq = time.freq
    if freq is None:
        freq = xr.infer_freq(time)
    elif hasattr(freq, "freqstr"):
        # When freq is a Offset
        freq = freq.freqstr

    freq_base, freq_is_start = parse_offset(freq)[1:3]

    # Normalizing without using `.normalize` because cftime doesn't have it
    floor = {"hour": 0, "minute": 0, "second": 0, "microsecond": 0, "nanosecond": 0}
    if freq_base in ["h", "min", "s", "ms", "us", "ns"]:
        floor.pop("hour")
    if freq_base in ["min", "s", "ms", "us", "ns"]:
        floor.pop("minute")
    if freq_base in ["s", "ms", "us", "ns"]:
        floor.pop("second")
    if freq_base in ["us", "ns"]:
        floor.pop("microsecond")
    if freq_base == "ns":
        floor.pop("nanosecond")

    if isinstance(time, xr.CFTimeIndex):
        period = xr.coding.cftime_offsets.to_offset(freq)
        is_on_offset = period.onOffset
        eps = pd.Timedelta(precision or "1us").to_pytimedelta()
        day = pd.Timedelta("1D").to_pytimedelta()
        floor.pop("nanosecond")  # unsupported by cftime
    else:
        period = pd.tseries.frequencies.to_offset(freq)
        is_on_offset = period.is_on_offset
        eps = pd.Timedelta(precision or "1ns")
        day = pd.Timedelta("1D")

    def shift_time(t):
        if not is_on_offset(t):
            if freq_is_start:
                t = period.rollback(t)
            else:
                t = period.rollforward(t)
        return t.replace(**floor)

    time_real = list(map(shift_time, time))

    cls = time.__class__
    if freq_is_start:
        tbnds = [cls(time_real), cls([t + period - eps for t in time_real])]
    else:
        tbnds = [
            cls([t - period + day for t in time_real]),
            cls([t + day - eps for t in time_real]),
        ]
    return xr.DataArray(
        tbnds, dims=("bnds", "time"), coords={"time": time}, name="time_bnds"
    ).transpose()


def climatological_mean_doy(
    arr: xr.DataArray, window: int = 5
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate the climatological mean and standard deviation for each day of the year.

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
    """Return whether array values are within bounds for each day of the year.

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
    base: xr.DataArray, start: DayOfYearStr | None = None
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculate dayofyear to days since, or the inverse.

    Parameters
    ----------
    base : xr.DataArray
        1D time coordinate.
    start : DayOfYearStr, optional
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
        days_in_year, base.dt.year, vectorize=True, kwargs={"calendar": calendar}
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
    start: DayOfYearStr | None = None,
    calendar: str | None = None,
) -> xr.DataArray:
    """Convert day-of-year data to days since a given date.

    This is useful for computing meaningful statistics on doy data.

    Parameters
    ----------
    da : xr.DataArray
        Array of "day-of-year", usually int dtype, must have a `time` dimension.
        Sampling frequency should be finer or similar to yearly and coarser than daily.
    start : date of year str, optional
        A date in "MM-DD" format, the base day of the new array. If None (default), the `time` axis is used.
        Passing `start` only makes sense if `da` has a yearly sampling frequency.
    calendar : str, optional
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
    >>> time = date_range("2020-07-01", "2021-07-01", freq="AS-JUL")
    >>> # July 8th 2020 and Jan 2nd 2022
    >>> da = DataArray([190, 2], dims=("time",), coords={"time": time})
    >>> # Convert to days since Oct. 2nd, of the data's year.
    >>> doy_to_days_since(da, start="10-02").values
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
    start: DayOfYearStr | None = None,
    calendar: str | None = None,
) -> xr.DataArray:
    """Reverse the conversion made by :py:func:`doy_to_days_since`.

    Converts data given in days since a specific date to day-of-year.

    Parameters
    ----------
    da : xr.DataArray
        The result of :py:func:`doy_to_days_since`.
    start : DateOfYearStr, optional
        `da` is considered as days since that start date (in the year of the time index).
        If None (default), it is read from the attributes.
    calendar : str, optional
        Calendar the "days since" were computed in.
        If None (default), it is read from the attributes.

    Returns
    -------
    xr.DataArray
      Same shape as `da`, values as `day of year`.

    Examples
    --------
    >>> from xarray import DataArray
    >>> time = date_range("2020-07-01", "2021-07-01", freq="AS-JUL")
    >>> da = DataArray(
    ...     [-86, 92],
    ...     dims=("time",),
    ...     coords={"time": time},
    ...     attrs={"units": "days since 10-02"},
    ... )
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
    datetime: pydt.datetime | cftime.datetime,
    new_doy: float | int | None = None,
    calendar: str = "default",
) -> cftime.datetime | pydt.datetime | float:
    """Convert a datetime object to another calendar.

    Nanosecond information are lost as cftime.datetime doesn't support them.

    Parameters
    ----------
    datetime : datetime.datetime or cftime.datetime
        A datetime object to convert.
    new_doy : float or int, optional
        Allows for redefining the day of year (thus ignoring month and day information from the source datetime).
        -1 is understood as a nan.
    calendar : str
        The target calendar

    Returns
    -------
    Union[cftime.datetime, datetime.datetime, np.nan]
        A datetime object of the target calendar with the same year, month, day and time as the source
        (month and day according to `new_doy` if given).
        If the month and day doesn't exist in the target calendar, returns np.nan. (Ex. 02-29 in "noleap")
    """
    if new_doy in [np.nan, -1]:
        return np.nan
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


def select_time(
    da: xr.DataArray | xr.Dataset,
    drop: bool = False,
    season: str | Sequence[str] | None = None,
    month: int | Sequence[int] | None = None,
    doy_bounds: tuple[int, int] | None = None,
    date_bounds: tuple[str, str] | None = None,
    include_bounds: bool | tuple[bool, bool] = True,
) -> DataType:
    """Select entries according to a time period.

    This conveniently improves xarray's :py:meth:`xarray.DataArray.where` and
    :py:meth:`xarray.DataArray.sel` with fancier ways of indexing over time elements.
    In addition to the data `da` and argument `drop`, only one of `season`, `month`,
    `doy_bounds` or `date_bounds` may be passed.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input data.
    drop : bool
        Whether to drop elements outside the period of interest or to simply mask them (default).
    season : string or sequence of strings, optional
        One or more of 'DJF', 'MAM', 'JJA' and 'SON'.
    month : integer or sequence of integers, optional
        Sequence of month numbers (January = 1 ... December = 12)
    doy_bounds : 2-tuple of integers, optional
        The bounds as (start, end) of the period of interest expressed in day-of-year, integers going from
        1 (January 1st) to 365 or 366 (December 31st).
        If calendar awareness is needed, consider using ``date_bounds`` instead.
    date_bounds : 2-tuple of strings, optional
        The bounds as (start, end) of the period of interest expressed as dates in the month-day (%m-%d) format.
    include_bounds : bool or 2-tuple of booleans
        Whether the bounds of `doy_bounds` or `date_bounds` should be inclusive or not.
        Either one value for both or a tuple. Default is True, meaning bounds are inclusive.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Selected input values. If ``drop=False``, this has the same length as ``da`` (along dimension 'time'),
        but with masked (NaN) values outside the period of interest.

    Examples
    --------
    Keep only the values of fall and spring.

    >>> ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    >>> ds.time.size
    1461
    >>> out = select_time(ds, drop=True, season=["MAM", "SON"])
    >>> out.time.size
    732

    Or all values between two dates (included).

    >>> out = select_time(ds, drop=True, date_bounds=("02-29", "03-02"))
    >>> out.time.values
    array(['1990-03-01T00:00:00.000000000', '1990-03-02T00:00:00.000000000',
           '1991-03-01T00:00:00.000000000', '1991-03-02T00:00:00.000000000',
           '1992-02-29T00:00:00.000000000', '1992-03-01T00:00:00.000000000',
           '1992-03-02T00:00:00.000000000', '1993-03-01T00:00:00.000000000',
           '1993-03-02T00:00:00.000000000'], dtype='datetime64[ns]')
    """
    N = sum(arg is not None for arg in [season, month, doy_bounds, date_bounds])
    if N > 1:
        raise ValueError(f"Only one method of indexing may be given, got {N}.")

    if N == 0:
        return da

    def _get_doys(_start, _end, _inclusive):
        if _start <= _end:
            _doys = np.arange(_start, _end + 1)
        else:
            _doys = np.concatenate((np.arange(_start, 367), np.arange(0, _end + 1)))
        if not _inclusive[0]:
            _doys = _doys[1:]
        if not _inclusive[1]:
            _doys = _doys[:-1]
        return _doys

    if isinstance(include_bounds, bool):
        include_bounds = (include_bounds, include_bounds)

    if season is not None:
        if isinstance(season, str):
            season = [season]
        mask = da.time.dt.season.isin(season)

    elif month is not None:
        if isinstance(month, int):
            month = [month]
        mask = da.time.dt.month.isin(month)

    elif doy_bounds is not None:
        mask = da.time.dt.dayofyear.isin(_get_doys(*doy_bounds, include_bounds))

    elif date_bounds is not None:
        # This one is a bit trickier.
        start, end = date_bounds
        time = da.time
        calendar = get_calendar(time)
        if calendar not in uniform_calendars:
            # For non-uniform calendars, we can't simply convert dates to doys
            # conversion to all_leap is safe for all non-uniform calendar as it doesn't remove any date.
            time = convert_calendar(time, "all_leap")
            # values of time are the _old_ calendar
            # and the new calendar is in the coordinate
            calendar = "all_leap"

        # Get doy of date, this is now safe because the calendar is uniform.
        doys = _get_doys(
            to_cftime_datetime(f"2000-{start}", calendar).dayofyr,
            to_cftime_datetime(f"2000-{end}", calendar).dayofyr,
            include_bounds,
        )
        mask = time.time.dt.dayofyear.isin(doys)
        # Needed if we converted calendar, this puts back the correct coord
        mask["time"] = da.time

    else:
        raise ValueError(
            "Must provide either `season`, `month`, `doy_bounds` or `date_bounds`."
        )

    return da.where(mask, drop=drop)


def _month_is_first_period_month(time, freq):
    """Returns True if the given time is from the first month of freq."""
    if isinstance(time, cftime.datetime):
        frq_monthly = xr.coding.cftime_offsets.to_offset("MS")
        frq = xr.coding.cftime_offsets.to_offset(freq)
        if frq_monthly.onOffset(time):
            return frq.onOffset(time)
        return frq.onOffset(frq_monthly.rollback(time))
    # Pandas
    time = pd.Timestamp(time)
    frq_monthly = pd.tseries.frequencies.to_offset("MS")
    frq = pd.tseries.frequencies.to_offset(freq)
    if frq_monthly.is_on_offset(time):
        return frq.is_on_offset(time)
    return frq.is_on_offset(frq_monthly.rollback(time))


def stack_periods(
    da: xr.Dataset | xr.DataArray,
    window: int = 30,
    stride: int | None = None,
    min_length: int | None = None,
    freq: str = "YS",
    dim: str = "period",
    start: str = "1970-01-01",
    align_days: bool = True,
    pad_value=dtypes.NA,
):
    """Construct a multi-period array.

    Stack different equal-length periods of `da` into a new 'period' dimension.

    This is similar to ``da.rolling(time=window).construct(dim, stride=stride)``, but adapted for arguments
    in terms of a base temporal frequency that might be non-uniform (years, months, etc.).
    It is reversible for some cases (see `stride`).
    A rolling-construct method will be much more performant for uniform periods (days, weeks).

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        An xarray object with a `time` dimension.
        Must have a uniform timestep length.
        Output might be strange if this does not use a uniform calendar (noleap, 360_day, all_leap).
    window : int
        The length of the moving window as a multiple of ``freq``.
    stride : int, optional
        At which interval to take the windows, as a multiple of ``freq``.
        For the operation to be reversible with :py:func:`unstack_periods`, it must divide `window` into an odd number of parts.
        Default is `window` (no overlap between periods).
    min_length : int, optional
        Windows shorter than this are not included in the output.
        Given as a multiple of ``freq``. Default is ``window`` (every window must be complete).
        Similar to the ``min_periods`` argument of  ``da.rolling``.
        If ``freq`` is annual or quarterly and ``min_length == ``window``, the first period is considered complete
        if the first timestep is in the first month of the period.
    freq : str
        Units of ``window``, ``stride`` and ``min_length``, as a frequency string.
        Must be larger or equal to the data's sampling frequency.
        Note that this function offers an easier interface for non-uniform period (like years or months)
        but is much slower than a rolling-construct method.
    dim : str
        The new dimension name.
    start : str
        The `start` argument passed to :py:func:`xarray.date_range` to generate the new placeholder
        time coordinate.
    align_days : bool
        When True (default), an error is raised if the output would have unaligned days across periods.
        If `freq = 'YS'`, day-of-year alignment is checked and if `freq` is "MS" or "QS", we check day-in-month.
        Only uniform-calendar will pass the test for `freq='YS'`.
        For other frequencies, only the `360_day` calendar will work.
        This check is ignored if the sampling rate of the data is coarser than "D".
    pad_value : Any
        When some periods are shorter than others, this value is used to pad them at the end.
        Passed directly as argument ``fill_value`` to :py:func:`xarray.concat`,
        the default is the same as on that function.

    Return
    ------
    xr.DataArray
        A DataArray with a new `period` dimension and a `time` dimension with the length of the longest window.
        The new time coordinate has the same frequency as the input data but is generated using
        :py:func:`xarray.date_range` with the given `start` value.
        That coordinate is the same for all periods, depending on the choice of ``window`` and ``freq``, it might make sense.
        But for unequal periods or non-uniform calendars, it will certainly not.
        If ``stride`` is a divisor of ``window``, the correct timeseries can be reconstructed with :py:func:`unstack_periods`.
        The coordinate of `period` is the first timestep of each window.
    """
    from xclim.core.units import (  # Import in function to avoid cyclical imports
        ensure_cf_units,
        infer_sampling_units,
    )

    stride = stride or window
    min_length = min_length or window
    if stride > window:
        raise ValueError(
            f"Stride must be less than or equal to window. Got {stride} > {window}."
        )

    srcfreq = xr.infer_freq(da.time)
    cal = da.time.dt.calendar
    use_cftime = da.time.dtype == "O"

    if (
        compare_offsets(srcfreq, "<=", "D")
        and align_days
        and (
            (freq.startswith(("Y", "A")) and cal not in uniform_calendars)
            or (freq.startswith(("Q", "M")) and window > 1 and cal != "360_day")
        )
    ):
        if freq.startswith(("Y", "A")):
            u = "year"
        else:
            u = "month"
        raise ValueError(
            f"Stacking {window}{freq} periods will result in unaligned day-of-{u}. "
            f"Consider converting the calendar of your data to one with uniform {u} lengths, "
            "or pass `align_days=False` to disable this check."
        )

    # Convert integer inputs to freq strings
    mult, *args = parse_offset(freq)
    win_frq = construct_offset(mult * window, *args)
    strd_frq = construct_offset(mult * stride, *args)
    minl_frq = construct_offset(mult * min_length, *args)

    # The same time coord as da, but with one extra element.
    # This way, the last window's last index is not returned as None by xarray's grouper.
    time2 = xr.DataArray(
        xr.date_range(
            da.time[0].item(),
            freq=srcfreq,
            calendar=cal,
            periods=da.time.size + 1,
            use_cftime=use_cftime,
        ),
        dims=("time",),
        name="time",
    )

    periods = []
    # longest = 0
    # Iterate over strides, but recompute the full window for each stride start
    for _, strd_slc in da.resample(time=strd_frq).groups.items():
        win_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(time=win_frq)
        # Get slice for first group
        win_slc = win_resamp._group_indices[0]
        if min_length < window:
            # If we ask for a min_length period instead is it complete ?
            min_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(
                time=minl_frq
            )
            min_slc = min_resamp._group_indices[0]
            open_ended = min_slc.stop is None
        else:
            # The end of the group slice is None if no outside-group value was found after the last element
            # As we added an extra step to time2, we avoid the case where a group ends exactly on the last element of ds
            open_ended = win_slc.stop is None
        if open_ended:
            # Too short, we got to the end
            break
        if (
            strd_slc.start == 0
            and parse_offset(freq)[1] in "YAQ"
            and min_length == window
            and not _month_is_first_period_month(da.time[0].item(), freq)
        ):
            # For annual or quarterly frequencies (which can be anchor-based),
            # if the first time is not in the first month of the first period,
            # then the first period is incomplete but by a fractional amount.
            continue
        periods.append(
            slice(
                strd_slc.start + win_slc.start,
                (
                    (strd_slc.start + win_slc.stop)
                    if win_slc.stop is not None
                    else da.time.size
                ),
            )
        )

    # Make coordinates
    lengths = xr.DataArray(
        [slc.stop - slc.start for slc in periods],
        dims=(dim,),
        attrs={"long_name": "Length of each period"},
    )
    longest = lengths.max().item()
    # Length as a pint-ready array : with proper units, but values are not usable as indexes anymore
    m, u = infer_sampling_units(da)
    lengths = lengths * m
    lengths.attrs["units"] = ensure_cf_units(u)
    # Start points for each period and remember parameters for unstacking
    starts = xr.DataArray(
        [da.time[slc.start].item() for slc in periods],
        dims=(dim,),
        attrs={
            "long_name": "Start of the period",
            # Save parameters so that we can unstack.
            "window": window,
            "stride": stride,
            "freq": freq,
            "unequal_lengths": int(len(np.unique(lengths)) > 1),
        },
    )
    # The "fake" axis that all periods share
    fake_time = xr.date_range(
        start, periods=longest, freq=srcfreq, calendar=cal, use_cftime=use_cftime
    )
    # Slice and concat along new dim. We drop the index and add a new one so that xarray can concat them together.
    out = xr.concat(
        [
            da.isel(time=slc)
            .drop_vars("time")
            .assign_coords(time=np.arange(slc.stop - slc.start))
            for slc in periods
        ],
        dim,
        join="outer",
        fill_value=pad_value,
    )
    out = out.assign_coords(
        time=(("time",), fake_time, da.time.attrs.copy()),
        **{f"{dim}_length": lengths, dim: starts},
    )
    out.time.attrs.update(long_name="Placeholder time axis")
    return out


def unstack_periods(da: xr.DataArray | xr.Dataset, dim: str = "period"):
    """Unstack an array constructed with :py:func:`stack_periods`.

    Can only work with periods stacked with a ``stride`` that divides ``window`` in an odd number of sections.
    When ``stride`` is smaller than ``window``, only the center-most stride of each window is kept,
    except for the beginning and end which are taken from the first and last windows.

    Parameters
    ----------
    da : xr.DataArray
        As constructed by :py:func:`stack_periods`, attributes of the period coordinates must have been preserved.
    dim : str
        The period dimension name.

    Notes
    -----
    The following table shows which strides are included (``o``) in the unstacked output.

    In this example, ``stride`` was a fifth of ``window`` and ``min_length`` was four (4) times ``stride``.
    The row index ``i`` the period index in the stacked dataset,
    columns are the stride-long section of the original timeseries.

    .. table:: Unstacking example with ``stride < window``.

        === === === === === === === ===
         i   0   1   2   3   4   5   6
        === === === === === === === ===
         3               x   x   o   o
         2           x   x   o   x   x
         1       x   x   o   x   x
         0   o   o   o   x   x
        === === === === === === === ===
    """
    from xclim.core.units import infer_sampling_units

    try:
        starts = da[dim]
        window = starts.attrs["window"]
        stride = starts.attrs["stride"]
        freq = starts.attrs["freq"]
        unequal_lengths = bool(starts.attrs["unequal_lengths"])
    except (AttributeError, KeyError) as err:
        raise ValueError(
            f"`unstack_periods` can't find the window, stride and freq attributes on the {dim} coordinates."
        ) from err

    if unequal_lengths:
        try:
            lengths = da[f"{dim}_length"]
        except KeyError as err:
            raise ValueError(
                f"`unstack_periods` can't find the `{dim}_length` coordinate."
            ) from err
        # Get length as number of points
        m, _ = infer_sampling_units(da.time)
        lengths = lengths // m
    else:
        # It is acceptable to lose "{dim}_length" if they were all equal
        lengths = xr.DataArray([da.time.size] * da[dim].size, dims=(dim,))

    # Convert from the fake axis to the real one
    time_as_delta = da.time - da.time[0]
    if da.time.dtype == "O":
        # cftime can't add with np.timedelta64 (restriction comes from numpy which refuses to add O with m8)
        time_as_delta = pd.TimedeltaIndex(
            time_as_delta
        ).to_pytimedelta()  # this array is O, numpy complies
    else:
        # Xarray will return int when iterating over datetime values, this returns timestamps
        starts = pd.DatetimeIndex(starts)

    def _reconstruct_time(_time_as_delta, _start):
        times = _time_as_delta + _start
        return xr.DataArray(times, dims=("time",), coords={"time": times}, name="time")

    # Easy case:
    if window == stride:
        # just concat them all
        periods = []
        for i, (start, length) in enumerate(zip(starts.values, lengths.values)):
            real_time = _reconstruct_time(time_as_delta, start)
            periods.append(
                da.isel(**{dim: i}, drop=True)
                .isel(time=slice(0, length))
                .assign_coords(time=real_time.isel(time=slice(0, length)))
            )
        return xr.concat(periods, "time")

    # Difficult and ambiguous case
    if (window / stride) % 2 != 1:
        raise NotImplementedError(
            "`unstack_periods` can't work with strides that do not divide the window into an odd number of parts."
            f"Got {window} / {stride} which is not an odd integer."
        )

    # Non-ambiguous overlapping case
    Nwin = window // stride
    mid = (Nwin - 1) // 2  # index of the center window

    mult, *args = parse_offset(freq)
    strd_frq = construct_offset(mult * stride, *args)

    periods = []
    for i, (start, length) in enumerate(zip(starts.values, lengths.values)):
        real_time = _reconstruct_time(time_as_delta, start)
        slices = real_time.resample(time=strd_frq)._group_indices
        if i == 0:
            slc = slice(slices[0].start, min(slices[mid].stop, length))
        elif i == da.period.size - 1:
            slc = slice(slices[mid].start, min(slices[Nwin - 1].stop or length, length))
        else:
            slc = slice(slices[mid].start, min(slices[mid].stop, length))
        periods.append(
            da.isel(**{dim: i}, drop=True)
            .isel(time=slc)
            .assign_coords(time=real_time.isel(time=slc))
        )

    return xr.concat(periods, "time")
