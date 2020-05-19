# -*- coding: utf-8 -*-
"""
Health checks
=============

Functions performing basic health checks on xarray.DataArrays.
"""
import datetime as dt
import fnmatch

import numpy as np
import pandas as pd
import xarray as xr
from boltons.funcutils import wraps

from .options import cfcheck
from .options import CHECK_MISSING
from .options import datacheck
from .options import MISSING_METHODS
from .options import MISSING_OPTIONS
from .options import OPTIONS
from .options import register_missing_method
from .utils import ValidationError

# Dev notes
# ---------
#
# I suggest we use `check` for weak checking, and `assert` for strong checking.
# Weak checking would log problems in a log, while strong checking would raise an error.
#
# ANSWER to the above:
#   The use of set_options with a switch for raising vs warning vs logging would render these terms useless.
#
# `functools.wraps` is used to copy the docstring and the function's original name from the source
# function to the decorated function. This allows sphinx to correctly find and document functions.


# TODO: Implement pandas infer_freq in xarray with CFTimeIndex. >> PR pydata/xarray#4033
@cfcheck
def check_valid(var, key, expected):
    r"""Check that a variable's attribute has the expected value. Warn user otherwise."""

    att = getattr(var, key, None)
    if att is None:
        raise ValidationError(f"Variable does not have a `{key}` attribute.")
    if not fnmatch.fnmatch(att, expected):
        raise ValidationError(
            f"Variable has a non-conforming {key}. Got `{att}`, expected `{expected}`",
        )


@datacheck
def check_daily(var):
    r"""Assert that the series is daily and monotonic (no jumps in time index).

    A ValueError is raised otherwise."""

    t0, t1 = var.time[:2]

    # This won't work for non-standard calendars. Needs to be implemented in xarray. Comment for now
    if isinstance(t0.values, np.datetime64):
        if pd.infer_freq(var.time.to_pandas()) != "D":
            raise ValidationError("time series is not recognized as daily.")

    # Check that the first time step is one day.
    if np.timedelta64(dt.timedelta(days=1)) != (t1 - t0).data:
        raise ValidationError("time series is not daily.")

    # Check that the series does not go backward in time
    if not var.time.to_pandas().is_monotonic_increasing:
        raise ValidationError("time index is not monotonically increasing.")


def check_valid_temperature(var, units):
    r"""Check that variable is air temperature."""

    check_valid(var, "standard_name", "air_temperature")
    check_valid(var, "units", units)
    check_daily(var)


def check_valid_discharge(var):
    r"""Check that the variable is a discharge."""
    #
    check_valid(var, "standard_name", "water_volume_transport_in_river_channel")
    check_valid(var, "units", "m3 s-1")


def valid_daily_min_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmin, *args, **kwds):
        check_valid_temperature(tasmin, units)
        check_valid(tasmin, "cell_methods", "time: minimum within days")
        return comp(tasmin, **kwds)

    return func


def valid_daily_mean_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tas, *args, **kwds):
        check_valid_temperature(tas, units)
        check_valid(tas, "cell_methods", "time: mean within days")
        return comp(tas, *args, **kwds)

    return func


def valid_daily_max_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmax, *args, **kwds):
        check_valid_temperature(tasmax, units)
        check_valid(tasmax, "cell_methods", "time: maximum within days")
        return comp(tasmax, *args, **kwds)

    return func


def valid_daily_max_min_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on valid min and max temperature datasets."""

    @wraps(comp)
    def func(tasmax, tasmin, **kwds):
        valid_daily_max_temperature(tasmax, units)
        valid_daily_min_temperature(tasmin, units)

        return comp(tasmax, tasmin, **kwds)

    return func


def valid_daily_mean_discharge(comp):
    r"""Decorator to check that a computation runs on valid discharge data."""

    @wraps(comp)
    def func(q, **kwds):
        check_valid_discharge(q)
        return comp(q, **kwds)

    return func


def valid_missing_data_threshold(comp, threshold=0):
    r"""Check that the relative number of missing data points does not exceed a threshold."""
    # TODO
    raise NotImplementedError


def check_is_dataarray(comp):
    r"""Decorator to check that a computation has an instance of xarray.DataArray
     as first argument."""

    @wraps(comp)
    def func(data_array, *args, **kwds):
        assert isinstance(data_array, xr.DataArray)
        return comp(data_array, *args, **kwds)

    return func


# This function can probably be made simpler once CFPeriodIndex is implemented.
class MissingBase:
    def __init__(self, da, freq, **indexer):
        self.null, self.count = self.prepare(da, freq, **indexer)

    @staticmethod
    def split_freq(freq):
        if freq is None:
            return "", None

        if "-" in freq:
            return freq.split("-")

        return freq, None

    @staticmethod
    def is_null(da, freq, **indexer):
        """Return a boolean array indicating which values are null."""
        from xclim.indices import generic

        selected = generic.select_time(da, **indexer)
        if selected.time.size == 0:
            raise ValueError("No data for selected period.")

        null = selected.isnull()
        if freq:
            return null.resample(time=freq)

        return null

    def prepare(self, da, freq, **indexer):
        """Prepare arrays to be fed to the `is_missing` function.

        Parameters
        ----------
        da : xr.DataArray
          Input data.
        freq : str
          Resampling frequency defining the periods defined in
          http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.
        **indexer : {dim: indexer, }, optional
          Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
          values, month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given,
          all values are considered.

        Returns
        -------
        xr.DataArray, xr.DataArray
          Boolean array indicating which values are null, array of expected number of valid values.

        Notes
        -----
        If `freq=None` and an indexer is given, then missing values during period at the start or end of array won't be
        flagged.
        """
        from xclim.indices import generic

        null = self.is_null(da, freq, **indexer)

        pfreq, anchor = self.split_freq(freq)

        c = null.sum(dim="time")

        # Otherwise simply use the start and end dates to find the expected number of days.
        if pfreq.endswith("S"):
            start_time = c.indexes["time"]
            end_time = start_time.shift(1, freq=freq)
        elif pfreq:
            end_time = c.indexes["time"]
            start_time = end_time.shift(-1, freq=freq)
        else:
            i = da.time.to_index()
            start_time = i[:1]
            end_time = i[-1:]

        if indexer:
            # Create a full synthetic time series and compare the number of days with the original series.
            t0 = str(start_time[0].date())
            t1 = str(end_time[-1].date())
            if isinstance(da.indexes["time"], xr.CFTimeIndex):
                cal = da.time.encoding.get("calendar")
                t = xr.cftime_range(t0, t1, freq="D", calendar=cal)
            else:
                t = pd.date_range(t0, t1, freq="D")

            sda = xr.DataArray(data=np.ones(len(t)), coords={"time": t}, dims=("time",))
            st = generic.select_time(sda, **indexer)
            if freq:
                count = st.notnull().resample(time=freq).sum(dim="time")
            else:
                count = st.notnull().sum(dim="time")

        else:
            n = (end_time - start_time).days
            if freq:
                count = xr.DataArray(n.values, coords={"time": c.time}, dims="time")
            else:
                count = xr.DataArray(n.values[0] + 1)

        return null, count

    def is_missing(self, null, count, **kwargs):
        """Return whether or not the values within each period should be considered missing or not."""
        raise NotImplementedError

    @staticmethod
    def validate(**kwargs):
        """Return whether or not arguments are valid."""
        return True

    def __call__(self, **kwargs):
        if not self.validate(**kwargs):
            raise ValueError("Invalid arguments")
        return self.is_missing(self.null, self.count, **kwargs)


@register_missing_method("any")
class MissingAny(MissingBase):
    def is_missing(self, null, count, **kwargs):
        cond0 = null.count(dim="time") != count  # Check total number of days
        cond1 = null.sum(dim="time") > 0  # Check if any is missing
        return cond0 | cond1


@register_missing_method("wmo")
class MissingWMO(MissingAny):
    def __init__(self, da, freq, **indexer):
        # Force computation on monthly frequency
        if not freq.startswith("M"):
            raise ValueError
        super().__init__(da, freq, **indexer)

    def is_missing(self, null, count, nm=11, nc=5):
        import xclim.indices.run_length as rl

        # Check total number of days
        cond0 = null.count(dim="time") != count

        # Check if more than threshold is missing
        cond1 = null.sum(dim="time") >= nm

        # Check for consecutive missing values
        cond2 = null.map(rl.longest_run, dim="time") >= nc

        return cond0 | cond1 | cond2

    @staticmethod
    def validate(nm, nc):
        return nm < 31 and nc < 31


@register_missing_method("pct")
class MissingPct(MissingBase):
    def is_missing(self, null, count, tolerance=0.1):
        if tolerance < 0 or tolerance > 1:
            raise ValueError("tolerance should be between 0 and 1.")

        n = count - null.count(dim="time") + null.sum(dim="time")
        return n / count >= tolerance

    @staticmethod
    def validate(tolerance):
        return 0 <= tolerance <= 1


@register_missing_method("at_least_n")
class AtLeastNValid(MissingBase):
    def is_missing(self, null, count, n=20):
        """The result of a reduction operation is considered missing if less than `n` values are valid."""
        nvalid = null.count(dim="time") - null.sum(dim="time")
        return nvalid < n

    @staticmethod
    def validate(n):
        return n > 0


def missing_any(da, freq, **indexer):
    r"""Return whether there are missing days in the array.

    Parameters
    ----------
    da : DataArray
      Input array at daily frequency.
    freq : str
      Resampling frequency.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    out : DataArray
      A boolean array set to True if period has missing values.
    """
    return MissingAny(da, freq, **indexer)()


def missing_wmo(da, freq, nm=11, nc=5, **indexer):
    r"""Return whether a series fails WMO criteria for missing days.

    The World Meteorological Organisation recommends that where monthly means are computed from daily values,
    it should considered missing if either of these two criteria are met:

      – observations are missing for 11 or more days during the month;
      – observations are missing for a period of 5 or more consecutive days during the month.

    Stricter criteria are sometimes used in practice, with a tolerance of 5 missing values or 3 consecutive missing
    values.

    Parameters
    ----------
    da : DataArray
      Input array at daily frequency.
    freq : str
      Resampling frequency.
    nm : int
      Number of missing values per month that should not be exceeded.
    nc : int
      Number of consecutive missing values per month that should not be exceeded.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    out : DataArray
      A boolean array set to True if period has missing values.
    """
    missing = MissingWMO(da, "M", **indexer)(nm=nm, nc=nc)
    return missing.resample(time=freq).any()


def missing_pct(da, freq, tolerance, **indexer):
    r"""Return whether there are more missing days in the array than a given percentage.

    Parameters
    ----------
    da : DataArray
      Input array at daily frequency.
    freq : str
      Resampling frequency.
    tolerance : float
      Fraction of missing values that is tolerated.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
      values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    out : DataArray
      A boolean array set to True if period has missing values.
    """
    return MissingPct(da, freq, **indexer)(tolerance=tolerance)


def at_least_n_valid(da, freq, n=1, **indexer):
    r"""Return whether there are at least a given number of valid values.

        Parameters
        ----------
        da : DataArray
          Input array at daily frequency.
        freq : str
          Resampling frequency.
        n : int
          Minimum of valid values required.
        **indexer : {dim: indexer, }, optional
          Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
          values, month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given,
          all values are considered.

        Returns
        -------
        out : DataArray
          A boolean array set to True if period has missing values.
        """
    return AtLeastNValid(da, freq, **indexer)(n=n)


def missing_from_context(da, freq, **indexer):
    """Return whether each element of the resampled da should be considered missing according
    to the currently set options in `xclim.set_options`.

    See `xclim.set_options` and `xclim.core.options.register_missing_method`.
    """
    name = OPTIONS[CHECK_MISSING]
    cls = MISSING_METHODS[name]
    opts = OPTIONS[MISSING_OPTIONS][name]

    return cls(da, freq, **indexer)(**opts)
