# noqa: D205,D400
"""
Missing values identification
=============================

Indicators may use different criteria to determine whether or not a computed indicator value should be
considered missing. In some cases, the presence of any missing value in the input time series should result in a
missing indicator value for that period. In other cases, a minimum number of valid values or a percentage of missing
values should be enforced. The World Meteorological Organisation  (WMO) suggests criteria based on the number of
consecutive and overall missing values per month.

`xclim` has a registry of missing value detection algorithms that can be extended by users to customize the behavior
of indicators. Once registered, algorithms can be be used within indicators by setting the `missing` attribute of an
`Indicator` subclass. By default, `xclim` registers the following algorithms:

 * `any`: A result is missing if any input value is missing.
 * `at_least_n`: A result is missing if less than a given number of valid values are present.
 * `pct`: A result is missing if more than a given fraction of values are missing.
 * `wmo`: A result is missing if 11 days are missing, or 5 consecutive values are missing in a month.
 * `skip`: Skip missing value detection.
 * `from_context`: Look-up the missing value algorithm from options settings. See :func:`xclim.set_options`.

To define another missing value algorithm, subclass :class:`MissingBase` and decorate it with
`xclim.core.options.register_missing_method`.

"""
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr

from xclim.core.calendar import date_range, get_calendar
from xclim.core.options import (
    CHECK_MISSING,
    MISSING_METHODS,
    MISSING_OPTIONS,
    OPTIONS,
    register_missing_method,
)
from xclim.indices import generic

__all__ = [
    "missing_wmo",
    "missing_any",
    "missing_pct",
    "at_least_n_valid",
    "missing_from_context",
    "register_missing_method",
]

_np_timedelta64 = {"D": "timedelta64[D]", "H": "timedelta64[h]"}


class MissingBase:
    """Base class used to determined where Indicator outputs should be masked.

    Subclasses should implement `is_missing` and `validate` methods.

    Decorate subclasses with `xclim.core.options.register_missing_method` to add them
    to the registry before using them in an Indicator.
    """

    def __init__(self, da, freq, src_timestep, **indexer):
        if src_timestep is None:
            raise ValueError("`src_timestep` must be either 'D', 'H' or 'M'.")
        self.null, self.count = self.prepare(da, freq, src_timestep, **indexer)

    @classmethod
    def execute(cls, da, freq, src_timestep, options, indexer):
        """Create the instance and call it in one operation."""
        obj = cls(da, freq, src_timestep, **indexer)
        return obj(**options)

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
        selected = generic.select_time(da, **indexer)
        if selected.time.size == 0:
            raise ValueError("No data for selected period.")

        null = selected.isnull()
        if freq:
            return null.resample(time=freq)

        return null

    def prepare(self, da, freq, src_timestep, **indexer):
        """Prepare arrays to be fed to the `is_missing` function.

        Parameters
        ----------
        da : xr.DataArray
          Input data.
        freq : str
          Resampling frequency defining the periods defined in
          https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.
        src_timestep : {"D", "H"}
          Expected input frequency.
        **indexer : {dim: indexer}, optional
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
        # This function can probably be made simpler once CFPeriodIndex is implemented.
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

        if indexer or "M" in src_timestep:
            # Create a full synthetic time series and compare the number of days with the original series.
            t = date_range(
                start_time[0],
                end_time[-1],
                freq=src_timestep,
                calendar=get_calendar(da),
            )

            sda = xr.DataArray(data=np.ones(len(t)), coords={"time": t}, dims=("time",))
            st = generic.select_time(sda, **indexer)
            if freq:
                count = st.notnull().resample(time=freq).sum(dim="time")
            else:
                count = st.notnull().sum(dim="time")

        else:
            delta = end_time - start_time
            n = delta.astype(_np_timedelta64[src_timestep])

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
        """Return whether or not options arguments are valid."""
        return True

    def __call__(self, **kwargs):
        if not self.validate(**kwargs):
            raise ValueError("Invalid arguments")
        return self.is_missing(self.null, self.count, **kwargs)


# -----------------------------------------------
# --- Missing value identification algorithms ---
# -----------------------------------------------


@register_missing_method("any")
class MissingAny(MissingBase):
    r"""Return whether there are missing days in the array.

    Parameters
    ----------
    da : DataArray
      Input array.
    freq : str
      Resampling frequency.
    src_timestep : {"D", "H", "M"}
      Expected input frequency.
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

    def is_missing(self, null, count):  # noqa
        cond0 = null.count(dim="time") != count  # Check total number of days
        cond1 = null.sum(dim="time") > 0  # Check if any is missing
        return cond0 | cond1


@register_missing_method("wmo")
class MissingWMO(MissingAny):
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
      Input array.
    freq : str
      Resampling frequency.
    nm : int
      Number of missing values per month that should not be exceeded.
    nc : int
      Number of consecutive missing values per month that should not be exceeded.
    src_timestep : {"D"}
      Expected input frequency. Only daily values are supported.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
      values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    out : DataArray
      A boolean array set to True if period has missing values.

    Notes
    -----
    If used at frequencies larger than a month, for example on an annual or seasonal basis, the function will return
    True if any month within a period is missing.
    """

    def __init__(self, da, freq, src_timestep, **indexer):
        # Force computation on monthly frequency
        if src_timestep != "D":
            raise ValueError(
                "WMO method to estimate missing data is only defined for daily series."
            )

        if not freq.startswith("M"):
            raise ValueError

        super().__init__(da, freq, src_timestep, **indexer)

    @classmethod
    def execute(cls, da, freq, src_timestep, options, indexer):
        """Create the instance and call it in one operation."""
        if freq[0] not in ["Y", "A", "Q", "M"]:
            raise ValueError(
                "MissingWMO can only be used with Monthly or longer frequencies."
            )
        obj = cls(da, "M", src_timestep, **indexer)
        miss = obj(**options)
        # Replace missing months by NaNs
        mda = miss.where(miss == 0)
        return MissingAny(mda, freq, "M", **indexer)()

    def is_missing(self, null, count, nm=11, nc=5):
        from xclim.indices import run_length as rl

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
    r"""Return whether there are more missing days in the array than a given percentage.

    Parameters
    ----------
    da : DataArray
      Input array.
    freq : str
      Resampling frequency.
    tolerance : float
      Fraction of missing values that is tolerated [0,1].
    src_timestep : {"D", "H"}
      Expected input frequency.
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

    def is_missing(self, null, count, tolerance=0.1):
        if tolerance < 0 or tolerance > 1:
            raise ValueError("tolerance should be between 0 and 1.")

        n = count - null.count(dim="time").fillna(0) + null.sum(dim="time").fillna(0)
        return n / count >= tolerance

    @staticmethod
    def validate(tolerance):
        return 0 <= tolerance <= 1


@register_missing_method("at_least_n")
class AtLeastNValid(MissingBase):
    r"""Return whether there are at least a given number of valid values.

    Parameters
    ----------
    da : DataArray
      Input array.
    freq : str
      Resampling frequency.
    n : int
      Minimum of valid values required.
    src_timestep : {"D", "H"}
      Expected input frequency.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
      values, month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given,
      all values are considered.

    Returns
    -------
    out : DataArray
      A boolean array set to True if period has missing values.
    """

    def __init__(self, da, freq, src_timestep, **indexer):
        # No need to compute count, so no check required on `src_timestep`.
        self.null = self.is_null(da, freq, **indexer)
        self.count = None  # Not needed

    def is_missing(self, null, count, n: int = 20):
        """Check for missing results after a reduction operation.

        The result of a reduction operation is considered missing if less than `n` values are valid.
        """
        nvalid = null.count(dim="time").fillna(0) - null.sum(dim="time").fillna(0)
        return nvalid < n

    @staticmethod
    def validate(n: int) -> bool:
        return n > 0


@register_missing_method("skip")
class Skip(MissingBase):
    def __init__(self, da, freq=None, src_timestep=None, **indexer):
        pass

    def is_missing(self, null, count):
        """Return whether or not the values within each period should be considered missing or not."""
        return False

    def __call__(self):
        return False


@register_missing_method("from_context")
class FromContext(MissingBase):
    """Return whether each element of the resampled da should be considered missing according to the currently set options in `xclim.set_options`.

    See `xclim.set_options` and `xclim.core.options.register_missing_method`.
    """

    @classmethod
    def execute(cls, da, freq, src_timestep, options, indexer):

        name = OPTIONS[CHECK_MISSING]
        kls = MISSING_METHODS[name]
        opts = OPTIONS[MISSING_OPTIONS][name]

        return kls.execute(da, freq, src_timestep, opts, indexer)


# --------------------------
# --- Shortcut functions ---
# --------------------------
# These stand-alone functions hide the fact the the algorithms are implemented in a class and make their use more
# user-friendly. This can also be useful for testing.


def missing_any(da, freq, src_timestep=None, **indexer):  # noqa: D103
    src_timestep = src_timestep or xr.infer_freq(da.time)
    return MissingAny(da, freq, src_timestep, **indexer)()


def missing_wmo(da, freq, nm=11, nc=5, src_timestep=None, **indexer):  # noqa: D103
    src_timestep = src_timestep or xr.infer_freq(da.time)
    return MissingWMO.execute(
        da, freq, src_timestep, options=dict(nm=nm, nc=nc), indexer=indexer
    )


def missing_pct(da, freq, tolerance, src_timestep=None, **indexer):  # noqa: D103
    src_timestep = src_timestep or xr.infer_freq(da.time)
    return MissingPct(da, freq, src_timestep, **indexer)(tolerance=tolerance)


def at_least_n_valid(da, freq, n=1, src_timestep=None, **indexer):  # noqa: D103
    src_timestep = src_timestep or xr.infer_freq(da.time)
    return AtLeastNValid(da, freq, src_timestep, **indexer)(n=n)


def missing_from_context(da, freq, src_timestep=None, **indexer):  # noqa: D103
    src_timestep = src_timestep or xr.infer_freq(da.time)
    return FromContext.execute(da, freq, src_timestep, options={}, indexer=indexer)


missing_any.__doc__ = MissingAny.__doc__
missing_wmo.__doc__ = MissingWMO.__doc__
missing_pct.__doc__ = MissingPct.__doc__
at_least_n_valid.__doc__ = AtLeastNValid.__doc__
missing_from_context.__doc__ = FromContext.__doc__
