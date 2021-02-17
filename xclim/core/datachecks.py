# noqa: D205,D400
"""
Data checks
===========

Utilities designed to check the validity of data inputs.
"""
import xarray as xr

from ..indices.run_length import suspicious_run
from .calendar import clim_mean_doy, within_bnds_doy
from .options import datacheck
from .utils import ValidationError


@datacheck
def check_freq(var, freq, strict=True):
    """Raise an error if not series has not the expected temporal frequency or is not monotonically increasing.

    Parameters
    ----------
    var : xr.DataArray
      Input array.
    freq : str
      The temporal frequency defined using the Pandas frequency strings, e.g. 'A', 'M', 'D', 'H', 'T',
      'S'. Note that a 3-hourly time series is declared as '3H'.
    strict : bool
      Whether or not multiples of the frequency are considered invalid. With `strict` set to False, a '3H' series
      will not raise an error if freq is set to 'H'.
    """
    v_freq = xr.infer_freq(var.time)
    if v_freq != freq:
        if (freq in v_freq) and not strict:
            return
        raise ValidationError(
            "Time series has temporal frequency `{v_freq}`, expected `{freq}`."
        )


@datacheck
def check_daily(var):
    """Raise an error if not series has a frequency other that daily, or is not monotonically increasing.

    Note that this does not check for gaps in the series.
    """
    if xr.infer_freq(var.time) != "D":
        raise ValidationError("time series is not recognized as daily.")


# TODO: Migrated from Data Quality Assurance Checks
def flag(arr, conditions):
    for msg, func in conditions.items():
        if func(arr):
            UserWarning(msg)


# TODO: Migrated from Data Quality Assurance Checks
def outside_climatology(arr, n=5):
    """Check if any value is outside `n` standard deviations from the day of year mean."""
    mu, sig = clim_mean_doy(arr, window=5)
    return not within_bnds_doy(arr, mu + n * sig, mu - n * sig).all()


# TODO: Migrated from Data Quality Assurance Checks
def icclim_precipitation_flags():
    """Return a dictionary of conditions that would flag a suspicious precipitation time series."""
    conditions = {
        "Contains negative values": lambda x: (x < 0).any(),
        "Values too large (> 300mm)": lambda x: (x > 300).any(),
        "Many 1mm repetitions": lambda x: suspicious_run(x, window=10, thresh=1.0),
        "Many 5mm repetitions": lambda x: suspicious_run(x, window=5, thresh=5.0)
        # TODO: Create a check for dry days in precipitation runs
        # . . . dry periods receive flag = 1 (suspect), if the amount of dry days lies
        # outside a 14·bivariate standard deviation ?
        # 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation
    }

    return conditions


# TODO: Migrated from Data Quality Assurance Checks
def icclim_tasmean_flags():
    """Return a dictionary of conditions that would flag a suspicious tas time series."""
    conditions = {
        "Extremely low (< -90℃)": lambda x: (x < -90).any(),
        "Extremely high (> 60℃)": lambda x: (x > 60).any(),
        "Exceeds maximum temperature": lambda x, tasmax: (x > tasmax).any(),
        "Below minimum temperature": lambda x, tasmin: (x < tasmin).any(),
        "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
        "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
    }

    return conditions


# TODO: Migrated from Data Quality Assurance Checks
def icclim_tasmax_flags():
    """Return a dictionary of conditions that would flag a suspicious tasmax time series."""
    conditions = {
        "Extremely low (< -90℃)": lambda x: (x < -90).any(),
        "Extremely high (> 60℃)": lambda x: (x > 60).any(),
        "Below mean temperature": lambda x, tas: (x < tas).any(),
        "Below minimum temperature": lambda x, tasmin: (x < tasmin).any(),
        "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
        "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
    }

    return conditions


# TODO: Migrated from Data Quality Assurance Checks
def icclim_tasmin_flags():
    """Return a dictionary of conditions that would flag a suspicious tasmin time series."""
    conditions = {
        "Extremely low (< -90℃)": lambda x: (x < -90).any(),
        "Extremely high (> 60℃)": lambda x: (x > 60).any(),
        "Exceeds maximum temperature": lambda x, tasmax: (x > tasmax).any(),
        "Exceeds mean temperature": lambda x, tas: (x > tas).any(),
        "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
        "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
    }

    return conditions
