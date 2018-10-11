import datetime as dt
from functools import wraps
from warnings import warn

import numpy as np
import pandas as pd


# Dev notes
# ---------
#
# I suggest we use `check` for weak checking, and `assert` for strong checking.
# Weak checking would log problems in a log, while strong checking would raise an error.
#
# `functools.wraps` is used to copy the docstring and the function's original name from the source
# function to the decorated function. This allows sphinx to correctly find and document functions.


# TODO: Implement pandas infer_freq in xarray with CFTimeIndex.

def check_valid(var, key, expected):
    """Check that a variable's attribute has the expected value. Warn user otherwise."""
    att = getattr(var, key, None)
    if att is None:
        e = 'Variable does not have a `{}` attribute.'.format(key)
        warn(e)
    elif att != expected:
        e = 'Variable has a non-conforming {}. Got `{}`, expected `{}`'.format(key, att, expected)
        warn(e)


def assert_daily(var):
    """Assert that the series is daily and monotonic (no jumps in time index).

    A ValueError is raised otherwise."""
    t0, t1 = var.time[:2]

    # This won't work for non-standard calendars. Needs to be implemented in xarray.
    if pd.infer_freq(var.time.to_pandas()) != 'D':
        raise ValueError("time series is not recognized as daily.")

    # Check that the first time step is one day.
    if np.timedelta64(dt.timedelta(days=1)) != (t1 - t0).data:
        raise ValueError("time series is not daily.")

    # Check that the series has the same time step throughout
    if not var.time.to_pandas().is_monotonic_increasing:
        raise ValueError("time index is not monotonically increasing.")


def check_valid_temperature(var, units):
    """Check that variable is a temperature."""
    check_valid(var, 'standard_name', 'air_temperature')
    check_valid(var, 'units', units)
    assert_daily(var)


def check_valid_discharge(var):
    """Check that the variable is a discharge."""
    #
    check_valid(var, 'standard_name', 'water_volume_transport_in_river_channel')
    check_valid(var, 'units', 'm3 s-1')


def valid_missing_data_threshold(comp, threshold=0):
    """Check that the relative number of missing data points does not exceed a threshold."""
    # TODO
    pass


def missing_any(da, freq):
    """Return a boolean DataArray indicating whether any group contains missing data."""
    # TODO: Handle cases where the series does not fill an entire month or year.
    return da.isnull().resample(time=freq).any()


def check_is_dataarray(comp):
    """Decorator to check that a computation has an instance of xarray.DataArray
     as first argument."""

    @wraps(comp)
    def func(data_array, *args, **kwds):
        import xarray as xr
        assert (isinstance(data_array, xr.DataArray))
        return comp(data_array, *args, **kwds)

    return func


# TODO: Define a unit conversion system precipitation [mm h-1, Kg m-2 s-1] metrics.
def convert_pr(da, required_units='mm'):
    raise NotImplementedError
