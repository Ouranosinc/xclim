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


def valid_daily_min_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmin, *args, **kwds):
        check_valid_temperature(tasmin, units)
        check_valid(tasmin, 'cell_methods', 'time: minimum within days')
        return comp(tasmin, **kwds)

    return func


def valid_daily_mean_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tas, *args, **kwds):
        check_valid_temperature(tas, units)
        check_valid(tas, 'cell_methods', 'time: mean within days')
        return comp(tas, *args, **kwds)

    return func


def valid_daily_max_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmax, *args, **kwds):
        check_valid_temperature(tasmax, units)
        check_valid(tasmax, 'cell_methods', 'time: maximum within days')
        return comp(tasmax, *args, **kwds)

    return func


def valid_daily_max_min_temperature(comp, units='K'):
    """Decorator to check that a computation runs on valid min and max temperature datasets."""

    @wraps(comp)
    def func(tasmax, tasmin, **kwds):
        valid_daily_max_temperature(tasmax, units)
        valid_daily_min_temperature(tasmin, units)

        return comp(tasmin, tasmax, **kwds)

    return func


def valid_daily_mean_discharge(comp):
    """Decorator to check that a computation runs on valid discharge data."""

    @wraps(comp)
    def func(q, **kwds):
        check_valid_discharge(q)
        return comp(q, **kwds)

    return func


def valid_missing_data_threshold(comp, threshold=0):
    """Check that the relative number of missing data points does not exceed a threshold."""
    # TODO
    pass


def check_is_dataarray(comp):
    """Decorator to check that a computation has an instance of xarray.DataArray
     as first argument."""

    @wraps(comp)
    def func(data_array, *args, **kwds):
        import xarray as xr
        assert (isinstance(data_array, xr.DataArray))
        return comp(data_array, *args, **kwds)

    return func


def convert_temp(da, required_units='K'):
    if da.units == required_units:
        return da
    elif da.units == 'C' and required_units == 'K':
        return da + 273.15
    elif da.units == 'K' and required_units == 'C':
        return da - 273.15
    else:
        raise AttributeError(da.units)
