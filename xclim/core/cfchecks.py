# -*- coding: utf-8 -*-
"""
Health checks
=============

Functions performing basic health checks on xarray.DataArrays.
"""
import fnmatch

import xarray as xr
from boltons.funcutils import wraps

from xclim.core.options import cfcheck
from xclim.core.utils import ValidationError

# Dev notes
# ---------
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


def check_valid_temperature(var, units):
    r"""Check that variable is air temperature."""

    check_valid(var, "standard_name", "air_temperature")
    check_valid(var, "units", units)


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
