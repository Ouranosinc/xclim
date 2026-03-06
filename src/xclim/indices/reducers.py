"""
Reducer Operators Submodule
===========================

Functions that reduce the time axis of a variable to complement the basic functions already available on DataArrays.

Functions defined here should usually take a DataArray as their first argument and
at least a ``dim`` keyword argument to specific which dimension to reduce, defaulting
to ``"time"``. Functions should not handle units.

Functions should be added to :py:data:`XCLIM_OPS`.
"""

from __future__ import annotations

from collections.abc import Callable

import xarray as xr


def doymax(da: xr.DataArray) -> xr.DataArray:
    """
    Return the day of year of the maximum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.

    Returns
    -------
    xr.DataArray
        The day of year of the maximum value.
        If all values are the same, NaN is returned.
    """
    tmax = da.idxmax("time")
    std = da.std("time")
    tmax = tmax.where(std != 0)
    return tmax.dt.dayofyear


def doymin(da: xr.DataArray) -> xr.DataArray:
    """
    Return the day of year of the minimum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.

    Returns
    -------
    xr.DataArray
        The day of year of the minimum value.
        If all values are the same, NaN is returned.
    """
    tmax = da.idxmin("time")
    std = da.std("time")
    tmax = tmax.where(std != 0)
    return tmax.dt.dayofyear


XCLIM_OPS: dict[str, Callable] = {"doymin": doymin, "doymax": doymax}
"""A dictionary of additional time-reducing operations known to xclim."""
