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

from xclim.core.utils import lazy_indexing, uses_dask


def doymax(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """
    Return the day of year of the maximum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.
    dim : str
        Name of the dimension to reduce.

    Returns
    -------
    xr.DataArray
        The day of year of the maximum value.
    """
    i = da.argmax(dim=dim)
    doy = da.time.dt.dayofyear

    if uses_dask(da):
        out = lazy_indexing(doy, i, dim).astype(doy.dtype)
    else:
        out = doy.isel(time=i)
    return out


def doymin(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """
    Return the day of year of the minimum value.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to process.
    dim : str
        Name of the dimension to reduce.

    Returns
    -------
    xr.DataArray
        The day of year of the minimum value.
    """
    i = da.argmin(dim=dim)
    doy = da.time.dt.dayofyear

    if uses_dask(da):
        out = lazy_indexing(doy, i, dim).astype(doy.dtype)
    else:
        out = doy.isel(time=i)

    return out


XCLIM_OPS: dict[str, Callable] = {"doymin": doymin, "doymax": doymax}
"""A dictionary of additional time-reducing operations known to xclim."""
