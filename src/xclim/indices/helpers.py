"""
Indices Helper Functions Submodule
==================================

Functions that encapsulate logic and can be shared by many indices,
but are not particularly index-like themselves (those should go in the :py:mod:`xclim.indices.generic` module).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from inspect import stack
from typing import Literal

import numpy as np
import xarray as xr

try:
    from flox.xarray import rechunk_for_blockwise

    flox_err = None
except ImportError:
    rechunk_for_blockwise = None

import pandas as pd

from xclim.core.options import MAP_BLOCKS, OPTIONS
from xclim.core.units import convert_units_to
from xclim.core.utils import sel_with_nans, uses_dask

__all__ = ["accumulate_between_times", "interpolate_to_time", "resample_map", "wind_speed_height_conversion"]


def _get_dt(freq: str):
    """
    Get the time delta, in seconds for a given pandas frequency. Only valid for freq <= 'D'.

    Parameters
    ----------
    freq : str
        Pandas time frequency.

    Returns
    -------
    float
        Total seconds between two timestamps with this frequency.
    """
    return pd.date_range(freq=freq, periods=2, start="2000-01-01").diff()[1].total_seconds()


def accumulate_between_times(
    da: xr.DataArray, prev_time: xr.DataArray, curr_time: xr.DataArray, freq: str | None = None
) -> xr.DataArray:
    """
    Accumulate (sum) between the given time DataArray (usually solar noon yesterday and solar noon today).

    Parameters
    ----------
    da : xr.DataArray
        DataArray with variable `var` to accumulate.
    prev_time : xr.DataArray
        Time occurrence of the previous event (indexed at the current time).
    curr_time : xr.DataArray
        Time occurrence of the current event (indexed at the current time).
    freq : str | None
        Pandas frequency for ds.time. Defaults to xr.infer_freq(da).

    Returns
    -------
    xr.DataArray
        Variable accumulated between prev_time and curr_time.
    """
    if freq is None:
        freq = xr.infer_freq(da)
    dt = _get_dt(freq)
    da_cum = da.cumsum("time")

    curr_fl = curr_time.dt.floor(freq)
    curr_ratio = (curr_time - curr_fl).dt.total_seconds() / dt

    prev_fl = prev_time.dt.floor(freq)
    prev_ratio = (prev_time - prev_fl).dt.total_seconds() / dt

    d_tilcurr = sel_with_nans(da_cum, "time", curr_fl - pd.Timedelta(dt, "s"))
    d_curr = sel_with_nans(da, "time", curr_fl)
    d_tilprev = sel_with_nans(da_cum, "time", prev_fl - pd.Timedelta(dt, "s"))
    d_prev = sel_with_nans(da, "time", prev_fl)
    da_accum = (d_tilcurr + curr_ratio * d_curr) - (d_tilprev + prev_ratio * d_prev)

    return da_accum


def interpolate_to_time(da: xr.DataArray, curr_time: xr.DataArray, freq: str | None = None) -> xr.DataArray:
    """
    Interpolate Dataset to the given time DataArray (such as Solar noon times).

    This is equivalent to ds[var].interp(time=curr_time), but tends to be faster.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to interpolate, with dimension time and variable `var`.
    curr_time : xr.DataArray
        Time array to interpolate.
    freq : str or None
        Pandas frequency for ds.time. Defaults to xr.infer_freq(ds).

    Returns
    -------
    xr.DataArray
        DataArray of interpolated times.
    """
    if freq is None:
        freq = xr.infer_freq(da)
    dt = _get_dt(freq)

    curr_time_fl = curr_time.dt.floor(freq)
    curr_time_cl = curr_time.dt.ceil(freq)

    curr_ratio = (curr_time - curr_time_fl).dt.total_seconds() / dt

    d_curr_fl = sel_with_nans(da, "time", curr_time_fl)
    d_curr_cl = sel_with_nans(da, "time", curr_time_cl)

    da_interp = (1 - curr_ratio) * d_curr_fl + curr_ratio * d_curr_cl
    return da_interp


def _wrap_radians(da):
    with xr.set_options(keep_attrs=True):
        return ((da + np.pi) % (2 * np.pi)) - np.pi


def wind_speed_height_conversion(
    ua: xr.DataArray,
    h_source: str,
    h_target: str,
    method: Literal["log"] = "log",
) -> xr.DataArray:
    r"""
    Wind speed at two meters.

    Parameters
    ----------
    ua : xarray.DataArray
        Wind speed at height `h`.
    h_source : str
        Height of the input wind speed `ua` (e.g. `h == "10 m"` for a wind speed at `10 meters`).
    h_target : str
        Height of the output wind speed.
    method : {"log"}
        Method used to convert wind speed from one height to another.

    Returns
    -------
    xarray.DataArray
        Wind speed at height `h_target`.

    References
    ----------
    :cite:cts:`allen_crop_1998`
    """
    h_source = convert_units_to(h_source, "m")
    h_target = convert_units_to(h_target, "m")
    if method == "log":
        if min(h_source, h_target) < 1 + 5.42 / 67.8:
            raise ValueError(
                f"The height {min(h_source, h_target)}m is too small for method {method}. "
                f"Heights must be greater than {1 + 5.42 / 67.8}"
            )
        with xr.set_options(keep_attrs=True):
            return ua * np.log(67.8 * h_target - 5.42) / np.log(67.8 * h_source - 5.42)
    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")


def _gather_lat(da: xr.DataArray) -> xr.DataArray:
    """
    Gather latitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "latitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Latitude coordinate.
    """
    try:
        lat = da.cf["latitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = f"{n_func} could not find latitude coordinate in DataArray. Try passing it explicitly (`lat=ds.lat`)."
        raise ValueError(msg) from err


def _gather_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Gather longitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "longitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Longitude coordinate.
    """
    try:
        lat = da.cf["longitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = f"{n_func} could not find longitude coordinate in DataArray. Try passing it explicitly (`lon=ds.lon`)."
        raise ValueError(msg) from err


def resample_map(
    obj: xr.DataArray | xr.Dataset,
    dim: str,
    freq: str,
    func: Callable,
    map_blocks: bool | Literal["from_context"] = "from_context",
    resample_kwargs: dict | None = None,
    map_kwargs: dict | None = None,
) -> xr.DataArray | xr.Dataset:
    r"""
    Wrap xarray's resample(...).map() with a :py:func:`xarray.map_blocks`.

    Ensures that the chunking is appropriate using `flox`.

    Parameters
    ----------
    obj : DataArray or Dataset
        The xarray object to resample.
    dim : str
        Dimension over which to resample.
    freq : str
        Resampling frequency along `dim`.
    func : callable
        Function to map on each resampled group.
    map_blocks : bool or "from_context"
        If True, the resample().map() call is wrapped inside a `map_blocks`.
        If False, this does not do anything special.
        If "from_context", xclim's "resample_map_blocks" option is used.
        If the object is not using dask, this is set to False.
    resample_kwargs : dict, optional
        Other arguments to pass to `obj.resample()`.
    map_kwargs : dict, optional
        Arguments to pass to `map`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Resampled object.
    """
    resample_kwargs = resample_kwargs or {}
    map_kwargs = map_kwargs or {}
    if map_blocks == "from_context":
        map_blocks = OPTIONS[MAP_BLOCKS]

    if not uses_dask(obj) or not map_blocks:
        return obj.resample({dim: freq}, **resample_kwargs).map(func, **map_kwargs)

    if rechunk_for_blockwise is None:
        msg = f"Using {MAP_BLOCKS}=True requires flox."
        raise ValueError(msg) from flox_err

    # Make labels, a unique integer for each resample group
    labels = xr.full_like(obj[dim], -1, dtype=np.int32)
    for lbl, group_slice in enumerate(obj[dim].resample({dim: freq}).groups.values()):
        labels[group_slice] = lbl

    obj_rechunked = rechunk_for_blockwise(obj, dim, labels)

    def _resample_map(obj_chnk, dm, frq, rs_kws, fun, mp_kws):
        return obj_chnk.resample({dm: frq}, **rs_kws).map(fun, **mp_kws)

    # Template. We are hoping that this takes a negligible time as it is never loaded.
    template = obj_rechunked.resample(**{dim: freq}, **resample_kwargs).first()

    # New chunks along the time dim : infer the number of elements resulting from the resampling of each chunk
    if isinstance(obj_rechunked, xr.Dataset):
        chunksizes = obj_rechunked.chunks[dim]
    else:
        chunksizes = obj_rechunked.chunks[obj_rechunked.get_axis_num(dim)]
    new_chunks = []
    i = 0
    for chunksize in chunksizes:
        new_chunks.append(len(np.unique(labels[i : i + chunksize])))
        i += chunksize
    template = template.chunk({dim: tuple(new_chunks)})

    return obj_rechunked.map_blocks(_resample_map, (dim, freq, resample_kwargs, func, map_kwargs), template=template)


def _add_one_day(time: xr.DataArray) -> xr.DataArray:
    """
    Add one day to a time coordinate.

    Depending on the calendar/dtype of the time array we need to use numpy's or datetime's (for cftimes) timedelta.

    Parameters
    ----------
    time : xr.DataArray
        Time coordinate.

    Returns
    -------
    xr.DataArray
        Next day.
    """
    if time.dtype == "O":
        return time + timedelta(days=1)
    return time + np.timedelta64(1, "D")
