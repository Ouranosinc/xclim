# -*- coding: utf-8 -*-
"""
Run length algorithms submodule
===============================

Computation of statistics on runs of True values in boolean arrays.
"""
import logging
from datetime import datetime
from functools import partial
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from warnings import warn

import dask.array as dsk
import numpy as np
import xarray as xr


logging.captureWarnings(True)
npts_opt = 9000


def get_npts(da: xr.DataArray) -> int:
    """Return the number of gridpoints in a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
      N-dimensional input array

    Returns
    -------
    int
      Product of input DataArray coordinate sizes excluding the dimension 'time'
    """
    coords = list(da.coords)
    coords.remove("time")
    npts = 1
    for c in coords:
        npts *= da[c].size
    return npts


def rle(da: xr.DataArray, dim: str = "time", max_chunk: int = 1_000_000):
    n = len(da[dim])
    i = xr.DataArray(np.arange(da[dim].size), dims=dim).chunk({"time": 1})
    ind = xr.broadcast(i, da)[0].chunk(da.chunks)
    b = ind.where(~da)  # find indexes where false
    end1 = (
        da.where(b[dim] == b[dim][-1], drop=True) * 0 + n
    )  # add additional end value index (deal with end cases)
    start1 = (
        da.where(b[dim] == b[dim][0], drop=True) * 0 - 1
    )  # add additional start index (deal with end cases)
    b = xr.concat([start1, b, end1], dim)

    # Ensure bfill operates on entire (unchunked) time dimension
    # Determine appropraite chunk size for other dims - do not exceed 'max_chunk' total size per chunk (default 1000000)
    ndims = len(b.shape)
    chunk_dim = b[dim].size
    # divide extra dims into equal size
    # Note : even if calculated chunksize > dim.size result will have chunk==dim.size
    chunksize_ex_dims = None
    if ndims > 1:
        chunksize_ex_dims = np.round(np.power(max_chunk / chunk_dim, 1 / (ndims - 1)))
    chunks = dict()
    chunks[dim] = -1
    for dd in b.dims:
        if dd != dim:
            chunks[dd] = chunksize_ex_dims
    b = b.chunk(chunks)

    # back fill nans with first position after
    z = b.bfill(dim=dim)
    # calculate lengths
    d = z.diff(dim=dim) - 1
    d = d.where(d >= 0)
    return d


def longest_run(
    da: xr.DataArray, dim: str = "time", ufunc_1dim: Union[str, bool] = "auto"
):
    """Return the length of the longest consecutive run of True values.

    Parameters
    ----------
    da : xr.DataArray
      N-dimensional array (boolean)
    dim : str
      Dimension along which to calculate consecutive run; Default: 'time'.
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
      for dataarray with a small number of gridpoints.
    Returns
    -------
    N-dimensional array (int)
      Length of longest run of True values along dimension
    """
    if ufunc_1dim == "auto":
        npts = get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        rl_long = longest_run_ufunc(da)
    else:
        d = rle(da, dim=dim)
        rl_long = d.max(dim=dim)

    return rl_long


def windowed_run_events(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "auto",
) -> xr.DataArray:
    """Return the number of runs of a minimum length.

    Parameters
    ----------
    da: xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum run length.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
      for dataarray with a small number of gridpoints.
    Returns
    -------
    xr.DataArray
      Number of distinct runs of a minimum length (int).
    """
    if ufunc_1dim == "auto":
        npts = get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        out = windowed_run_events_ufunc(da, window)
    else:
        d = rle(da, dim=dim)
        out = (d >= window).sum(dim=dim)
    return out


def windowed_run_count(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "auto",
) -> xr.DataArray:
    """Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    da: xr.DataArray
      Input N-dimensional DataArray (boolean).
    window : int
      Minimum run length.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points. Using 1D_ufunc=True is typically more efficient
      for dataarray with a small number of gridpoints.

    Returns
    -------
    xr.DataArray
      Total number of true values part of a consecutive runs of at least `window` long.
    """
    if ufunc_1dim == "auto":
        npts = get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        out = windowed_run_count_ufunc(da, window)
    else:
        d = rle(da, dim=dim)
        out = d.where(d >= window, 0).sum(dim=dim)
    return out


def first_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    coord: Optional[Union[str, bool]] = False,
    ufunc_1dim: Union[str, bool] = "auto",
):
    """Return the index of the first item of the first run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[str]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
      for dataarray with a small number of gridpoints.

    Returns
    -------
    out : xr.DataArray
      Index (or coordinate if `coord` is not False) of first item in first valid run. Returns np.nan if there are no valid run.
    """
    if ufunc_1dim == "auto":
        if isinstance(da.data, dsk.Array) and len(da.chunks[da.dims.index(dim)]) > 1:
            ufunc_1dim = False
        else:
            npts = get_npts(da)
            ufunc_1dim = npts <= npts_opt

    da = da.fillna(0)  # We expect a boolean array, but there could be NaNs nonetheless

    if ufunc_1dim:
        out = first_run_ufunc(x=da, window=window, dim=dim)

    else:
        da = da.astype("int")
        i = xr.DataArray(np.arange(da[dim].size), dims=dim)
        ind = xr.broadcast(i, da)[0].transpose(*da.dims)
        if isinstance(da.data, dsk.Array):
            ind = ind.chunk(da.chunks)
        wind_sum = da.rolling(time=window).sum(allow_lazy=True, skipna=False)
        out = ind.where(wind_sum >= window).min(dim=dim) - (
            window - 1
        )  # remove window -1 as rolling result index is last element of the moving window

    if coord:
        crd = da[dim]
        if isinstance(coord, str):
            crd = getattr(crd.dt, coord)

        return lazy_indexing(crd, out)

    return out


def last_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    coord: Optional[Union[str, bool]] = False,
    ufunc_1dim: Union[str, bool] = "auto",
):
    """Return the index of the last item of the last run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[str]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points.  Using `1D_ufunc=True` is typically more efficient
      for a DataArray with a small number of grid points.

    Returns
    -------
    out : xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    reversed_da = da.sortby(dim, ascending=False)
    out = first_run(
        reversed_da, window=window, dim=dim, coord=coord, ufunc_1dim=ufunc_1dim
    )
    if not coord:
        return reversed_da[dim].size - out - 1
    return out


def run_length_with_date(
    da: xr.DataArray, window: int, date: str = "07-01", dim: str = "time",
):
    """Return the length of the longest consecutive run of True values found
    to be semi-continuous before and after a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    date:
      The date that a run must include to be considered valid.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').

    Returns
    -------
    out : xr.DataArray
      Length of longest run of True values along a given dimension inclusive of a given date.

    Notes
    -----
    The run can include holes of False or NaN values, so long as they do not exceed the window size.
    """
    include_date = datetime.strptime(date, "%m-%d").timetuple().tm_yday

    def _crld(rlda, rlwindow, rldoy, rldim):
        mid_index = np.where(rlda.time.dt.dayofyear == rldoy)[0]
        if (
            mid_index.size == 0
        ):  # The date is not within the group. Happens at boundaries.
            all_nans = xr.full_like(rlda.isel(time=0), np.nan)
            all_nans.attrs = {}
            return all_nans
        end = first_run(
            (~rlda).where(rlda.time >= rlda.time[mid_index][0]),
            window=rlwindow,
            dim=rldim,
        )
        beg = first_run(rlda, window=rlwindow, dim=rldim)
        sl = end - beg
        sl = xr.where(
            beg.isnull() & end.notnull(), 0, sl
        )  # If series is never triggered
        sl = xr.where(
            beg.notnull() & end.isnull(), rlda.time.size - beg, sl
        )  # If series is not ended by end of resample time frequency
        return sl.where(sl >= 0)

    return _crld(rlda=da, rlwindow=window, rldoy=include_date, rldim=dim)


def run_end_after_date(
    da: xr.DataArray,
    window: int,
    date: str = "07-01",
    dim: str = "time",
    coord: str = "dayofyear",
):
    """Return the index of the first item after the end of a run after a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    date : str
      The date after which to look for the end of a run.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[str]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    out : xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    after_date = datetime.strptime(date, "%m-%d").timetuple().tm_yday

    def _cred(rlda, rlwindow, rldoy, rldim, rlcoord):
        mid_idx = np.where(rlda.time.dt.dayofyear == rldoy)[0]
        if (
            mid_idx.size == 0
        ):  # The date is not within the group. Happens at boundaries.
            all_nans = xr.full_like(rlda.isel(time=0), np.nan)
            all_nans.attrs = {}
            return all_nans
        end = first_run(
            (~rlda).where(rlda.time >= rlda.time[mid_idx][0]),
            window=rlwindow,
            dim=rldim,
            coord=rlcoord,
        )
        beg = first_run(
            rlda.where(rlda.time < rlda.time[mid_idx][0]), window=rlwindow, dim=rldim,
        )
        end = xr.where(
            end.isnull() & beg.notnull(), rlda.time.isel(time=-1).dt.dayofyear, end
        )
        return end.where(beg.notnull())

    return _cred(rlda=da, rlwindow=window, rldoy=after_date, rldim=dim, rlcoord=coord)


def last_run_before_date(
    da: xr.DataArray,
    window: int,
    date: str = "07-01",
    dim: str = "time",
    coord: str = "dayofyear",
):
    """Return the index of the last item of the last run before a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    date : str
      The date before which to look for the last event.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[str]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    out : xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    before_date = datetime.strptime(date, "%m-%d").timetuple().tm_yday

    def _clrbd(rlda, rlwindow, rlbefore_date, rldim, rlcoord):
        mid_idx = np.where(rlda.time.dt.dayofyear == rlbefore_date)[0]
        if (
            mid_idx.size == 0
        ):  # The date is not within the group. Happens at boundaries.
            all_nans = xr.full_like(rlda.isel(time=0), np.nan)
            all_nans.attrs = dict()
            return all_nans
        run = rlda.where(rlda.time <= rlda.time[mid_idx][0])
        return last_run(run, window=rlwindow, dim=rldim, coord=rlcoord)

    return _clrbd(
        rlda=da, rlwindow=window, rlbefore_date=before_date, rldim=dim, rlcoord=coord
    )


def rle_1d(
    arr: Union[int, float, bool, Sequence[Union[int, float, bool]]]
) -> Tuple[np.array, np.array, np.array]:
    """Return the length, starting position and value of consecutive identical values.

    Parameters
    ----------
    arr : Sequence[Union[int, float, bool]]
      Array of values to be parsed.

    Returns
    -------
    values : np.array
      The values taken by arr over each run
    run lengths : np.array
      The length of each run
    start position : np.array
      The starting index of each run

    Examples
    --------
    >>> a = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    >>> rle_1d(a)
    (array([1, 2, 3]), array([2, 4, 6]), array([0, 2, 6]))
    """
    ia = np.asarray(arr)
    n = len(ia)

    if n == 0:
        e = "run length array empty"
        warn(e)
        # Returning None makes some other 1d func below fail.
        return np.array(np.nan), 0, np.array(np.nan)

    y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)  # must include last element position
    rl = np.diff(np.append(-1, i))  # run lengths
    pos = np.cumsum(np.append(0, rl))[:-1]  # positions
    return ia[i], rl, pos


def first_run_1d(arr: Sequence[Union[int, float]], window: int) -> int:
    """Return the index of the first item of a run of at least a given length.

    Parameters
    ----------
    arr : Sequence[Union[int, float]]
      Input array
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Index of first item in first valid run. Returns np.nan if there are no valid run.
    """
    v, rl, pos = rle_1d(arr)
    ind = np.where(v * rl >= window, pos, np.inf).min()

    if np.isinf(ind):
        return np.nan
    return ind


def longest_run_1d(arr: Sequence[bool]) -> int:
    """Return the length of the longest consecutive run of identical values.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool)

    Returns
    -------
    int
      Length of longest run.
    """
    v, rl = rle_1d(arr)[:2]
    return np.where(v, rl, 0).max()


def windowed_run_count_1d(arr: Sequence[bool], window: int) -> int:
    """Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool)
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Total number of true values part of a consecutive run at least `window` long.
    """
    v, rl = rle_1d(arr)[:2]
    return np.where(v * rl >= window, rl, 0).sum()


def windowed_run_events_1d(arr: Sequence[bool], window: int):
    """Return the number of runs of a minimum length.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool)
    window : int
      Minimum run length.

    Returns
    -------
    out : func
      Number of distinct runs of a minimum length.
    """
    v, rl, pos = rle_1d(arr)
    return (v * rl >= window).sum()


def windowed_run_count_ufunc(x: Sequence[bool], window: int) -> xr.apply_ufunc:
    """Dask-parallel version of windowed_run_count_1d, ie the number of consecutive true values in
    array for runs at least as long as given duration.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool)
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        windowed_run_count_1d,
        x,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int],
        keep_attrs=True,
        kwargs={"window": window},
    )


def windowed_run_events_ufunc(x: Sequence[bool], window: int) -> xr.apply_ufunc:
    """Dask-parallel version of windowed_run_events_1d, ie the number of runs at least as long as given duration.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool)
    window : int
      Minimum run length

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        windowed_run_events_1d,
        x,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int],
        keep_attrs=True,
        kwargs={"window": window},
    )


def longest_run_ufunc(x: Sequence[bool]) -> xr.apply_ufunc:
    """Dask-parallel version of longest_run_1d, ie the maximum number of consecutive true values in
    array.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool)

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        longest_run_1d,
        x,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int],
        keep_attrs=True,
    )


def first_run_ufunc(x: xr.DataArray, window: int, dim: str = "time",) -> xr.apply_ufunc:
    """Dask-parallel version of first_run_1d, ie the first entry in array of consecutive true values.

    Parameters
    ----------
    x : xr.DataArray
      Input array (bool)
    window : int
    dim: Optional[str]

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """

    ind = xr.apply_ufunc(
        first_run_1d,
        x,
        input_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float],
        keep_attrs=True,
        kwargs={"window": window},
    )

    return ind


def lazy_indexing(da: xr.DataArray, index: xr.DataArray):
    """Get values of `da` at indices `index` in a NaN-aware and lazy manner.

    Parameters
    ----------
    da : xr.DataArray
      1D Input array
    index : xr.DataArray
      N-d integer indices

    Returns
    -------
    xr.DataArray
      Values of `da` at indices `index`
    """

    def _index_from_1d_array(array, indices):
        return array[
            indices,
        ]

    invalid = index.isnull()
    index = index.fillna(0).astype(int)
    func = partial(_index_from_1d_array, da)

    out = index.map_blocks(func)
    return out.where(~invalid)
