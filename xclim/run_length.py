# -*- coding: utf-8 -*-
"""Run length algorithms module"""
import logging
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from warnings import warn

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
    ufunc_1dim: Union[str, bool] = "auto",
):
    """Return the index of the first item of a run of at least a given length.

        Parameters
        ----------
        da : xr.DataArray
          Input N-dimensional DataArray (boolean)
        window : int
          Minimum duration of consecutive run to accumulate values.
        dim : str
          Dimension along which to calculate consecutive run (default: 'time').
        ufunc_1dim : Union[str, bool]
          Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
          usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
          for dataarray with a small number of gridpoints.

        Returns
        -------
        out : N-dimensional xarray data array (int)
          Index of first item in first valid run. Returns np.nan if there are no valid run.
        """
    if ufunc_1dim == "auto":
        npts = get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        out = first_run_ufunc(da, window)

    else:
        dims = list(da.dims)
        if "time" not in dims:
            da["time"] = da[dim]
            da.swap_dims({dim: "time"})
        da = da.astype("int")
        i = xr.DataArray(np.arange(da[dim].size), dims=dim).chunk({"time": 1})
        ind = xr.broadcast(i, da)[0].chunk(da.chunks)
        wind_sum = da.rolling(time=window).sum(allow_lazy=True, skipna=False)
        out = ind.where(wind_sum >= window).min(dim=dim) - (
            window - 1
        )  # remove window -1 as rolling result index is last element of the moving window
    return out


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
        return None, None, None

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


def windowed_run_events_1d(arr, window: int):
    """Return the number of runs of a minimum length.

    Parameters
    ----------
    arr : bool array
      Input array

    window : int
      Minimum run length.

    Returns
    -------
    out : func
      Number of distinct runs of a minimum length.
    """
    v, rl, pos = rle_1d(arr)
    return (v * rl >= window).sum()


def windowed_run_count_ufunc(x, window) -> xr.apply_ufunc:
    """Dask-parallel version of windowed_run_count_1d, ie the number of consecutive true values in
    array for runs at least as long as given duration.

    Parameters
    ----------
    x : bool array
      Input array
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


def first_run_ufunc(
    x: xr.DataArray, window, index: Optional[str] = None
) -> xr.apply_ufunc:
    """Dask-parallel version of first_run_1d, ie the first entry in array of consecutive true values.

    Parameters
    ----------
    x : xr.DataArray
      Input array (bool)
    window : int
    index: Optional[str]

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """

    ind = xr.apply_ufunc(
        first_run_1d,
        x,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float],
        keep_attrs=True,
        kwargs={"window": window},
    )

    if index is not None and ~(np.isnan(ind)):
        val = getattr(x.indexes["time"], index)
        i = int(ind.data)
        ind.data = val[i]

    return ind
