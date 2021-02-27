# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Run length algorithms submodule
===============================

Computation of statistics on runs of True values in boolean arrays.
"""
from datetime import datetime
from functools import partial
from typing import Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import xarray as xr
from dask import array as dsk

from xclim.core.utils import DateStr, DayOfYearStr

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
    return da.size // da.time.size


def rle(
    da: xr.DataArray, dim: str = "time", max_chunk: int = 1_000_000
) -> xr.DataArray:
    """Generate basic run length function.

    Parameters
    ----------
    da : xr.DataArray
    dim : str
    max_chunk : int

    Returns
    -------
    xr.DataArray
    """
    n = len(da[dim])
    # Need to chunk here to ensure the broadcasting is not made in memory
    i = xr.DataArray(np.arange(da[dim].size), dims=dim).chunk({"time": -1})
    ind, da = xr.broadcast(i, da)
    # Rechunk, but with broadcasted da
    ind = ind.chunk(da.chunks)
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
    chunksize_ex_dims = None  # TODO: This raises type assignment errors in mypy
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
) -> xr.DataArray:
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
      for DataArray with a small number of grid points.

    Returns
    -------
    xr.DataArray
      Length of longest run of True values along dimension (int).
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
) -> xr.DataArray:
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
    xr.DataArray
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
        wind_sum = da.rolling(time=window).sum(skipna=False)
        out = ind.where(wind_sum >= window).min(dim=dim) - (window - 1)
        # remove window - 1 as rolling result index is last element of the moving window

    if coord:
        crd = da[dim]
        if isinstance(coord, str):
            crd = getattr(crd.dt, coord)

        out = lazy_indexing(crd, out)

    if dim in out.coords:
        out = out.drop_vars(dim)

    return out


def last_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    coord: Optional[Union[str, bool]] = False,
    ufunc_1dim: Union[str, bool] = "auto",
) -> xr.DataArray:
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
    xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    reversed_da = da.sortby(dim, ascending=False)
    out = first_run(
        reversed_da, window=window, dim=dim, coord=coord, ufunc_1dim=ufunc_1dim
    )
    if not coord:
        return reversed_da[dim].size - out - 1
    return out


def season_length(
    da: xr.DataArray,
    window: int,
    date: Optional[DayOfYearStr] = None,
    dim: str = "time",
) -> xr.DataArray:
    """Return the length of the longest semi-consecutive run of True values (optionally including a given date).

    A "season" is a run of True values that may include breaks under a given length (`window`).
    The start is computed as the first run of `window` True values, then end as the first subsequent run
    of  `window` False values. If a date is passed, it must be included in the season.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive values to start and end the season.
    date: DayOfYearStr, optional
      The date (in MM-DD format) that a run must include to be considered valid.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').

    Returns
    -------
    xr.DataArray
      Length of longest run of True values along a given dimension (inclusive of a given date) without breaks longer than a given length.

    Notes
    -----
    The run can include holes of False or NaN values, so long as they do not exceed the window size.

    If a date is given, the season end is forced to be later or equal to this date. This means that
    even if the "real" season has been over for a long time, this is the date used in the length calculation.
    Example : Length of the "warm season", where T > 25°C, with date = 1st August. Let's say
    the temperature is over 25 for all june, but july and august have very cold temperatures.
    Instead of returning 30 days (june), the function will return 61 days (july + june).
    """
    beg = first_run(da, window=window, dim=dim)
    # Invert the condition and mask all values after beginning
    # we fillna(0) as so to differentiate series with no runs and all-nan series
    not_da = (~da).where(da.time.copy(data=np.arange(da.time.size)) >= beg.fillna(0))

    # Mask also values after "date"
    mid_idx = index_of_date(da.time, date, max_idxs=1, default=0)
    if mid_idx.size == 0:
        # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel(time=0), np.nan, float).drop_vars("time")
    not_da = not_da.where(da.time >= da.time[mid_idx][0])

    end = first_run(
        not_da,
        window=window,
        dim=dim,
    )

    sl = end - beg
    sl = xr.where(
        beg.notnull() & end.isnull(), da.time.size - beg, sl
    )  # If series is not ended by end of resample time frequency
    if date is not None:
        sl = sl.where(beg < mid_idx.squeeze())
    sl = xr.where(beg.isnull() & end.notnull(), 0, sl)  # If series is never triggered
    return sl


def run_end_after_date(
    da: xr.DataArray,
    window: int,
    date: DayOfYearStr = "07-01",
    dim: str = "time",
    coord: Optional[Union[bool, str]] = "dayofyear",
) -> xr.DataArray:
    """Return the index of the first item after the end of a run after a given date.

    The run must begin before the date.

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
    coord : Optional[Union[bool, str]]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    mid_idx = index_of_date(da.time, date, max_idxs=1, default=0)
    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel(time=0), np.nan, float).drop_vars("time")

    end = first_run(
        (~da).where(da.time >= da.time[mid_idx][0]),
        window=window,
        dim=dim,
        coord=coord,
    )
    beg = first_run(da.where(da.time < da.time[mid_idx][0]), window=window, dim=dim)
    end = xr.where(
        end.isnull() & beg.notnull(), da.time.isel(time=-1).dt.dayofyear, end
    )
    return end.where(beg.notnull()).drop_vars("time")


def first_run_after_date(
    da: xr.DataArray,
    window: int,
    date: DayOfYearStr = "07-01",
    dim: str = "time",
    coord: Optional[Union[bool, str]] = "dayofyear",
) -> xr.DataArray:
    """Return the index of the first item of the first run after a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    date : DayOfYearStr
      The date after which to look for the run.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[Union[bool, str]]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.DataArray
      Index (or coordinate if `coord` is not False) of first item in the first valid run. Returns np.nan if there are no valid run.
    """
    mid_idx = index_of_date(da.time, date, max_idxs=1, default=0)
    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel(time=0), np.nan, float).drop_vars("time")

    return first_run(
        da.where(da.time >= da.time[mid_idx][0]),
        window=window,
        dim=dim,
        coord=coord,
    )


def last_run_before_date(
    da: xr.DataArray,
    window: int,
    date: DayOfYearStr = "07-01",
    dim: str = "time",
    coord: Optional[Union[bool, str]] = "dayofyear",
) -> xr.DataArray:
    """Return the index of the last item of the last run before a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean)
    window : int
      Minimum duration of consecutive run to accumulate values.
    date : DayOfYearStr
      The date before which to look for the last event.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[Union[bool, str]]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.DataArray
      Index (or coordinate if `coord` is not False) of last item in last valid run. Returns np.nan if there are no valid run.
    """
    mid_idx = index_of_date(da.time, date, default=-1)

    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel(time=0), np.nan, float).drop_vars("time")

    run = da.where(da.time <= da.time[mid_idx][0])
    return last_run(run, window=window, dim=dim, coord=coord)


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
    >>> from xclim.indices.run_length import rle_1d
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
    func
      Number of distinct runs of a minimum length.
    """
    v, rl, pos = rle_1d(arr)
    return (v * rl >= window).sum()


def windowed_run_count_ufunc(x: Sequence[bool], window: int) -> xr.DataArray:
    """Dask-parallel version of windowed_run_count_1d, ie: the number of consecutive true values in array for runs at least as long as given duration.

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


def windowed_run_events_ufunc(x: Sequence[bool], window: int) -> xr.DataArray:
    """Dask-parallel version of windowed_run_events_1d, ie: the number of runs at least as long as given duration.

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


def longest_run_ufunc(x: Union[xr.DataArray, Sequence[bool]]) -> xr.DataArray:
    """Dask-parallel version of longest_run_1d, ie: the maximum number of consecutive true values in array.

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
    x: Union[xr.DataArray, Sequence[bool]],
    window: int,
    dim: str = "time",
) -> xr.DataArray:
    """Dask-parallel version of first_run_1d, ie: the first entry in array of consecutive true values.

    Parameters
    ----------
    x : Union[xr.DataArray, Sequence[bool]]
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


def lazy_indexing(
    da: xr.DataArray, index: xr.DataArray, dim: Optional[str] = None
) -> xr.DataArray:
    """Get values of `da` at indices `index` in a NaN-aware and lazy manner.

    Two case

    Parameters
    ----------
    da : xr.DataArray
      Input array. If not 1D, `dim` must be given and must not appear in index.
    index : xr.DataArray
      N-d integer indices, if da is not 1D, all dimensions of index must be in da
    dim : str, optional
      Dimension along which to index, unused if `da` is 1D,
      should not be present in `index`.

    Returns
    -------
    xr.DataArray
      Values of `da` at indices `index`
    """
    if da.ndim == 1:
        # Case where da is 1D and index is N-D
        # Slightly better performance using map_blocks, over an apply_ufunc
        def _index_from_1d_array(array, indices):
            return array[
                indices,
            ]

        idx_ndim = index.ndim
        if idx_ndim == 0:
            # The 0-D index case, we add a dummy dimension to help dask
            dim = xr.core.utils.get_temp_dimname(da.dims, "x")
            index = index.expand_dims(dim)
        invalid = index.isnull()  # Which indexes to mask
        # NaN-indexing doesn't work, so fill with 0 and cast to int
        index = index.fillna(0).astype(int)
        # for each chunk of index, take corresponding values from da
        func = partial(_index_from_1d_array, da)
        out = index.map_blocks(func)
        # mask where index was NaN
        out = out.where(~invalid)
        if idx_ndim == 0:
            # 0-D case, drop useless coords and dummy dim
            out = out.drop_vars(da.dims[0]).squeeze()
        return out

    # Case where index.dims is a subset of da.dims.
    if dim is None:
        diff_dims = set(da.dims) - set(index.dims)
        if len(diff_dims) == 0:
            raise ValueError(
                "da must have at least one dimension more than index for lazy_indexing."
            )
        if len(diff_dims) > 1:
            raise ValueError(
                "If da has more than one dimension more than index, the indexing dim must be given through `dim`"
            )
        dim = diff_dims.pop()

    def _index_from_nd_array(array, indices):
        return np.take_along_axis(array, indices[..., np.newaxis], axis=-1)[..., 0]

    return xr.apply_ufunc(
        _index_from_nd_array,
        da,
        index.astype(int),
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[da.dtype],
    )


def index_of_date(
    time: xr.DataArray,
    date: Optional[Union[DateStr, DayOfYearStr]],
    max_idxs: Optional[int] = None,
    default: int = 0,
) -> np.ndarray:
    """Get the index of a date in a time array.

    Parameters
    ----------
    time : xr.DataArray
      An array of datetime values, any calendar.
    date : DayOfYearStr or DateStr, optional
      A string in the "yyyy-mm-dd" or "mm-dd" format.
      If None, returns default.
    max_idxs: int, optional
      Maximum number of returned indexes.
    default: int
      Index to return if date is None.

    Raises
    ------
    ValueError
      If there are most instances of `date` in `time` than `max_idxs`.

    Returns
    -------
    ndarray
      1D array of integers, indexes of `date` in `time`.
    """
    if date is None:
        return np.array([default])
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        year_cond = time.dt.year == date.year
    except ValueError:
        date = datetime.strptime(date, "%m-%d")
        year_cond = True

    idxs = np.where(
        year_cond & (time.dt.month == date.month) & (time.dt.day == date.day)
    )[0]
    if max_idxs is not None and idxs.size > max_idxs:
        raise ValueError(
            f"More than {max_idxs} instance of date {date} found in the coordinate array."
        )
    return idxs
