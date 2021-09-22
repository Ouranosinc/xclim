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
from xarray.core.utils import get_temp_dimname

from xclim.core.options import OPTIONS, RUN_LENGTH_UFUNC
from xclim.core.utils import DateStr, DayOfYearStr, uses_dask

npts_opt = 9000
"""
Arrays with less than this number of data points per slice will trigger
the use of the ufunc version of run lengths algorithms.
"""


def use_ufunc(
    ufunc_1dim: Union[bool, str],
    da: xr.DataArray,
    dim: str = "time",
    index: str = "first",
) -> bool:
    """Return whether the ufunc version of run length algorithms should be used with this DataArray or not.

    If ufunc_1dim is 'from_context', the parameter is read from xclim's global (or context) options.
    If it is 'auto', this returns False for dask-backed array and for arrays with more than :py:const:`npts_opt`
    points per slice along `dim`.

    Parameters
    ----------
    ufunc_1dim: {'from_context', 'auto', True, False}
    da : xr.DataArray
      Input array.
    dim: str
      The dimension along which to find runs.
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.

    Returns
    -------
    bool
      If ufunc_1dim is "auto", returns True if the array is on dask or too large.
      Otherwise, returns ufunc_1dim.
    """

    if ufunc_1dim == "from_context":
        ufunc_1dim = OPTIONS[RUN_LENGTH_UFUNC]

    if ufunc_1dim == "auto":
        ufunc_1dim = not uses_dask(da) and (da.size // da[dim].size) < npts_opt

    return index == "first" and ufunc_1dim


def rle(
    da: xr.DataArray,
    dim: str = "time",
    max_chunk: int = 1_000_000,
    index: str = "first",
) -> xr.DataArray:
    """Generate basic run length function.

    Parameters
    ----------
    da : xr.DataArray
      Input array.
    dim : str
      Dimension name.
    max_chunk : int
      Maximum chunk size.
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray
      Values are 0 where da is False (out of runs).
    """
    use_dask = isinstance(da.data, dsk.Array)

    # Ensure boolean
    da = da.astype(bool)
    if index == "last":
        da = da.reindex({dim: da[dim][::-1]})

    n = len(da[dim])
    # Need to chunk here to ensure the broadcasting is not made in memory
    i = xr.DataArray(np.arange(da[dim].size), dims=dim)
    if use_dask:
        i = i.chunk({dim: -1})

    ind, da = xr.broadcast(i, da)
    if use_dask:
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
    # Determine appropriate chunk size for other dims - do not exceed 'max_chunk' total size per chunk (default 1000000)
    ndims = len(b.shape)
    if use_dask:
        chunk_dim = b[dim].size
        # divide extra dims into equal size
        # Note : even if calculated chunksize > dim.size result will have chunk==dim.size
        chunksize_ex_dims = None  # TODO: This raises type assignment errors in mypy
        if ndims > 1:
            chunksize_ex_dims = np.round(
                np.power(max_chunk / chunk_dim, 1 / (ndims - 1))
            )
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
    d = d.isel({dim: slice(None, -1)}).where(da, 0)
    if index == "last":
        d = d.reindex({dim: d[dim][::-1]})
    return d


def rle_statistics(
    da: xr.DataArray,
    reducer: str = "max",
    window: int = 1,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "from_context",
    index: str = "first",
) -> xr.DataArray:
    """Return the length of consecutive run of True values, according to a reducing operator.

    Parameters
    ----------
    da : xr.DataArray
      N-dimensional array (boolean).
    reducer: str
      Name of the reducing function.
    window : int
      Minimal length of consecutive runs to be included in the statistics.
    dim : str
      Dimension along which to calculate consecutive run; Default: 'time'.
    ufunc_1dim : Union[str, bool]
      Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
      usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
      for DataArray with a small number of grid points.
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.


    Returns
    -------
    xr.DataArray
      Length of runs of True values along dimension, according to the reducing function (float)
      If there are no runs (but the data is valid), returns 0.
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index)

    if ufunc_1dim:
        rl_stat = statistics_run_ufunc(da, reducer, window, dim)
    else:
        d = rle(da, dim=dim, index=index)
        rl_stat = getattr(d.where(d >= window), reducer)(dim=dim)
        rl_stat = xr.where((d.isnull() | (d < window)).all(dim=dim), 0, rl_stat)

    return rl_stat


def longest_run(
    da: xr.DataArray,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "from_context",
    index: str = "first",
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
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.


    Returns
    -------
    xr.DataArray
      Length of longest run of True values along dimension (int).
    """
    return rle_statistics(
        da, reducer="max", dim=dim, ufunc_1dim=ufunc_1dim, index=index
    )


def windowed_run_events(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "auto",
    index: str = "first",
) -> xr.DataArray:
    """Return the number of runs of a minimum length.

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
      usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
      for dataarray with a small number of gridpoints.
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.


    Returns
    -------
    xr.DataArray
      Number of distinct runs of a minimum length (int).
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index)

    if ufunc_1dim:
        out = windowed_run_events_ufunc(da, window, dim)
    else:
        d = rle(da, dim=dim, index=index)
        out = (d >= window).sum(dim=dim)
    return out


def windowed_run_count(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "from_context",
    index: str = "first",
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
    index: {'first', 'last'}
      If 'first', the run length is indexed with the first element in the run.
      If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray
      Total number of `True` values part of a consecutive runs of at least `window` long.
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index)

    if ufunc_1dim:
        out = windowed_run_count_ufunc(da, window, dim)
    else:
        d = rle(da, dim=dim, index=index)
        out = d.where(d >= window, 0).sum(dim=dim)
    return out


def first_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    coord: Optional[Union[str, bool]] = False,
    ufunc_1dim: Union[str, bool] = "from_context",
) -> xr.DataArray:
    """Return the index of the first item of the first run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean).
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
      Index (or coordinate if `coord` is not False) of first item in first valid run.
      Returns np.nan if there are no valid runs.
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim)

    da = da.fillna(0)  # We expect a boolean array, but there could be NaNs nonetheless
    if ufunc_1dim:
        out = first_run_ufunc(x=da, window=window, dim=dim)

    else:
        da = da.astype("int")
        i = xr.DataArray(np.arange(da[dim].size), dims=dim)
        ind = xr.broadcast(i, da)[0].transpose(*da.dims)
        if isinstance(da.data, dsk.Array):
            ind = ind.chunk(da.chunks)
        wind_sum = da.rolling({dim: window}).sum(skipna=False)
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
    ufunc_1dim: Union[str, bool] = "from_context",
) -> xr.DataArray:
    """Return the index of the last item of the last run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean).
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
      Index (or coordinate if `coord` is not False) of last item in last valid run.
      Returns np.nan if there are no valid runs.
    """
    reversed_da = da.sortby(dim, ascending=False)
    out = first_run(
        reversed_da, window=window, dim=dim, coord=coord, ufunc_1dim=ufunc_1dim
    )
    if not coord:
        return reversed_da[dim].size - out - 1
    return out


# TODO: Add window arg
# TODO: Inverse window arg to tolerate holes?
def run_bounds(
    mask: xr.DataArray, dim: str = "time", coord: Optional[Union[bool, str]] = True
):
    """Return the start and end dates of boolean runs along a dimension.

    Parameters
    ----------
    mask : xr.DataArray
      Boolean array.
    dim : str
      Dimension along which to look for runs.
    coord : bool or str
      If True, return values of the coordinate, if a string, returns values from `dim.dt.<coord>`.
      if False, return indexes.

    Returns
    -------
    xr.DataArray
      With ``dim`` reduced to "events" and "bounds". The events dim is as long as needed, padded with NaN or NaT.
    """
    if uses_dask(mask):
        raise NotImplementedError(
            "Dask arrays not supported as we can't know the final event number before computing."
        )

    diff = xr.concat((mask.isel({dim: 0}).astype(int), mask.astype(int).diff(dim)), dim)

    nstarts = (diff == 1).sum(dim).max().item()

    def _get_indices(arr, *, N):
        out = np.full((N,), np.nan, dtype=float)
        inds = np.where(arr)[0]
        out[: len(inds)] = inds
        return out

    starts = xr.apply_ufunc(
        _get_indices,
        diff == 1,
        input_core_dims=[[dim]],
        output_core_dims=[["events"]],
        kwargs={"N": nstarts},
        vectorize=True,
    )

    ends = xr.apply_ufunc(
        _get_indices,
        diff == -1,
        input_core_dims=[[dim]],
        output_core_dims=[["events"]],
        kwargs={"N": nstarts},
        vectorize=True,
    )

    if coord:
        crd = mask[dim]
        if isinstance(coord, str):
            crd = getattr(crd.dt, coord)

        starts = lazy_indexing(crd, starts).drop(dim)
        ends = lazy_indexing(crd, ends).drop(dim)
    return xr.concat((starts, ends), "bounds")


def keep_longest_run(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Keep the longest run along a dimension.

    Parameters
    ----------
    da : xr.DataArray
      Boolean array.
    dim : str
      Dimension along which to check for the longest run.

    Returns
    -------
    xr.DataArray
      Boolean array similar to da but with only one run, the (first) longest.
    """
    # Get run lengths
    rls = rle(da, dim)
    out = xr.where(
        # Construct an integer array and find the max
        rls[dim].copy(data=np.arange(rls[dim].size)) == rls.argmax(dim),
        rls + 1,  # Add one to the First longest run
        rls,
    )
    out = out.ffill(dim) == out.max(dim)
    return da.copy(data=out.transpose(*da.dims).data)  # Keep everything the same


def season(
    da: xr.DataArray,
    window: int,
    date: Optional[DayOfYearStr] = None,
    dim: str = "time",
    coord: Optional[Union[str, bool]] = False,
) -> xr.Dataset:
    """Return the bounds of a season (along dim).

    A "season" is a run of True values that may include breaks under a given length (`window`).
    The start is computed as the first run of `window` True values, then end as the first subsequent run
    of  `window` False values. If a date is passed, it must be included in the season.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean).
    window : int
      Minimum duration of consecutive values to start and end the season.
    date: DayOfYearStr, optional
      The date (in MM-DD format) that a run must include to be considered valid.
    dim : str
      Dimension along which to calculate consecutive run (default: 'time').
    coord : Optional[str]
      If not False, the function returns values along `dim` instead of indexes.
      If `dim` has a datetime dtype, `coord` can also be a str of the name of the
      DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.Dataset
      "dim" is reduced to "season_bnds" with 2 elements : season start and season end, both indices of da[dim].

    Notes
    -----
    The run can include holes of False or NaN values, so long as they do not exceed the window size.

    If a date is given, the season start and end are forced to be on each side of this date. This means that
    even if the "real" season has been over for a long time, this is the date used in the length calculation.
    Example : Length of the "warm season", where T > 25°C, with date = 1st August. Let's say
    the temperature is over 25 for all june, but july and august have very cold temperatures.
    Instead of returning 30 days (june), the function will return 61 days (july + june).
    """
    beg = first_run(da, window=window, dim=dim)
    # Invert the condition and mask all values after beginning
    # we fillna(0) as so to differentiate series with no runs and all-nan series
    not_da = (~da).where(da[dim].copy(data=np.arange(da[dim].size)) >= beg.fillna(0))

    # Mask also values after "date"
    mid_idx = index_of_date(da[dim], date, max_idxs=1, default=0)
    if mid_idx.size == 0:
        # The date is not within the group. Happens at boundaries.
        base = da.isel({dim: 0})  # To have the proper shape
        beg = xr.full_like(base, np.nan, float).drop_vars(dim)
        end = xr.full_like(base, np.nan, float).drop_vars(dim)
        length = xr.full_like(base, np.nan, float).drop_vars(dim)
    else:
        if date is not None:
            # If the beginning was after the mid date, both bounds are NaT.
            valid_start = beg < mid_idx.squeeze()
        else:
            valid_start = True

        not_da = not_da.where(da[dim] >= da[dim][mid_idx][0])
        end = first_run(
            not_da,
            window=window,
            dim=dim,
        )
        # If there was a beginning but no end, season goes to the end of the array
        no_end = beg.notnull() & end.isnull()

        # Length
        length = end - beg

        # No end:  length is actually until the end of the array, so it is missing 1
        length = xr.where(no_end, da[dim].size - beg, length)
        # Where the begining was before the mid date, invalid.
        length = length.where(valid_start)
        # Where there were data points, but no season : put 0 length
        length = xr.where(beg.isnull() & end.notnull(), 0, length)

        # No end: end defaults to the last element (this differs from length, but heh)
        end = xr.where(no_end, da[dim].size - 1, end)

        # Where the beginning was before the mid date
        beg = beg.where(valid_start)
        end = end.where(valid_start)

    if coord:
        crd = da[dim]
        if isinstance(coord, str):
            crd = getattr(crd.dt, coord)
            coordstr = coord
        else:
            coordstr = dim
        beg = lazy_indexing(crd, beg)
        end = lazy_indexing(crd, end)
    else:
        coordstr = "index"

    out = xr.Dataset({"start": beg, "end": end, "length": length})

    out.start.attrs.update(
        long_name="Start of the season.",
        description=f"First {coordstr} of a run of at least {window} steps respecting the condition.",
    )
    out.end.attrs.update(
        long_name="End of the season.",
        description=f"First {coordstr} of a run of at least {window} "
        "steps breaking the condition, starting after `start`.",
    )
    out.length.attrs.update(
        long_name="Length of the season.",
        description="Number of steps of the original series in the season, between 'start' and 'end'.",
    )
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
      Input N-dimensional DataArray (boolean).
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
    seas = season(da, window, date, dim, coord=False)
    return seas.length


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
      Input N-dimensional DataArray (boolean).
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
      Index (or coordinate if `coord` is not False) of last item in last valid run.
      Returns np.nan if there are no valid runs.
    """
    mid_idx = index_of_date(da[dim], date, max_idxs=1, default=0)
    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel({dim: 0}), np.nan, float).drop_vars(dim)

    end = first_run(
        (~da).where(da[dim] >= da[dim][mid_idx][0]),
        window=window,
        dim=dim,
        coord=coord,
    )
    beg = first_run(da.where(da[dim] < da[dim][mid_idx][0]), window=window, dim=dim)

    if coord:
        last = da[dim][-1]
        if isinstance(coord, str):
            last = getattr(last.dt, coord)
    else:
        last = da[dim].size - 1

    end = xr.where(end.isnull() & beg.notnull(), last, end)
    return end.where(beg.notnull()).drop_vars(dim, errors="ignore")


def first_run_after_date(
    da: xr.DataArray,
    window: int,
    date: Optional[DayOfYearStr] = "07-01",
    dim: str = "time",
    coord: Optional[Union[bool, str]] = "dayofyear",
) -> xr.DataArray:
    """Return the index of the first item of the first run after a given date.

    Parameters
    ----------
    da : xr.DataArray
      Input N-dimensional DataArray (boolean).
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
      Index (or coordinate if `coord` is not False) of first item in the first valid run.
      Returns np.nan if there are no valid runs.
    """
    mid_idx = index_of_date(da[dim], date, max_idxs=1, default=0)
    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel({dim: 0}), np.nan, float).drop_vars(dim)

    return first_run(
        da.where(da[dim] >= da[dim][mid_idx][0]),
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
      Input N-dimensional DataArray (boolean).
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
      Index (or coordinate if `coord` is not False) of last item in last valid run.
      Returns np.nan if there are no valid runs.
    """
    mid_idx = index_of_date(da[dim], date, default=-1)

    if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
        return xr.full_like(da.isel({dim: 0}), np.nan, float).drop_vars(dim)

    run = da.where(da[dim] <= da[dim][mid_idx][0])
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
      The values taken by arr over each run.
    run lengths : np.array
      The length of each run.
    start position : np.array
      The starting index of each run.

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
      Input array.
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Index of first item in first valid run.
      Returns np.nan if there are no valid runs.
    """
    v, rl, pos = rle_1d(arr)
    ind = np.where(v * rl >= window, pos, np.inf).min()

    if np.isinf(ind):
        return np.nan
    return ind


def statistics_run_1d(arr: Sequence[bool], reducer: str, window: int = 1) -> int:
    """Return statistics on lengths of run of identical values.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool)
    reducer : {'mean', 'sum', 'min', 'max', 'std'}
      Reducing function name.
    window: int
      Minimal length of runs to be included in the statistics

    Returns
    -------
    int
      Statistics on length of runs.
    """
    v, rl = rle_1d(arr)[:2]
    if not np.any(v) or np.all(v * rl < window):
        return 0
    func = getattr(np, f"nan{reducer}")
    return func(np.where(v * rl >= window, rl, np.NaN))


def windowed_run_count_1d(arr: Sequence[bool], window: int) -> int:
    """Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool).
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Total number of true values part of a consecutive run at least `window` long.
    """
    v, rl = rle_1d(arr)[:2]
    return np.where(v * rl >= window, rl, 0).sum()


def windowed_run_events_1d(arr: Sequence[bool], window: int) -> xr.DataArray:
    """Return the number of runs of a minimum length.

    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool).
    window : int
      Minimum run length.

    Returns
    -------
    xr.DataArray
      Number of distinct runs of a minimum length.
    """
    v, rl, pos = rle_1d(arr)
    return (v * rl >= window).sum()


def windowed_run_count_ufunc(
    x: Union[xr.DataArray, Sequence[bool]], window: int, dim: str
) -> xr.DataArray:
    """Dask-parallel version of windowed_run_count_1d, ie: the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool).
    window : int
      Minimum duration of consecutive run to accumulate values.
    dim : str
      Dimension along which to calculate windowed run.

    Returns
    -------
    xr.DataArray
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        windowed_run_count_1d,
        x,
        input_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
        keep_attrs=True,
        kwargs={"window": window},
    )


def windowed_run_events_ufunc(
    x: Union[xr.DataArray, Sequence[bool]], window: int, dim: str
) -> xr.DataArray:
    """Dask-parallel version of windowed_run_events_1d, ie: the number of runs at least as long as given duration.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool).
    window : int
      Minimum run length.
    dim : str
      Dimension along which to calculate windowed run.

    Returns
    -------
    xr.DataArray
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        windowed_run_events_1d,
        x,
        input_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
        keep_attrs=True,
        kwargs={"window": window},
    )


def statistics_run_ufunc(
    x: Union[xr.DataArray, Sequence[bool]],
    reducer: str,
    window: int = 1,
    dim: str = "time",
) -> xr.DataArray:
    """Dask-parallel version of statistics_run_1d, ie: the {reducer} number of consecutive true values in array.

    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool)
    reducer: {'min', 'max', 'mean', 'sum', 'std'}
      Reducing function name.
    window : int
      Minimal length of runs.
    dim : str
      The dimension along which the runs are found.

    Returns
    -------
    xr.DataArray
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        statistics_run_1d,
        x,
        input_core_dims=[[dim]],
        kwargs={"reducer": reducer, "window": window},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


def first_run_ufunc(
    x: Union[xr.DataArray, Sequence[bool]],
    window: int,
    dim: str,
) -> xr.DataArray:
    """Dask-parallel version of first_run_1d, ie: the first entry in array of consecutive true values.

    Parameters
    ----------
    x : Union[xr.DataArray, Sequence[bool]]
      Input array (bool).
    window : int
      Minimum run length.
    dim : str
      The dimension along which the runs are found.

    Returns
    -------
    xr.DataArray
      A function operating along the time dimension of a dask-array.
    """
    ind = xr.apply_ufunc(
        first_run_1d,
        x,
        input_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
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
      Values of `da` at indices `index`.
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
            dim = get_temp_dimname(da.dims, "x")
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
    numpy.ndarray
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


def suspicious_run_1d(
    arr: np.ndarray,
    window: int = 10,
    op: str = ">",
    thresh: Optional[float] = None,
) -> np.ndarray:
    """Return True where the array contains a run of identical values.

    Parameters
    ----------
    arr : numpy.ndarray
      Array of values to be parsed.
    window : int
      Minimum run length
    op : {">", ">=", "==", "<", "<= "eq", "gt", "lt", "gteq", "lteq"}, optional
      Operator for threshold comparison. Defaults to ">".
    thresh : float, optional
      Threshold above which values are checked for identical values.

    Returns
    -------
    numpy.ndarray
      Whether or not the data points are part of a run of identical values.
    """
    v, rl, pos = rle_1d(arr)
    sus_runs = rl >= window
    if thresh is not None:
        if op in {">", "gt"}:
            sus_runs = sus_runs & (v > thresh)
        elif op in {"<", "lt"}:
            sus_runs = sus_runs & (v < thresh)
        elif op in {"==", "eq"}:
            sus_runs = sus_runs & (v == thresh)
        elif op in {">=", "gteq"}:
            sus_runs = sus_runs & (v >= thresh)
        elif op in {"<=", "lteq"}:
            sus_runs = sus_runs & (v <= thresh)
        else:
            raise NotImplementedError(f"{op}")

    out = np.zeros_like(arr, dtype=bool)
    for st, l in zip(pos[sus_runs], rl[sus_runs]):
        out[st : st + l] = True
    return out


def suspicious_run(
    arr: xr.DataArray,
    dim: str = "time",
    window: int = 10,
    op: str = ">",
    thresh: Optional[float] = None,
) -> xr.DataArray:
    """Return True where the array contains has runs of identical values, vectorized version.

    In opposition to other run length functions, here the output has the same shape as the input.

    Parameters
    ----------
    arr : xr.DataArray
      Array of values to be parsed.
    dim: str
      Dimension along which to check for runs (default: "time").
    window : int
      Minimum run length
    thresh : float, optional
      Threshold above which values are checked for identical values.
    op: {">", ">=", "==", "<", "<= "eq", "gt", "lt", "gteq", "lteq"}
      Operator for threshold comparison, defaults to ">".

    Returns
    -------
    xarray.DataArray
    """
    return xr.apply_ufunc(
        suspicious_run_1d,
        arr,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        keep_attrs=True,
        kwargs=dict(window=window, op=op, thresh=thresh),
    )
