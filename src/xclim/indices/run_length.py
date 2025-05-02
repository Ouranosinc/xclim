"""
Run-Length Algorithms Submodule
===============================

Computation of statistics on runs of True values in boolean arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit

from xclim.core import DateStr, DayOfYearStr
from xclim.core.options import OPTIONS, RUN_LENGTH_UFUNC
from xclim.core.utils import get_temp_dimname, split_auxiliary_coordinates, uses_dask
from xclim.indices.helpers import resample_map

npts_opt = 9000
"""
Arrays with less than this number of data points per slice will trigger
the use of the ufunc version of run lengths algorithms.
"""


def use_ufunc(
    ufunc_1dim: bool | str,
    da: xr.DataArray,
    dim: str = "time",
    freq: str | None = None,
    index: str = "first",
) -> bool:
    """
    Return whether the ufunc version of run length algorithms should be used with this DataArray or not.

    If ufunc_1dim is 'from_context', the parameter is read from xclim's global (or context) options.
    If it is 'auto', this returns False for dask-backed array and for arrays with more than :py:const:`npts_opt`
    points per slice along `dim`.

    Parameters
    ----------
    ufunc_1dim : {'from_context', 'auto', True, False}
        The method for handling the ufunc parameters.
    da : xr.DataArray
        Input array.
    dim : str
        The dimension along which to find runs.
    freq : str
        Resampling frequency.
    index : {'first', 'last'}
        If 'first' (default), the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    bool
        If ufunc_1dim is "auto", returns True if the array is on dask or too large.
        Otherwise, returns ufunc_1dim.
    """
    if ufunc_1dim is True and freq is not None:
        raise ValueError("Resampling after run length operations is not implemented for 1d method")

    if ufunc_1dim == "from_context":
        ufunc_1dim = OPTIONS[RUN_LENGTH_UFUNC]

    if ufunc_1dim == "auto":
        ufunc_1dim = not uses_dask(da) and (da.size // da[dim].size) < npts_opt
    # If resampling after run length is set up for the computation, the 1d method is not implemented
    # Unless ufunc_1dim is specifically set to False (in which case we flag an error above),
    # we simply forbid this possibility.
    return (index == "first") and ufunc_1dim and (freq is None)


def resample_and_rl(
    da: xr.DataArray,
    resample_before_rl: bool,
    compute: Callable,
    *args,
    freq: str,
    dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    r"""
    Wrap run length algorithms to control if resampling occurs before or after the algorithms.

    Parameters
    ----------
    da : xr.DataArray
        N-dimensional array (boolean).
    resample_before_rl : bool
        Determines whether if input arrays of runs `da` should be separated in period before
        or after the run length algorithms are applied.
    compute : Callable
        Run length function to apply.
    *args : Any
        Positional arguments needed in `compute`.
    freq : str
        Resampling frequency.
    dim : str
        The dimension along which to find runs.
    **kwargs : Any
        Keyword arguments needed in `compute`.

    Returns
    -------
    xr.DataArray
        Output of compute resampled according to frequency {freq}.
    """
    if resample_before_rl:
        out = resample_map(
            da,
            dim,
            freq,
            compute,
            map_kwargs=dict(args=args, freq=None, dim=dim, **kwargs),
        )
    else:
        out = compute(da, *args, dim=dim, freq=freq, **kwargs)
    return out


def _cumsum_reset(
    da: xr.DataArray,
    dim: str = "time",
    index: str = "last",
    reset_on_zero: bool = True,
) -> xr.DataArray:
    """
    Compute the cumulative sum for each series of numbers separated by zero.

    Parameters
    ----------
    da : xr.DataArray
        Input array.
    dim : str
        Dimension name along which the cumulative sum is taken.
    index : {'first', 'last'}
        If 'first', the largest value of the cumulative sum is indexed with the first element in the run.
        If 'last'(default), with the last element in the run.
    reset_on_zero : bool
        If True, the cumulative sum is reset on each zero value of `da`. Otherwise, the cumulative sum resets
        on NaNs. Default is True.

    Returns
    -------
    xr.DataArray
        An array with cumulative sums.
    """
    if index == "first":
        da = da[{dim: slice(None, None, -1)}]

    # Example: da == 100110111 -> cs_s == 100120123
    cs = da.cumsum(dim=dim)  # cumulative sum  e.g. 111233456
    cond = da == 0 if reset_on_zero else da.isnull()  # reset condition
    cs2 = cs.where(cond)  # keep only numbers at positions of zeroes e.g. N11NN3NNN (default)
    cs2[{dim: 0}] = 0  # put a zero in front e.g. 011NN3NNN
    cs2 = cs2.ffill(dim=dim)  # e.g. 011113333
    out = cs - cs2

    if index == "first":
        out = out[{dim: slice(None, None, -1)}]

    return out


# TODO: Check if rle would be more performant with ffill/bfill instead of two times [{dim: slice(None, None, -1)}]
def rle(
    da: xr.DataArray,
    dim: str = "time",
    index: str = "first",
) -> xr.DataArray:
    """
    Run length.

    Despite its name, this is not an actual run-length encoder : it returns an array of the same shape
    as the input with 0 where the input was <= 0, nan where the input was > 0, except on the first (or last) element
    of each run of consecutive > 0 values, where it is set to the sum of the elements within the run.
    For an actual run length encoder, see :py:func:`rle_1d`.

    Usually, the input would be a boolean mask and the first element of each run would then be set to
    the run's length (thus the name), but the function also accepts int and float inputs.

    Parameters
    ----------
    da : xr.DataArray
        Input array.
    dim : str
        Dimension name.
    index : {'first', 'last'}
        If 'first' (default), the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray
        The run length array.
    """
    if da.dtype == bool:
        da = da.astype(int)

    # "first" case: Algorithm is applied on inverted array and output is inverted back
    if index == "first":
        da = da[{dim: slice(None, None, -1)}]

    # Get cumulative sum for each series of 1, e.g. da == 100110111 -> cs_s == 100120123
    cs_s = _cumsum_reset(da, dim)

    # Keep total length of each series (and also keep 0's), e.g. 100120123 -> 100N20NN3
    # Keep numbers with a 0 to the right and also the last number
    cs_s = cs_s.where(da.shift({dim: -1}, fill_value=0) == 0)
    out = cs_s.where(da > 0, 0)  # Reinsert 0's at their original place

    # Inverting back if needed e.g. 100N20NN3 -> 3NN02N001. This is the output of
    # `rle` for 111011001 with index == "first"
    if index == "first":
        out = out[{dim: slice(None, None, -1)}]

    return out


def rle_statistics(
    da: xr.DataArray,
    reducer: str,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    ufunc_1dim: str | bool = "from_context",
    index: str = "first",
) -> xr.DataArray:
    """
    Return the length of consecutive run of True values, according to a reducing operator.

    Parameters
    ----------
    da : xr.DataArray
        N-dimensional array (boolean).
    reducer : str
        Name of the reducing function.
    window : int
        Minimal length of consecutive runs to be included in the statistics.
    dim : str
        Dimension along which to calculate consecutive run; Default: 'time'.
    freq : str
        Resampling frequency.
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        It can be modified globally through the "run_length_ufunc" global option.
    index : {'first', 'last'}
        If 'first' (default), the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray, [int]
        Length of runs of True values along dimension, according to the reducing function (float)
        If there are no runs (but the data is valid), returns 0.
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index, freq=freq)
    if ufunc_1dim:
        rl_stat = statistics_run_ufunc(da, reducer, window, dim)
    else:
        d = rle(da, dim=dim, index=index)

        def get_rl_stat(d):
            rl_stat = getattr(d.where(d >= window), reducer)(dim=dim)
            rl_stat = xr.where((d.isnull() | (d < window)).all(dim=dim), 0, rl_stat)
            return rl_stat

        if freq is None:
            rl_stat = get_rl_stat(d)
        else:
            rl_stat = resample_map(d, dim, freq, get_rl_stat)
    return rl_stat


def longest_run(
    da: xr.DataArray,
    dim: str = "time",
    freq: str | None = None,
    ufunc_1dim: str | bool = "from_context",
    index: str = "first",
) -> xr.DataArray:
    """
    Return the length of the longest consecutive run of True values.

    Parameters
    ----------
    da : xr.DataArray
        N-dimensional array (boolean).
    dim : str
        Dimension along which to calculate consecutive run; Default: 'time'.
    freq : str
        Resampling frequency.
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        It can be modified globally through the "run_length_ufunc" global option.
    index : {'first', 'last'}
        If 'first', the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray, [int]
        Length of the longest run of True values along dimension (int).
    """
    return rle_statistics(
        da,
        reducer="max",
        window=1,
        dim=dim,
        freq=freq,
        ufunc_1dim=ufunc_1dim,
        index=index,
    )


def windowed_run_events(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    ufunc_1dim: str | bool = "from_context",
    index: str = "first",
) -> xr.DataArray:
    """
    Return the number of runs of a minimum length.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum run length.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    freq : str
        Resampling frequency.
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        Ignored when `window=1`. It can be modified globally through the "run_length_ufunc" global option.
    index : {'first', 'last'}
        If 'first', the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray, [int]
        Number of distinct runs of a minimum length (int).
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index, freq=freq)

    if ufunc_1dim:
        out = windowed_run_events_ufunc(da, window, dim)

    else:
        if window == 1:
            shift = 1 * (index == "first") + -1 * (index == "last")
            d = xr.where(da.shift({dim: shift}, fill_value=0) == 0, 1, 0)
            d = d.where(da == 1, 0)
        else:
            d = rle(da, dim=dim, index=index)
            d = xr.where(d >= window, 1, 0)
        if freq is not None:
            d = d.resample({dim: freq})
        out = d.sum(dim=dim)

    return out


def windowed_run_count(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    ufunc_1dim: str | bool = "from_context",
    index: str = "first",
) -> xr.DataArray:
    """
    Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum run length.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    freq : str
        Resampling frequency.
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points. Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        Ignored when `window=1`. It can be modified globally through the "run_length_ufunc" global option.
    index : {'first', 'last'}
        If 'first', the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray, [int]
        Total number of `True` values part of a consecutive runs of at least `window` long.
    """
    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, index=index, freq=freq)

    if ufunc_1dim:
        out = windowed_run_count_ufunc(da, window, dim)

    elif window == 1 and freq is None:
        out = da.sum(dim=dim)

    else:
        d = rle(da, dim=dim, index=index)
        d = d.where(d >= window, 0)
        if freq is not None:
            d = d.resample({dim: freq})
        out = d.sum(dim=dim)

    return out


def windowed_max_run_sum(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    index: str = "first",
) -> xr.DataArray:
    """
    Return the maximum sum of consecutive float values for runs at least as long as the given window length.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum run length.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    freq : str
        Resampling frequency.
    index : {'first', 'last'}
        If 'first', the run length is indexed with the first element in the run.
        If 'last', with the last element in the run.

    Returns
    -------
    xr.DataArray, [int]
        Total number of `True` values part of a consecutive runs of at least `window` long.
    """
    if window == 1 and freq is None:
        out = rle(da, dim=dim, index=index).max(dim=dim)

    else:
        d_rse = rle(da, dim=dim, index=index)
        d_rle = rle((da > 0).astype(bool), dim=dim, index=index)

        d = d_rse.where(d_rle >= window, 0)
        if freq is not None:
            d = d.resample({dim: freq})
        out = d.max(dim=dim)

    return out


def _boundary_run(
    da: xr.DataArray,
    window: int,
    dim: str,
    freq: str | None,
    coord: str | bool | None,
    ufunc_1dim: str | bool,
    position: str,
) -> xr.DataArray:
    """
    Return the index of the first item of the first or last run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive run to accumulate values.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run.
    freq : str
        Resampling frequency.
    coord : Optional[str]
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        Ignored when `window=1`. It can be modified globally through the "run_length_ufunc" global option.
    position : {"first", "last"}
        Determines if the algorithm finds the "first" or "last" run

    Returns
    -------
    xr.DataArray
        Index (or coordinate if `coord` is not False) of first item in first (last) valid run.
        Returns np.nan if there are no valid runs.
    """

    # FIXME: The logic here should not use outside scope variables, but rather pass them as arguments.
    def coord_transform(out, da):
        """Transforms indexes to coordinates if needed, and drops obsolete dim."""
        if coord:
            crd = da[dim]
            if isinstance(coord, str):
                crd = getattr(crd.dt, coord)
            out = lazy_indexing(crd, out)

        if dim in out.coords:
            out = out.drop_vars(dim)
        return out

    # FIXME: The logic here should not use outside scope variables, but rather pass them as arguments.
    # general method to get indices (or coords) of first run
    def find_boundary_run(runs, position):
        if position == "last":
            runs = runs[{dim: slice(None, None, -1)}]
        dmax_ind = runs.argmax(dim=dim)
        # If there are no runs, dmax_ind will be 0: We must replace this with NaN
        out = dmax_ind.where(dmax_ind != runs.argmin(dim=dim))
        if position == "last":
            out = runs[dim].size - out - 1
            runs = runs[{dim: slice(None, None, -1)}]
        out = coord_transform(out, runs)
        return out

    ufunc_1dim = use_ufunc(ufunc_1dim, da, dim=dim, freq=freq)

    da = da.fillna(0)  # We expect a boolean array, but there could be NaNs nonetheless
    if window == 1:
        if freq is not None:
            out = resample_map(da, dim, freq, find_boundary_run, map_kwargs=dict(position=position))
        else:
            out = find_boundary_run(da, position)

    elif ufunc_1dim:
        if position == "last":
            da = da[{dim: slice(None, None, -1)}]
        out = first_run_ufunc(x=da, window=window, dim=dim)
        if position == "last" and not coord:
            out = da[dim].size - out - 1
            da = da[{dim: slice(None, None, -1)}]
        out = coord_transform(out, da)

    else:
        # _cusum_reset_on_zero() is an intermediate step in rle, which is sufficient here
        d = _cumsum_reset(da, dim=dim, index=position)
        d = xr.where(d >= window, 1, 0)
        # for "first" run, return "first" element in the run (and conversely for "last" run)
        if freq is not None:
            out = resample_map(d, dim, freq, find_boundary_run, map_kwargs=dict(position=position))
        else:
            out = find_boundary_run(d, position)

    return out


def first_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    coord: str | bool | None = False,
    ufunc_1dim: str | bool = "from_context",
) -> xr.DataArray:
    """
    Return the index of the first item of the first run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive run to accumulate values.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    freq : str
        Resampling frequency.
    coord : Optional[str]
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
        for DataArray with a small number of grid points.
        Ignored when `window=1`. It can be modified globally through the "run_length_ufunc" global option.

    Returns
    -------
    xr.DataArray
        Index (or coordinate if `coord` is not False) of first item in first valid run.
        Returns np.nan if there are no valid runs.
    """
    out = _boundary_run(
        da,
        window=window,
        dim=dim,
        freq=freq,
        coord=coord,
        ufunc_1dim=ufunc_1dim,
        position="first",
    )
    return out


def last_run(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    freq: str | None = None,
    coord: str | bool | None = False,
    ufunc_1dim: str | bool = "from_context",
) -> xr.DataArray:
    """
    Return the index of the last item of the last run of at least a given length.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive run to accumulate values.
        When equal to 1, an optimized version of the algorithm is used.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    freq : str
        Resampling frequency.
    coord : Optional[str]
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').
    ufunc_1dim : Union[str, bool]
        Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
        usage based on number of data points.  Using `1D_ufunc=True` is typically more efficient
        for a DataArray with a small number of grid points.
        Ignored when `window=1`. It can be modified globally through the "run_length_ufunc" global option.

    Returns
    -------
    xr.DataArray
        Index (or coordinate if `coord` is not False) of last item in last valid run.
        Returns np.nan if there are no valid runs.
    """
    out = _boundary_run(
        da,
        window=window,
        dim=dim,
        freq=freq,
        coord=coord,
        ufunc_1dim=ufunc_1dim,
        position="last",
    )
    return out


# TODO: Add window arg
# TODO: Inverse window arg to tolerate holes?
def run_bounds(mask: xr.DataArray, dim: str = "time", coord: bool | str = True):
    """
    Return the start and end dates of boolean runs along a dimension.

    Parameters
    ----------
    mask : xr.DataArray
        Boolean array.
    dim : str
        Dimension along which to look for runs.
    coord : bool or str
        If `True`, return values of the coordinate, if a string, returns values from `dim.dt.<coord>`.
        If `False`, return indexes.

    Returns
    -------
    xr.DataArray
        With ``dim`` reduced to "events" and "bounds". The events dim is as long as needed, padded with NaN or NaT.
    """
    if uses_dask(mask):
        raise NotImplementedError("Dask arrays not supported as we can't know the final event number before computing.")

    diff = xr.concat((mask.isel({dim: [0]}).astype(int), mask.astype(int).diff(dim)), dim)

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

        starts = lazy_indexing(crd, starts)
        ends = lazy_indexing(crd, ends)
    return xr.concat((starts, ends), "bounds")


def keep_longest_run(da: xr.DataArray, dim: str = "time", freq: str | None = None) -> xr.DataArray:
    """
    Keep the longest run along a dimension.

    Parameters
    ----------
    da : xr.DataArray
        Boolean array.
    dim : str
        Dimension along which to check for the longest run.
    freq : str
        Resampling frequency.

    Returns
    -------
    xr.DataArray, [bool]
        Boolean array similar to da but with only one run, the (first) longest.
    """
    # Get run lengths
    rls = rle(da, dim)

    def _get_out(_rls):  # numpydoc ignore=GL08
        _out = xr.where(
            # Construct an integer array and find the max
            _rls[dim].copy(data=np.arange(_rls[dim].size)) == _rls.argmax(dim),
            _rls + 1,  # Add one to the First longest run
            _rls,
        )
        _out = _out.ffill(dim) == _out.max(dim)
        return _out

    if freq is not None:
        out = resample_map(rls, dim, freq, _get_out)
    else:
        out = _get_out(rls)

    return da.copy(data=out.transpose(*da.dims).data)


def runs_with_holes(
    da_start: xr.DataArray,
    window_start: int,
    da_stop: xr.DataArray,
    window_stop: int,
    dim: str = "time",
) -> xr.DataArray:
    """
    Extract events, i.e. runs whose starting and stopping points are defined through run length conditions.

    Parameters
    ----------
    da_start : xr.DataArray
        Input array where run sequences are searched to define the start points in the main runs.
    window_start : int
        Number of True (1) values needed to start a run in `da_start`.
    da_stop : xr.DataArray
        Input array where run sequences are searched to define the stop points in the main runs.
    window_stop : int
        Number of True (1) values needed to start a run in `da_stop`.
    dim : str
        Dimension name.

    Returns
    -------
    xr.DataArray
        Output array with 1's when in a run sequence and with 0's elsewhere.

    Notes
    -----
    A season (as defined in ``season``) could be considered as an event with ``window_stop == window_start``
    and ``da_stop == 1 - da_start``, although it has more constraints on when to start and stop a run through
    the `date` argument and only one season can be found.
    """
    da_start = da_start.astype(int).fillna(0)
    da_stop = da_stop.astype(int).fillna(0)

    start_runs = _cumsum_reset(da_start, dim=dim, index="first")
    stop_runs = _cumsum_reset(da_stop, dim=dim, index="first")
    start_positions = xr.where(start_runs >= window_start, 1, np.nan)
    stop_positions = xr.where(stop_runs >= window_stop, 0, np.nan)

    # start positions (1) are f-filled until a stop position (0) is met
    runs = stop_positions.combine_first(start_positions).ffill(dim=dim).fillna(0)
    return runs


def season_start(
    da: xr.DataArray,
    window: int,
    mid_date: DayOfYearStr | None = None,
    dim: str = "time",
    coord: str | bool | None = False,
) -> xr.DataArray:
    """
    Start of a season.

    See :py:func:`season`.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive values to start and end the season.
    mid_date : DayOfYearStr, optional
        The date (in MM-DD format) that a season must include to be considered valid.
    dim : str
        Dimension along which to calculate the season (default: 'time').
    coord : Optional[str]
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.DataArray
        Start of the season, units depend on `coord`.

    See Also
    --------
    season : Calculate the bounds of a season along a dimension.
    season_end : End of a season.
    season_length : Length of a season.
    """
    return first_run_before_date(da, window=window, date=mid_date, dim=dim, coord=coord)


def season_end(
    da: xr.DataArray,
    window: int,
    mid_date: DayOfYearStr | None = None,
    dim: str = "time",
    coord: str | bool | None = False,
    _beg: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    End of a season.

    See :py:func:`season`. Similar to :py:func:`first_run_after_date` but here a season
    must have a start for an end to be valid. Also, if no end is found but a start was found
    the end is set to the last element of the series.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive values to start and end the season.
    mid_date : DayOfYearStr, optional
        The date (in MM-DD format) that a run must include to be considered valid.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    coord : str, optional
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').
    _beg : xr.DataArray, optional
        If given, the start of the season. This is used to avoid recomputing the start.

    Returns
    -------
    xr.DataArray
        End of the season, units depend on `coord`.
        If there is a start is found but no end, the end is set to the last element.

    See Also
    --------
    season : Calculate the bounds of a season along a dimension.
    season_start : Start of a season.
    season_length : Length of a season.
    """
    # Fast path for `season` and `season_length`
    if _beg is not None:
        beg = _beg
    else:
        beg = season_start(da, window=window, dim=dim, mid_date=mid_date, coord=False)
    # Invert the condition and mask all values after beginning
    # we fillna(0) as so to differentiate series with no runs and all-nan series
    not_da = (~da).where(da[dim].copy(data=np.arange(da[dim].size)) >= beg.fillna(0))
    end = first_run_after_date(not_da, window=window, dim=dim, date=mid_date, coord=False)
    if _beg is None:
        # Where end is null and beg is not null (valid data, no end detected), put the last index
        # Don't do this in the fast path, so that the length can use this information
        end = xr.where(end.isnull() & beg.notnull(), da[dim].size - 1, end)
        end = end.where(beg.notnull())
    if coord:
        crd = da[dim]
        if isinstance(coord, str):
            crd = getattr(crd.dt, coord)
        end = lazy_indexing(crd, end)
    return end


def season(
    da: xr.DataArray,
    window: int,
    mid_date: DayOfYearStr | None = None,
    dim: str = "time",
    stat: str | None = None,
    coord: str | bool | None = False,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the bounds of a season along a dimension.

    A "season" is a run of True values that may include breaks under a given length (`window`).
    The start is computed as the first run of `window` True values, and the end as the first subsequent run
    of  `window` False values. The end element is the first element after the season.
    If a date is given, it must be included in the season
    (i.e. the start cannot occur later and the end cannot occur earlier).

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive values to start and end the season.
    mid_date : DayOfYearStr, optional
        The date (in MM-DD format) that a run must include to be considered valid.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    stat : str, optional
        Not currently implemented.
        If not None, return a statistic of the season. The statistic is calculated on the season's values.
    coord : Optional[str]
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (ex: 'dayofyear').

    Returns
    -------
    xr.Dataset
        The Dataset variables:
            start : start of the season (index or units depending on ``coord``)
            end : end of the season (index or units depending on ``coord``)
            length : length of the season (in number of elements along ``dim``)

    See Also
    --------
    season_start : Start of a season.
    season_end : End of a season.
    season_length : Length of a season.

    Notes
    -----
    The run can include holes of False or NaN values, so long as they do not exceed the window size.

    If a date is given, the season start and end are forced to be on each side of this date. This means that
    even if the "real" season has been over for a long time, this is the date used in the length calculation.
    e.g. Length of the "warm season", where T > 25°C, with date = 1st August. Let's say the temperature is over
    25 for all June, but July and august have very cold temperatures. Instead of returning 30 days (June),
    the function will return 61 days (July + June).

    The season's length is always the difference between the end and the start. Except if no season end was
    found before the end of the data. In that case the end is set to last element and the length is set to
    the data size minus the start index. Thus, for the specific case, :math:`length = end - start + 1`,
    because the end falls on the last element of the season instead of the subsequent one.
    """
    beg = season_start(da, window=window, dim=dim, mid_date=mid_date, coord=False)
    # Use fast path in season_end : no recomputing of start,
    # no masking of end where beg.isnull(), and don't set end if none found
    end = season_end(da, window=window, dim=dim, mid_date=mid_date, _beg=beg, coord=False)
    # Three cases :
    #           start     no start
    # end       e - s        0
    # no end   size - s      0
    # da is boolean, so we have no way of knowing if the absence of season
    # is due to missing values or to an actual absence.
    length = xr.where(
        beg.isnull(),
        0,
        # Where there is no end, from the start to the boundary
        xr.where(end.isnull(), da[dim].size - beg, end - beg),
    )
    # Now masks ends
    # Still give an end if we didn't find any : put the last element
    # This breaks the length = end - beg, but yields a truer length
    end = xr.where(end.isnull() & beg.notnull(), da[dim].size - 1, end)
    end = end.where(beg.notnull())

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
    mid_date: DayOfYearStr | None = None,
    dim: str = "time",
) -> xr.DataArray:
    """
    Length of a season.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive values to start and end the season.
    mid_date : DayOfYearStr, optional
        The date (in MM-DD format) that a run must include to be considered valid.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').

    Returns
    -------
    xr.DataArray, [int]
        Length of the season, in number of elements along dimension `time`.

    See Also
    --------
    season : Calculate the bounds of a season along a dimension.
    season_start : Start of a season.
    season_end : End of a season.
    """
    seas = season(da, window, mid_date, dim, coord=False)
    return seas.length


def run_end_after_date(
    da: xr.DataArray,
    window: int,
    date: DayOfYearStr = "07-01",
    dim: str = "time",
    coord: bool | str | None = "dayofyear",
) -> xr.DataArray:
    """
    Return the index of the first item after the end of a run after a given date.

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
    date: DayOfYearStr | None = "07-01",
    dim: str = "time",
    coord: bool | str | None = "dayofyear",
) -> xr.DataArray:
    """
    Return the index of the first item of the first run after a given date.

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
    coord: bool | str | None = "dayofyear",
) -> xr.DataArray:
    """
    Return the index of the last item of the last run before a given date.

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


def first_run_before_date(
    da: xr.DataArray,
    window: int,
    date: DayOfYearStr | None = "07-01",
    dim: str = "time",
    coord: bool | str | None = "dayofyear",
) -> xr.DataArray:
    """
    Return the index of the first item of the first run before a given date.

    Parameters
    ----------
    da : xr.DataArray
        Input N-dimensional DataArray (boolean).
    window : int
        Minimum duration of consecutive run to accumulate values.
    date : DayOfYearStr
        The date before which to look for the run.
    dim : str
        Dimension along which to calculate consecutive run (default: 'time').
    coord : bool or str, optional
        If not False, the function returns values along `dim` instead of indexes.
        If `dim` has a datetime dtype, `coord` can also be a str of the name of the
        DateTimeAccessor object to use (e.g. 'dayofyear').

    Returns
    -------
    xr.DataArray
        Index (or coordinate if `coord` is not False) of first item in the first valid run.
        Returns np.nan if there are no valid runs.
    """
    if date is not None:
        mid_idx = index_of_date(da[dim], date, max_idxs=1, default=0)
        if mid_idx.size == 0:  # The date is not within the group. Happens at boundaries.
            return xr.full_like(da.isel({dim: 0}), np.nan, float).drop_vars(dim)
        # Mask anything after the mid_date + window - 1
        # Thus, the latest run possible can begin on the day just before mid_idx
        da = da.where(da[dim] < da[dim][mid_idx + window - 1][0])

    return first_run(
        da,
        window=window,
        dim=dim,
        coord=coord,
    )


@njit
def _rle_1d(ia):
    y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
    i = np.append(np.nonzero(y)[0], ia.size - 1)  # must include last element position
    rl = np.diff(np.append(-1, i))  # run lengths
    pos = np.cumsum(np.append(0, rl))[:-1]  # positions
    return ia[i], rl, pos


def rle_1d(
    arr: int | float | bool | Sequence[int | float | bool],
) -> tuple[np.array, np.array, np.array]:
    """
    Return the length, starting position and value of consecutive identical values.

    In opposition to py:func:`rle`, this is an actuel run length encoder.

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
        warn("run length array empty")
        # Returning None makes some other 1d func below fail.
        return np.array(np.nan), 0, np.array(np.nan)
    return _rle_1d(ia)


def first_run_1d(arr: Sequence[int | float], window: int) -> int | np.nan:
    """
    Return the index of the first item of a run of at least a given length.

    Parameters
    ----------
    arr : sequence of int or float
        Input array.
    window : int
        Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int or np.nan
        Index of first item in first valid run.
        Returns np.nan if there are no valid runs.
    """
    v, rl, pos = rle_1d(arr)
    ind = np.where(v * rl >= window, pos, np.inf).min()  # noqa

    if np.isinf(ind):
        return np.nan
    return ind


def statistics_run_1d(arr: Sequence[bool], reducer: str, window: int) -> int:
    """
    Return statistics on lengths of run of identical values.

    Parameters
    ----------
    arr : sequence of bool
        Input array (bool).
    reducer : {"mean", "sum", "min", "max", "std", "count"}
        Reducing function name.
    window : int
        Minimal length of runs to be included in the statistics.

    Returns
    -------
    int
        Statistics on length of runs.
    """
    v, rl = rle_1d(arr)[:2]
    if not np.any(v) or np.all(v * rl < window):
        return 0
    if reducer == "count":
        return (v * rl >= window).sum()
    func = getattr(np, f"nan{reducer}")
    return func(np.where(v * rl >= window, rl, np.nan))


def windowed_run_count_1d(arr: Sequence[bool], window: int) -> int:
    """
    Return the number of consecutive true values in array for runs at least as long as given duration.

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
    """
    Return the number of runs of a minimum length.

    Parameters
    ----------
    arr : Sequence[bool]
        Input array (bool).
    window : int
        Minimum run length.

    Returns
    -------
    xr.DataArray, [int]
        Number of distinct runs of a minimum length.
    """
    v, rl, _ = rle_1d(arr)
    return (v * rl >= window).sum()


def windowed_run_count_ufunc(x: xr.DataArray | Sequence[bool], window: int, dim: str) -> xr.DataArray:
    """
    Dask-parallel version of windowed_run_count_1d.

    The number of consecutive true values in array for runs at least as long as given duration.

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


def windowed_run_events_ufunc(x: xr.DataArray | Sequence[bool], window: int, dim: str) -> xr.DataArray:
    """
    Dask-parallel version of windowed_run_events_1d.

    The number of runs at least as long as given duration.

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
    x: xr.DataArray | Sequence[bool],
    reducer: str,
    window: int,
    dim: str = "time",
) -> xr.DataArray:
    """
    Dask-parallel version of statistics_run_1d.

    The {reducer} number of consecutive true values in array.

    Parameters
    ----------
    x : sequence of bool
        Input array (bool).
    reducer : {'min', 'max', 'mean', 'sum', 'std'}
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
    x: xr.DataArray | Sequence[bool],
    window: int,
    dim: str,
) -> xr.DataArray:
    """
    Dask-parallel version of first_run_1d.

    The first entry in array of consecutive true values.

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


def lazy_indexing(da: xr.DataArray, index: xr.DataArray, dim: str | None = None) -> xr.DataArray:
    """
    Get values of `da` at indices `index` in a NaN-aware and lazy manner.

    Parameters
    ----------
    da : xr.DataArray
        Input array. If not 1D, `dim` must be given and must not appear in index.
    index : xr.DataArray
        N-d integer indices, if DataArray is not 1D, all dimensions of index must be in DataArray.
    dim : str, optional
        Dimension along which to index, unused if `da` is 1D, should not be present in `index`.

    Returns
    -------
    xr.DataArray
        Values of `da` at indices `index`.
    """
    if da.ndim == 1:
        # Case where da is 1D and index is N-D
        # Slightly better performance using map_blocks, over an apply_ufunc
        def _index_from_1d_array(indices, array):
            return array[indices]

        idx_ndim = index.ndim
        if idx_ndim == 0:
            # The 0-D index case, we add a dummy dimension to help dask
            dim = get_temp_dimname(da.dims, "x")
            index = index.expand_dims(dim)
        # Which indexes to mask.
        invalid = index.isnull()
        # NaN-indexing doesn't work, so fill with 0 and cast to int
        index = index.fillna(0).astype(int)

        # No need for coords, we extract by integer index.
        # Renaming with no name to fix bug in xr 2024.01.0
        tmpname = get_temp_dimname(da.dims, "temp")
        da2 = xr.DataArray(da.data, dims=(tmpname,), name=None)
        # Map blocks chunks aux coords. Remove them to avoid the alignment check load in `where`
        index, auxcrd = split_auxiliary_coordinates(index)
        # for each chunk of index, take corresponding values from da
        out = index.map_blocks(_index_from_1d_array, args=(da2,)).rename(da.name)
        # mask where index was NaN. Drop any auxiliary coord, they are already on `out`.
        # Chunked aux coord would have the same name on both sides and xarray will want to check if they are equal,
        # which means loading them making lazy_indexing not lazy. same issue as above
        out = out.where(~invalid.drop_vars([crd for crd in invalid.coords if crd not in invalid.dims]))
        out = out.assign_coords(auxcrd.coords)
        if idx_ndim == 0:
            # 0-D case, drop useless coords and dummy dim
            out = out.drop_vars(da.dims[0], errors="ignore").squeeze()
        return out.drop_vars(dim or da.dims[0], errors="ignore")

    # Case where index.dims is a subset of da.dims.
    if dim is None:
        diff_dims = set(da.dims) - set(index.dims)
        if len(diff_dims) == 0:
            raise ValueError("da must have at least one dimension more than index for lazy_indexing.")
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
    date: DateStr | DayOfYearStr | None,
    max_idxs: int | None = None,
    default: int = 0,
) -> np.ndarray:
    """
    Get the index of a date in a time array.

    Parameters
    ----------
    time : xr.DataArray
        An array of datetime values, any calendar.
    date : DayOfYearStr or DateStr, optional
        A string in the "yyyy-mm-dd" or "mm-dd" format.
        If None, returns default.
    max_idxs : int, optional
        Maximum number of returned indexes.
    default : int
        Index to return if date is None.

    Returns
    -------
    numpy.ndarray
        1D array of integers, indexes of `date` in `time`.

    Raises
    ------
    ValueError
        If there are most instances of `date` in `time` than `max_idxs`.
    """
    if date is None:
        return np.array([default])
    if len(date.split("-")) == 2:
        date = f"1840-{date}"
        date = datetime.strptime(date, "%Y-%m-%d")
        year_cond = True
    else:
        date = datetime.strptime(date, "%Y-%m-%d")
        year_cond = time.dt.year == date.year

    idxs = np.where(year_cond & (time.dt.month == date.month) & (time.dt.day == date.day))[0]
    if max_idxs is not None and idxs.size > max_idxs:
        raise ValueError(f"More than {max_idxs} instance of date {date} found in the coordinate array.")
    return idxs


def suspicious_run_1d(
    arr: np.ndarray,
    window: int = 10,
    op: str = ">",
    thresh: float | None = None,
) -> np.ndarray:
    """
    Return `True` where the array contains a run of identical values.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of values to be parsed.
    window : int
        Minimum run length.
    op : {">", ">=", "==", "<", "<=", "eq", "gt", "lt", "gteq", "lteq", "ge", "le"}
        Operator for threshold comparison. Defaults to ">".
    thresh : float, optional
        Threshold compared against which values are checked for identical values.

    Returns
    -------
    numpy.ndarray
        Whether the data points are part of a run of identical values or not.
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
        elif op in {"!=", "ne"}:
            sus_runs = sus_runs & (v != thresh)
        elif op in {">=", "gteq", "ge"}:
            sus_runs = sus_runs & (v >= thresh)
        elif op in {"<=", "lteq", "le"}:
            sus_runs = sus_runs & (v <= thresh)
        else:
            raise NotImplementedError(f"{op}")

    out = np.zeros_like(arr, dtype=bool)
    for st, l in zip(pos[sus_runs], rl[sus_runs], strict=False):  # noqa: E741
        out[st : st + l] = True  # noqa: E741
    return out


def suspicious_run(
    arr: xr.DataArray,
    dim: str = "time",
    window: int = 10,
    op: str = ">",
    thresh: float | None = None,
) -> xr.DataArray:
    """
    Return `True` where the array contains has runs of identical values, vectorized version.

    In opposition to other run length functions, here the output has the same shape as the input.

    Parameters
    ----------
    arr : xr.DataArray
        Array of values to be parsed.
    dim : str
        Dimension along which to check for runs (default: "time").
    window : int
        Minimum run length.
    op : {">", ">=", "==", "<", "<=", "eq", "gt", "lt", "gteq", "lteq"}
        Operator for threshold comparison, defaults to ">".
    thresh : float, optional
        Threshold above which values are checked for identical values.

    Returns
    -------
    xarray.DataArray
        A boolean array of the same shape as the input, indicating where runs of identical values are found.
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
        kwargs={"window": window, "op": op, "thresh": thresh},
    )


def _find_events(da_start, da_stop, data, window_start, window_stop):
    """
    Actual finding of events for each period.

    Get basic blocks to work with, our runs with holes and the lengths of those runs.
    Series of ones indicating where we have continuous runs with pauses
    not exceeding `window_stop`
    """
    runs = runs_with_holes(da_start, window_start, da_stop, window_stop)

    # Compute the length of freezing rain events
    # I think int16 is safe enough, fillna first to suppress warning
    ds = rle(runs).fillna(0).astype(np.int16).to_dataset(name="event_length")
    # Time duration where the precipitation threshold is exceeded during an event
    # (duration of complete run - duration of holes in the run )
    ds["event_effective_length"] = _cumsum_reset(da_start.where(runs == 1), index="first", reset_on_zero=False).astype(
        np.int16
    )

    if data is not None:
        # Ex: Cumulated precipitation in a given freezing rain event
        ds["event_sum"] = _cumsum_reset(data.where(runs == 1), index="first", reset_on_zero=False)

    # Keep time as a variable, it will be used to keep start of events
    ds["event_start"] = ds["time"].broadcast_like(ds)  # .astype(int)
    # We convert to an integer for the filtering, time object won't do well in the apply_ufunc/vectorize
    time_min = ds.event_start.min()
    ds["event_start"] = ds.event_start.copy(
        data=(ds.event_start - time_min).values.astype("timedelta64[s]").astype(int)
    )

    # Filter events: Reduce time dimension
    def _filter_events(da, rl, max_event_number):
        out = np.full(max_event_number, np.nan)
        events_start = da[rl > 0]
        out[: len(events_start)] = events_start
        return out

    # Dask inputs need to be told their length before computing anything.
    max_event_number = int(np.ceil(da_start.time.size / (window_start + window_stop)))
    ds = xr.apply_ufunc(
        _filter_events,
        ds,
        ds.event_length,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["event"]],
        kwargs=dict(max_event_number=max_event_number),
        dask_gufunc_kwargs=dict(output_sizes={"event": max_event_number}),
        dask="parallelized",
        vectorize=True,
    )

    # convert back start to a time
    if time_min.dtype == "O":
        # Can't add a numpy array of timedeltas to an array of cftime (because they have non-compatible dtypes)
        # Also, we can't add cftime to NaTType. So we fill with negative timedeltas and mask them after the addition

        def _get_start_cftime(deltas, time_min=None):
            starts = time_min + pd.to_timedelta(deltas, "s").to_pytimedelta()
            starts[starts < time_min] = np.nan
            return starts

        ds["event_start"] = xr.apply_ufunc(
            _get_start_cftime,
            ds.event_start.fillna(-1),
            dask="parallelized",
            kwargs={"time_min": time_min.item()},
            output_dtypes=[time_min.dtype],
        )
    else:
        ds["event_start"] = ds.event_start.copy(data=time_min.values + ds.event_start.data.astype("timedelta64[s]"))

    ds["event"] = np.arange(1, ds.event.size + 1)
    ds["event_length"].attrs["units"] = "1"
    ds["event_effective_length"].attrs["units"] = "1"
    ds["event_start"].attrs["units"] = ""
    if data is not None:
        ds["event_sum"].attrs["units"] = data.units
    return ds


# TODO: Implement more event stats ? (max, effective sum, etc)
def find_events(
    condition: xr.DataArray,
    window: int,
    condition_stop: xr.DataArray | None = None,
    window_stop: int = 1,
    data: xr.DataArray | None = None,
    freq: str | None = None,
):
    """
    Find events (runs).

    An event starts with a run of ``window`` consecutive True values in the condition
    and stops with ``window_stop`` consecutive True values in the stop condition.

    This returns a Dataset with each event along an `event` dimension.
    It does not perform statistics over the events like other function in this module do.

    Parameters
    ----------
    condition : DataArray of bool
        The boolean mask, true where the start condition of the event is fulfilled.
    window : int
        The number of consecutive True values for an event to start.
    condition_stop : DataArray of bool, optional
        The stopping boolean mask, true where the end condition of the event is fulfilled.
        Defaults to the opposite of ``condition``.
    window_stop : int
        The number of consecutive True values in ``condition_stop`` for an event to end.
        Defaults to 1.
    data : DataArray, optional
        The actual data. If present, its sum within each event is added to the output.
    freq : str, optional
        A frequency to divide the data into periods. If absent, the output has not time dimension.
        If given, the events are searched within in each resample period independently.

    Returns
    -------
    xr.Dataset, same shape as the data (and the time dimension is resample or removed, according to ``freq``).
        The Dataset has the following variables:
            event_length: The number of time steps in each event
            event_effective_length: The number of time steps of even event where the start condition is true.
            event_start: The datetime of the start of the run.
            event_sum: The sum within each event, only considering steps where start condition is true (if ``data``).
    """
    if condition_stop is None:
        condition_stop = ~condition

    if freq is None:
        return _find_events(condition, condition_stop, data, window, window_stop)

    ds = xr.Dataset({"da_start": condition, "da_stop": condition_stop})
    if data is not None:
        ds = ds.assign(data=data)
    return ds.resample(time=freq).map(
        lambda grp: _find_events(grp.da_start, grp.da_stop, grp.get("data", None), window, window_stop)
    )
