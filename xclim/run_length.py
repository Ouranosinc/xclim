# -*- coding: utf-8 -*-
"""Run length algorithms module"""

import numpy as np
import xarray as xr
import logging
from warnings import warn

logging.captureWarnings(True)


# TODO: Need to benchmark and adapt for xarray.
def rle(arr):
    """Return the length, starting position and value of consecutive identical values.

    Parameters
    ----------
    arr : sequence
      Array of values to be parsed.

    Returns
    -------
    (values, run lengths, start positions)
    values : np.array
      The values taken by arr over each run
    run lengths : np.array
      The length of each run
    start position : np.array
      The starting index of each run

    Examples
    --------
    >>> a = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    >>> rle(a)
    (array([1, 2, 3]), array([2, 4, 6]), array([0, 2, 6]))

    """
    ia = np.asarray(arr)
    n = len(ia)

    if n == 0:
        e = 'run length array empty'
        warn(e)
        return None, None, None

    y = np.array(ia[1:] != ia[:-1])         # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)       # must include last element position
    rl = np.diff(np.append(-1, i))          # run lengths
    pos = np.cumsum(np.append(0, rl))[:-1]  # positions
    return ia[i], rl, pos


def windowed_run_count(arr, window):
    """Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    arr : bool array
      Input array
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Total number of true values part of a consecutive run at least `window` long.
    """
    v, rl = rle(arr)[:2]
    return np.where(v * rl >= window, rl, 0).sum()


def longest_run(arr):
    """Return the length of the longest consecutive run of identical values.

    Parameters
    ----------
    arr : bool array
      Input array

    Returns
    -------
    int
      Length of longest run.
    """
    v, rl = rle(arr)[:2]
    return np.where(v, rl, 0).max()


# DO NOT USE #
# This function does not work for some corner cases.
def xr_longest_run(da, freq):
    """Return the length of the longest run of true values along the time dimension.

    Parameters
    ----------
    da : xarray.DataArray
      Boolean array with time coordinates.
    freq : str
      Resampling frequency.

    """
    # Create an monotonously increasing index [0,1,2,...] along the time dimension.
    i = xr.DataArray(np.arange(da.time.size), dims='time')
    index = xr.broadcast(i, da)[0]

    ini = xr.DataArray([-1], coords={'time': da.time[:1]}, dims='time')
    end = xr.DataArray([da.time.size], coords={'time': da.time[-1:]}, dims='time')

    masked_da = xr.concat((ini, index.where(~da), end), dim='time')

    # Fill NaNs with the following valid value
    nan_filled_da = masked_da.bfill(dim='time')

    # Find the difference between start and end indices
    diff_ind = nan_filled_da.diff(dim='time') - 1

    # Find the longest run by period - but it does not work if all values are True.
    run = diff_ind.resample(time=freq).max(dim='time')

    # Replace periods where all values are true by the item count
    g = da.resample(time=freq)
    return run.where(~g.all(dim='time'), g.count(dim='time'))


def windowed_run_count_ufunc(x, window):
    """Dask-parallel version of windowed_run_count, ie the number of consecutive true values in
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
    return xr.apply_ufunc(windowed_run_count,
                          x,
                          input_core_dims=[['time'], ],
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.int, ],
                          keep_attrs=True,
                          kwargs={'window': window})


def longest_run_ufunc(x):
    """Dask-parallel version of longest_run, ie the maximum number of consecutive true values in
    array.

    Parameters
    ----------
    x : bool array
      Input array

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(longest_run,
                          x,
                          input_core_dims=[['time'], ],
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.int, ],
                          keep_attrs=True,
                          )
