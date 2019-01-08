# -*- coding: utf-8 -*-
"""Run length algorithms module"""

import numpy as np
import xarray as xr
import logging
from warnings import warn

logging.captureWarnings(True)


def Ndim_rle(da, dim='time'):
    n = len(da[dim])
    i = xr.DataArray(np.arange(da[dim].size), dims=dim)
    ind = xr.broadcast(i, da)[0]
    b = ind.where(~da)  # find indexes where false

    end1 = da.where(b[dim] == b[dim][-1], drop=True) * 0 + n  # add additional end value index (deal with end cases)
    start1 = da.where(b[dim] == b[dim][0], drop=True) * 0 - 1  # add additional start index (deal with end cases)
    b = xr.concat([start1, b, end1], dim)
    z = b.bfill(dim=dim)
    z = z.where(~np.isnan(z), n)  # backfill not filling last sequence from end1?
    d = z.diff(dim=dim) - 1
    d = d.where(d >= 0)
    return d


def Ndim_longest_run(da, dim='time'):
    """Return the length of the longest consecutive run of True values.

        Parameters
        ----------
        arr : N-dimensional array (boolean)
          Input array
        dim : Xarray dimension (default = 'time')
          Dimension along which to calculate consecutive run

        Returns
        -------
        N-dimensional array (int)
          Length of longest run of True values along dimension
        """

    d = Ndim_rle(da, dim=dim)
    rl_long = d.max(dim=dim)

    return rl_long


def Ndim_windowed_run_events(da, window, dim='time'):
    """Return the number of runs of a minimum length.

        Parameters
        ----------
        da: N-dimensional Xarray data array  (boolean)
          Input data array
        window : int
          Minimum run length.
        dim : Xarray dimension (default = 'time')
          Dimension along which to calculate consecutive run

        Returns
        -------
        out : N-dimensional xarray data array (int)
          Number of distinct runs of a minimum length.
        """
    d = Ndim_rle(da, dim=dim)
    out = (d >= window).sum(dim=dim)
    return out


def Ndim_windowed_run_count(da, window, dim='time'):
    """Return the number of consecutive true values in array for runs at least as long as given duration.

        Parameters
        ----------
        da: N-dimensional Xarray data array  (boolean)
          Input data array
        window : int
          Minimum run length.
        dim : Xarray dimension (default = 'time')
          Dimension along which to calculate consecutive run


        Returns
        -------
        out : N-dimensional xarray data array (int)
          Total number of true values part of a consecutive runs of at least `window` long.
        """
    d = Ndim_rle(da, dim=dim)
    out = d.where(d >= window, 0).sum(dim=dim)
    return out


def Ndim_first_run(da, window, dim='time'):
    """Return the index of the first item of a run of at least a given length.

        Parameters
        ----------
        ----------
        arr : bool array
          Input array
        window : int
          Minimum duration of consecutive run to accumulate values.

        Returns
        -------
        int
          Index of first item in first valid run. Returns np.nan if there are no valid run.
        """
    dims = list(da.dims)
    if not 'time' in dims:
        da['time'] = da[dim]
        da.swap_dims({dim: 'time'})
    da = da.astype('int')
    i = xr.DataArray(np.arange(da[dim].size), dims=dim)
    ind = xr.broadcast(i, da)[0]
    wind_sum = da.rolling(time=window).sum(dim=dim)
    out = ind.where(wind_sum >= window).min(dim=dim) - (
        window - 1)  # remove window -1 as rolling result index is last element of the moving window
    return out


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

    y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)  # must include last element position
    rl = np.diff(np.append(-1, i))  # run lengths
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


def first_run(arr, window):
    """Return the index of the first item of a run of at least a given length.

    Parameters
    ----------
    ----------
    arr : bool array
      Input array
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Index of first item in first valid run. Returns np.nan if there are no valid run.
    """
    v, rl, pos = rle(arr)
    ind = np.where(v * rl >= window, pos, np.inf).min()

    if np.isinf(ind):
        return np.nan
    return ind


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


def windowed_run_events(arr, window):
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
    v, rl, pos = rle(arr)
    return (v * rl >= window).sum()


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


def windowed_run_events_ufunc(x, window):
    """Dask-parallel version of windowed_run_events, ie the number of runs at least as long as given duration.

    Parameters
    ----------
    x : bool array
      Input array
    window : int
      Minimum run length

    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(windowed_run_events,
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


def first_run_ufunc(x, window, index=None):
    ind = xr.apply_ufunc(first_run,
                         x,
                         input_core_dims=[['time'], ],
                         vectorize=True,
                         dask='parallelized',
                         output_dtypes=[np.float, ],
                         keep_attrs=True,
                         kwargs={'window': window}
                         )

    if index is not None and ~np.isnan(ind):
        val = getattr(x.indexes['time'], index)
        i = int(ind.data)
        ind.data = val[i]

    return ind
