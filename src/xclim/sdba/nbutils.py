# pylint: disable=no-value-for-parameter
"""
Numba-accelerated Utilities
===========================
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numba import boolean, float32, float64, guvectorize, njit
from xarray import DataArray, apply_ufunc


try:
    from fastnanquantile.xrcompat import xr_apply_nanquantile

    USE_FASTNANQUANTILE = True
except ImportError:
    USE_FASTNANQUANTILE = False


@njit(
    fastmath={"arcp", "contract", "reassoc", "nsz", "afn"},
    nogil=True,
    cache=False,
)
def _get_indexes(
    arr: np.array, virtual_indexes: np.array, valid_values_count: np.array
) -> tuple[np.array, np.array]:
    """
    Get the valid indexes of arr neighbouring virtual_indexes.

    Parameters
    ----------
    arr : array-like
    virtual_indexes : array-like
    valid_values_count : array-like

    Returns
    -------
    array-like, array-like
        A tuple of virtual_indexes neighbouring indexes (previous and next)

    Notes
    -----
    This is a companion function to linear interpolation of quantiles.
    """
    previous_indexes = np.asarray(np.floor(virtual_indexes))
    next_indexes = np.asarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if (arr.dtype is np.dtype(np.float64)) or (arr.dtype is np.dtype(np.float32)):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


@njit(
    fastmath={"arcp", "contract", "reassoc", "nsz", "afn"},
    nogil=True,
    cache=False,
)
def _linear_interpolation(
    left: np.array,
    right: np.array,
    gamma: np.array,
) -> np.array:
    """
    Compute the linear interpolation weighted by gamma on each point of two same shape arrays.

    Parameters
    ----------
    left : array_like
        Left bound.
    right : array_like
        Right bound.
    gamma : array_like
        The interpolation weight.

    Returns
    -------
    array_like

    Notes
    -----
    This is a companion function for `_nan_quantile_1d`
    """
    diff_b_a = np.subtract(right, left)
    lerp_interpolation = np.asarray(np.add(left, diff_b_a * gamma))
    ind = gamma >= 0.5
    lerp_interpolation[ind] = right[ind] - diff_b_a[ind] * (1 - gamma[ind])
    return lerp_interpolation


@njit(
    fastmath={"arcp", "contract", "reassoc", "nsz", "afn"},
    nogil=True,
    cache=False,
)
def _nan_quantile_1d(
    arr: np.array,
    quantiles: np.array,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float | np.array:
    """
    Get the quantiles of the 1-dimensional array.

    A  linear interpolation is performed using alpha and beta.

    Notes
    -----
    By default, `alpha == beta == 1` which performs the 7th method of :cite:t:`hyndman_sample_1996`.
    with `alpha == beta == 1/3` we get the 8th method. alpha == beta == 1 reproduces the behaviour of `np.nanquantile`.
    """
    # We need at least two values to do an interpolation
    valid_values_count = (~np.isnan(arr)).sum()

    # Computation of indexes
    virtual_indexes = (
        valid_values_count * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1
    )
    virtual_indexes = np.asarray(virtual_indexes)
    previous_indexes, next_indexes = _get_indexes(
        arr, virtual_indexes, valid_values_count
    )
    # Sorting
    arr.sort()

    previous = arr[previous_indexes]
    next_elements = arr[next_indexes]

    # Linear interpolation
    gamma = np.asarray(virtual_indexes - previous_indexes, dtype=arr.dtype)
    interpolation = _linear_interpolation(previous, next_elements, gamma)
    # When an interpolation is in Nan range, (near the end of the sorted array) it means
    # we can clip to the array max value.
    result = np.where(
        np.isnan(interpolation), arr[np.intp(valid_values_count) - 1], interpolation
    )
    return result


@guvectorize(
    [(float32[:], float32, float32[:]), (float64[:], float64, float64[:])],
    "(n),()->()",
    nopython=True,
    cache=False,
)
def _vecquantiles(arr, rnk, res):
    if np.isnan(rnk):
        res[0] = np.nan
    else:
        res[0] = np.nanquantile(arr, rnk)


def vecquantiles(
    da: DataArray, rnk: DataArray, dim: str | Sequence[Hashable]
) -> DataArray:
    """
    For when the quantile (rnk) is different for each point.

    da and rnk must share all dimensions but dim.

    Parameters
    ----------
    da : xarray.DataArray
        The data to compute the quantiles on.
    rnk : xarray.DataArray
        The quantiles to compute.
    dim : str or sequence of str
        The dimension along which to compute the quantiles.

    Returns
    -------
    xarray.DataArray
        The quantiles computed along the `dim` dimension.
    """
    tem = get_temp_dimname(da.dims, "temporal")
    dims = [dim] if isinstance(dim, str) else dim
    da = da.stack({tem: dims})
    da = da.transpose(*rnk.dims, tem)

    res = DataArray(
        _vecquantiles(da.values, rnk.values),
        dims=rnk.dims,
        coords=rnk.coords,
        attrs=da.attrs,
    )
    return res


@njit
def _wrapper_quantile1d(arr, q):
    out = np.empty((arr.shape[0], q.size), dtype=arr.dtype)
    for index in range(out.shape[0]):
        out[index] = _nan_quantile_1d(arr[index], q)
    return out


def _quantile(arr, q, nreduce=None):
    nreduce = nreduce or arr.ndim
    if arr.ndim == nreduce:
        out = _nan_quantile_1d(arr.flatten(), q)
    else:
        # dimensions that are reduced by quantile
        red_axis = np.arange(len(arr.shape) - nreduce, len(arr.shape))
        reduction_dim_size = np.prod([arr.shape[idx] for idx in red_axis])
        # kept dimensions
        keep_axis = np.arange(len(arr.shape) - nreduce)
        final_shape = [arr.shape[idx] for idx in keep_axis] + [len(q)]
        # reshape as (keep_dims, red_dims), compute, reshape back
        arr = arr.reshape(-1, reduction_dim_size)
        out = _wrapper_quantile1d(arr, q)
        out = out.reshape(final_shape)
    return out


def quantile(da: DataArray, q: np.ndarray, dim: str | Sequence[Hashable]) -> DataArray:
    """
    Compute the quantiles from a fixed list `q`.

    Parameters
    ----------
    da : xarray.DataArray
        The data to compute the quantiles on.
    q : array-like
        The quantiles to compute.
    dim : str or sequence of str
        The dimension along which to compute the quantiles.

    Returns
    -------
    xarray.DataArray
        The quantiles computed along the `dim` dimension.
    """
    if USE_FASTNANQUANTILE is True:
        return xr_apply_nanquantile(da, dim=dim, q=q).rename({"quantile": "quantiles"})

    qc = np.array(q, dtype=da.dtype)
    dims = [dim] if isinstance(dim, str) else dim
    kwargs = {"nreduce": len(dims), "q": qc}
    res = (
        apply_ufunc(
            _quantile,
            da,
            input_core_dims=[dims],
            exclude_dims=set(dims),
            output_core_dims=[["quantiles"]],
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"output_sizes": {"quantiles": len(q)}},
            dask="parallelized",
            kwargs=kwargs,
        )
        .assign_coords(quantiles=q)
        .assign_attrs(da.attrs)
    )
    return res


@njit(
    [
        float32[:, :](float32[:, :]),
        float64[:, :](float64[:, :]),
    ],
    fastmath=False,
    nogil=True,
    cache=False,
)
def remove_NaNs(x):  # noqa
    """Remove NaN values from series."""
    remove = np.zeros_like(x[0, :], dtype=boolean)
    for i in range(x.shape[0]):
        remove = remove | np.isnan(x[i, :])
    return x[:, ~remove]


@njit(
    [
        float32(float32[:, :], float32[:, :]),
        float64(float64[:, :], float64[:, :]),
    ],
    fastmath=True,
    nogil=True,
    cache=False,
)
def _correlation(X, Y):
    """
    Compute a correlation as the mean of pairwise distances between points in X and Y.

    X is KxN and Y is KxM, the result is the mean of the MxN distances.
    Similar to scipy.spatial.distance.cdist(X, Y, 'euclidean')
    """
    d = 0
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            d1 = 0
            for k in range(X.shape[0]):
                d1 += (X[k, i] - Y[k, j]) ** 2
            d += np.sqrt(d1)
    return d / (X.shape[1] * Y.shape[1])


@njit(
    [
        float32(float32[:, :]),
        float64(float64[:, :]),
    ],
    fastmath=True,
    nogil=True,
    cache=False,
)
def _autocorrelation(X):
    """
    Mean of the NxN pairwise distances of points in X of shape KxN.

    Similar to scipy.spatial.distance.pdist(..., 'euclidean')
    """
    d = 0
    for i in range(X.shape[1]):
        for j in range(i):
            d1 = 0
            for k in range(X.shape[0]):
                d1 += (X[k, i] - X[k, j]) ** 2
            d += np.sqrt(d1)
    return (2 * d) / X.shape[1] ** 2


@guvectorize(
    [
        (float32[:, :], float32[:, :], float32[:]),
        (float64[:, :], float64[:, :], float64[:]),
    ],
    "(k, n),(k, m)->()",
    nopython=True,
    cache=False,
)
def _escore(tgt, sim, out):
    """
    E-score based on the Székely-Rizzo e-distances between clusters.

    tgt and sim are KxN and KxM, where dimensions are along K and observations along M and N.
    When N > 0, only this many points of target and sim are used, taken evenly distributed in the series.
    When std is True, X and Y are standardized according to the nanmean and nanstd (ddof = 1) of X.
    """
    sim = remove_NaNs(sim)
    tgt = remove_NaNs(tgt)

    n1 = sim.shape[1]
    n2 = tgt.shape[1]

    sXY = _correlation(tgt, sim)
    sXX = _autocorrelation(tgt)
    sYY = _autocorrelation(sim)

    w = n1 * n2 / (n1 + n2)
    out[0] = w * (sXY + sXY - sXX - sYY) / 2


@njit(
    fastmath=False,
    nogil=True,
    cache=False,
)
def _first_and_last_nonnull(arr):
    """For each row of arr, get the first and last non NaN elements."""
    out = np.empty((arr.shape[0], 2))
    for i in range(arr.shape[0]):
        idxs = np.where(~np.isnan(arr[i]))[0]
        if idxs.size > 0:
            out[i] = arr[i][idxs[np.array([0, -1])]]
        else:
            out[i] = np.array([np.nan, np.nan])
    return out


@njit(
    fastmath=False,
    nogil=True,
    cache=False,
)
def _extrapolate_on_quantiles(
    interp, oldx, oldg, oldy, newx, newg, method="constant"
):  # noqa
    """
    Apply extrapolation to the output of interpolation on quantiles with a given grouping.

    Arguments are the same as _interp_on_quantiles_2D.
    """
    bnds = _first_and_last_nonnull(oldx)
    xp = oldg[:, 0]
    toolow = newx < np.interp(newg, xp, bnds[:, 0])
    toohigh = newx > np.interp(newg, xp, bnds[:, 1])
    if method == "constant":
        constants = _first_and_last_nonnull(oldy)
        cnstlow = np.interp(newg, xp, constants[:, 0])
        cnsthigh = np.interp(newg, xp, constants[:, 1])
        interp[toolow] = cnstlow[toolow]
        interp[toohigh] = cnsthigh[toohigh]
    else:  # 'nan'
        interp[toolow] = np.nan
        interp[toohigh] = np.nan
    return interp


@njit(
    fastmath=False,
    nogil=True,
    cache=False,
)
def _pairwise_haversine_and_bins(lond, latd, transpose=False):
    """Inter-site distances with the haversine approximation."""
    N = lond.shape[0]
    lon = np.deg2rad(lond)
    lat = np.deg2rad(latd)
    dists = np.full((N, N), np.nan)
    for i in range(N - 1):
        for j in range(i + 1, N):
            dlon = lon[j] - lon[i]
            dists[i, j] = 6367 * np.arctan2(
                np.sqrt(
                    (np.cos(lat[j]) * np.sin(dlon)) ** 2
                    + (
                        np.cos(lat[i]) * np.sin(lat[j])
                        - np.sin(lat[i]) * np.cos(lat[j]) * np.cos(dlon)
                    )
                    ** 2
                ),
                np.sin(lat[i]) * np.sin(lat[j])
                + np.cos(lat[i]) * np.cos(lat[j]) * np.cos(dlon),
            )
            if transpose:
                dists[j, i] = dists[i, j]
    mn = np.nanmin(dists)
    mx = np.nanmax(dists)
    if transpose:
        np.fill_diagonal(dists, 0)
    return dists, mn, mx


# Copied from xarray
def get_temp_dimname(dims: Sequence[str], new_dim: str) -> str:
    """
    Get an new dimension name based on new_dim, that is not used in dims.
    If the same name exists, we add an underscore(s) in the head.
    """
    while new_dim in dims:
        new_dim = "_" + str(new_dim)
    return new_dim
