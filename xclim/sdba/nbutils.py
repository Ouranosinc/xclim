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
from xarray.core import utils

from xclim.core.utils import _nan_quantile

try:
    from fastnanquantile.xrcompat import xr_apply_nanquantile

    USE_FASTNANQUANTILE = True
except ImportError:
    USE_FASTNANQUANTILE = False


@guvectorize(
    [(float32[:], float32, float32[:]), (float64[:], float64, float64[:])],
    "(n),()->()",
    nopython=True,
    cache=False,
)
def _vecquantiles(arr, rnk, res):
    if np.isnan(rnk):
        res[0] = np.NaN
    else:
        res[0] = _nan_quantile(arr, rnk)


def vecquantiles(
    da: DataArray, rnk: DataArray, dim: str | Sequence[Hashable]
) -> DataArray:
    """For when the quantile (rnk) is different for each point.

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
    tem = utils.get_temp_dimname(da.dims, "temporal")
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
        out[index] = _nan_quantile(arr[index], q)
    return out


def _quantile(arr, q, nreduce):
    if arr.ndim == nreduce:
        out = _nan_quantile(arr.flatten(), q)
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


def quantile(da, q, dim):
    """Compute the quantiles from a fixed list `q`."""
    if USE_FASTNANQUANTILE is True:
        return xr_apply_nanquantile(da, dim=dim, q=q).rename({"quantile": "quantiles"})
    else:
        qc = np.array(q, dtype=da.dtype)
        dims = [dim] if isinstance(dim, str) else dim
        kwargs = dict(nreduce=len(dims), q=qc)
        res = (
            apply_ufunc(
                _quantile,
                da,
                input_core_dims=[dims],
                exclude_dims=set(dims),
                output_core_dims=[["quantiles"]],
                output_dtypes=[da.dtype],
                dask_gufunc_kwargs=dict(output_sizes={"quantiles": len(q)}),
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
    """Compute a correlation as the mean of pairwise distances between points in X and Y.

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
    """Mean of the NxN pairwise distances of points in X of shape KxN.

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
    """E-score based on the Székely-Rizzo e-distances between clusters.

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
    #    [
    #        float32[:,:](float32[:]),
    #        float64[:,:](float64[:]),
    #        float32[:,:](float32[:,:]),
    #        float64[:,:](float64[:,:]),
    #    ],
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
            out[i] = np.array([np.NaN, np.NaN])
    return out


@njit(
    #    [
    #        float64[:](float32[:],float32[:],float32[:],float32[:],float32[:],float32[:],optional(typeof("constant"))),
    #        float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],optional(typeof("constant"))),
    #    ],
    fastmath=False,
    nogil=True,
    cache=False,
)
def _extrapolate_on_quantiles(
    interp, oldx, oldg, oldy, newx, newg, method="constant"
):  # noqa
    """Apply extrapolation to the output of interpolation on quantiles with a given grouping.

    Arguments are the same as _interp_on_quantiles_2D.
    """
    bnds = _first_and_last_nonnull(oldx)
    xp = np.arange(bnds.shape[0])
    toolow = newx < np.interp(newg, xp, bnds[:, 0])
    toohigh = newx > np.interp(newg, xp, bnds[:, 1])
    if method == "constant":
        constants = _first_and_last_nonnull(oldy)
        cnstlow = np.interp(newg, xp, constants[:, 0])
        cnsthigh = np.interp(newg, xp, constants[:, 1])
        interp[toolow] = cnstlow[toolow]
        interp[toohigh] = cnsthigh[toohigh]
    else:  # 'nan'
        interp[toolow] = np.NaN
        interp[toohigh] = np.NaN
    return interp


@njit(
    #    [
    #        typeof((float64[:,:],float64,float64))(float32[:],float32[:],optional(boolean)),
    #        typeof((float64[:,:],float64,float64))(float64[:],float64[:],optional(boolean)),
    #    ],
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
