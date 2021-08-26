"""Numba-accelerated utils."""
import numpy as np
from numba import boolean, float32, float64, guvectorize, njit
from xarray import DataArray
from xarray.core import utils


@guvectorize(
    [(float32[:], float32, float32[:]), (float64[:], float64, float64[:])],
    "(n),()->()",
    nopython=True,
)
def _vecquantiles(arr, rnk, res):
    if np.isnan(rnk):
        res[0] = np.NaN
    else:
        res[0] = np.nanquantile(arr, rnk)


def vecquantiles(da, rnk, dim):
    """For when the quantile (rnk) is different for each point.

    da and rnk must share all dimensions but dim.
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


@njit(
    [
        float32[:, :](float32[:, :], float32[:]),
        float64[:, :](float64[:, :], float64[:]),
        float32[:](float32[:], float32[:]),
        float64[:](float64[:], float64[:]),
    ],
)
def _quantile(arr, q):
    if arr.ndim == 1:
        out = np.empty((q.size,), dtype=arr.dtype)
        out[:] = np.nanquantile(arr, q)
    else:
        out = np.empty((arr.shape[0], q.size), dtype=arr.dtype)
        for index in range(out.shape[0]):
            out[index] = np.nanquantile(arr[index], q)
    return out


def quantile(da, q, dim):
    """Compute the quantiles from a fixed list "q" """
    # We have two cases :
    # - When all dims are processed : we stack them and use _quantile1d
    # - When the quantiles are vectorized over some dims, these are also stacked and then _quantile2D is used.
    # All this stacking is so we can cover all ND+1D cases with one numba function.

    # Stack the dims and send to the last position
    # This is in case there are more than one
    dims = [dim] if isinstance(dim, str) else dim
    tem = utils.get_temp_dimname(da.dims, "temporal")
    da = da.stack({tem: dims})

    # So we cut in half the definitions to declare in numba
    # We still use q as the coords so it corresponds to what was done upstream
    if not hasattr(q, "dtype") or q.dtype != da.dtype:
        qc = np.array(q, dtype=da.dtype)
    else:
        qc = q

    if len(da.dims) > 1:
        # There are some extra dims
        extra = utils.get_temp_dimname(da.dims, "extra")
        da = da.stack({extra: set(da.dims) - {tem}})
        da = da.transpose(..., tem)
        res = DataArray(
            _quantile(da.values, qc),
            dims=(extra, "quantiles"),
            coords={extra: da[extra], "quantiles": q},
            attrs=da.attrs,
        ).unstack(extra)

    else:
        # All dims are processed
        res = DataArray(
            _quantile(da.values, qc),
            dims=("quantiles"),
            coords={"quantiles": q},
            attrs=da.attrs,
        )

    return res


@njit([float32[:, :](float32[:, :]), float64[:, :](float64[:, :])])
def remove_NaNs(x):
    remove = np.zeros_like(x[0, :], dtype=boolean)
    for i in range(x.shape[0]):
        remove = remove | np.isnan(x[i, :])
    return x[:, ~remove]


@njit([float32(float32[:]), float64(float64[:])], fastmath=True)
def _euclidean_norm(v):
    """Compute the euclidean norm of vector v."""
    return np.sqrt(np.sum(v ** 2))


@njit(
    [float32(float32[:, :], float32[:, :]), float64(float64[:, :], float64[:, :])],
    fastmath=True,
)
def _correlation(X, Y):
    """Compute a correlation as the mean of pairwise distances between points in X and Y.

    X is KxN and Y is KxM, the result is the mean of the MxN distances.
    Similar to scipy.spatial.distance.cdist(X, Y, 'euclidean')
    """
    d = 0
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            d += _euclidean_norm(X[:, i] - Y[:, j])
    return d / (X.shape[1] * Y.shape[1])


@njit([float32(float32[:, :]), float64(float64[:, :])], fastmath=True)
def _autocorrelation(X):
    """Mean of the NxN pairwise distances of points in X of shape KxN.

    Similar to scipy.spatial.distance.pdist(..., 'euclidean')
    """
    d = 0
    for i in range(X.shape[1]):
        for j in range(i + 1):
            d += (int(i != j) + 1) * _euclidean_norm(X[:, i] - X[:, j])
    return d / X.shape[1] ** 2


@guvectorize(
    [
        (float32[:, :], float32[:, :], float32[:]),
        (float64[:, :], float64[:, :], float64[:]),
    ],
    "(k, n),(k, m)->()",
)
def _escore(tgt, sim, out):
    """E-score based on the Skezely-Rizzo e-distances between clusters.

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
