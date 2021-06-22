"""Numba-accelerated utils."""
import numpy as np
import xarray as xr
from numba import boolean, float32, float64, guvectorize, int64, njit


@guvectorize(
    [(float32[:], float32, float32[:]), (float64[:], float64, float64[:])],
    "(n),()->()",
    nopython=True,
)
def _vecquantiles(arr, rnk, res):
    res[0] = np.nanquantile(arr, rnk)


def vecquantiles(da, rnk, dim):
    """For when the quantile (rnk) is different for each point.

    da and rnk must share all dimensions but dim.
    """
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    dims = [dim] if isinstance(dim, str) else dim
    da = da.stack({tem: dims})
    da = da.transpose(*rnk.dims, tem)

    res = xr.DataArray(
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
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    da = da.stack({tem: dims})

    # So we cut in half the definitions to declare in numba
    # We still use q as the coords so it corresponds to what was done upstream
    if not hasattr(q, "dtype") or q.dtype != da.dtype:
        qc = np.array(q, dtype=da.dtype)
    else:
        qc = q

    if len(da.dims) > 1:
        # There are some extra dims
        extra = xr.core.utils.get_temp_dimname(da.dims, "extra")
        da = da.stack({extra: set(da.dims) - {tem}})
        da = da.transpose(..., tem)
        res = xr.DataArray(
            _quantile(da.values, qc),
            dims=(extra, "quantiles"),
            coords={extra: da[extra], "quantiles": q},
            attrs=da.attrs,
        ).unstack(extra)

    else:
        # All dims are processed
        res = xr.DataArray(
            _quantile(da.values, qc),
            dims=("quantiles"),
            coords={"quantiles": q},
            attrs=da.attrs,
        )

    return res


@njit(
    [
        float32[:, :, :](float32[:, :], float32[:, :]),
        float64[:, :, :](float64[:, :], float64[:, :]),
    ],
    fastmath=True,
)
def _standardize_xy(x, y):
    """Standardize two MxN matrices according to the mean and standard deviation of only the first one.

    The standard deviation is computed as with np.nanstd(..., ddof=1).

    Returns a 2xMxN array.
    """
    out = np.empty((2, x.shape[0], x.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        xx = x[i, :]
        xx = xx[~np.isnan(xx)]
        mx = np.mean(xx)
        # Scaling so it imitates R. We can't use ddof=1 with numba
        stdx = np.std(xx) * np.sqrt(xx.size / (xx.size - 1))
        out[0, i, :] = (x[i, :] - mx) / stdx
        out[1, i, :] = (y[i, :] - mx) / stdx
    return out


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


@njit(
    [
        float32(float32[:, :], float32[:, :], int64, boolean),
        float64(float64[:, :], float64[:, :], int64, boolean),
    ],
    nogil=True,
)
def _escore(tgt, sim, N=0, std=False):
    """E-score based on the Skezely-Rizzo e-distances between clusters.

    tgt and sim are KxN and KxM, where dimensions are along K and observations along M and N.
    When N > 0, only this many points of target and sim are used, taken evenly distributed in the series.
    When std is True, X and Y are standardized according to the nanmean and nanstd (ddof = 1) of X.
    """
    n1 = sim.shape[1]
    n2 = tgt.shape[1]
    if N > 0:
        sim = sim[:, :: np.ceil(n1 / N)]
        tgt = tgt[:, :: np.ceil(n2 / N)]

    if std:
        out = _standardize_xy(tgt, sim)
        sim = out[0, ...]
        tgt = out[1, ...]

    sXY = _correlation(tgt, sim)
    sXX = _autocorrelation(tgt)
    sYY = _autocorrelation(sim)

    w = n1 * n2 / (n1 + n2)
    return w * (sXY + sXY - sXX - sYY) / 2


def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    N: int = 0,
    scale: bool = False,
    obs_dim: str = "time",
):
    r"""Energy score, or energy dissimilarity metric, based on [SkezelyRizzo]_ and [Cannon18]_.

    Parameters
    ----------
    tgt: DataArray
      Target observations, 2D.
    sim: DataArray
      Candidate observations. Must have the same (2) dimensions as `tgt`.
    N : int
      If larger than 0, the number of observations to use in the score computation. The points are taken
      evenly distributed along `obs_dim`.
    scale: boolean
      Whether to scale the data before computing the score. If True, both arrays as scaled according
      to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
      statistics excluding NaN values.
    obs_dim: str
      The name of the array dimension along which the observation points are listed. `tgt` and `sim` can
      have different length along this one, but must be equal along the other one.

    Returns
    -------
    e-score
        float

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,,n_j`
    proposed by Szekely and Rizzo (2005) is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of [Cannon18]_ to
    the metric. With that factor, the test becomes identical to the one defined by [BaringhausFranz]_.

    References
    ----------
    .. [SkezelyRizzo] Szekely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
    .. [BaringhausFranz] Baringhaus, L. and Franz, C. (2004) On a new multivariate two-sample test, Journal of Multivariate Analysis, 88(1), 190–206. https://doi.org/10.1016/s0047-259x(03)00079-4
    """

    pts_dim = set(tgt.dims) - {obs_dim}
    if len(pts_dim) > 1 or {pts_dim, obs_dim} != set(sim.dims):
        raise ValueError(
            f"Incorrect dimensions or number of dimensions. `sim` and `tgt` must both have the same 2 dimension names, including obs_dim={obs_dim}."
        )

    sim = sim.transpose(..., obs_dim)
    tgt = tgt.transpose(..., obs_dim)

    return _escore(tgt.values, sim.values, N=N, scale=scale)
