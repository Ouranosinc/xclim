"""Spatial Analogues module."""
# TODO: Hellinger distance
# TODO: Mahalanobis distance
# TODO: Comment on "significance" of results.
# Code adapted from flyingpigeon.dissimilarity, Nov 2020.
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from boltons.funcutils import wraps
from pkg_resources import parse_version
from scipy import __version__ as __scipy_version__
from scipy import spatial
from scipy.spatial import cKDTree as KDTree

metrics = {}


def spatial_analogs(
    target: xr.Dataset,
    candidates: xr.Dataset,
    dist_dim: str | Sequence[str] = "time",
    method: str = "kldiv",
    **kwargs,
):
    r"""Compute dissimilarity statistics between target points and candidate points.

    Spatial analogues based on the comparison of climate indices. The algorithm compares
    the distribution of the reference indices with the distribution of spatially
    distributed candidate indices and returns a value measuring the dissimilarity
    between both distributions over the candidate grid.

    Parameters
    ----------
    target : xr.Dataset
        Dataset of the target indices. Only indice variables should be included in the
        dataset's `data_vars`. They should have only the dimension(s) `dist_dim `in common with `candidates`.
    candidates : xr.Dataset
        Dataset of the candidate indices. Only indice variables should be included in
        the dataset's `data_vars`.
    dist_dim : str
        The dimension over which the *distributions* are constructed. This can be a multi-index dimension.
    method : {'seuclidean', 'nearest_neighbor', 'zech_aslan', 'kolmogorov_smirnov', 'friedman_rafsky', 'kldiv'}
        Which method to use when computing the dissimilarity statistic.
    \*\*kwargs
        Any other parameter passed directly to the dissimilarity method.

    Returns
    -------
    xr.DataArray
        The dissimilarity statistic over the union of candidates' and target's dimensions.
        The range depends on the method.
    """
    if parse_version(__scipy_version__) < parse_version("1.6.0") and method in [
        "kldiv",
        "nearest_neighbor",
    ]:
        raise RuntimeError(f"Spatial analogue method ({method}) requires scipy>=1.6.0.")

    # Create the target DataArray:
    target = target.to_array("_indices", "target")

    # Create the target DataArray
    # drop any (sub-)index along "dist_dim" that could conflict with target, and rename it.
    # The drop is the simplest solution that is compatible with both xarray <=2022.3.0 and >2022.3.1
    candidates = candidates.to_array("_indices", "candidates").rename(
        {dist_dim: "_dist_dim"}
    )
    if isinstance(candidates.indexes["_dist_dim"], pd.MultiIndex):
        candidates = candidates.drop_vars(
            ["_dist_dim"] + candidates.indexes["_dist_dim"].names,
            # in xarray <= 2022.3.0 the sub-indexes are not listed as separate coords,
            # instead, they are dropped when the multiindex is dropped.
            errors="ignore",
        )

    try:
        metric = metrics[method]
    except KeyError as e:
        raise ValueError(
            f"Method {method} is not implemented. Available methods are : {','.join(metrics.keys())}."
        ) from e

    if candidates.chunks is not None:
        candidates = candidates.chunk({"_indices": -1})
    if target.chunks is not None:
        target = target.chunk({"_indices": -1})

    # Compute dissimilarity
    diss = xr.apply_ufunc(
        metric,
        target,
        candidates,
        input_core_dims=[(dist_dim, "_indices"), ("_dist_dim", "_indices")],
        output_core_dims=[()],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs=kwargs,
    )
    diss.name = "dissimilarity"
    diss.attrs.update(
        long_name=f"Dissimilarity between target and candidates, using metric {method}.",
        indices=",".join(target._indices.values),  # noqa
        metric=method,
    )

    return diss


# ---------------------------------------------------------------------------- #
# -------------------------- Utility functions ------------------------------- #
# ---------------------------------------------------------------------------- #


def standardize(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize x and y by the square root of the product of their standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Array to be compared.
    y : np.ndarray
        Array to be compared.

    Returns
    -------
    (ndarray, ndarray)
        Standardized arrays.
    """
    s = np.sqrt(x.std(0, ddof=1) * y.std(0, ddof=1))
    return x / s, y / s


def metric(func):
    """Register a metric function in the `metrics` mapping and add some preparation/checking code.

    All metric functions accept 2D inputs. This reshapes 1D inputs to (n, 1) and (m, 1).
    All metric functions are invalid when any non-finite values are present in the inputs.
    """

    @wraps(func)
    def _metric_overhead(x, y, **kwargs):
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            return np.NaN

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        # If array is 1D, flip it.
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T

        if x.shape[1] != y.shape[1]:
            raise AttributeError("Shape mismatch")

        return func(x, y, **kwargs)

    metrics[func.__name__] = _metric_overhead
    return _metric_overhead


# ---------------------------------------------------------------------------- #
# ------------------------ Dissimilarity metrics ----------------------------- #
# ---------------------------------------------------------------------------- #


@metric
def seuclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Euclidean distance between the mean of a multivariate candidate sample with respect to the mean of a reference sample.

    This method is scale-invariant.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Reference sample.
    y : np.ndarray (m,d)
        Candidate sample.

    Returns
    -------
    float
        Standardized Euclidean Distance between the mean of the samples
        ranging from 0 to infinity.

    Notes
    -----
    This metric considers neither the information from individual points nor
    the standard deviation of the candidate distribution.

    References
    ----------
    :cite:cts:`veloz_identifying_2012`
    """
    mx = x.mean(axis=0)
    my = y.mean(axis=0)

    return spatial.distance.seuclidean(mx, my, x.var(axis=0, ddof=1))


@metric
def nearest_neighbor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute a dissimilarity metric based on the number of points in the pooled sample whose nearest neighbor belongs to the same distribution.

    This method is scale-invariant.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Reference sample.
    y : np.ndarray (m,d)
        Candidate sample.

    Returns
    -------
    float
        Nearest-Neighbor dissimilarity metric ranging from 0 to 1.

    References
    ----------
    :cite:cts:`henze_multivariate_1988`
    """
    x, y = standardize(x, y)

    nx, _ = x.shape

    # Pool the samples and find the nearest neighbours
    xy = np.vstack([x, y])
    tree = KDTree(xy)
    _, ind = tree.query(xy, k=2, eps=0, p=2, workers=2)

    # Identify points whose neighbors are from the same sample
    same = ~np.logical_xor(*(ind < nx).T)

    return same.mean()


@metric
def zech_aslan(x: np.ndarray, y: np.ndarray, *, dmin: float = 1e-12) -> float:
    r"""
    Compute a modified Zech-Aslan energy distance dissimilarity metric based on an analogy with the energy of a cloud of electrical charges.

    This method is scale-invariant.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Reference sample.
    y : np.ndarray (m,d)
        Candidate sample.
    dmin : float
        The cut-off for low distances to avoid singularities on identical points.

    Returns
    -------
    float
        Zech-Aslan dissimilarity metric ranging from -infinity to infinity.

    Notes
    -----
    The energy measure between two variables :math:`X`, :math:`Y` (target and candidates) of sizes :math:`n,d` and
    :math:`m,d` proposed by :cite:t:`aslan_new_2003` is defined by:

    .. math::

        e(X, Y) &= \left[\phi_{xx} + \phi_{yy} - \phi_{xy}\right] \\
        \phi_{xy} &= \frac{1}{n m} \sum_{i = 1}^n \sum_{j = 1}^m R\left[SED(X_i, Y_j)\right] \\
        \phi_{xx} &= \frac{1}{n^2} \sum_{i = 1}^n \sum_{j = i + 1}^n R\left[SED(X_i, X_j)\right] \\
        \phi_{yy} &= \frac{1}{m^2} \sum_{i = 1}^m \sum_{j = i + 1}^m R\left[SED(X_i, Y_j)\right] \\

    where :math:`X_i` denotes the i-th observation of :math:`X`. :math:`R` is a weight function and :math:`SED(A, B)`
    denotes the standardized Euclidean distance.

    .. math::

        R(r) &= \left\{\begin{array}{r l} -\ln r & \text{for } r > d_{min} \\ -\ln d_{min} & \text{for } r \leq d_{min} \end{array}\right. \\
        SED(X_i, Y_j) &= \sqrt{\sum_{k=1}^d \frac{\left(X_i(k) - Y_i(k)\right)^2}{\sigma_x(k)\sigma_y(k)}}

    where :math:`k` is a counter over dimensions (indices in the case of spatial analogs) and :math:`\sigma_x(k)` is the
    standard deviation of :math:`X` in dimension :math:`k`. Finally, :math:`d_{min}` is a cut-off to avoid poles when
    :math:`r \to 0`, it is controllable through the `dmin` parameter.

    This version corresponds the :math:`D_{ZAE}` test of :cite:t:`grenier_assessment_2013` (eq. 7), which is a version
    of :math:`\phi_{NM}` from :cite:t:`aslan_new_2003`, modified by using the standardized  euclidean distance, the log
    weight function and choosing :math:`d_{min} = 10^{-12}`.

    References
    ----------
    :cite:cts:`grenier_assessment_2013,zech_multivariate_2003,aslan_new_2003`
    """
    nx, _ = x.shape
    ny, _ = y.shape

    v = (x.std(axis=0, ddof=1) * y.std(axis=0, ddof=1)).astype(np.double)

    dx = spatial.distance.pdist(x, "seuclidean", V=v)
    dy = spatial.distance.pdist(y, "seuclidean", V=v)
    dxy = spatial.distance.cdist(x, y, "seuclidean", V=v)

    phix = -np.log(dx.clip(dmin)).sum() / (nx * (nx - 1))
    phiy = -np.log(dy.clip(dmin)).sum() / (ny * (ny - 1))
    phixy = -np.log(dxy.clip(dmin)).sum() / (nx * ny)
    return phix + phiy - phixy


@metric
def szekely_rizzo(x: np.ndarray, y: np.ndarray, *, standardize: bool = True) -> float:
    r"""
    Compute the Székely-Rizzo energy distance dissimilarity metric based on an analogy with Newton's gravitational potential energy.

    This method is scale-invariant when `standardize=True` (default), scale-dependent otherwise.

    Parameters
    ----------
    x : ndarray (n,d)
        Reference sample.
    y : ndarray (m,d)
        Candidate sample.
    standardize : bool
        If True (default), the standardized euclidean norm is used, instead of the conventional one.

    Returns
    -------
    float
        Székely-Rizzo's energy distance dissimilarity metric ranging from 0 to infinity.

    Notes
    -----
    The e-distance between two variables :math:`X`, :math:`Y` (target and candidates) of sizes :math:`n,d` and
    :math:`m,d` proposed by :cite:t:`szekely_testing_2004` is defined by:

    .. math::

        e(X, Y) = \frac{n m}{n + m} \left[2\phi_{xy} − \phi_{xx} − \phi_{yy} \right]

    where

    .. math::

        \phi_{xy} &= \frac{1}{n m} \sum_{i = 1}^n \sum_{j = 1}^m \left\Vert X_i − Y_j \right\Vert \\
        \phi_{xx} &= \frac{1}{n^2} \sum_{i = 1}^n \sum_{j = 1}^n \left\Vert X_i − X_j \right\Vert \\
        \phi_{yy} &= \frac{1}{m^2} \sum_{i = 1}^m \sum_{j = 1}^m \left\Vert X_i − Y_j \right\Vert \\

    and where :math:`\Vert\cdot\Vert` denotes the Euclidean norm, :math:`X_i` denotes the i-th observation of :math:`X`.
    When `standardized=False`, this corresponds to the :math:`T` test of :cite:t:`rizzo_energy_2016` (p. 28) and to the
    ``eqdist.e`` function of the `energy` R package (with two samples) and gives results twice as big as
    :py:func:`xclim.sdba.processing.escore`. The standardization was added following the logic of
    :cite:p:`grenier_assessment_2013` to make the metric scale-invariant.

    References
    ----------
    :cite:cts:`grenier_assessment_2013,szekely_testing_2004,rizzo_energy_2016`
    """
    n, _ = x.shape
    m, _ = y.shape

    # Mean of the distance pairs
    # We are not taking "mean" because of the condensed output format of pdist
    if standardize:
        v = (x.std(axis=0, ddof=1) * y.std(axis=0, ddof=1)).astype(np.double)
        sXY = spatial.distance.cdist(x, y, "seuclidean", V=v).sum() / (n * m)
        sXX = spatial.distance.pdist(x, "seuclidean", V=v).sum() * 2 / n**2
        sYY = spatial.distance.pdist(y, "seuclidean", V=v).sum() * 2 / m**2
    else:
        sXY = spatial.distance.cdist(x, y, "euclidean").sum() / (n * m)
        sXX = spatial.distance.pdist(x, "euclidean").sum() * 2 / n**2
        sYY = spatial.distance.pdist(y, "euclidean").sum() * 2 / m**2
    w = n * m / (n + m)
    return w * (sXY + sXY - sXX - sYY)


@metric
def friedman_rafsky(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a dissimilarity metric based on the Friedman-Rafsky runs statistics.

    The algorithm builds a minimal spanning tree (the subset of edges connecting all points that minimizes the total
    edge length) then counts the edges linking points from the same distribution. This method is scale-dependent.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Reference sample.
    y : np.ndarray (m,d)
        Candidate sample.

    Returns
    -------
    float
        Friedman-Rafsky dissimilarity metric ranging from 0 to (m+n-1)/(m+n).

    References
    ----------
    :cite:cts:`friedman_multivariate_1979`
    """
    from scipy.sparse.csgraph import (  # pylint: disable=import-outside-toplevel
        minimum_spanning_tree,
    )
    from sklearn import neighbors  # pylint: disable=import-outside-toplevel

    nx, _ = x.shape
    ny, _ = y.shape
    n = nx + ny

    xy = np.vstack([x, y])
    # Compute the NNs and the minimum spanning tree
    g = neighbors.kneighbors_graph(xy, n_neighbors=n - 1, mode="distance")
    mst = minimum_spanning_tree(g, overwrite=True)
    edges = np.array(mst.nonzero()).T

    # Number of points whose neighbor is from the other sample
    diff = np.logical_xor(*(edges < nx).T).sum()

    return 1.0 - (1.0 + diff) / n


@metric
def kolmogorov_smirnov(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic applied to two multivariate samples as described by Fasano and Franceschini.

    This method is scale-dependent.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Reference sample.
    y : np.ndarray (m,d)
        Candidate sample.

    Returns
    -------
    float
        Kolmogorov-Smirnov dissimilarity metric ranging from 0 to 1.

    References
    ----------
    :cite:cts:`fasano_multidimensional_1987`
    """

    def pivot(x, y):
        nx, d = x.shape
        ny, d = y.shape

        # Multiplicative factor converting d-dim booleans to a unique integer.
        mf = (2 ** np.arange(d)).reshape(1, d, 1)
        minlength = 2**d

        # Assign a unique integer according on whether or not x[i] <= sample
        ix = ((x.T <= np.atleast_3d(x)) * mf).sum(1)
        iy = ((x.T <= np.atleast_3d(y)) * mf).sum(1)

        # Count the number of samples in each quadrant
        cx = 1.0 * np.apply_along_axis(np.bincount, 0, ix, minlength=minlength) / nx
        cy = 1.0 * np.apply_along_axis(np.bincount, 0, iy, minlength=minlength) / ny

        # This is from https://github.com/syrte/ndtest/blob/master/ndtest.py
        # D = cx - cy
        # D[0,:] -= 1. / nx # I don't understand this...
        # dmin, dmax = -D.min(), D.max() + .1 / nx

        return np.max(np.abs(cx - cy))

    return max(pivot(x, y), pivot(y, x))


@metric
def kldiv(
    x: np.ndarray, y: np.ndarray, *, k: int | Sequence[int] = 1
) -> float | Sequence[float]:
    r"""Compute the Kullback-Leibler divergence between two multivariate samples.

    .. math
        D(P||Q) = \frac{d}{n} \sum_i^n \log\left\{\frac{r_k(x_i)}{s_k(x_i)}\right\} + \log\left\{\frac{m}{n-1}\right\}

    where :math:`r_k(x_i)` and :math:`s_k(x_i)` are, respectively, the euclidean distance to the kth neighbour of
    :math:`x_i` in the x array (excepting :math:`x_i`) and in the y array. This method is scale-dependent.

    Parameters
    ----------
    x : np.ndarray (n,d)
        Samples from distribution P, which typically represents the true distribution (reference).
    y : np.ndarray (m,d)
        Samples from distribution Q, which typically represents the approximate distribution (candidate)
    k : int or sequence
        The kth neighbours to look for when estimating the density of the distributions.
        Defaults to 1, which can be noisy.

    Returns
    -------
    float or sequence
        The estimated Kullback-Leibler divergence D(P||Q) computed from the distances to the kth neighbour.

    Notes
    -----
    In information theory, the Kullback–Leibler divergence :cite:p:`perez-cruz_kullback-leibler_2008` is a non-symmetric
    measure of the difference between two probability distributions P and Q, where P is the "true" distribution and Q an
    approximation. This nuance is important because :math:`D(P||Q)` is not equal to :math:`D(Q||P)`.

    For probability distributions P and Q of a continuous random variable, the K–L  divergence is defined as:

    .. math::

        D_{KL}(P||Q) = \int p(x) \log\left(\frac{p(x)}{q(x)}\right) dx

    This formula assumes we have a representation of the probability densities :math:`p(x)` and :math:`q(x)`.
    In many cases, we only have samples from the distribution, and most methods first estimate the densities from the
    samples and then proceed to compute the K-L divergence. In :cite:t:`perez-cruz_kullback-leibler_2008`, the author
    proposes an algorithm to estimate the K-L divergence directly from the sample using an empirical CDF. Even though the
    CDFs do not converge to their true values, the paper proves that the K-L divergence almost surely does converge to
    its true value.

    References
    ----------
    :cite:cts:`perez-cruz_kullback-leibler_2008`
    """
    mk = np.iterable(k)
    ka = np.atleast_1d(k)

    nx, d = x.shape
    ny, d = y.shape

    # Limit the number of dimensions to 10, too slow otherwise.
    if d > 10:
        raise ValueError(f"Too many dimensions: {d}.")

    # Not enough data to draw conclusions.
    if nx < 5 or ny < 5:
        return np.nan if not mk else [np.nan] * len(k)

    # Build a KD tree representation of the samples.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the k'th nearest neighbour from each points in x for both x and y.
    # We get the values for K + 1 to make sure the output is a 2D array.
    kmax = max(ka) + 1
    r, _ = xtree.query(x, k=kmax, eps=0, p=2, workers=2)
    s, _ = ytree.query(x, k=kmax, eps=0, p=2, workers=2)

    # There is a mistake in the paper. In Eq. 14, the right side misses a
    # negative sign on the first term of the right-hand side.
    out = []
    for ki in ka:
        # The 0th nearest neighbour of x[i] in x is x[i] itself.
        # Hence, we take the k'th + 1, which in 0-based indexing is given by
        # index k.
        out.append(
            -np.log(r[:, ki] / s[:, ki - 1]).sum() * d / nx + np.log(ny / (nx - 1.0))
        )

    if mk:
        return out
    return out[0]
