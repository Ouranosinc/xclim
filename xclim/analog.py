# -*- encoding: utf8 -*-
# noqa: D205,D400
"""
Spatial analogs
===============

Spatial analogues are maps showing which areas have a present-day climate that is analogous
to the future climate of a given place. This type of map can be useful for climate adaptation
to see how well regions are coping today under specific climate conditions. For example,
officials from a city located in a temperate region that may be expecting more heatwaves in
the future can learn from the experience of another city where heatwaves are a common occurrence,
leading to more proactive intervention plans to better deal with new climate conditions.

Spatial analogues are estimated by comparing the distribution of climate indices computed at
the target location over the future period with the distribution of the same climate indices
computed over a reference period for multiple candidate regions. A number of methodological
choices thus enter the computation:

    - Climate indices of interest,
    - Metrics measuring the difference between both distributions,
    - Reference data from which to compute the base indices,
    - A future climate scenario to compute the target indices.

The climate indices chosen to compute the spatial analogues are usually annual values of
indices relevant to the intended audience of these maps. For example, in the case of the
wine grape industry, the climate indices examined could include the length of the frost-free
season, growing degree-days, annual winter minimum temperature andand annual number of
very cold days [Roy2017]_.


Methods to compute the (dis)similarity between samples
------------------------------------------------------

This module implements five of the six methods described in [Grenier2013]_ to measure
the dissimilarity between two samples. Some of these algorithms can be used to
test whether or not two samples have been drawn from the same distribution.
Here, they are used to find areas with analog climate conditions to a target
climate.

Methods available
~~~~~~~~~~~~~~~~~
 * Standardized Euclidean distance
 * Nearest Neighbour distance
 * Zech-Aslan energy statistic
 * Friedman-Rafsky runs statistic
 * Kolmogorov-Smirnov statistic
 * Kullback-Leibler divergence

All methods accept arrays, the first is the reference (n, D) and
the second is the candidate (m, D). Where the climate indicators
vary along D and the distribution dimension along n or m. All methods output
a single float.


.. rubric:: References

.. [Roy2017] Roy, P., Grenier, P., Barriault, E. et al. Climatic Change (2017) 143: 43. `<doi:10.1007/s10584-017-1960-x>`_
.. [Grenier2013]  Grenier, P., A.-C. Parent, D. Huard, F. Anctil, and D. Chaumont, 2013: An assessment of six dissimilarity metrics for climate analogs. J. Appl. Meteor. Climatol., 52, 733–752, `<doi:10.1175/JAMC-D-12-0170.1>`_
"""
# Code adapted from flyingpigeon.dissimilarity, Nov 2020.
from typing import Sequence, Tuple, Union

import numpy as np
import xarray as xr
from boltons.funcutils import wraps
from pkg_resources import parse_version
from scipy import __version__ as __scipy_version__
from scipy import spatial
from scipy.spatial import cKDTree as KDTree

# TODO: Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
# based on distances. J Stat Planning & Inference 143: 1249-1272

# TODO: Hellinger distance
metrics = dict()


def spatial_analogs(
    target: xr.Dataset,
    candidates: xr.Dataset,
    dist_dim: Union[str, Sequence[str]] = "time",
    method: str = "kldiv",
    **kwargs,
):
    """Compute dissimilarity statistics between target points and candidate points.

    Spatial analogs based on the comparison of climate indices. The algorithm compares
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
    **kwargs
      Any other parameter passed directly to the dissimilarity method.

    Returns
    -------
    xr.DataArray
      The dissimilarity statistic over the union of candidates' and target's dimensions.
    """
    if parse_version(__scipy_version__) < parse_version("1.6.0") and method in [
        "kldiv",
        "nearest_neighbor",
    ]:
        raise RuntimeError(f"Spatial analog method ({method}) requires scipy>=1.6.0.")

    # Create the target DataArray:
    target = xr.concat(
        target.data_vars.values(),
        xr.DataArray(list(target.data_vars.keys()), dims=("indices",), name="indices"),
    )

    # Create the target DataArray with different dist_dim
    c_dist_dim = "candidate_dist_dim"
    candidates = xr.concat(
        candidates.data_vars.values(),
        xr.DataArray(
            list(candidates.data_vars.keys()),
            dims=("indices",),
            name="indices",
        ),
    ).rename({dist_dim: c_dist_dim})

    try:
        metric = metrics[method]
    except KeyError:
        raise ValueError(
            f"Method {method} is not implemented. Available methods are : {','.join(metrics.keys())}."
        )

    # Compute dissimilarity
    diss = xr.apply_ufunc(
        metric,
        target,
        candidates,
        input_core_dims=[(dist_dim, "indices"), (c_dist_dim, "indices")],
        output_core_dims=[()],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        **kwargs,
    )
    diss.name = "dissimilarity"
    diss.attrs.update(
        long_name=f"Dissimilarity between target and candidates, using metric {method}.",
        indices=",".join(target.indices.values),
        metric=method,
    )

    return diss


# ---------------------------------------------------------------------------- #
# -------------------------- Utility functions ------------------------------- #
# ---------------------------------------------------------------------------- #


def standardize(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize x and y by the square root of the product of their standard deviation.

    Parameters
    ----------
    x: np.ndarray
      Array to be compared.
    y: np.ndarray
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

    All metric functions accept 2D inputs. This reshape 1D inputs to (n, 1) and (m, 1).
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
    """
    Compute the Euclidean distance between the mean of a multivariate candidate sample with respect to the mean of a reference sample.

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
    Veloz et al. (2011) Identifying climatic analogs for Wisconsin under
    21st-century climate-change scenarios. Climatic Change,
    DOI 10.1007/s10584-011-0261-z.
    """
    mx = x.mean(axis=0)
    my = y.mean(axis=0)

    return spatial.distance.seuclidean(mx, my, x.var(axis=0, ddof=1))


@metric
def nearest_neighbor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute a dissimilarity metric based on the number of points in the pooled sample whose nearest neighbor belongs to the same distribution.

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
    Henze N. (1988) A Multivariate two-sample test based on the number of
    nearest neighbor type coincidences. Ann. of Stat., Vol. 16, No.2, 772-783.
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
def zech_aslan(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Zech-Aslan energy distance dissimimilarity metric based on an analogy with the energy of a cloud of electrical charges.

    Parameters
    ----------
    x : np.ndarray (n,d)
      Reference sample.
    y : np.ndarray (m,d)
      Candidate sample.

    Returns
    -------
    float
      Zech-Aslan dissimilarity metric ranging from -infinity to infinity.

    References
    ----------
    Zech G. and Aslan B. (2003) A Multivariate two-sample test based on the
    concept of minimum energy. PHYStat2003, SLAC, Stanford, CA, Sep 8-11.
    Aslan B. and Zech G. (2008) A new class of binning-free, multivariate
    goodness-of-fit tests: the energy tests. arXiV:hep-ex/0203010v5.
    """
    nx, d = x.shape
    ny, d = y.shape

    v = (x.std(axis=0, ddof=1) * y.std(axis=0, ddof=1)).astype(np.double)

    dx = spatial.distance.pdist(x, "seuclidean", V=v)
    dy = spatial.distance.pdist(y, "seuclidean", V=v)
    dxy = spatial.distance.cdist(x, y, "seuclidean", V=v)

    phix = -np.log(dx).sum() / nx / (nx - 1)
    phiy = -np.log(dy).sum() / ny / (ny - 1)
    phixy = np.log(dxy).sum() / nx / ny
    return phix + phiy + phixy


@metric
def skezely_rizzo(x, y):
    """
    Compute the Skezely-Rizzo energy distance dissimimilarity metric based on an analogy with the energy of a cloud of electrical charges.

    Parameters
    ----------
    x : ndarray (n,d)
      Reference sample.
    y : ndarray (m,d)
      Candidate sample.

    Returns
    -------
    float
      Skezely-Rizzo dissimilarity metric ranging from -infinity to infinity.

    References
    ----------
    TODO
    """
    raise NotImplementedError
    # nx, d = x.shape
    # ny, d = y.shape
    #
    # v = x.std(0, ddof=1) * y.std(0, ddof=1)
    #
    # dx = spatial.distance.pdist(x, 'seuclidean', V=v)
    # dy = spatial.distance.pdist(y, 'seuclidean', V=v)
    # dxy = spatial.distance.cdist(x, y, 'seuclidean', V=v)
    #
    # phix = -np.log(dx).sum() / nx / (nx - 1)
    # phiy = -np.log(dy).sum() / ny / (ny - 1)
    # phixy = np.log(dxy).sum() / nx / ny

    # z = dxy.sum() * 2. / (nx*ny) - (1./nx**2) *

    # z = (2 / (n * m)) * sum(dxy(:)) - (1 / (n ^ 2)) * sum(2 * dx) - (1 /
    #  (m ^ 2)) * sum(2 * dy);
    # z = ((n * m) / (n + m)) * z;


@metric
def friedman_rafsky(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a dissimilarity metric based on the Friedman-Rafsky runs statistics.

    The algorithm builds a minimal spanning tree (the subset of edges
    connecting all points that minimizes the total edge length) then counts
    the edges linking points from the same distribution.

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
    Friedman J.H. and Rafsky L.C. (1979) Multivariate generaliations of the
    Wald-Wolfowitz and Smirnov two-sample tests. Annals of Stat. Vol.7, No. 4, 697-717.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from sklearn import neighbors

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
    """
    Compute the Kolmogorov-Smirnov statistic applied to two multivariate samples as described by Fasano and Franceschini.

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
    Fasano G. and Francheschini A. (1987) A multidimensional version
    of the Kolmogorov-Smirnov test. Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170.
    """

    def pivot(x, y):
        nx, d = x.shape
        ny, d = y.shape

        # Multiplicative factor converting d-dim booleans to a unique integer.
        mf = (2 ** np.arange(d)).reshape(1, d, 1)
        minlength = 2 ** d

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
    x: np.ndarray, y: np.ndarray, *, k: Union[int, Sequence[int]] = 1
) -> Union[float, Sequence[float]]:
    r"""
    Compute the Kullback-Leibler divergence between two multivariate samples.

    .. math
        D(P||Q) = "\"frac{d}{n} "\"sum_i^n "\"log{"\"frac{r_k(x_i)}{s_k(x_i)}} + "\"log{"\"frac{m}{n-1}}

    where r_k(x_i) and s_k(x_i) are, respectively, the euclidean distance
    to the kth neighbour of x_i in the x array (excepting x_i) and
    in the y array.

    Parameters
    ----------
    x : np.ndarray (n,d)
      Samples from distribution P, which typically represents the true
      distribution (reference).
    y : np.ndarray (m,d)
      Samples from distribution Q, which typically represents the
      approximate distribution (candidate)
    k : int or sequence
      The kth neighbours to look for when estimating the density of the
      distributions. Defaults to 1, which can be noisy.

    Returns
    -------
    float or sequence
      The estimated Kullback-Leibler divergence D(P||Q) computed from
      the distances to the kth neighbour.

    Notes
    -----
    In information theory, the Kullback–Leibler divergence is a non-symmetric
    measure of the difference between two probability distributions P and Q,
    where P is the "true" distribution and Q an approximation. This nuance is
    important because D(P||Q) is not equal to D(Q||P).

    For probability distributions P and Q of a continuous random variable,
    the K–L  divergence is defined as:

        D_{KL}(P||Q) = "\"int p(x) "\"log{p()/q(x)} dx

    This formula assumes we have a representation of the probability
    densities p(x) and q(x).  In many cases, we only have samples from the
    distribution, and most methods first estimate the densities from the
    samples and then proceed to compute the K-L divergence. In Perez-Cruz,
    the authors propose an algorithm to estimate the K-L divergence directly
    from the sample using an empirical CDF. Even though the CDFs do not
    converge to their true values, the paper proves that the K-L divergence
    almost surely does converge to its true value.

    References
    ----------
    Kullback-Leibler Divergence Estimation of Continuous Distributions (2008).
    Fernando Pérez-Cruz.
    """
    mk = np.iterable(k)
    ka = np.atleast_1d(k)

    nx, d = x.shape
    ny, d = y.shape

    # Limit the number of dimensions to 10, too slow otherwise.
    if d > 10:
        raise ValueError("Too many dimensions: {}.".format(d))

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
    # negative sign on the first term of the right hand side.
    out = list()
    for ki in ka:
        # The 0th nearest neighbour of x[i] in x is x[i] itself.
        # Hence we take the k'th + 1, which in 0-based indexing is given by
        # index k.
        out.append(
            -np.log(r[:, ki] / s[:, ki - 1]).sum() * d / nx + np.log(ny / (nx - 1.0))
        )

    if mk:
        return out
    return out[0]
