# Tests taken from flyingpigeon on Nov 2020
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import integrate, stats
from sklearn import datasets

import xclim.analog as xca


def matlab_sample(n=30):
    """
    In some of the following tests I'm using Matlab code written by Patrick
    Grenier for the paper "An Assessment of Six Dissimilarity Metrics for
    Climate Analogs" to compare against the functions here. The sample
    created here is identical to the sample used to drive the Matlab code.

    Parameters
    ----------
    n : int
      Sample size.

    Returns
    -------
    2D array, 2D array
       Synthetic samples (3, n)
    """
    z = 1.0 * (np.arange(n) + 1) / n - 0.5

    x = np.vstack([z * 2 + 30, z * 3 + 40, z]).T

    y = np.vstack([z * 2.2 + 31, z[::-1] * 2.8 + 38, z * 1.1]).T

    return x, y


@pytest.fixture
def exact_randn(random):
    def _randn(mean, std, shape):
        """Return a random normal sample with exact mean and standard deviation."""
        r = random.standard_normal(shape)
        r1 = r / r.std(0, ddof=1) * np.array(std)
        return r1 - r1.mean(0) + np.array(mean)

    return _randn


def test_exact_randn(exact_randn):
    mu, std = [2, 3], [1, 2]
    r = exact_randn(mu, std, [10, 2])
    assert_almost_equal(r.mean(0), mu)
    assert_almost_equal(r.std(0, ddof=1), std)


@pytest.mark.slow
@pytest.mark.parametrize("method", xca.metrics.keys())
def test_spatial_analogs(method, open_dataset):
    diss = open_dataset("SpatialAnalogs/dissimilarity.nc")
    data = open_dataset("SpatialAnalogs/indicators.nc")

    target = data.sel(lat=46.1875, lon=-72.1875, time=slice("1970", "1990"))
    candidates = data.sel(time=slice("1970", "1990"))

    out = xca.spatial_analogs(target, candidates, method=method)
    # Special case since scikit-learn updated to 1.2.0 (and again at 1.3)
    if method == "friedman_rafsky":
        diss[method][42, 105] = np.nan
        out[42, 105] = np.nan
    np.testing.assert_allclose(diss[method], out, rtol=1e-3, atol=1e-3)


def test_unsupported_spatial_analog_method(open_dataset):
    method = "KonMari"

    data = open_dataset("SpatialAnalogs/indicators.nc")
    target = data.sel(lat=46.1875, lon=-72.1875, time=slice("1970", "1990"))
    candidates = data.sel(time=slice("1970", "1990"))

    match_statement = f"Method `KonMari` is not implemented. Available methods are: {','.join(xca.metrics.keys())}"

    with pytest.raises(ValueError, match=match_statement):
        xca.spatial_analogs(target, candidates, method=method)


def test_spatial_analogs_multi_index(open_dataset):
    # Test multi-indexes
    diss = open_dataset("SpatialAnalogs/dissimilarity.nc")
    data = open_dataset("SpatialAnalogs/indicators.nc")

    target = data.sel(lat=46.1875, lon=-72.1875, time=slice("1970", "1990"))
    candidates = data.sel(time=slice("1970", "1990"))

    target_stacked = target.stack(sample=["time"])
    candidates_stacked = candidates.stack(sample=["time"])

    method = "seuclidean"
    out = xca.spatial_analogs(target_stacked, candidates_stacked, dist_dim="sample", method=method)
    np.testing.assert_allclose(diss[method], out, rtol=1e-3, atol=1e-3)

    # Check that it works as well when time dimensions don't have the same length.
    candidates = data.sel(time=slice("1970", "1991"))
    xca.spatial_analogs(target_stacked, candidates_stacked, dist_dim="sample", method=method)


class TestSEuclidean:
    def test_simple(self, exact_randn):
        d = 2
        n, m = 25, 30

        x = exact_randn(0, 1, (n, d))
        y = exact_randn([1, 2], 1, (m, d))
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, np.hypot(1, 2), 2)

        # Variance of the candidate sample does not affect answer.
        x = exact_randn(0, 1, (n, d))
        y = exact_randn([1, 2], 2, (m, d))
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, np.hypot(1, 2), 2)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, 2.8463, 4)


class TestNN:
    def test_simple(self, random):
        d = 2
        n, m = 200, 200
        x = random.standard_normal((n, d))
        y = random.standard_normal((m, d))

        # Almost identical samples
        dm = xca.nearest_neighbor(x + 0.001, x)
        assert_almost_equal(dm, 0, 2)

        # Same distribution but mixed
        dm = xca.nearest_neighbor(x, y)
        assert_almost_equal(dm, 0.5, 1)

        # Two completely different distributions
        dm = xca.nearest_neighbor(x + 10, y)
        assert_almost_equal(dm, 1, 2)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.nearest_neighbor(x, y)
        assert_almost_equal(dm, 1, 4)


class TestZAE:
    def test_simple(self, random):
        d = 2
        n = 200
        # m = 200
        x = random.standard_normal((n, d))
        # y = random.standard_normal(m, d)

        # Almost identical samples
        dm = xca.zech_aslan(x + 0.001, x)
        assert dm < 0

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.zech_aslan(x, y)
        assert_almost_equal(dm, 0.77802, 4)


class TestFR:
    def test_simple(self):
        # Over these 7 points, there are 2 with edges within the same sample.
        # [1,2]-[2,2] & [3,2]-[4,2]
        # |
        # |   x
        # | o o x x
        # | x  o
        # |_ _ _ _ _ _ _
        x = np.array([[1, 2], [2, 2], [3, 1]])
        y = np.array([[1, 1], [2, 4], [3, 2], [4, 2]])

        dm = xca.friedman_rafsky(x, y)
        assert_almost_equal(dm, 2.0 / 7, 3)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.friedman_rafsky(x, y)
        assert_almost_equal(dm, 0.96667, 4)


class TestKS:
    def test_1D_ks_2samp(self, random):
        # Compare with scipy.stats.ks_2samp
        x = random.standard_normal(50) + 1
        y = random.standard_normal(50)
        s, _p = stats.ks_2samp(x, y)
        dm = xca.kolmogorov_smirnov(x, y)
        assert_almost_equal(dm, s, 3)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.kolmogorov_smirnov(x, y)
        assert_almost_equal(dm, 0.96667, 4)


def analytical_KLDiv(p, q):
    """
    Return the Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p, q : scipy.frozen_rv
      Scipy frozen distribution instances, e.g. stats.norm(0,1)

    Returns
    -------
    out : float
      The Kullback-Leibler divergence computed by numerically integrating
      p(x)*log(p(x)/q(x)).
    """

    def func(x):
        return p.pdf(x) * np.log(p.pdf(x) / q.pdf(x))

    a = 1e-5
    return integrate.quad(func, max(p.ppf(a), q.ppf(a)), min(p.isf(a), q.isf(a)))[0]


@pytest.mark.slow
class TestKLDIV:
    #
    def test_against_analytic(self, random):
        p = stats.norm(2, 1)
        q = stats.norm(2.6, 1.4)

        ra = analytical_KLDiv(p, q)

        N = 10000
        # x, y = p.rvs(N, random_state=random), q.rvs(N, random_state=random)

        re = xca.kldiv(p.rvs(N, random_state=random), q.rvs(N, random_state=random))

        assert_almost_equal(re, ra, 1)

    def test_accuracy(self, random):
        p = stats.norm(0, 1)
        q = stats.norm(0.2, 0.9)

        k = np.arange(1, 16)

        out = []
        n = 500
        for _i in range(500):
            out.append(xca.kldiv(p.rvs(n, random_state=random), q.rvs(n, random_state=random), k=k))
        out = np.array(out)

        # Compare with analytical value
        err = out - analytical_KLDiv(p, q)

        m = err.mean(0)
        assert_almost_equal(np.mean(m[0:2]), 0, 2)

    #
    def test_different_sample_size(self, random):
        p = stats.norm(2, 1)
        q = stats.norm(2.6, 1.4)

        ra = analytical_KLDiv(p, q)

        n = 6000
        # Same sample size for x and y
        re = [xca.kldiv(p.rvs(n, random_state=random), q.rvs(n, random_state=random)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

        # Different sample sizes
        re = [xca.kldiv(p.rvs(n * 2, random_state=random), q.rvs(n, random_state=random)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

        re = [xca.kldiv(p.rvs(n, random_state=random), q.rvs(n * 2, random_state=random)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

    #
    def test_mvnormal(self, random):
        """Compare the results to the figure 2 in the paper."""
        n = 30000
        p = random.normal(0, 1, size=(n, 2))
        q = random.multivariate_normal([0.5, -0.5], [[0.5, 0.1], [0.1, 0.3]], size=n)

        assert_almost_equal(xca.kldiv(p, q), 1.39, 1)
        assert_almost_equal(xca.kldiv(q, p), 0.62, 1)


def test_szekely_rizzo():
    iris = pd.DataFrame(datasets.load_iris().data)

    # first 80 against last 70
    x = iris.iloc[:80, :].to_xarray().to_array().T
    y = iris.iloc[80:, :].to_xarray().to_array().T

    np.testing.assert_allclose(xca.szekely_rizzo(x, y, standardize=False), 116.1987, atol=5e-5)

    # first 50 against last 100
    x = iris.iloc[:50, :].to_xarray().to_array().T
    y = iris.iloc[50:, :].to_xarray().to_array().T

    np.testing.assert_allclose(xca.szekely_rizzo(x, y, standardize=False), 199.6205, atol=5e-5)
