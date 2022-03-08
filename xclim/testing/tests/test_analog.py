# Tests taken from flyingpigeon on Nov 2020
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from pkg_resources import parse_version
from scipy import __version__ as __scipy_version__
from scipy import integrate, stats

import xclim.analog as xca
from xclim.testing import open_dataset


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


def randn(mean, std, shape):
    """Return a random normal sample with exact mean and standard deviation."""
    r = np.random.randn(*shape)
    r1 = r / r.std(0, ddof=1) * np.array(std)
    return r1 - r1.mean(0) + np.array(mean)


def test_randn():
    mu, std = [2, 3], [1, 2]
    r = randn(mu, std, [10, 2])
    assert_almost_equal(r.mean(0), mu)
    assert_almost_equal(r.std(0, ddof=1), std)


@pytest.mark.slow
@pytest.mark.parametrize("method", xca.metrics.keys())
def test_spatial_analogs(method):
    if method == "skezely_rizzo":
        pytest.skip("Method not implemented.")

    if method in ["nearest_neighbor", "kldiv"] and parse_version(
        __scipy_version__
    ) < parse_version("1.6.0"):
        pytest.skip("Method not supported in scipy<1.6.0")

    diss = open_dataset("SpatialAnalogs/dissimilarity")
    data = open_dataset("SpatialAnalogs/indicators")

    target = data.sel(lat=46.1875, lon=-72.1875, time=slice("1970", "1990"))
    candidates = data.sel(time=slice("1970", "1990"))

    out = xca.spatial_analogs(target, candidates, method=method)
    np.testing.assert_allclose(diss[method], out, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_spatial_analogs_multidim():
    diss = open_dataset("SpatialAnalogs/dissimilarity")
    data = open_dataset("SpatialAnalogs/indicators")

    targets = data.sel(
        lat=slice(46, 47), lon=slice(-73, -72), time=slice("1970", "1990")
    )
    targets = targets.stack(locations=["lat", "lon"])
    candidates = data.sel(time=slice("1970", "1990"))

    out = xca.spatial_analogs(targets, candidates, method="seuclidean")
    assert out.dims == ("locations", "lat", "lon")

    np.testing.assert_array_almost_equal(
        diss.seuclidean, out.sel(locations=(46.1875, -72.1875)), 5
    )
    assert out.attrs["indices"] == "meantemp,totalpr"


def test_spatial_analogs_multi_index():
    # Test multi-indexes
    diss = open_dataset("SpatialAnalogs/dissimilarity")
    data = open_dataset("SpatialAnalogs/indicators")

    target = data.sel(lat=46.1875, lon=-72.1875, time=slice("1970", "1990"))
    candidates = data.sel(time=slice("1970", "1990"))

    target_stacked = target.stack(sample=["time"])
    candidates_stacked = candidates.stack(sample=["time"])

    method = "seuclidean"
    out = xca.spatial_analogs(
        target_stacked, candidates_stacked, dist_dim="sample", method=method
    )
    np.testing.assert_allclose(diss[method], out, rtol=1e-3, atol=1e-3)

    # Check that it works as well when time dimensions don't have the same length.
    candidates = data.sel(time=slice("1970", "1991"))
    xca.spatial_analogs(
        target_stacked, candidates_stacked, dist_dim="sample", method=method
    )


class TestSEuclidean:
    def test_simple(self):
        d = 2
        n, m = 25, 30

        x = randn(0, 1, (n, d))
        y = randn([1, 2], 1, (m, d))
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, np.hypot(1, 2), 2)

        # Variance of the candidate sample does not affect answer.
        x = randn(0, 1, (n, d))
        y = randn([1, 2], 2, (m, d))
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, np.hypot(1, 2), 2)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.seuclidean(x, y)
        assert_almost_equal(dm, 2.8463, 4)


@pytest.mark.skipif(
    parse_version(__scipy_version__) < parse_version("1.6.0"),
    reason="Not supported in scipy<1.6.0",
)
class TestNN:
    def test_simple(self):
        d = 2
        n, m = 200, 200
        np.random.seed(1)
        x = np.random.randn(n, d)
        y = np.random.randn(m, d)

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
    def test_simple(self):
        d = 2
        n = 200
        # m = 200
        x = np.random.randn(n, d)
        # y = np.random.randn(m, d)

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
    def test_1D_ks_2samp(self):
        # Compare with scipy.stats.ks_2samp
        x = np.random.randn(50) + 1
        y = np.random.randn(50)
        s, p = stats.ks_2samp(x, y)
        dm = xca.kolmogorov_smirnov(x, y)
        assert_almost_equal(dm, s, 3)

    def test_compare_with_matlab(self):
        x, y = matlab_sample()
        dm = xca.kolmogorov_smirnov(x, y)
        assert_almost_equal(dm, 0.96667, 4)


def analytical_KLDiv(p, q):
    """Return the Kullback-Leibler divergence between two distributions.

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
@pytest.mark.skipif(
    parse_version(__scipy_version__) < parse_version("1.6.0"),
    reason="Not supported in scipy<1.6.0",
)
class TestKLDIV:
    #
    def test_against_analytic(self):
        p = stats.norm(2, 1)
        q = stats.norm(2.6, 1.4)

        ra = analytical_KLDiv(p, q)

        N = 10000
        np.random.seed(2)
        # x, y = p.rvs(N), q.rvs(N)

        re = xca.kldiv(p.rvs(N), q.rvs(N))

        assert_almost_equal(re, ra, 1)

    def accuracy_vs_kth(self, n=100, trials=100):
        """Evalute the accuracy of the algorithm as a function of k.

        Parameters
        ----------
        N : int
          Number of random samples.
        trials : int
          Number of independent drawing experiments.

        Returns
        -------
        (err, stddev) The mean error and standard deviation around the
        analytical value for different values of k from 1 to 15.
        """
        p = stats.norm(0, 1)
        q = stats.norm(0.2, 0.9)

        k = np.arange(1, 16)

        out = []
        for i in range(trials):
            out.append(xca.kldiv(p.rvs(n), q.rvs(n), k=k))
        out = np.array(out)

        # Compare with analytical value
        err = out - analytical_KLDiv(p, q)

        # Return mean and standard deviation
        return err.mean(0), err.std(0)

    #
    def test_accuracy(self):
        m, _ = self.accuracy_vs_kth(n=500, trials=500)
        assert_almost_equal(np.mean(m[0:2]), 0, 2)

    #
    def test_different_sample_size(self):
        p = stats.norm(2, 1)
        q = stats.norm(2.6, 1.4)

        ra = analytical_KLDiv(p, q)

        n = 6000
        # Same sample size for x and y
        re = [xca.kldiv(p.rvs(n), q.rvs(n)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

        # Different sample sizes
        re = [xca.kldiv(p.rvs(n * 2), q.rvs(n)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

        re = [xca.kldiv(p.rvs(n), q.rvs(n * 2)) for i in range(30)]
        assert_almost_equal(np.mean(re), ra, 2)

    #
    def test_mvnormal(self):
        """Compare the results to the figure 2 in the paper."""
        from numpy.random import multivariate_normal, normal

        n = 30000
        p = normal(0, 1, size=(n, 2))
        np.random.seed(1)
        q = multivariate_normal([0.5, -0.5], [[0.5, 0.1], [0.1, 0.3]], size=n)

        assert_almost_equal(xca.kldiv(p, q), 1.39, 1)
        assert_almost_equal(xca.kldiv(q, p), 0.62, 1)
