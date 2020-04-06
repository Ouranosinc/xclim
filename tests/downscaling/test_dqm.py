# Tests for detrended quantile mapping
import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling import examples
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import invert
from xclim.downscaling.utils import MULTIPLICATIVE


class TestDQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        ns = 10000
        u = np.random.rand(ns)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sim = fut = series(x, name)
        obs = series(y, name)

        p, qm = examples.dqm(
            obs, sim, fut, kind=kind, group="time", nquantiles=50, interp="linear"
        )

        q = qm.quantiles
        ex = apply_correction(xd.ppf(q), invert(xd.mean(), kind), kind)
        ey = yd.ppf(q)
        expected = get_correction(ex, ey, kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], obs[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize(
        "spatial_dims", [None, {"lat": np.arange(20), "lon": np.arange(20)}]
    )
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name, spatial_dims):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        n = 10000
        u = np.random.rand(n)

        # Define distributions
        xd = uniform(loc=2, scale=0.1)
        yd = uniform(loc=4, scale=0.1)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sim, obs = series(x, name), mon_series(y, name)

        trend = np.linspace(-0.2, 0.2, n) + (1 if kind == MULTIPLICATIVE else 0)
        obs_t = mon_series(apply_correction(y, trend, kind), name)
        fut = series(apply_correction(x, trend, kind), name)

        if spatial_dims:
            sim = sim.expand_dims(**spatial_dims)
            obs = obs.expand_dims(**spatial_dims)
            fut = fut.expand_dims(**spatial_dims)
            obs_t = obs_t.expand_dims(**spatial_dims)

        p, qm = examples.dqm(obs, sim, fut, kind=kind, group="time.month", nquantiles=5)
        mqm = qm.qf.mean(dim="quantiles")

        expected = apply_correction(mon_triangular, 4, kind)

        if spatial_dims:
            mqm = mqm.isel({crd: 0 for crd in spatial_dims.keys()})
        np.testing.assert_array_almost_equal(mqm, expected, 1)
        np.testing.assert_array_almost_equal(p, obs_t, 1)


class TestQDM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        x : U(1,1)
        y : U(1,2)

        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=4)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sim = fut = series(x, name)
        obs = series(y, name)
        p, qm = examples.qdm(
            obs, sim, fut, kind=kind, group="time", nquantiles=10, interp="linear"
        )

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf.T, expected, 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], obs[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=2)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sim = fut = series(x, name)
        obs = mon_series(y, name)
        p, qm = examples.qdm(
            obs, sim, fut, kind=kind, group="time.month", nquantiles=40
        )

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(
            mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind
        )
        np.testing.assert_array_almost_equal(qm.qf.sel(quantiles=q), expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, obs, 1)


class TestEQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sim = fut = series(x, name)
        obs = series(y, name)
        p, qm = examples.eqm(
            obs, sim, fut, kind=kind, group="time", nquantiles=50, interp="linear"
        )

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], obs[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=2, scale=0.1)
        yd = uniform(loc=4, scale=0.1)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sim = fut = series(x, name)
        obs = series(y, name)

        p, qm = examples.eqm(obs, sim, fut, kind=kind, group="time.month", nquantiles=5)
        mqm = qm.qf.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, obs, 2)
