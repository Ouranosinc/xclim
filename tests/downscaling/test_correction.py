import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling.correction import DetrendedQuantileMapping
from xclim.downscaling.correction import EmpiricalQuantileMapping
from xclim.downscaling.correction import LOCI
from xclim.downscaling.correction import QuantileDeltaMapping
from xclim.downscaling.correction import Scaling
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import equally_spaced_nodes
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import invert
from xclim.downscaling.utils import MULTIPLICATIVE


@pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
class TestLoci:
    def test_time(self, series, group, dec):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=0, scale=3)
        x = xd.ppf(u)

        sim = fut = series(x, "pr")
        y = x * 2
        thresh = 2
        obs_fit = series(y, "pr").where(y > thresh, 0.1)
        obs = series(y, "pr")

        loci = LOCI(group=group, thresh=thresh)
        loci.train(obs_fit, sim)
        np.testing.assert_array_almost_equal(loci.ds.sim_thresh, 1, dec)
        np.testing.assert_array_almost_equal(loci.ds.cf, 2, dec)

        p = loci.predict(fut)
        np.testing.assert_array_almost_equal(p, obs, dec)


class TestScaling:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_time(self, kind, name, series):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sim = fut = series(x, name)
        obs = series(apply_correction(x, 2, kind), name)

        scaling = Scaling(group="time", kind=kind)
        scaling.train(obs, sim)
        np.testing.assert_array_almost_equal(scaling.ds.cf, 2)

        p = scaling.predict(fut)
        np.testing.assert_array_almost_equal(p, obs)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sim = fut = series(x, name)
        obs = mon_series(apply_correction(x, 2, kind), name)

        # Test train
        scaling = Scaling(group="time.month", kind=kind)
        scaling.train(obs, sim)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(scaling.ds.cf, expected)

        # Test predict
        p = scaling.predict(fut)
        np.testing.assert_array_almost_equal(p, obs)


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

        DQM = DetrendedQuantileMapping(
            kind=kind, group="time", nquantiles=50, interp="linear"
        )
        DQM.train(obs, sim)
        p = DQM.predict(fut)

        q = DQM.ds.quantiles
        ex = apply_correction(xd.ppf(q), invert(xd.mean(), kind), kind)
        ey = apply_correction(yd.ppf(q), invert(yd.mean(), kind), kind)
        expected = get_correction(ex, ey, kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(DQM.ds.cf[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], obs[middle], 1)

        # Test with future not equal to sim
        ff = series(np.ones(ns) * 1.1, name)
        fut2 = apply_correction(fut, ff, kind)
        obs2 = apply_correction(obs, ff, kind)

        p2 = DQM.predict(fut2)

        np.testing.assert_array_almost_equal(p2[middle], obs2[middle], 1)

        # Test with actual trend in fut
        trend = series(
            np.linspace(-0.2, 0.2, ns) + (1 if kind == MULTIPLICATIVE else 0), name
        )
        fut3 = apply_correction(fut, trend, kind)
        obs3 = apply_correction(obs, trend, kind)
        p3 = DQM.predict(fut3)
        np.testing.assert_array_almost_equal(p3[middle], obs3[middle], 1)

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

        trend = np.linspace(-0.2, 0.2, n) + int(kind == MULTIPLICATIVE)
        obs_t = mon_series(apply_correction(y, trend, kind), name)
        fut = series(apply_correction(x, trend, kind), name)

        if spatial_dims:
            sim = sim.expand_dims(**spatial_dims)
            obs = obs.expand_dims(**spatial_dims)
            fut = fut.expand_dims(**spatial_dims)
            obs_t = obs_t.expand_dims(**spatial_dims)

        DQM = DetrendedQuantileMapping(kind=kind, group="time.month", nquantiles=5)
        DQM.train(obs, sim)
        mqm = DQM.ds.cf.mean(dim="quantiles")
        p = DQM.predict(fut)

        if spatial_dims:
            mqm = mqm.isel({crd: 0 for crd in spatial_dims.keys()})
        np.testing.assert_array_almost_equal(mqm, int(kind == MULTIPLICATIVE), 1)
        np.testing.assert_array_almost_equal(p, obs_t, 1)

    def test_cannon(self, cannon_2015_rvs):
        obs, hist, fut = cannon_2015_rvs(15000)

        DQM = DetrendedQuantileMapping(kind="*", group="time")
        DQM.train(obs, hist)
        p = DQM.predict(fut)

        np.testing.assert_almost_equal(p.mean(), 41.6, 0)
        np.testing.assert_almost_equal(p.std(), 15.0, 0)


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

        QDM = QuantileDeltaMapping(
            kind=kind, group="time", nquantiles=10, interp="linear"
        )
        QDM.train(obs, sim)
        p = QDM.predict(fut)

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QDM.ds.cf.T, expected, 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (u > 1e-2) * (u < 0.99)
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

        QDM = QuantileDeltaMapping(kind=kind, group="time.month", nquantiles=40)
        QDM.train(obs, sim)
        p = QDM.predict(fut)

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(
            mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind
        )
        np.testing.assert_array_almost_equal(QDM.ds.cf.sel(quantiles=q), expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, obs, 1)

    def test_cannon(self, cannon_2015_dist, cannon_2015_rvs):
        obs, hist, fut = cannon_2015_rvs(15000, random=False)

        # Quantile mapping
        QDM = QuantileDeltaMapping(kind="*", group="time", nquantiles=50)
        QDM.train(obs, hist)
        bc_fut = QDM.predict(fut)

        # Theoretical results
        # obs, hist, fut = cannon_2015_dist
        # u1 = equally_spaced_nodes(1001, None)
        # u = np.convolve(u1, [0.5, 0.5], mode="valid")
        # pu = obs.ppf(u) * fut.ppf(u) / hist.ppf(u)
        # pu1 = obs.ppf(u1) * fut.ppf(u1) / hist.ppf(u1)
        # pdf = np.diff(u1) / np.diff(pu1)

        # mean = np.trapz(pdf * pu, pu)
        # mom2 = np.trapz(pdf * pu ** 2, pu)
        # std = np.sqrt(mom2 - mean ** 2)

        np.testing.assert_almost_equal(bc_fut.mean(), 41.5, 1)
        np.testing.assert_almost_equal(bc_fut.std(), 16.7, 0)


class TestQM:
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
        QM = EmpiricalQuantileMapping(
            kind=kind, group="time", nquantiles=50, interp="linear"
        )
        QM.train(obs, sim)
        p = QM.predict(fut)

        q = QM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QM.ds.cf[2:-2], expected[2:-2], 1)

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
        obs = mon_series(y, name)

        QM = EmpiricalQuantileMapping(kind=kind, group="time.month", nquantiles=5)
        QM.train(obs, sim)
        p = QM.predict(fut)
        mqm = QM.ds.cf.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, obs, 2)
