import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm
from scipy.stats import uniform

sdba = pytest.importorskip("xclim.sdba")  # noqa
from xclim.sdba.adjustment import DetrendedQuantileMapping
from xclim.sdba.adjustment import EmpiricalQuantileMapping
from xclim.sdba.adjustment import LOCI
from xclim.sdba.adjustment import QuantileDeltaMapping
from xclim.sdba.adjustment import Scaling
from xclim.sdba.utils import ADDITIVE
from xclim.sdba.utils import apply_correction
from xclim.sdba.utils import get_correction
from xclim.sdba.utils import invert
from xclim.sdba.utils import MULTIPLICATIVE


@pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
class TestLoci:
    def test_time(self, series, group, dec):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=0, scale=3)
        x = xd.ppf(u)

        hist = sim = series(x, "pr")
        y = x * 2
        thresh = 2
        ref_fit = series(y, "pr").where(y > thresh, 0.1)
        ref = series(y, "pr")

        loci = LOCI(group=group, thresh=thresh)
        loci.train(ref_fit, hist)
        np.testing.assert_array_almost_equal(loci.ds.hist_thresh, 1, dec)
        np.testing.assert_array_almost_equal(loci.ds.af, 2, dec)

        p = loci.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref, dec)

        assert "history" in p.attrs
        assert "Bias-adjusted with LOCI(" in p.attrs["history"]


class TestScaling:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_time(self, kind, name, series):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        hist = sim = series(x, name)
        ref = series(apply_correction(x, 2, kind), name)

        scaling = Scaling(group="time", kind=kind)
        scaling.train(ref, hist)
        np.testing.assert_array_almost_equal(scaling.ds.af, 2)

        p = scaling.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        hist = sim = series(x, name)
        ref = mon_series(apply_correction(x, 2, kind), name)

        # Test train
        scaling = Scaling(group="time.month", kind=kind)
        scaling.train(ref, hist)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(scaling.ds.af, expected)

        # Test predict
        p = scaling.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref)


class TestDQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        hist: U
        ref: Normal

        Predict on hist to get ref
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
        hist = sim = series(x, name)
        ref = series(y, name)

        DQM = DetrendedQuantileMapping(kind=kind, group="time", nquantiles=50,)
        DQM.train(ref, hist)
        p = DQM.adjust(sim, interp="linear")

        q = DQM.ds.quantiles
        ex = apply_correction(xd.ppf(q), invert(xd.mean(), kind), kind)
        ey = apply_correction(yd.ppf(q), invert(yd.mean(), kind), kind)
        expected = get_correction(ex, ey, kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(DQM.ds.af[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

        # Test with simure not equal to hist
        ff = series(np.ones(ns) * 1.1, name)
        sim2 = apply_correction(sim, ff, kind)
        ref2 = apply_correction(ref, ff, kind)

        p2 = DQM.adjust(sim2, interp="linear")

        np.testing.assert_array_almost_equal(p2[middle], ref2[middle], 1)

        # Test with actual trend in sim
        trend = series(
            np.linspace(-0.2, 0.2, ns) + (1 if kind == MULTIPLICATIVE else 0), name
        )
        sim3 = apply_correction(sim, trend, kind)
        ref3 = apply_correction(ref, trend, kind)
        p3 = DQM.adjust(sim3, interp="linear")
        np.testing.assert_array_almost_equal(p3[middle], ref3[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize(
        "spatial_dims", [None, {"lat": np.arange(20), "lon": np.arange(20)}]
    )
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name, spatial_dims):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
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
        hist, ref = series(x, name), mon_series(y, name)

        trend = np.linspace(-0.2, 0.2, n) + int(kind == MULTIPLICATIVE)
        ref_t = mon_series(apply_correction(y, trend, kind), name)
        sim = series(apply_correction(x, trend, kind), name)

        if spatial_dims:
            hist = hist.expand_dims(**spatial_dims).chunk({"lat": 10})
            sim = sim.expand_dims(**spatial_dims).chunk({"lat": 10})
            ref_t = ref_t.expand_dims(**spatial_dims)

        DQM = DetrendedQuantileMapping(kind=kind, group="time.month", nquantiles=5)
        DQM.train(ref, hist)
        mqm = DQM.ds.af.mean(dim="quantiles")
        p = DQM.adjust(sim)

        if spatial_dims:
            mqm = mqm.isel({crd: 0 for crd in spatial_dims.keys()})
        np.testing.assert_array_almost_equal(mqm, int(kind == MULTIPLICATIVE), 1)
        np.testing.assert_array_almost_equal(p, ref_t, 1)

    def test_cannon(self, cannon_2015_rvs):
        ref, hist, sim = cannon_2015_rvs(15000)

        DQM = DetrendedQuantileMapping(kind="*", group="time")
        DQM.train(ref, hist)
        p = DQM.adjust(sim)

        np.testing.assert_almost_equal(p.mean(), 41.6, 0)
        np.testing.assert_almost_equal(p.std(), 15.0, 0)

    def test_group_norm(self, tas_series):
        x = np.arange(365)
        ref = tas_series(273 - 10 * np.cos(2 * np.pi * x / 365), start="2001-01-01")
        hist = sim = tas_series(
            273 - 8 * np.cos(2 * np.pi * x / 365), start="2001-01-01"
        )
        DQM = DetrendedQuantileMapping(group="time.month")
        DQM.train(ref, hist)
        scen = DQM.adjust(sim, interp="linear")
        xr.testing.assert_allclose(ref, scen, rtol=1.5e-4)


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
        hist = sim = series(x, name)
        ref = series(y, name)

        QDM = QuantileDeltaMapping(kind=kind, group="time", nquantiles=10,)
        QDM.train(ref, hist)
        p = QDM.adjust(sim, interp="linear")

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QDM.ds.af.T, expected, 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (u > 1e-2) * (u < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize("add_dims", [True, False])
    def test_mon_U(
        self, mon_series, series, mon_triangular, add_dims, kind, name, use_dask
    ):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
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
        hist = sim = series(x, name)
        if use_dask:
            sim = sim.chunk({"time": -1})
        if add_dims:
            hist = hist.expand_dims(site=[0, 1, 2, 3, 4])
            sim = sim.expand_dims(site=[0, 1, 2, 3, 4])
            sel = {"site": 0}
        else:
            sel = {}

        ref = mon_series(y, name)

        QDM = QuantileDeltaMapping(kind=kind, group="time.month", nquantiles=40)
        QDM.train(ref, hist)
        p = QDM.adjust(sim)

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(
            mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind
        )
        np.testing.assert_array_almost_equal(
            QDM.ds.af.sel(quantiles=q, **sel), expected, 1
        )

        # Test predict
        np.testing.assert_array_almost_equal(p.isel(**sel), ref, 1)

    def test_cannon(self, cannon_2015_dist, cannon_2015_rvs):
        ref, hist, sim = cannon_2015_rvs(15000, random=False)

        # Quantile mapping
        QDM = QuantileDeltaMapping(kind="*", group="time", nquantiles=50)
        QDM.train(ref, hist)
        bc_sim = QDM.adjust(sim)

        # Theoretical results
        # ref, hist, sim = cannon_2015_dist
        # u1 = equally_spaced_nodes(1001, None)
        # u = np.convolve(u1, [0.5, 0.5], mode="valid")
        # pu = ref.ppf(u) * sim.ppf(u) / hist.ppf(u)
        # pu1 = ref.ppf(u1) * sim.ppf(u1) / hist.ppf(u1)
        # pdf = np.diff(u1) / np.diff(pu1)

        # mean = np.trapz(pdf * pu, pu)
        # mom2 = np.trapz(pdf * pu ** 2, pu)
        # std = np.sqrt(mom2 - mean ** 2)

        np.testing.assert_almost_equal(bc_sim.mean(), 41.5, 1)
        np.testing.assert_almost_equal(bc_sim.std(), 16.7, 0)


class TestQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        hist: U
        ref: Normal

        Predict on hist to get ref
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        hist = sim = series(x, name)
        ref = series(y, name)
        QM = EmpiricalQuantileMapping(kind=kind, group="time", nquantiles=50,)
        QM.train(ref, hist)
        p = QM.adjust(sim, interp="linear")

        q = QM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QM.ds.af[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
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
        hist = sim = series(x, name)
        ref = mon_series(y, name)

        QM = EmpiricalQuantileMapping(kind=kind, group="time.month", nquantiles=5)
        QM.train(ref, hist)
        p = QM.adjust(sim)
        mqm = QM.ds.af.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, ref, 2)
