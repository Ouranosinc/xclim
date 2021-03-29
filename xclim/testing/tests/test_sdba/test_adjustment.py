import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm, uniform

from xclim.sdba.adjustment import (
    LOCI,
    BaseAdjustment,
    DetrendedQuantileMapping,
    EmpiricalQuantileMapping,
    PrincipalComponents,
    QuantileDeltaMapping,
    Scaling,
)
from xclim.sdba.base import Grouper
from xclim.sdba.utils import (
    ADDITIVE,
    MULTIPLICATIVE,
    apply_correction,
    get_correction,
    invert,
)

from .utils import nancov


@pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
class TestLoci:
    def test_time_and_from_ds(self, series, group, dec, tmp_path):
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

        assert "xclim_history" in p.attrs
        assert "Bias-adjusted with LOCI(" in p.attrs["xclim_history"]

        file = tmp_path / "test_loci.nc"
        loci.ds.to_netcdf(file)

        ds = xr.open_dataset(file)
        loci2 = LOCI.from_dataset(ds)

        xr.testing.assert_equal(loci.ds, loci2.ds)

        p2 = loci2.adjust(sim)
        np.testing.assert_array_equal(p, p2)


@pytest.mark.slow
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


@pytest.mark.slow
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

        DQM = DetrendedQuantileMapping(
            kind=kind,
            group="time",
            nquantiles=50,
        )
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

        # PB 13-01-21 : This seems the same as the next test.
        # Test with sim not equal to hist
        # ff = series(np.ones(ns) * 1.1, name)
        # sim2 = apply_correction(sim, ff, kind)
        # ref2 = apply_correction(ref, ff, kind)

        # p2 = DQM.adjust(sim2, interp="linear")

        # np.testing.assert_array_almost_equal(p2[middle], ref2[middle], 1)

        # Test with actual trend in sim
        trend = series(
            np.linspace(-0.2, 0.2, ns) + (1 if kind == MULTIPLICATIVE else 0), name
        )
        sim3 = apply_correction(sim, trend, kind)
        ref3 = apply_correction(ref, trend, kind)
        p3 = DQM.adjust(sim3, interp="linear")
        np.testing.assert_array_almost_equal(p3[middle], ref3[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize("add_dims", [True, False])
    def test_mon_U(self, mon_series, series, kind, name, add_dims):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
        """
        n = 5000
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

        if add_dims:
            hist = hist.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            sim = sim.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            ref_t = ref_t.expand_dims(lat=[0, 1, 2])

        DQM = DetrendedQuantileMapping(kind=kind, group="time.month", nquantiles=5)
        DQM.train(ref, hist)
        mqm = DQM.ds.af.mean(dim="quantiles")
        p = DQM.adjust(sim)

        if add_dims:
            mqm = mqm.isel(lat=0)
        np.testing.assert_array_almost_equal(mqm, int(kind == MULTIPLICATIVE), 1)
        np.testing.assert_allclose(p, ref_t, rtol=0.1, atol=0.5)

    def test_cannon_and_from_ds(self, cannon_2015_rvs, tmp_path):
        ref, hist, sim = cannon_2015_rvs(15000)

        DQM = DetrendedQuantileMapping(kind="*", group="time")
        DQM.train(ref, hist)
        p = DQM.adjust(sim)

        np.testing.assert_almost_equal(p.mean(), 41.6, 0)
        np.testing.assert_almost_equal(p.std(), 15.0, 0)

        file = tmp_path / "test_dqm.nc"
        DQM.ds.to_netcdf(file)

        ds = xr.open_dataset(file)
        DQM2 = DetrendedQuantileMapping.from_dataset(ds)

        xr.testing.assert_equal(DQM.ds, DQM2.ds)

        p2 = DQM2.adjust(sim)
        np.testing.assert_array_equal(p, p2)


@pytest.mark.slow
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

        QDM = QuantileDeltaMapping(
            kind=kind,
            group="time",
            nquantiles=10,
        )
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
        p = QDM.adjust(sim, interp="linear" if kind == "+" else "nearest")

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(
            mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind
        )
        np.testing.assert_array_almost_equal(
            QDM.ds.af.sel(quantiles=q, **sel), expected, 1
        )

        # Test predict
        np.testing.assert_allclose(p.isel(**sel), ref, rtol=0.1, atol=0.2)
        # np.testing.assert_array_almost_equal(p.isel(**sel), ref, 1)

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


@pytest.mark.slow
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
        QM = EmpiricalQuantileMapping(
            kind=kind,
            group="time",
            nquantiles=50,
        )
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


class TestPrincipalComponents:
    @pytest.mark.parametrize(
        "group,crd_dims,pts_dims",
        (
            ["time", ["lat"], None],  # Lon as vectorizing dim
            ["time", None, None],  # Lon as second coord dims
            ["time", ["lat"], ["lon"]],  # Lon as a Points dim
            # Testing time grouping, vectorization on lon
            [Grouper("time.month"), ["lat"], None],
        ),
    )
    def test_simple(self, group, crd_dims, pts_dims):
        n = 15 * 365
        m = 2  # A dummy dimension to test vectorizing.
        ref_y = norm.rvs(loc=10, scale=1, size=(m, n))
        ref_x = norm.rvs(loc=3, scale=2, size=(m, n))
        sim_x = norm.rvs(loc=4, scale=2, size=(m, n))
        sim_y = sim_x + norm.rvs(loc=1, scale=1, size=(m, n))

        ref = xr.DataArray([ref_x, ref_y], dims=("lat", "lon", "time"))
        ref["time"] = xr.cftime_range("1990-01-01", periods=n, calendar="noleap")
        sim = xr.DataArray([sim_x, sim_y], dims=("lat", "lon", "time"))
        sim["time"] = ref["time"]

        PCA = PrincipalComponents(group=group, crd_dims=crd_dims, pts_dims=pts_dims)
        PCA.train(ref, sim)
        scen = PCA.adjust(sim)

        group = group if isinstance(group, Grouper) else Grouper("time")
        crds = crd_dims or ["lat", "lon"]
        pts = (pts_dims or []) + ["time"]

        vec = list({"lat", "lon"} - set(crds) - set(pts))
        refs = ref.stack(crd=crds)
        sims = sim.stack(crd=crds)
        scens = scen.stack(crd=crds)

        def _assert(ds):
            cov_ref = nancov(ds.ref.transpose("crd", "pt"))
            cov_sim = nancov(ds.sim.transpose("crd", "pt"))
            cov_scen = nancov(ds.scen.transpose("crd", "pt"))

            # PC adjustment makes the covariance of scen match the one of ref.
            np.testing.assert_allclose(cov_ref - cov_scen, 0, atol=1e-6)
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(cov_ref - cov_sim, 0, atol=1e-6)

        def _group_assert(ds, dim):
            ds = ds.stack(pt=pts)
            if len(vec) == 1:
                for v in ds[vec[0]]:
                    _assert(ds.sel({vec[0]: 0}))
            else:
                _assert(ds)
            return ds.unstack("pt")

        group.apply(_group_assert, {"ref": refs, "sim": sims, "scen": scens})

    @pytest.mark.parametrize(
        "group", [Grouper("time"), Grouper("time.month", window=11)]
    )
    def test_real_data(self, group):
        ds = xr.tutorial.open_dataset("air_temperature")

        ref = ds.air.isel(lat=21, lon=[40, 52]).drop_vars(["lon", "lat"])
        sim = ds.air.isel(lat=18, lon=[17, 35]).drop_vars(["lon", "lat"])

        PCA = PrincipalComponents(group=group)
        PCA.train(ref, sim)
        scen = PCA.adjust(sim)

        def dist(ref, sim):
            """Pointwise distance between ref and sim in the PC space."""
            return np.sqrt(((ref - sim) ** 2).sum("lon"))

        # Most points are closer after transform.
        assert (dist(ref, sim) < dist(ref, scen)).mean() < 0.05

        # "Error" is very small
        assert (ref - scen).mean() < 5e-3


def test_raise_on_multiple_chunks(tas_series):
    ref = tas_series(np.arange(730)).chunk({"time": 365})
    Adj = BaseAdjustment(group=Grouper("time.month"))
    with pytest.raises(ValueError):
        Adj.train(ref, ref)
