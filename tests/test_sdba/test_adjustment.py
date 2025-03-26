# pylint: disable=no-member
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import genpareto, norm, uniform

from xclim.core.calendar import stack_periods
from xclim.core.options import set_options
from xclim.core.units import convert_units_to
from xclim.sdba import adjustment
from xclim.sdba.adjustment import (
    LOCI,
    OTC,
    BaseAdjustment,
    DetrendedQuantileMapping,
    EmpiricalQuantileMapping,
    ExtremeValues,
    MBCn,
    PrincipalComponents,
    QuantileDeltaMapping,
    Scaling,
    dOTC,
)
from xclim.sdba.base import Grouper
from xclim.sdba.processing import (
    jitter_under_thresh,
    stack_variables,
    uniform_noise_like,
    unstack_variables,
)
from xclim.sdba.utils import (
    ADDITIVE,
    MULTIPLICATIVE,
    apply_correction,
    get_correction,
    invert,
)


def nancov(X):
    """Numpy's cov but dropping observations with NaNs."""
    X_na = np.isnan(X).any(axis=0)
    return np.cov(X[:, ~X_na])


class TestBaseAdjustment:
    def test_harmonize_units(self, series, random):
        n = 10
        u = random.random(n)
        da = series(u, "tas")
        da2 = da.copy()
        da2 = convert_units_to(da2, "degC")
        (da, da2), _ = BaseAdjustment._harmonize_units(da, da2)
        assert da.units == da2.units

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_harmonize_units_multivariate(self, series, random, use_dask):
        n = 10
        u = random.random(n)
        ds = xr.merge([series(u, "tas"), series(u * 100, "pr")])
        ds2 = ds.copy()
        ds2["tas"] = convert_units_to(ds2["tas"], "degC")
        ds2["pr"] = convert_units_to(ds2["pr"], "nm/day")
        da, da2 = stack_variables(ds), stack_variables(ds2)
        if use_dask:
            da, da2 = da.chunk({"multivar": 1}), da2.chunk({"multivar": 1})

        (da, da2), _ = BaseAdjustment._harmonize_units(da, da2)
        ds, ds2 = unstack_variables(da), unstack_variables(da2)
        assert (ds.tas.units == ds2.tas.units) & (ds.pr.units == ds2.pr.units)

    def test_matching_times(self, series, random):
        n = 10
        u = random.random(n)
        da = series(u, "tas", start="2000-01-01")
        da2 = series(u, "tas", start="2010-01-01")
        with pytest.raises(
            ValueError,
            match="`ref` and `hist` have distinct time arrays, this is not supported for BaseAdjustment adjustment.",
        ):
            BaseAdjustment._check_matching_times(ref=da, hist=da2)

    def test_matching_time_sizes(self, series, random):
        n = 10
        u = random.random(n)
        da = series(u, "tas", start="2000-01-01")
        da2 = da.isel(time=slice(0, 5)).copy()
        with pytest.raises(
            ValueError,
            match="Inputs have different size for the time array, this is not supported for BaseAdjustment adjustment.",
        ):
            BaseAdjustment._check_matching_time_sizes(da, da2)


class TestLoci:
    @pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
    def test_time_and_from_ds(self, series, group, dec, tmp_path, random, open_dataset):
        n = 10000
        u = random.random(n)

        xd = uniform(loc=0, scale=3)
        x = xd.ppf(u)

        hist = sim = series(x, "pr")
        y = x * 2
        thresh = 2
        ref_fit = series(y, "pr").where(y > thresh, 0.1)
        ref = series(y, "pr")

        loci = LOCI.train(ref_fit, hist, group=group, thresh=f"{thresh} kg m-2 s-1")
        np.testing.assert_array_almost_equal(loci.ds.hist_thresh, 1, dec)
        np.testing.assert_array_almost_equal(loci.ds.af, 2, dec)

        p = loci.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref, dec)

        assert "history" in p.attrs
        assert "Bias-adjusted with LOCI(" in p.attrs["history"]

        file = tmp_path / "test_loci.nc"
        loci.ds.to_netcdf(file, engine="h5netcdf")

        ds = open_dataset(file)
        loci2 = LOCI.from_dataset(ds)

        xr.testing.assert_equal(loci.ds, loci2.ds)

        p2 = loci2.adjust(sim)
        np.testing.assert_array_equal(p, p2)

    @pytest.mark.requires_internet
    @pytest.mark.enable_socket
    def test_reduce_dims(self, ref_hist_sim_tuto):
        ref, hist, _sim = ref_hist_sim_tuto()
        hist = hist.expand_dims(member=[0, 1])
        ref = ref.expand_dims(member=hist.member)
        LOCI.train(ref, hist, group="time", thresh="283 K", add_dims=["member"])


@pytest.mark.slow
class TestScaling:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_time(self, kind, name, series, random):
        n = 10000
        u = random.random(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        hist = sim = series(x, name)
        ref = series(apply_correction(x, 2, kind), name)
        if kind == ADDITIVE:
            ref = convert_units_to(ref, "degC")

        scaling = Scaling.train(ref, hist, group="time", kind=kind)
        np.testing.assert_array_almost_equal(scaling.ds.af, 2)

        p = scaling.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name, random):
        n = 10000
        u = random.random(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        hist = sim = series(x, name)
        ref = mon_series(apply_correction(x, 2, kind), name)

        # Test train
        scaling = Scaling.train(ref, hist, group="time.month", kind=kind)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(scaling.ds.af, expected)

        # Test predict
        p = scaling.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref)

    def test_add_dim(self, series, mon_series, random):
        n = 10000
        u = random.random((n, 4))

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        hist = sim = series(x, "tas")
        ref = mon_series(apply_correction(x, 2, "+"), "tas")

        group = Grouper("time.month", add_dims=["lon"])

        scaling = Scaling.train(ref, hist, group=group, kind="+")
        assert "lon" not in scaling.ds
        p = scaling.adjust(sim)
        assert "lon" in p.dims
        np.testing.assert_array_almost_equal(p.transpose(*ref.dims), ref)


@pytest.mark.slow
class TestDQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name, random):
        """
        Train on
        hist: U
        ref: Normal

        Predict on hist to get ref
        """
        ns = 10000
        u = random.random(ns)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        hist = sim = series(x, name)
        ref = series(y, name)

        DQM = DetrendedQuantileMapping.train(
            ref,
            hist,
            kind=kind,
            group="time",
            nquantiles=50,
        )
        p = DQM.adjust(sim, interp="linear")

        q = DQM.ds.quantiles
        ex = apply_correction(xd.ppf(q), invert(xd.mean(), kind), kind)
        ey = apply_correction(yd.ppf(q), invert(yd.mean(), kind), kind)
        expected = get_correction(ex, ey, kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(DQM.ds.af[:, 2:-2], expected[np.newaxis, 2:-2], 1)

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
        trend = series(np.linspace(-0.2, 0.2, ns) + (1 if kind == MULTIPLICATIVE else 0), name)
        sim3 = apply_correction(sim, trend, kind)
        ref3 = apply_correction(ref, trend, kind)
        p3 = DQM.adjust(sim3, interp="linear")
        np.testing.assert_array_almost_equal(p3[middle], ref3[middle], 1)

    @pytest.mark.xfail(
        raises=ValueError,
        reason="This test sometimes fails due to a block/indexing error",
        strict=False,
    )
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize("add_dims", [True, False])
    def test_mon_U(self, mon_series, series, kind, name, add_dims, random):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
        """
        n = 5000
        u = random.random(n)

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
            ref = ref.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            hist = hist.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            sim = sim.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            ref_t = ref_t.expand_dims(lat=[0, 1, 2])

        DQM = DetrendedQuantileMapping.train(ref, hist, kind=kind, group="time.month", nquantiles=5)
        mqm = DQM.ds.af.mean(dim="quantiles")
        p = DQM.adjust(sim)

        if add_dims:
            mqm = mqm.isel(lat=0)
        np.testing.assert_array_almost_equal(mqm, int(kind == MULTIPLICATIVE), 1)
        # FIXME: This test sometimes fails due to a block/indexing error
        np.testing.assert_allclose(p.transpose(..., "time"), ref_t, rtol=0.1, atol=0.5)

    def test_cannon_and_from_ds(self, cannon_2015_rvs, tmp_path, open_dataset, random):
        ref, hist, sim = cannon_2015_rvs(15000, random=random)

        DQM = DetrendedQuantileMapping.train(ref, hist, kind="*", group="time")
        p = DQM.adjust(sim)

        np.testing.assert_almost_equal(p.mean(), 41.6, 0)
        np.testing.assert_almost_equal(p.std(), 15.0, 0)

        file = tmp_path / "test_dqm.nc"
        DQM.ds.to_netcdf(file, engine="h5netcdf")

        ds = open_dataset(file)
        DQM2 = DetrendedQuantileMapping.from_dataset(ds)

        xr.testing.assert_equal(DQM.ds, DQM2.ds)

        p2 = DQM2.adjust(sim)
        np.testing.assert_array_equal(p, p2)


@pytest.mark.slow
class TestQDM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name, random):
        """
        Train on
        x : U(1,1)
        y : U(1,2)

        """
        u = random.random(10000)

        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=4)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        hist = sim = series(x, name)
        ref = series(y, name)

        QDM = QuantileDeltaMapping.train(
            ref.astype("float32"),
            hist.astype("float32"),
            kind=kind,
            group="time",
            nquantiles=10,
        )
        p = QDM.adjust(sim.astype("float32"), interp="linear")

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)[np.newaxis, :]

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QDM.ds.af, expected, 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (u > 1e-2) * (u < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

        # Test dtype control of map_blocks
        assert QDM.ds.af.dtype == "float32"
        assert p.dtype == "float32"

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize("add_dims", [True, False])
    def test_mon_U(self, mon_series, series, mon_triangular, add_dims, kind, name, use_dask, random):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
        """
        u = random.random(10000)

        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=2)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        ref = mon_series(y, name)
        hist = sim = series(x, name)
        if use_dask:
            ref = ref.chunk({"time": -1})
            hist = hist.chunk({"time": -1})
            sim = sim.chunk({"time": -1})
        if add_dims:
            ref = ref.expand_dims(site=[0, 1, 2, 3, 4]).drop_vars("site")
            hist = hist.expand_dims(site=[0, 1, 2, 3, 4]).drop_vars("site")
            sim = sim.expand_dims(site=[0, 1, 2, 3, 4]).drop_vars("site")
            sel = {"site": 0}
        else:
            sel = {}

        QDM = QuantileDeltaMapping.train(ref, hist, kind=kind, group="time.month", nquantiles=40)
        p = QDM.adjust(sim, interp="linear" if kind == "+" else "nearest")

        q = QDM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind)
        np.testing.assert_array_almost_equal(QDM.ds.af.sel(quantiles=q, **sel), expected, 1)

        # Test predict
        np.testing.assert_allclose(p, ref.transpose(*p.dims), rtol=0.1, atol=0.2)

    def test_seasonal(self, series, random):
        u = random.random(10000)
        kind = "+"
        name = "tas"
        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=4)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        hist = sim = series(x, name)
        ref = series(y, name)

        QDM = QuantileDeltaMapping.train(
            ref.astype("float32"),
            hist.astype("float32"),
            kind=kind,
            group="time.season",
            nquantiles=10,
        )
        p = QDM.adjust(sim.astype("float32"), interp="linear")

        # Test predict
        # Accept discrepancies near extremes
        middle = (u > 1e-2) * (u < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

    def test_cannon_and_diagnostics(self, cannon_2015_dist, cannon_2015_rvs):
        ref, hist, sim = cannon_2015_rvs(15000, random=False)

        # Quantile mapping
        with set_options(sdba_extra_output=True):
            QDM = QuantileDeltaMapping.train(ref, hist, kind="*", group="time", nquantiles=50)
            scends = QDM.adjust(sim)

        assert isinstance(scends, xr.Dataset)

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
        bc_sim = scends.scen
        np.testing.assert_almost_equal(bc_sim.mean(), 41.5, 1)
        np.testing.assert_almost_equal(bc_sim.std(), 16.7, 0)


@pytest.mark.slow
class TestQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name, random):
        """
        Train on
        hist: U
        ref: Normal

        Predict on hist to get ref
        """
        u = random.random(10000)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        hist = sim = series(x, name)
        ref = series(y, name)

        QM = EmpiricalQuantileMapping.train(
            ref,
            hist,
            kind=kind,
            group="time",
            nquantiles=50,
        )
        p = QM.adjust(sim, interp="linear")

        q = QM.ds.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)[np.newaxis, :]
        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QM.ds.af[:, 2:-2], expected[:, 2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name, random):
        """
        Train on
        hist: U
        ref: U + monthly cycle

        Predict on hist to get ref
        """
        u = random.random(10000)

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

        QM = EmpiricalQuantileMapping.train(ref, hist, kind=kind, group="time.month", nquantiles=5)
        p = QM.adjust(sim)
        mqm = QM.ds.af.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, ref, 2)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_add_dims(self, use_dask, open_dataset):
        with set_options(sdba_encode_cf=use_dask):
            if use_dask:
                chunks = {"location": -1}
            else:
                chunks = None

            dsim = open_dataset(
                "sdba/CanESM2_1950-2100.nc",
                chunks=chunks,
                drop_variables=["lat", "lon"],
            ).tasmax
            hist = dsim.sel(time=slice("1981", "2010"))
            sim = dsim.sel(time=slice("2041", "2070"))

            ref = (
                open_dataset(
                    "sdba/ahccd_1950-2013.nc",
                    chunks=chunks,
                    drop_variables=["lat", "lon"],
                )
                .sel(time=slice("1981", "2010"))
                .tasmax
            )
            ref = convert_units_to(ref, "K")
            # The idea is to have ref defined only over 1 location
            # But sdba needs the same dimensions on ref and hist for Grouping with add_dims
            ref = ref.where(ref.location == "Amos")

            # With add_dims, "does it run" test
            group = Grouper("time.dayofyear", window=5, add_dims=["location"])
            EQM = EmpiricalQuantileMapping.train(ref, hist, group=group)
            EQM.adjust(sim).load()

            # Without, sanity test.
            group = Grouper("time.dayofyear", window=5)
            EQM2 = EmpiricalQuantileMapping.train(ref, hist, group=group)
            scen2 = EQM2.adjust(sim).load()
            assert scen2.sel(location=["Kugluktuk", "Vancouver"]).isnull().all()

    def test_different_times_training(self, series, random):
        n = 10
        u = random.random(n)
        ref = series(u, "tas", start="2000-01-01")
        u2 = random.random(n)
        hist = series(u2, "tas", start="2000-01-01")
        hist_fut = series(u2, "tas", start="2001-01-01")
        ds = EmpiricalQuantileMapping.train(ref, hist).ds
        EmpiricalQuantileMapping._allow_diff_training_times = True
        ds_fut = EmpiricalQuantileMapping.train(ref, hist_fut).ds
        EmpiricalQuantileMapping._allow_diff_training_times = False
        assert (ds.af == ds_fut.af).all()


@pytest.mark.slow
class TestMBCn:
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("group, window", [["time", 1], ["time.dayofyear", 31]])
    @pytest.mark.parametrize("period_dim", [None, "period"])
    def test_simple(self, open_dataset, use_dask, group, window, period_dim):
        group, window, period_dim, use_dask = "time", 1, None, False
        with set_options(sdba_encode_cf=use_dask):
            if use_dask:
                chunks = {"location": -1}
            else:
                chunks = None
            ref, dsim = (
                open_dataset(
                    f"sdba/{file}",
                    chunks=chunks,
                    drop_variables=["lat", "lon"],
                )
                .isel(location=1, drop=True)
                .expand_dims(location=["Amos"])
                for file in ["ahccd_1950-2013.nc", "CanESM2_1950-2100.nc"]
            )
            ref, hist = (ds.sel(time=slice("1981", "2010")).isel(time=slice(365 * 4)) for ds in [ref, dsim])
            dsim = dsim.sel(time=slice("1981", None))
            sim = (stack_periods(dsim).isel(period=slice(1, 2))).isel(time=slice(365 * 4))

            ref, hist, sim = (stack_variables(ds) for ds in [ref, hist, sim])

        MBCN = MBCn.train(
            ref,
            hist,
            base_kws=dict(nquantiles=50, group=Grouper(group, window)),
            adj_kws=dict(interp="linear"),
        )
        p = MBCN.adjust(sim=sim, ref=ref, hist=hist, period_dim=period_dim)
        # 'does it run' test
        p.load()


class TestPrincipalComponents:
    @pytest.mark.parametrize("group", (Grouper("time.month"), Grouper("time", add_dims=["lon"])))
    def test_simple(self, group, random):
        n = 15 * 365
        m = 2  # A dummy dimension to test vectorizing.
        ref_y = norm.rvs(loc=10, scale=1, size=(m, n), random_state=random)
        ref_x = norm.rvs(loc=3, scale=2, size=(m, n), random_state=random)
        sim_x = norm.rvs(loc=4, scale=2, size=(m, n), random_state=random)
        sim_y = sim_x + norm.rvs(loc=1, scale=1, size=(m, n), random_state=random)

        ref = xr.DataArray([ref_x, ref_y], dims=("lat", "lon", "time"), attrs={"units": "degC"})
        ref["time"] = xr.date_range("1990-01-01", periods=n, calendar="noleap")
        sim = xr.DataArray([sim_x, sim_y], dims=("lat", "lon", "time"), attrs={"units": "degC"})
        sim["time"] = ref["time"]

        PCA = PrincipalComponents.train(ref, sim, group=group, crd_dim="lat")
        scen = PCA.adjust(sim)

        def _assert(ds):
            cov_ref = nancov(ds.ref.transpose("lat", "pt"))
            cov_sim = nancov(ds.sim.transpose("lat", "pt"))
            cov_scen = nancov(ds.scen.transpose("lat", "pt"))

            # PC adjustment makes the covariance of scen match the one of ref.
            np.testing.assert_allclose(cov_ref - cov_scen, 0, atol=1e-6)
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(cov_ref - cov_sim, 0, atol=1e-6)

        def _group_assert(ds, dim):
            if "lon" not in dim:
                for lon in ds.lon:
                    _assert(ds.sel(lon=lon).stack(pt=dim))
            else:
                _assert(ds.stack(pt=dim))
            return ds

        group.apply(_group_assert, {"ref": ref, "sim": sim, "scen": scen})

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("pcorient", ["full", "simple"])
    def test_real_data(self, atmosds, use_dask, pcorient):
        ds0 = xr.Dataset({"tasmax": atmosds.tasmax, "tasmin": atmosds.tasmin, "tas": atmosds.tas})
        ref = stack_variables(ds0).isel(location=3)
        hist0 = ds0
        with xr.set_options(keep_attrs=True):
            hist0["tasmax"] = 1.001 * hist0.tasmax
            hist0["tasmin"] = hist0.tasmin - 0.25
            hist0["tas"] = hist0.tas + 1

        hist = stack_variables(hist0).isel(location=3)
        with xr.set_options(keep_attrs=True):
            sim = hist + 5
            sim["time"] = sim.time + np.timedelta64(10, "Y").astype("<m8[ns]")

        if use_dask:
            ref = ref.chunk()
            hist = hist.chunk()
            sim = sim.chunk()

        PCA = PrincipalComponents.train(ref, hist, crd_dim="multivar", best_orientation=pcorient)
        scen = PCA.adjust(sim)

        def dist(ref, sim):
            """Pointwise distance between ref and sim in the PC space."""
            sim["time"] = ref.time
            return np.sqrt(((ref - sim) ** 2).sum("multivar"))

        # Most points are closer after transform.
        assert (dist(ref, sim) < dist(ref, scen)).mean() < 0.01

        ref = unstack_variables(ref)
        scen = unstack_variables(scen)
        # "Error" is very small
        assert (ref - scen).mean().tasmin < 5e-3


# TODO: below we use `.adjust(scen,sim)`, but in the function signature, `sim` comes before
# are we testing the right thing below?
class TestExtremeValues:
    @pytest.mark.parametrize(
        "c_thresh,q_thresh,frac,power",
        [
            ["1 mm/d", 0.95, 0.25, 1],
            ["1 mm/d", 0.90, 1e-6, 1],
            ["0.007 m/week", 0.95, 0.25, 2],
        ],
    )
    def test_simple(self, c_thresh, q_thresh, frac, power, random):
        n = 45 * 365

        def gen_testdata(c, s):
            base = np.clip(norm.rvs(loc=0, scale=s, size=(n,), random_state=random), 0, None)
            qv = np.quantile(base[base > 1], q_thresh)
            base[base > qv] = genpareto.rvs(c, loc=qv, scale=s, size=base[base > qv].shape, random_state=random)
            return xr.DataArray(
                base,
                dims=("time",),
                coords={"time": xr.date_range("1990-01-01", periods=n, calendar="noleap")},
                attrs={"units": "mm/day", "thresh": qv},
            )

        ref = jitter_under_thresh(gen_testdata(-0.1, 2), "1e-3 mm/d")
        hist = jitter_under_thresh(gen_testdata(-0.1, 2), "1e-3 mm/d")
        sim = gen_testdata(-0.15, 2.5)

        EQM = EmpiricalQuantileMapping.train(ref, hist, group="time.dayofyear", nquantiles=15, kind="*")

        scen = EQM.adjust(sim)

        EX = ExtremeValues.train(ref, hist, cluster_thresh=c_thresh, q_thresh=q_thresh)
        qv = (ref.thresh + hist.thresh) / 2
        np.testing.assert_allclose(EX.ds.thresh, qv, atol=0.15, rtol=0.01)

        scen2 = EX.adjust(scen, sim, frac=frac, power=power)

        # What to test???
        # Test if extreme values of sim are still extreme
        exval = sim > EX.ds.thresh
        assert (scen2.where(exval) > EX.ds.thresh).sum() > (scen.where(exval) > EX.ds.thresh).sum()

    @pytest.mark.slow
    def test_real_data(self, open_dataset):
        dsim = open_dataset("sdba/CanESM2_1950-2100.nc").chunk()
        dref = open_dataset("sdba/ahccd_1950-2013.nc").chunk()

        ref = convert_units_to(dref.sel(time=slice("1950", "2009")).pr, "mm/d", context="hydro")
        hist = convert_units_to(dsim.sel(time=slice("1950", "2009")).pr, "mm/d", context="hydro")

        quantiles = np.linspace(0.01, 0.99, num=50)

        with xr.set_options(keep_attrs=True):
            ref = ref + uniform_noise_like(ref, low=1e-6, high=1e-3)
            hist = hist + uniform_noise_like(hist, low=1e-6, high=1e-3)

        EQM = EmpiricalQuantileMapping.train(
            ref, hist, group=Grouper("time.dayofyear", window=31), nquantiles=quantiles
        )

        scen = EQM.adjust(hist, interp="linear", extrapolation="constant")

        EX = ExtremeValues.train(ref, hist, cluster_thresh="1 mm/day", q_thresh=0.97)
        new_scen = EX.adjust(scen, hist, frac=0.000000001)
        new_scen.load()

    def test_nan_values(self):
        times = xr.date_range("1990-01-01", periods=365, calendar="noleap")
        ref = xr.DataArray(
            np.arange(365),
            dims=("time"),
            coords={"time": times},
            attrs={"units": "mm/day"},
        )
        hist = (ref.copy() * np.nan).assign_attrs(ref.attrs)
        EX = ExtremeValues.train(ref, hist, cluster_thresh="10 mm/day", q_thresh=0.9)
        new_scen = EX.adjust(sim=hist, scen=ref)
        assert new_scen.isnull().all()


class TestOTC:
    def test_compare_sbck(self, random, series):
        pytest.importorskip("ot")
        pytest.importorskip("SBCK", minversion="0.4.0")
        ns = 1000
        u = random.random(ns)

        ref_xd = uniform(loc=1000, scale=100)
        ref_yd = norm(loc=0, scale=100)
        hist_xd = norm(loc=-500, scale=100)
        hist_yd = uniform(loc=-1000, scale=100)

        ref_x = ref_xd.ppf(u)
        ref_y = ref_yd.ppf(u)
        hist_x = hist_xd.ppf(u)
        hist_y = hist_yd.ppf(u)

        # Constructing an histogram such that every bin contains
        # at most 1 point should ensure that ot is deterministic
        dx_ref = np.diff(np.sort(ref_x)).min()
        dx_hist = np.diff(np.sort(hist_x)).min()
        dx = min(dx_ref, dx_hist) * 9 / 10

        dy_ref = np.diff(np.sort(ref_y)).min()
        dy_hist = np.diff(np.sort(hist_y)).min()
        dy = min(dy_ref, dy_hist) * 9 / 10

        bin_width = [dx, dy]

        ref_tas = series(ref_x, "tas")
        ref_pr = series(ref_y, "pr")
        ref = xr.merge([ref_tas, ref_pr])
        ref = stack_variables(ref)

        hist_tas = series(hist_x, "tas")
        hist_pr = series(hist_y, "pr")
        hist = xr.merge([hist_tas, hist_pr])
        hist = stack_variables(hist)

        scen = OTC.adjust(ref, hist, bin_width=bin_width, jitter_inside_bins=False)

        otc_sbck = adjustment.SBCK_OTC
        scen_sbck = otc_sbck.adjust(ref, hist, hist, multi_dim="multivar", bin_width=bin_width)

        scen = scen.to_numpy().T
        scen_sbck = scen_sbck.to_numpy()
        assert np.allclose(scen, scen_sbck)


# TODO: Add tests for normalization methods
class TestdOTC:
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("cov_factor", ["std", "cholesky"])
    # FIXME: Should this comparison not fail if `standardization` != `None`?
    def test_compare_sbck(self, random, series, use_dask, cov_factor):
        pytest.importorskip("ot")
        pytest.importorskip("SBCK", minversion="0.4.0")
        ns = 1000
        u = random.random(ns)

        ref_xd = uniform(loc=1000, scale=100)
        ref_yd = norm(loc=0, scale=100)
        hist_xd = norm(loc=-500, scale=100)
        hist_yd = uniform(loc=-1000, scale=100)
        sim_xd = norm(loc=0, scale=100)
        sim_yd = uniform(loc=0, scale=100)

        ref_x = ref_xd.ppf(u)
        ref_y = ref_yd.ppf(u)
        hist_x = hist_xd.ppf(u)
        hist_y = hist_yd.ppf(u)
        sim_x = sim_xd.ppf(u)
        sim_y = sim_yd.ppf(u)

        # Constructing an histogram such that every bin contains
        # at most 1 point should ensure that ot is deterministic
        dx_ref = np.diff(np.sort(ref_x)).min()
        dx_hist = np.diff(np.sort(hist_x)).min()
        dx_sim = np.diff(np.sort(sim_x)).min()
        dx = min(dx_ref, dx_hist, dx_sim) * 9 / 10

        dy_ref = np.diff(np.sort(ref_y)).min()
        dy_hist = np.diff(np.sort(hist_y)).min()
        dy_sim = np.diff(np.sort(sim_y)).min()
        dy = min(dy_ref, dy_hist, dy_sim) * 9 / 10

        bin_width = [dx, dy]

        ref_tas = series(ref_x, "tas")
        ref_pr = series(ref_y, "pr")
        hist_tas = series(hist_x, "tas")
        hist_pr = series(hist_y, "pr")
        sim_tas = series(sim_x, "tas")
        sim_pr = series(sim_y, "pr")

        if use_dask:
            ref_tas = ref_tas.chunk({"time": -1})
            ref_pr = ref_pr.chunk({"time": -1})
            hist_tas = hist_tas.chunk({"time": -1})
            hist_pr = hist_pr.chunk({"time": -1})
            sim_tas = sim_tas.chunk({"time": -1})
            sim_pr = sim_pr.chunk({"time": -1})

        ref = xr.merge([ref_tas, ref_pr])
        hist = xr.merge([hist_tas, hist_pr])
        sim = xr.merge([sim_tas, sim_pr])

        ref = stack_variables(ref)
        hist = stack_variables(hist)
        sim = stack_variables(sim)

        scen = dOTC.adjust(
            ref,
            hist,
            sim,
            bin_width=bin_width,
            jitter_inside_bins=False,
            cov_factor=cov_factor,
        )

        dotc_sbck = adjustment.SBCK_dOTC
        scen_sbck = dotc_sbck.adjust(
            ref,
            hist,
            sim,
            multi_dim="multivar",
            bin_width=bin_width,
            cov_factor=cov_factor,
        )

        scen = scen.to_numpy().T
        scen_sbck = scen_sbck.to_numpy()
        assert np.allclose(scen, scen_sbck)

    # just check it runs
    def test_different_times(self, tasmax_series, tasmin_series):
        pytest.importorskip("ot")
        # `sim` has a different time than `ref,hist` (but same size)
        ref = xr.merge(
            [
                tasmax_series(np.arange(730).astype(float), start="2000-01-01").chunk({"time": -1}),
                tasmin_series(np.arange(730).astype(float), start="2000-01-01").chunk({"time": -1}),
            ]
        )
        hist = ref.copy()
        sim = xr.merge(
            [
                tasmax_series(np.arange(730).astype(float), start="2020-01-01").chunk({"time": -1}),
                tasmin_series(np.arange(730).astype(float), start="2020-01-01").chunk({"time": -1}),
            ]
        )
        ref, hist, sim = (stack_variables(arr) for arr in [ref, hist, sim])
        dOTC.adjust(ref, hist, sim)


def test_raise_on_multiple_chunks(tas_series):
    ref = tas_series(np.arange(730).astype(float)).chunk({"time": 365})
    with pytest.raises(ValueError):
        EmpiricalQuantileMapping.train(ref, ref, group=Grouper("time.month"))


def test_default_grouper_understood(tas_series):
    ref = tas_series(np.arange(730).astype(float))

    EQM = EmpiricalQuantileMapping.train(ref, ref)
    EQM.adjust(ref)
    assert EQM.group.dim == "time"


class TestSBCKutils:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method",
        [m for m in dir(adjustment) if m.startswith("SBCK_")],
    )
    @pytest.mark.parametrize("use_dask", [True])  # do we gain testing both?
    def test_sbck(self, method, use_dask, random):
        SBCK = pytest.importorskip("SBCK", minversion="0.4.0")  # noqa

        n = 10 * 365
        m = 2  # A dummy dimension to test vectorization.
        ref_y = norm.rvs(loc=10, scale=1, size=(m, n), random_state=random)
        ref_x = norm.rvs(loc=3, scale=2, size=(m, n), random_state=random)
        hist_x = norm.rvs(loc=11, scale=1.2, size=(m, n), random_state=random)
        hist_y = norm.rvs(loc=4, scale=2.2, size=(m, n), random_state=random)
        sim_x = norm.rvs(loc=12, scale=2, size=(m, n), random_state=random)
        sim_y = norm.rvs(loc=3, scale=1.8, size=(m, n), random_state=random)

        ref = xr.Dataset(
            {
                "tasmin": xr.DataArray(ref_x, dims=("lon", "time"), attrs={"units": "degC"}),
                "tasmax": xr.DataArray(ref_y, dims=("lon", "time"), attrs={"units": "degC"}),
            }
        )
        ref["time"] = xr.date_range("1990-01-01", periods=n, calendar="noleap")

        hist = xr.Dataset(
            {
                "tasmin": xr.DataArray(hist_x, dims=("lon", "time"), attrs={"units": "degC"}),
                "tasmax": xr.DataArray(hist_y, dims=("lon", "time"), attrs={"units": "degC"}),
            }
        )
        hist["time"] = ref["time"]

        sim = xr.Dataset(
            {
                "tasmin": xr.DataArray(sim_x, dims=("lon", "time"), attrs={"units": "degC"}),
                "tasmax": xr.DataArray(sim_y, dims=("lon", "time"), attrs={"units": "degC"}),
            }
        )
        sim["time"] = xr.date_range("2090-01-01", periods=n, calendar="noleap")

        if use_dask:
            ref = ref.chunk({"lon": 1})
            hist = hist.chunk({"lon": 1})
            sim = sim.chunk({"lon": 1})

        if "TSMBC" in method:
            kws = {"lag": 1}
        elif "MBCn" in method:
            kws = {"metric": SBCK.metrics.energy}
        else:
            kws = {}

        scen = getattr(adjustment, method).adjust(
            stack_variables(ref),
            stack_variables(hist),
            stack_variables(sim),
            multi_dim="multivar",
            **kws,
        )
        unstack_variables(scen).load()
