from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import genpareto, norm, uniform

from xclim.core.options import set_options
from xclim.core.units import convert_units_to
from xclim.sdba import adjustment
from xclim.sdba.adjustment import (
    LOCI,
    DetrendedQuantileMapping,
    EmpiricalQuantileMapping,
    ExtremeValues,
    PrincipalComponents,
    QuantileDeltaMapping,
    Scaling,
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
from xclim.testing.sdba_utils import nancov  # noqa


class TestLoci:
    @pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
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

        loci = LOCI.train(ref_fit, hist, group=group, thresh=f"{thresh} kg m-2 s-1")
        np.testing.assert_array_almost_equal(loci.ds.hist_thresh, 1, dec)
        np.testing.assert_array_almost_equal(loci.ds.af, 2, dec)

        p = loci.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref, dec)

        assert "history" in p.attrs
        assert "Bias-adjusted with LOCI(" in p.attrs["history"]

        file = tmp_path / "test_loci.nc"
        loci.ds.to_netcdf(file)

        ds = xr.open_dataset(file)
        loci2 = LOCI.from_dataset(ds)

        xr.testing.assert_equal(loci.ds, loci2.ds)

        p2 = loci2.adjust(sim)
        np.testing.assert_array_equal(p, p2)

    def test_reduce_dims(self, ref_hist_sim_tuto):
        ref, hist, sim = ref_hist_sim_tuto()
        hist = hist.expand_dims(member=[0, 1])
        ref = ref.expand_dims(member=hist.member)
        LOCI.train(ref, hist, group="time", thresh="283 K", add_dims=["member"])


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
        if kind == ADDITIVE:
            ref = convert_units_to(ref, "degC")

        scaling = Scaling.train(ref, hist, group="time", kind=kind)
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
        scaling = Scaling.train(ref, hist, group="time.month", kind=kind)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(scaling.ds.af, expected)

        # Test predict
        p = scaling.adjust(sim)
        np.testing.assert_array_almost_equal(p, ref)

    def test_add_dim(self, series, mon_series):
        n = 10000
        u = np.random.rand(n, 4)

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
        np.testing.assert_array_almost_equal(
            DQM.ds.af[:, 2:-2], expected[np.newaxis, 2:-2], 1
        )

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
            ref = ref.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            hist = hist.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            sim = sim.expand_dims(lat=[0, 1, 2]).chunk({"lat": 1})
            ref_t = ref_t.expand_dims(lat=[0, 1, 2])

        DQM = DetrendedQuantileMapping.train(
            ref, hist, kind=kind, group="time.month", nquantiles=5
        )
        mqm = DQM.ds.af.mean(dim="quantiles")
        p = DQM.adjust(sim)

        if add_dims:
            mqm = mqm.isel(lat=0)
        np.testing.assert_array_almost_equal(mqm, int(kind == MULTIPLICATIVE), 1)
        np.testing.assert_allclose(p.transpose(..., "time"), ref_t, rtol=0.1, atol=0.5)

    def test_cannon_and_from_ds(self, cannon_2015_rvs, tmp_path):
        ref, hist, sim = cannon_2015_rvs(15000)

        DQM = DetrendedQuantileMapping.train(ref, hist, kind="*", group="time")
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

        QDM = QuantileDeltaMapping.train(
            ref, hist, kind=kind, group="time.month", nquantiles=40
        )
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
        np.testing.assert_allclose(p, ref.transpose(*p.dims), rtol=0.1, atol=0.2)

    def test_cannon_and_diagnostics(self, cannon_2015_dist, cannon_2015_rvs):
        ref, hist, sim = cannon_2015_rvs(15000, random=False)

        # Quantile mapping
        with set_options(sdba_extra_output=True):
            QDM = QuantileDeltaMapping.train(
                ref, hist, kind="*", group="time", nquantiles=50
            )
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

        QM = EmpiricalQuantileMapping.train(
            ref, hist, kind=kind, group="time.month", nquantiles=5
        )
        p = QM.adjust(sim)
        mqm = QM.ds.af.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        np.testing.assert_array_almost_equal(p, ref, 2)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_add_dims(self, use_dask, open_dataset):
        with set_options(sdba_encode_cf=use_dask):
            if use_dask:
                chunks = {"location": -1}
            else:
                chunks = None
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
            ref = ref.isel(location=1, drop=True).expand_dims(location=["Amos"])

            dsim = open_dataset(
                "sdba/CanESM2_1950-2100.nc",
                chunks=chunks,
                drop_variables=["lat", "lon"],
            ).tasmax
            hist = dsim.sel(time=slice("1981", "2010"))
            sim = dsim.sel(time=slice("2041", "2070"))

            # With add_dims, "does it run" test
            group = Grouper("time.dayofyear", window=5, add_dims=["location"])
            EQM = EmpiricalQuantileMapping.train(ref, hist, group=group)
            EQM.adjust(sim).load()

            # Without, sanity test.
            group = Grouper("time.dayofyear", window=5)
            EQM2 = EmpiricalQuantileMapping.train(ref, hist, group=group)
            scen2 = EQM2.adjust(sim).load()
            assert scen2.sel(location=["Kugluktuk", "Vancouver"]).isnull().all()


class TestPrincipalComponents:
    @pytest.mark.parametrize(
        "group", (Grouper("time.month"), Grouper("time", add_dims=["lon"]))
    )
    def test_simple(self, group):
        n = 15 * 365
        m = 2  # A dummy dimension to test vectorizing.
        ref_y = norm.rvs(loc=10, scale=1, size=(m, n))
        ref_x = norm.rvs(loc=3, scale=2, size=(m, n))
        sim_x = norm.rvs(loc=4, scale=2, size=(m, n))
        sim_y = sim_x + norm.rvs(loc=1, scale=1, size=(m, n))

        ref = xr.DataArray(
            [ref_x, ref_y], dims=("lat", "lon", "time"), attrs={"units": "degC"}
        )
        ref["time"] = xr.cftime_range("1990-01-01", periods=n, calendar="noleap")
        sim = xr.DataArray(
            [sim_x, sim_y], dims=("lat", "lon", "time"), attrs={"units": "degC"}
        )
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
        ref = stack_variables(
            xr.Dataset(
                {"tasmax": atmosds.tasmax, "tasmin": atmosds.tasmin, "tas": atmosds.tas}
            )
        ).isel(location=3)
        hist = stack_variables(
            xr.Dataset(
                {
                    "tasmax": 1.001 * atmosds.tasmax,
                    "tasmin": atmosds.tasmin - 0.25,
                    "tas": atmosds.tas + 1,
                }
            )
        ).isel(location=3)
        with xr.set_options(keep_attrs=True):
            sim = hist + 5
            sim["time"] = sim.time + np.timedelta64(10, "Y").astype("<m8[ns]")

        if use_dask:
            ref = ref.chunk()
            hist = hist.chunk()
            sim = sim.chunk()

        PCA = PrincipalComponents.train(
            ref, hist, crd_dim="multivar", best_orientation=pcorient
        )
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


class TestExtremeValues:
    @pytest.mark.parametrize(
        "c_thresh,q_thresh,frac,power",
        [
            ["1 mm/d", 0.95, 0.25, 1],
            ["1 mm/d", 0.90, 1e-6, 1],
            ["0.007 m/week", 0.95, 0.25, 2],
        ],
    )
    def test_simple(self, c_thresh, q_thresh, frac, power):
        n = 45 * 365

        def gen_testdata(c, s):
            base = np.clip(norm.rvs(loc=0, scale=s, size=(n,)), 0, None)
            qv = np.quantile(base[base > 1], q_thresh)
            base[base > qv] = genpareto.rvs(
                c, loc=qv, scale=s, size=base[base > qv].shape
            )
            return xr.DataArray(
                base,
                dims=("time",),
                coords={
                    "time": xr.cftime_range("1990-01-01", periods=n, calendar="noleap")
                },
                attrs={"units": "mm/day", "thresh": qv},
            )

        ref = jitter_under_thresh(gen_testdata(-0.1, 2), "1e-3 mm/d")
        hist = jitter_under_thresh(gen_testdata(-0.1, 2), "1e-3 mm/d")
        sim = gen_testdata(-0.15, 2.5)

        EQM = EmpiricalQuantileMapping.train(
            ref, hist, group="time.dayofyear", nquantiles=15, kind="*"
        )

        scen = EQM.adjust(sim)

        EX = ExtremeValues.train(ref, hist, cluster_thresh=c_thresh, q_thresh=q_thresh)
        qv = (ref.thresh + hist.thresh) / 2
        np.testing.assert_allclose(EX.ds.thresh, qv, atol=0.15, rtol=0.01)

        scen2 = EX.adjust(scen, sim, frac=frac, power=power)

        # What to test???
        # Test if extreme values of sim are still extreme
        exval = sim > EX.ds.thresh
        assert (scen2.where(exval) > EX.ds.thresh).sum() > (
            scen.where(exval) > EX.ds.thresh
        ).sum()

    @pytest.mark.slow
    def test_real_data(self, open_dataset):
        dsim = open_dataset("sdba/CanESM2_1950-2100.nc").chunk()
        dref = open_dataset("sdba/ahccd_1950-2013.nc").chunk()

        ref = convert_units_to(
            dref.sel(time=slice("1950", "2009")).pr, "mm/d", context="hydro"
        )
        hist = convert_units_to(
            dsim.sel(time=slice("1950", "2009")).pr, "mm/d", context="hydro"
        )

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
        "method", [m for m in dir(adjustment) if m.startswith("SBCK_")]
    )
    @pytest.mark.parametrize("use_dask", [True])  # do we gain testing both?
    def test_sbck(self, method, use_dask):
        SBCK = pytest.importorskip("SBCK", minversion="0.4.0")  # noqa

        n = 10 * 365
        m = 2  # A dummy dimension to test vectorization.
        ref_y = norm.rvs(loc=10, scale=1, size=(m, n))
        ref_x = norm.rvs(loc=3, scale=2, size=(m, n))
        hist_x = norm.rvs(loc=11, scale=1.2, size=(m, n))
        hist_y = norm.rvs(loc=4, scale=2.2, size=(m, n))
        sim_x = norm.rvs(loc=12, scale=2, size=(m, n))
        sim_y = norm.rvs(loc=3, scale=1.8, size=(m, n))

        ref = xr.Dataset(
            {
                "tasmin": xr.DataArray(
                    ref_x, dims=("lon", "time"), attrs={"units": "degC"}
                ),
                "tasmax": xr.DataArray(
                    ref_y, dims=("lon", "time"), attrs={"units": "degC"}
                ),
            }
        )
        ref["time"] = xr.cftime_range("1990-01-01", periods=n, calendar="noleap")

        hist = xr.Dataset(
            {
                "tasmin": xr.DataArray(
                    hist_x, dims=("lon", "time"), attrs={"units": "degC"}
                ),
                "tasmax": xr.DataArray(
                    hist_y, dims=("lon", "time"), attrs={"units": "degC"}
                ),
            }
        )
        hist["time"] = ref["time"]

        sim = xr.Dataset(
            {
                "tasmin": xr.DataArray(
                    sim_x, dims=("lon", "time"), attrs={"units": "degC"}
                ),
                "tasmax": xr.DataArray(
                    sim_y, dims=("lon", "time"), attrs={"units": "degC"}
                ),
            }
        )
        sim["time"] = xr.cftime_range("2090-01-01", periods=n, calendar="noleap")

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
