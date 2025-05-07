"""Test xsdba integration."""

# sdba may or may not be imported, which fails some QA tests
# pylint: disable=E0606

from __future__ import annotations

import importlib.util as _util

import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm, uniform

xsdba_installed = _util.find_spec("xsdba")
if xsdba_installed:
    from xclim import sdba


@pytest.mark.skipif(not xsdba_installed, reason="`xsdba` is not installed")
def test_simple(timeseries):
    ref = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True).tas

    sim = timeseries(
        np.concatenate([np.ones(365 * 2) * 2, np.ones(365) * 3]),
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    ).tas

    ADJ = sdba.EmpiricalQuantileMapping.train(ref=ref, hist=sim.sel(time=slice("2001", "2003")))
    ADJ.adjust(sim=sim)


@pytest.mark.skipif(xsdba_installed, reason="Import failure of `sdba` only tested if `xsdba` is not installed")
def test_import_failure():
    error_msg = "The `xclim.sdba` module has been split into its own package: `xsdba`"
    with pytest.raises(ImportError) as e:
        import xclim.sdba  # noqa
    assert error_msg in e.value.args[0]


@pytest.mark.skipif(not xsdba_installed, reason="`xsdba` is not installed")
class TestBaseAdjustment:
    def test_harmonize_units(self, tas_series, random):
        n = 10
        u = random.random(n)
        da = tas_series(u)
        da2 = da.copy()
        da2 = sdba.units.convert_units_to(da2, "degC")
        (da, da2), _ = sdba.BaseAdjustment._harmonize_units(da, da2)
        assert da.units == da2.units

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_harmonize_units_multivariate(self, tas_series, pr_series, random, use_dask):
        n = 10
        u = random.random(n)
        ds = xr.merge(
            [
                tas_series(u).to_dataset(name="tas"),
                pr_series(u * 100).to_dataset(name="pr"),
            ]
        )
        ds2 = ds.copy()
        ds2["tas"] = sdba.units.convert_units_to(ds2["tas"], "degC")
        ds2["pr"] = sdba.units.convert_units_to(ds2["pr"], "kg mm-2 s-1")
        da, da2 = sdba.stack_variables(ds), sdba.stack_variables(ds2)
        if use_dask:
            da, da2 = da.chunk({"multivar": 1}), da2.chunk({"multivar": 1})

        (da, da2), _ = sdba.BaseAdjustment._harmonize_units(da, da2)
        ds, ds2 = sdba.unstack_variables(da), sdba.unstack_variables(da2)
        assert (ds.tas.units == ds2.tas.units) & (ds.pr.units == ds2.pr.units)

    def test_matching_times(self, tas_series, random):
        n = 10
        u = random.random(n)
        da = tas_series(u, start="2000-01-01")
        da2 = tas_series(u, start="2010-01-01")
        with pytest.raises(
            ValueError,
            match="`ref` and `hist` have distinct time arrays, this is not supported for BaseAdjustment adjustment.",
        ):
            sdba.BaseAdjustment._check_matching_times(ref=da, hist=da2)

    def test_matching_time_sizes(self, tas_series, random):
        n = 10
        u = random.random(n)
        da = tas_series(u, start="2000-01-01")
        da2 = da.isel(time=slice(0, 5)).copy()
        with pytest.raises(
            ValueError,
            match="Inputs have different size for the time array, this is not supported for BaseAdjustment adjustment.",
        ):
            sdba.BaseAdjustment._check_matching_time_sizes(da, da2)


@pytest.mark.skipif(not xsdba_installed, reason="`xsdba` is not installed")
# This is an optional dependency in `xsdba`
class TestOTC:
    def test_compare_sbck(self, open_dataset):
        pytest.importorskip("ot")
        pytest.importorskip("SBCK", minversion="0.4.0")
        ds = open_dataset("sdba/ahccd_1950-2013.nc").isel(location=0)
        da = sdba.stack_variables(ds).sel(time=slice("1950", "1960"))
        sdba.dOTC.adjust(ref=da, hist=da, sim=da)


# Test values
@pytest.mark.slow
@pytest.mark.skipif(not xsdba_installed, reason="`xsdba` is not installed")
class TestQM:
    @pytest.mark.parametrize("kind,units", [("+", "K"), ("*", "kg m-2 s-1")])
    def test_quantiles(self, timeseries, kind, units, random):
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
        var = {"K": "tas", "kg m-2 s-1": "pr"}[units]
        hist = sim = timeseries(x, var)
        ref = timeseries(y, var)

        QM = sdba.EmpiricalQuantileMapping.train(
            ref,
            hist,
            kind=kind,
            group="time",
            nquantiles=50,
        )
        p = QM.adjust(sim, interp="linear")

        q = QM.ds.coords["quantiles"]
        expected = sdba.utils.get_correction(xd.ppf(q), yd.ppf(q), kind)[np.newaxis, :]
        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(QM.ds.af[:, 2:-2], expected[:, 2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        np.testing.assert_array_almost_equal(p[middle], ref[middle], 1)
