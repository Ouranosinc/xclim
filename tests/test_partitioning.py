from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.ensembles import hawkins_sutton
from xclim.ensembles._filters import _concat_hist, _model_in_all_scens, _single_member


def test_hawkins_sutton_smoke(open_dataset):
    """Just a smoke test."""
    dims = {"run": "member", "scen": "scenario"}
    da = (
        open_dataset("uncertainty_partitioning/cmip5_pr_global_mon.nc")
        .pr.sel(time=slice("1950", None))
        .rename(dims)
    )
    da1 = _model_in_all_scens(da)
    dac = _concat_hist(da1, scenario="historical")
    das = _single_member(dac)
    hawkins_sutton(das)


def test_hawkins_sutton_synthetic(random):
    """Test logic of Hawkins-Sutton's implementation using synthetic data."""
    # Time, scenario, model
    # Here the scenarios don't change over time, so there should be no model variability (since it's relative to the
    # reference period.
    sm = np.arange(10, 41, 10)  # Scenario mean
    mm = np.arange(-6, 7, 1)  # Model mean
    mean = mm[np.newaxis, :] + sm[:, np.newaxis]

    # Natural variability
    r = random.standard_normal((4, 13, 60))

    x = r + mean[:, :, np.newaxis]
    time = xr.date_range("1970-01-01", periods=60, freq="Y")
    da = xr.DataArray(x, dims=("scenario", "model", "time"), coords={"time": time})
    m, v = hawkins_sutton(da)
    # Mean uncertainty over time
    vm = v.mean(dim="time")

    # Check that the mean relative to the baseline is zero
    np.testing.assert_array_almost_equal(m.mean(dim="time"), 0, decimal=1)

    # Check that the scenario uncertainty is zero
    np.testing.assert_array_almost_equal(vm.sel(uncertainty="scenario"), 0, decimal=1)

    # Check that model uncertainty > variability
    assert vm.sel(uncertainty="model") > vm.sel(uncertainty="variability")

    # Smoke test with polynomial of order 2
    fit = da.polyfit(dim="time", deg=2, skipna=True)
    sm = xr.polyval(coord=da.time, coeffs=fit.polyfit_coefficients).where(da.notnull())
    hawkins_sutton(da, sm=sm)

    # Test with a multiplicative variable and time evolving scenarios
    r = random.standard_normal((4, 13, 60)) + np.arange(60)
    x = r + mean[:, :, np.newaxis]
    da = xr.DataArray(x, dims=("scenario", "model", "time"), coords={"time": time})
    m, v = hawkins_sutton(da, kind="*")
    su = v.sel(uncertainty="scenario")
    # We expect the scenario uncertainty to grow over time
    # The scenarios all have the same absolute slope, but since their reference mean is different, the relative increase
    # is not the same and this creates a spread over time across "relative" scenarios.
    assert (
        su.sel(time=slice("2020", None)).mean()
        > su.sel(time=slice("2000", "2010")).mean()
    )
