from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.ensembles import (
    concat_hist,
    hawkins_sutton,
    model_in_all_scens,
    single_member,
)


def test_hawkins_sutton_smoke(open_dataset):
    """Just a smoke test - looking for data for a hard validation."""
    dims = {"run": "member", "scen": "scenario"}
    da = (
        open_dataset(
            "uncertainty_partitioning/cmip5_pr_global_mon.nc", branch="hawkins_sutton"
        )
        .pr.sel(time=slice("1950", None))
        .rename(dims)
    )
    da1 = model_in_all_scens(da)
    dac = concat_hist(da1, scenario="historical")
    das = single_member(dac)
    hawkins_sutton(das)


def test_hawkins_sutton_synthetic():
    """Create synthetic data to test logic of Hawkins-Sutton's implementation."""
    r = np.random.randn(4, 11, 60)
    # Time, scenario, model
    sm = np.arange(10, 41, 10)
    mm = np.arange(-5, 6, 1)
    mean = mm[np.newaxis, :] + sm[:, np.newaxis]

    # Here the scenarios don't change over time, so there should be no model variability (since it's relative to the
    # reference period.
    x = r + mean[:, :, np.newaxis]
    time = xr.date_range("1970-01-01", periods=60, freq="Y")
    da = xr.DataArray(x, dims=("scenario", "model", "time"), coords={"time": time})
    m, v = hawkins_sutton(da)

    np.testing.assert_array_almost_equal(m, 0, decimal=1)

    vm = v.mean(dim="time")

    np.testing.assert_array_almost_equal(vm.sel(uncertainty="scenario"), 0, decimal=2)
    np.testing.assert_array_almost_equal(
        vm.sel(uncertainty="variability"), 0, decimal=1
    )
    np.testing.assert_array_almost_equal(
        vm.sel(uncertainty="model"), vm.sel(uncertainty="variability"), decimal=1
    )
