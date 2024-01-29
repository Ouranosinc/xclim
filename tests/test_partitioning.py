from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.ensembles import fractional_uncertainty, hawkins_sutton, lafferty_sriver
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
    time = xr.date_range("1970-01-01", periods=60, freq="YE")
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


def test_lafferty_sriver_synthetic(random):
    """Test logic of Lafferty & Sriver's implementation using synthetic data."""
    # Time, scenario, model, downscaling
    # Here the scenarios don't change over time, so there should be no model variability (since it's relative to the
    # reference period.
    sm = np.arange(10, 41, 10)  # Scenario mean (4)
    mm = np.arange(-6, 7, 1)  # Model mean (13)
    dm = np.arange(-2, 3, 1)  # Downscaling mean (5)
    mean = (
        dm[np.newaxis, np.newaxis, :]
        + mm[np.newaxis, :, np.newaxis]
        + sm[:, np.newaxis, np.newaxis]
    )

    # Natural variability
    r = random.standard_normal((4, 13, 5, 60))

    x = r + mean[:, :, :, np.newaxis]
    time = xr.date_range("1970-01-01", periods=60, freq="YE")
    da = xr.DataArray(
        x, dims=("scenario", "model", "downscaling", "time"), coords={"time": time}
    )
    m, v = lafferty_sriver(da)
    # Mean uncertainty over time
    vm = v.mean(dim="time")

    # Check that the mean uncertainty
    np.testing.assert_array_almost_equal(m.mean(dim="time"), 25, decimal=1)

    # Check that model uncertainty > variability
    assert vm.sel(uncertainty="model") > vm.sel(uncertainty="variability")

    # Smoke test with polynomial of order 2
    fit = da.polyfit(dim="time", deg=2, skipna=True)
    sm = xr.polyval(coord=da.time, coeffs=fit.polyfit_coefficients).where(da.notnull())
    lafferty_sriver(da, sm=sm)


def test_lafferty_sriver(lafferty_sriver_ds):
    g, u = lafferty_sriver(lafferty_sriver_ds.tas)

    fu = fractional_uncertainty(u)

    # Assertions based on expected results from
    # https://github.com/david0811/lafferty-sriver_2023_npjCliAtm/blob/main/unit_test/unit_test_check.ipynb
    assert fu.sel(time="2020", uncertainty="downscaling") > fu.sel(
        time="2020", uncertainty="model"
    )
    assert fu.sel(time="2020", uncertainty="variability") > fu.sel(
        time="2020", uncertainty="scenario"
    )
    assert (
        fu.sel(time="2090", uncertainty="scenario").data
        > fu.sel(time="2020", uncertainty="scenario").data
    )
    assert (
        fu.sel(time="2090", uncertainty="downscaling").data
        < fu.sel(time="2020", uncertainty="downscaling").data
    )

    def graph():
        """Return graphic like in https://github.com/david0811/lafferty-sriver_2023_npjCliAtm/blob/main/unit_test/unit_test_check.ipynb"""
        from matplotlib import pyplot as plt

        udict = {
            "Scenario": fu.sel(uncertainty="scenario").to_numpy().flatten(),
            "Model": fu.sel(uncertainty="model").to_numpy().flatten(),
            "Downscaling": fu.sel(uncertainty="downscaling").to_numpy().flatten(),
            "Variability": fu.sel(uncertainty="variability").to_numpy().flatten(),
        }

        fig, ax = plt.subplots()
        ax.stackplot(
            np.arange(2015, 2101),
            udict.values(),
            labels=udict.keys(),
            alpha=1,
            colors=["#00CC89", "#6869B3", "#CC883C", "#FFFF99"],
            edgecolor="white",
            lw=1.5,
        )
        ax.set_xlim([2020, 2095])
        ax.set_ylim([0, 100])
        ax.legend(loc="upper left")
        plt.show()

    # graph()
