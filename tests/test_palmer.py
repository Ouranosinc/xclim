"""Tests for Palmer drought indices."""

import numpy as np
import pandas as pd
import xarray as xr

from xclim.compute import palmer_drought_indices


def test_palmer_smoke():
    """Verify Palmer computation runs successfully."""
    time = pd.date_range("2000-01-01", periods=120, freq="MS")

    pr = xr.DataArray(
        np.full(120, 50.0),
        coords={"time": time},
        dims=("time",),
        name="pr",
    )

    pet = xr.DataArray(
        np.full(120, 40.0),
        coords={"time": time},
        dims=("time",),
        name="pet",
    )

    awhc = xr.DataArray(
        150.0,
        name="awhc",
    )

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    assert isinstance(ds, xr.Dataset)
    assert "pdsi" in ds.data_vars


def test_palmer_outputs_present():

    time = pd.date_range("2000-01-01", periods=120, freq="MS")

    pr = xr.DataArray(
        np.full(120, 50.0),
        coords={"time": time},
        dims=("time",),
    )

    pet = xr.DataArray(
        np.full(120, 40.0),
        coords={"time": time},
        dims=("time",),
    )

    awhc = xr.DataArray(150.0)

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    expected = {
        "pdsi",
        "phdi",
        "pmdi",
        "scpdsi",
        "z",
    }

    assert expected.issubset(ds.data_vars)


def test_palmer_preserves_time_dimension():

    ntime = 120

    time = pd.date_range("2000-01-01", periods=ntime, freq="MS")

    pr = xr.DataArray(
        np.full(ntime, 50.0),
        coords={"time": time},
        dims=("time",),
    )

    pet = xr.DataArray(
        np.full(ntime, 40.0),
        coords={"time": time},
        dims=("time",),
    )

    awhc = xr.DataArray(150.0)

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    assert ds.sizes["time"] == ntime


def test_palmer_gridded_input():

    time = pd.date_range("2000-01-01", periods=60, freq="MS")

    pr = xr.DataArray(
        np.full((60, 3, 4), 50.0),
        coords={
            "time": time,
            "y": np.arange(3),
            "x": np.arange(4),
        },
        dims=("time", "y", "x"),
    )

    pet = xr.DataArray(
        np.full((60, 3, 4), 35.0),
        coords=pr.coords,
        dims=pr.dims,
    )

    awhc = xr.DataArray(
        np.full((3, 4), 150.0),
        coords={
            "y": np.arange(3),
            "x": np.arange(4),
        },
        dims=("y", "x"),
    )

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    assert ds["pdsi"].sizes["time"] == 60
    assert ds["pdsi"].sizes["y"] == 3
    assert ds["pdsi"].sizes["x"] == 4


def test_palmer_produces_finite_values():

    time = pd.date_range("2000-01-01", periods=120, freq="MS")

    # create a deterministic random-number generator
    rng = np.random.default_rng(42)

    pr = xr.DataArray(
        rng.uniform(20, 100, size=120),
        coords={"time": time},
        dims=("time",),
    )

    pet = xr.DataArray(
        rng.uniform(20, 80, size=120),
        coords={"time": time},
        dims=("time",),
    )

    awhc = xr.DataArray(150.0)

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    assert np.isfinite(ds["pdsi"]).any()


def test_pdsi_regression_wet_to_dry_transition():
    # Regression values generated using the current Palmer implementation
    # for a synthetic series transitioning from wet (50 mm/month)
    # to drought (10 mm/month) conditions.

    expected_pdsi = -4.342777742626958
    expected_z = -1.1403694898678955

    time = pd.date_range("2000-01-01", periods=120, freq="MS")

    pr = xr.DataArray(
        np.concatenate(
            [
                np.full(60, 50.0),  # wet period
                np.full(60, 10.0),  # drought period
            ]
        ),
        coords={"time": time},
        dims="time",
    )

    pet = xr.DataArray(
        np.full(120, 40.0),
        coords={"time": time},
        dims="time",
    )

    awhc = xr.DataArray(150.0)

    ds = palmer_drought_indices(
        pr=pr,
        pet=pet,
        awhc=awhc,
        monthly_inputs=True,
    )

    np.testing.assert_allclose(
        ds["pdsi"].isel(time=-1),
        expected_pdsi,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ds["z"].isel(time=-1),
        expected_z,
        rtol=1e-5,
    )
