from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim import sdba


def test_bias(open_dataset):
    sim = open_dataset("sdba/CanESM2_1950-2100.nc").sel(time="1950-01-01").tasmax
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time="1950-01-01").tasmax
    test = sdba.measures.bias(sim, ref).values
    np.testing.assert_array_almost_equal(test, [[6.430237, 39.088974, 5.2402344]])


def test_relative_bias(open_dataset):
    sim = open_dataset("sdba/CanESM2_1950-2100.nc").sel(time="1950-01-01").tasmax
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time="1950-01-01").tasmax
    test = sdba.measures.relative_bias(sim, ref).values
    np.testing.assert_array_almost_equal(test, [[0.02366494, 0.16392256, 0.01920133]])


def test_circular_bias():
    sim = xr.DataArray(
        data=np.array([1, 1, 1, 2, 365, 300]), attrs={"units": "", "long_name": "test"}
    )
    ref = xr.DataArray(
        data=np.array([2, 365, 300, 1, 1, 1]), attrs={"units": "", "long_name": "test"}
    )
    test = sdba.measures.circular_bias(sim, ref).values
    np.testing.assert_array_almost_equal(test, [1, 1, 66, -1, -1, -66])


def test_ratio(open_dataset):
    sim = open_dataset("sdba/CanESM2_1950-2100.nc").sel(time="1950-01-01").tasmax
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time="1950-01-01").tasmax
    test = sdba.measures.ratio(sim, ref).values
    np.testing.assert_array_almost_equal(test, [[1.023665, 1.1639225, 1.0192013]])


def test_rmse(open_dataset):
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc").sel(time=slice("1950", "1953")).tasmax
    )
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time=slice("1950", "1953")).tasmax
    test = sdba.measures.rmse(sim, ref).values
    np.testing.assert_array_almost_equal(test, [5.4499755, 18.124086, 12.387193], 4)


def test_mae(open_dataset):
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc").sel(time=slice("1950", "1953")).tasmax
    )
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time=slice("1950", "1953")).tasmax
    test = sdba.measures.mae(sim, ref).values
    np.testing.assert_array_almost_equal(test, [4.159672, 14.2148, 9.768536], 4)


def test_annual_cycle_correlation(open_dataset):
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc").sel(time=slice("1950", "1953")).tasmax
    )
    ref = open_dataset("sdba/nrcan_1950-2013.nc").sel(time=slice("1950", "1953")).tasmax
    test = (
        sdba.measures.annual_cycle_correlation(sim, ref, window=31)
        .sel(location="Vancouver")
        .values
    )
    np.testing.assert_array_almost_equal(test, [0.94580488], 4)


@pytest.mark.slow
def test_scorr(open_dataset):
    ref = open_dataset("NRCANdaily/nrcan_canada_daily_tasmin_1990.nc").tasmin
    sim = open_dataset("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc").tasmax
    scorr = sdba.measures.scorr(sim.isel(lon=slice(0, 50)), ref.isel(lon=slice(0, 50)))

    np.testing.assert_allclose(scorr, [97374.2146243])


def test_taylordiagram(open_dataset):
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1953"), location="Amos")
        .tasmax
    )
    ref = (
        open_dataset("sdba/nrcan_1950-2013.nc")
        .sel(time=slice("1950", "1953"), location="Amos")
        .tasmax
    )
    test = sdba.measures.taylordiagram(sim, ref).values
    np.testing.assert_array_almost_equal(test, [13.12244701, 6.76166582, 0.73230199], 4)
