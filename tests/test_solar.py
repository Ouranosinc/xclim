import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim.solar as sx
from xclim.core.utils import sel_with_nans


@pytest.mark.parametrize(["method", "tol"], [("astral", 5), ("pvlib", 5), ("internal", 180)])
def test_solar_noon(method, tol):
    if method != "internal" and not importlib.util.find_spec(method):
        pytest.skip(f"{method} library is not installed")
    # from https://gml.noaa.gov/grad/solcalc/
    location = ["San Jose", "Montreal"]
    lat = [37.77, 45.55]
    lon = [-122.42, -73.633]
    utcoffset = np.array([pd.Timedelta(-7, "h").to_numpy(), pd.Timedelta(-4, "h").to_numpy()])
    noaa_noon = np.array(
        [
            pd.Timestamp("2026-07-23 13:16:12").to_numpy(),
            pd.Timestamp("2026-07-23 13:01:03").to_numpy(),
        ]
    )

    coords = xr.Dataset(
        dict(
            lat=("location", lat),
            lon=("location", lon),
            utcoffset=("location", utcoffset),
            noon=("location", noaa_noon),
        ),
        coords=dict(location=location),
    )

    din = xr.Dataset(
        {},
        coords={
            "time": [pd.Timestamp("2026-07-23")],
        },
    ).assign_coords(coords)

    out = sx.solar_noon(
        ds=din,
        method="astral",
    )
    # output is in UTC, translate to timezone:
    assert np.abs((out + coords.utcoffset - coords.noon).dt.total_seconds()).max().item() < tol


def test_solar_noon_all_close():
    if not importlib.util.find_spec("pvlib") or not importlib.util.find_spec("astral"):
        pytest.skip("astral and pvlib libraries are not installed")
    time_ds = xr.Dataset(
        {},
        coords={
            "time": pd.date_range(start=pd.Timestamp.now(), periods=30, freq="D"),
            "lat": np.random.uniform(low=-90, high=90, size=1),
            # pvlib sometimes shifts along international date line, avoid those latitudes for comparison's sake
            "lon": np.random.uniform(low=-175, high=175, size=30),
        },
    )
    time_ds["time"] = time_ds.time.dt.floor("D")
    out_astral = sx.solar_noon(ds=time_ds, method="astral")
    out_pvlib = sx.solar_noon(ds=time_ds, method="pvlib")
    out_internal = sx.solar_noon(ds=time_ds, method="internal")
    # ensure within 60 seconds of each other.
    max_diff_astral = np.abs((out_astral - out_pvlib).dt.total_seconds()).max()
    assert max_diff_astral < 60
    # ensure within 5 minutes of each other.
    max_diff_xclim = np.abs((out_internal - out_pvlib).dt.total_seconds()).max()
    assert max_diff_xclim < 300


@pytest.mark.parametrize("method", ["astral", "pvlib", "internal"])
@pytest.mark.parametrize("uses_dask", [True, False])
def test_interp(method, uses_dask):
    if method != "internal" and not importlib.util.find_spec(method):
        pytest.skip(f"{method} library is not installed")
    ds = xr.Dataset(
        {"tas": (("lon", "lat", "time"), np.broadcast_to(np.linspace(0, 1, 25), shape=(12, 1, 25)))},
        coords=dict(
            time=pd.date_range(start="2000-01-01", periods=25, freq="h"),
            lat=[0],
            lon=np.linspace(-175, 175, 12),
        ),
    )
    if uses_dask:
        ds = ds.chunk(time=-1, lat=1, lon=2)

    ds_solar = sx.interpolate_to_solar_noon(ds, solar_method=method).compute()
    # fraction of day in noon:
    noon_frac = (ds_solar.noon - ds_solar.time).dt.total_seconds() / (24 * 60 * 60)
    np.testing.assert_allclose(ds_solar.tas.isel(time=0, lat=0), noon_frac.isel(time=0))


@pytest.mark.parametrize("method", ["astral", "pvlib", "internal"])
@pytest.mark.parametrize("uses_dask", [True, False])
def test_accum(method, uses_dask):
    if method != "internal" and not importlib.util.find_spec(method):
        pytest.skip(f"{method} library is not installed")
    arr = np.linspace(0, 1, 11)
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.broadcast_to(arr, shape=(100, 1, 11)))},
        coords=dict(
            time=pd.date_range(start="2000-01-01", periods=100, freq="h"),
            lat=[0],
            lon=np.linspace(-175, 175, 11),
        ),
    )
    if uses_dask:
        ds = ds.chunk(time=-1, lat=1, lon=2)

    ds_solar = sx.interpolate_to_solar_noon(ds, solar_method=method, method="accumulate").compute()
    # length of day
    day_frac = (ds_solar.noon.isel(time=2) - ds_solar.noon.isel(time=1)).dt.total_seconds() / (24 * 60 * 60)

    np.testing.assert_allclose(
        ds_solar.tas.isel(time=2, lat=0),
        arr * 24 * day_frac,  # summed approximately 24 times, plus the day fraction.
    )


@pytest.mark.parametrize(["uses_dask", "lazy"], [(True, True), (True, False), (False, False)])
def test_sel_with_nans(uses_dask, lazy):
    tas = xr.DataArray(
        np.linspace(0, 1, 125).reshape((5, 5, 5)),
        coords={"time": np.arange(5), "lat": np.arange(5), "lon": np.arange(5)},
    )
    time = xr.DataArray([-1, 0, 1, 2, 3, 4, 5, 3, 2, 1, 0], dims=("time"))
    if uses_dask:
        tas = tas.chunk(time=1, lat=2, lon=3)
        time = time.chunk(time=2)

    tas_sel = sel_with_nans(tas, "time", time, fill=-1, lazy=lazy).compute()
    time = time.compute()
    assert (tas_sel.isel(time=[0, 6]) == -1).all()

    assert (tas.isel(time=[0, 1, 2, 3, 4, 3, 2, 1, 0]) == tas_sel.where(time.isin(tas.time), drop=True)).all()
