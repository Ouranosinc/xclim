import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim.solar as sx


@pytest.mark.parametrize("method,tol", [("astral", 30), ("pvlib", 5), ("internal", 180)])
def test_solar_noon(method, tol):
    # from https://gml.noaa.gov/grad/solcalc/
    approx = np.array(
        [
            pd.Timestamp("2026-07-23T13:16:12").to_numpy(),
            pd.Timestamp("2026-07-23T13:01:03").to_numpy(),
        ]
    )
    out = sx.solar_noon(
        ds=xr.Dataset(
            {},
            coords={
                "time": [pd.Timestamp("2026-07-23")],
                "lat": [37.77, 45.55],  # San Jose, Montréal
                "lon": [-122.42, -73.633],
            },
        ),
        method=method,
    )
    # output is in UTC, translate to timezone:
    tz = np.array([pd.Timedelta(-7, "h").to_numpy(), pd.Timedelta(-4, "h").to_numpy()])
    out = out + tz
    assert np.abs((out - approx).dt.total_seconds()).max() < tol


def test_solar_noon_all_close():
    time_ds = xr.Dataset(
        {},
        coords={
            "time": pd.date_range(start=pd.Timestamp.now(), periods=365, freq="D"),
            "lat": np.random.uniform(low=-90, high=90, size=1),
            # pvlib sometimes shifts along international date line, avoid those latitudes for comparison's sake
            "lon": np.random.uniform(low=-175, high=175, size=100),
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
