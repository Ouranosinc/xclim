from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core.calendar import date_range
from xclim.core.units import convert_units_to
from xclim.indices import helpers


@pytest.mark.parametrize("method,rtol", [("spencer", 5e3), ("simple", 1e2)])
def test_solar_declinaton(method, rtol):
    # Expected values from https://gml.noaa.gov/grad/solcalc/azel.html
    times = xr.DataArray(
        pd.to_datetime(
            ["1793-01-21T10:22:00", "1969-07-20T20:17:40", "2022-05-20T16:55:48"]
        ),
        dims=("time",),
    )
    exp = [-19.83, 20.64, 20.00]
    np.testing.assert_allclose(
        helpers.solar_declination(times, method=method),
        np.deg2rad(exp),
        atol=rtol * 2 * np.deg2rad(23.44),  # % of the possible range
    )


@pytest.mark.parametrize("method", ["spencer", "simple"])
def test_extraterrestrial_radiation(method):
    # Expected values from https://www.engr.scu.edu/~emaurer/tools/calc_solar_cgi.pl
    # This source is not authoritative, thus the large rtol
    times = xr.DataArray(
        xr.date_range("1900-01-01", "1900-01-03", freq="D"),
        dims=("time",),
        name="time",
    )
    lat = xr.DataArray(
        [48.8656, 29.5519, -54],
        dims=("time",),
        coords={"time": times},
        attrs={"units": "degree_north"},
    )
    exp = [99.06, 239.98, 520.01]
    np.testing.assert_allclose(
        convert_units_to(
            helpers.extraterrestrial_solar_radiation(times, lat, method=method), "W m-2"
        ),
        exp,
        rtol=3e-2,
    )


@pytest.mark.parametrize("method", ["spencer", "simple"])
def test_day_lengths(method):
    time_data = date_range("1992-12-01", "1994-01-01", freq="D", calendar="standard")
    data = xr.DataArray(
        np.ones((time_data.size, 7)),
        dims=("time", "lat"),
        coords={"time": time_data, "lat": [-60, -45, -30, 0, 30, 45, 60]},
    )
    data.lat.attrs["units"] = "degree_north"

    dl = helpers.day_lengths(dates=data.time, lat=data.lat, method=method)

    events = dict(
        solstice=[
            ["1992-12-21", [[18.49, 15.43, 13.93, 12.0, 10.07, 8.57, 5.51]]],
            ["1993-06-21", [[5.51, 8.57, 10.07, 12.0, 13.93, 15.43, 18.49]]],
            ["1993-12-21", [[18.49, 15.43, 13.93, 12.0, 10.07, 8.57, 5.51]]],
        ],
        equinox=[
            ["1993-03-20", [[12] * 7]]
        ],  # True equinox on 1993-03-20 at 14:41 GMT. Some relative tolerance is needed.
    )

    for event, evaluations in events.items():
        for e in evaluations:
            if event == "solstice":
                np.testing.assert_array_almost_equal(
                    dl.sel(time=e[0]).transpose(), np.array(e[1]), 2
                )
            elif event == "equinox":
                np.testing.assert_allclose(
                    dl.sel(time=e[0]).transpose(), np.array(e[1]), rtol=2e-1
                )


def test_cosine_of_solar_zenith_angle():
    time = xr.date_range("1900-01-01T00:30", "1900-01-03", freq="H")
    time = xr.DataArray(time, dims=("time",), coords={"time": time}, name="time")
    lat = xr.DataArray(
        [0, 45, 70], dims=("site",), name="lat", attrs={"units": "degree_north"}
    )
    lon = xr.DataArray(
        [-40, 0, 80], dims=("site",), name="lon", attrs={"units": "degree_east"}
    )
    dec = helpers.solar_declination(time)

    czda = helpers.cosine_of_solar_zenith_angle(
        time, dec, lat, lon, stat="average", sunlit=True
    )
    # Data Generated with PyWGBT
    # raw = coszda(
    #     (time + pd.Timedelta('30 m')).data,
    #     convert_units_to(lat, 'rad').data[np.newaxis, :],
    #     convert_units_to(lon, 'rad').data[np.newaxis, :],
    #     1
    # )
    # exp_cza = xr.DataArray(raw, dims=('time', 'd', 'site'), coords={'lat': lat, 'lon': lon, 'time': time}).squeeze('d')
    exp_czda = np.array(
        [
            [0.0, 0.0610457, 0.0],
            [0.09999178, 0.18221077, 0.0],
            [0.31387116, 0.285383, 0.0],
            [0.52638271, 0.35026199, 0.0],
            [0.70303168, 0.37242693, 0.0],
        ]
    )
    np.testing.assert_allclose(czda[7:12, :], exp_czda, rtol=1e-3)

    # Same code as above, but with function "cosza".
    cza = helpers.cosine_of_solar_zenith_angle(
        time, dec, lat, lon, stat="average", sunlit=False
    )
    exp_cza = np.array(
        [
            [-0.83153798, -0.90358335, -0.34065474],
            [-0.90358299, -0.83874813, -0.26062708],
            [-0.91405234, -0.73561867, -0.18790995],
            [-0.86222963, -0.60121893, -0.12745608],
        ]
    )
    np.testing.assert_allclose(cza[:4, :], exp_cza, rtol=1e-3)
