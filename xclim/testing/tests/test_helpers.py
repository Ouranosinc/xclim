from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core.calendar import date_range, datetime_to_decimal_year
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

    day_angle = ((datetime_to_decimal_year(times) % 1) * 360).assign_attrs(
        units="degree"
    )
    np.testing.assert_allclose(
        helpers.solar_declination(day_angle, method=method),
        np.deg2rad(exp),
        atol=rtol * 2 * np.deg2rad(23.44),  # % of the possible range
    )


@pytest.mark.parametrize("method", ["spencer", "simple"])
def test_extraterrestrial_radiation(method):
    # Expected values from https://www.engr.scu.edu/~emaurer/tools/calc_solar_cgi.pl
    # This source is not authoritative, thus the large rtol
    times = xr.DataArray(
        pd.to_datetime(
            ["1793-01-21T10:22:00", "1969-07-20T20:17:40", "2022-05-20T16:55:48"]
        ),
        dims=("time",),
        name="time",
    )
    lat = xr.DataArray(
        [48.8656, 29.5519, 45.5435],
        dims=("time",),
        coords={"time": times},
        attrs={"units": "degree_north"},
    )
    exp = [120.90, 477.51, 470.74]
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
