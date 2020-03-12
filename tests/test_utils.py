#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test for utils
import os

import numpy as np
import pytest
import xarray as xr
from data import RH_testdata

from xclim.core.utils import relative_humidity
from xclim.core.utils import sfcwind_2_uas_vas
from xclim.core.utils import uas_vas_2_sfcwind
from xclim.core.utils import walk_map

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")
K2C = 273.15


class TestWindConversion:
    da_uas = xr.DataArray(
        np.array([[3.6, -3.6], [-1, 0]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_uas.attrs["units"] = "km/h"
    da_vas = xr.DataArray(
        np.array([[3.6, 3.6], [-1, -18]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_vas.attrs["units"] = "km/h"
    da_wind = xr.DataArray(
        np.array([[np.hypot(3.6, 3.6), np.hypot(3.6, 3.6)], [np.hypot(1, 1), 18]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_wind.attrs["units"] = "km/h"
    da_windfromdir = xr.DataArray(
        np.array([[225, 135], [0, 360]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )

    def test_uas_vas_2_sfcwind(self):
        wind, windfromdir = uas_vas_2_sfcwind(self.da_uas, self.da_vas)

        assert np.all(
            np.around(wind.values, decimals=10)
            == np.around(self.da_wind.values / 3.6, decimals=10)
        )
        assert np.all(
            np.around(windfromdir.values, decimals=10)
            == np.around(self.da_windfromdir.values, decimals=10)
        )

    def test_sfcwind_2_uas_vas(self):
        uas, vas = sfcwind_2_uas_vas(self.da_wind, self.da_windfromdir)

        assert np.all(np.around(uas.values, decimals=10) == np.array([[1, -1], [0, 0]]))
        assert np.all(
            np.around(vas.values, decimals=10)
            == np.around(np.array([[1, 1], [-(np.hypot(1, 1)) / 3.6, -5]]), decimals=10)
        )


def test_walk_map():
    d = {"a": -1, "b": {"c": -2}}
    o = walk_map(d, lambda x: 0)
    assert o["a"] == 0
    assert o["b"]["c"] == 0


# TODO: Find a better data source to put more reasonable rtol and atol
def test_relative_humidity_dewpoint():
    xr.testing.assert_allclose(
        relative_humidity(tas=RH_testdata.tas, dtas=RH_testdata.dtas),
        RH_testdata.rh,
        0.1,
        2,
    )


@pytest.mark.parametrize("method", ["tetens30", "sonntag90", "goffgratch46", "wmo08"])
@pytest.mark.parametrize(
    "invalid_values,exp0", [("clip", 100), ("fill", np.nan), (None, 188)]
)
def test_relative_humidity(
    tas_series, rh_series, huss_series, ps_series, method, invalid_values, exp0
):
    tas = tas_series(np.array([-10, -10, 10, 20, 35, 50, 75, 95]) + 273.16)
    rh_exp = rh_series([exp0, 63, 66, 34, 14, 6, 1, 0])
    ps = ps_series([101325] * 8)
    huss = huss_series([0.003, 0.001] + [0.005] * 7)

    rh = relative_humidity(
        tas=tas,
        huss=huss,
        ps=ps,
        method=method,
        invalid_values=invalid_values,
        ice_thresh="0 degC",
    )

    np.testing.assert_allclose(rh, rh_exp, atol=1, rtol=0.01)
