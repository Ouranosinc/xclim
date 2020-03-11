#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test for utils
import os
from inspect import signature

import numpy as np
import pytest
import xarray as xr

from .data import RH_testdata
from xclim.core.utils import sfcwind_2_uas_vas
from xclim.core.utils import tas_dtas_2_rh
from xclim.core.utils import uas_vas_2_sfcwind
from xclim.core.utils import walk_map
from xclim.core.utils import wrapped_partial

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
def test_tas_dtas_2_rh():
    xr.testing.assert_allclose(
        tas_dtas_2_rh(RH_testdata.tas, RH_testdata.dtas), RH_testdata.rh, 0.1, 2
    )


def test_wrapped_partial():
    def func(a, b=1, c=1):
        """Docstring"""
        return (a, b, c)

    newf = wrapped_partial(func, b=2)
    assert list(signature(newf).parameters.keys()) == ["a", "c"]
    assert newf(1) == (1, 2, 1)

    newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    assert list(signature(newf).parameters.keys()) == ["a", "c"]
    assert newf(1) == (1, 2, 2)
    assert newf.__doc__ == func.__doc__
