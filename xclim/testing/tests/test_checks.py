from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import set_options
from xclim.core import cfchecks, datachecks
from xclim.core.utils import ValidationError
from xclim.indicators.atmos import tg_mean

K2C = 273.15


def setup_module(module):
    set_options(cf_compliance="raise", data_validation="raise")


def teardown_module(module):
    set_options(cf_compliance="warn", data_validation="raise")


TestObj = namedtuple("TestObj", ["test"])


@pytest.fixture(scope="module", params=[xr.cftime_range, pd.date_range])
def date_range(request):
    return request.param


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a string", "a string"),
        ("a long string", "a * string"),
        ("a string", ["not correct", "a string"]),
    ],
)
def test_check_valid_ok(value, expected):
    d = TestObj(value)
    cfchecks.check_valid(d, "test", expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "a string"),
        ("a long string", "a * strings"),
        ("a string", ["not correct", "also not correct"]),
    ],
)
def test_check_valid_raise(value, expected):
    d = TestObj(value)
    with pytest.raises(ValidationError):
        cfchecks.check_valid(d, "test", expected)


class TestDateHandling:
    tas_attrs = {
        "units": "K",
        "cell_methods": "time: mean within days",
        "standard_name": "air_temperature",
    }

    def test_assert_daily(self, date_range):
        n = 365  # one day short of a full year
        times = date_range("2000-01-01", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)], attrs=self.tas_attrs)
        tg_mean(da)

    # Bad frequency
    def test_bad_frequency(self, date_range):
        with pytest.raises(ValidationError):
            n = 365
            times = date_range("2000-01-01", freq="12H", periods=n)
            da = xr.DataArray(np.arange(n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)

    # Decreasing index
    def test_decreasing_index(self, date_range):
        with pytest.raises(ValidationError):
            n = 365
            times = date_range("2000-01-01", freq="12H", periods=n)
            da = xr.DataArray(
                np.arange(n), [("time", times[::-1])], attrs=self.tas_attrs
            )
            tg_mean(da)

    # Missing one day between the two years
    def test_missing_one_day_between_two_years(self, date_range):
        with pytest.raises(ValidationError):
            n = 365
            times = date_range("2000-01-01", freq="1D", periods=n)
            times = times.append(date_range("2001-01-01", freq="1D", periods=n))
            da = xr.DataArray(np.arange(2 * n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)

    # Duplicate dates
    def test_duplicate_dates(self, date_range):
        with pytest.raises(ValidationError):
            n = 365
            times = date_range("2000-01-01", freq="1D", periods=n)
            times = times.append(date_range("2000-12-29", freq="1D", periods=n))
            da = xr.DataArray(np.arange(2 * n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)


class TestDataCheck:
    def test_check_hourly(self, date_range):
        tas_attrs = {
            "units": "K",
            "standard_name": "air_temperature",
        }

        n = 100
        time = date_range("2000-01-01", freq="H", periods=n)
        da = xr.DataArray(np.random.rand(n), [("time", time)], attrs=tas_attrs)
        datachecks.check_freq(da, "H")

        time = date_range("2000-01-01", freq="3H", periods=n)
        da = xr.DataArray(np.random.rand(n), [("time", time)], attrs=tas_attrs)
        with pytest.raises(ValidationError):
            datachecks.check_freq(da, "H")

        datachecks.check_freq(da, "H", strict=False)
