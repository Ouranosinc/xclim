import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from xarray.coding.cftimeindex import CFTimeIndex

from xclim.core.calendar import adjust_doy_calendar
from xclim.core.calendar import infer_doy_max
from xclim.core.calendar import percentile_doy
from xclim.core.calendar import time_bnds


TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


@pytest.fixture(
    params=[dict(start="2004-01-01T12:07:01", periods=27, freq="3MS")], ids=["3MS"]
)
def time_range_kwargs(request):
    return request.param


@pytest.fixture()
def datetime_index(time_range_kwargs):
    return pd.date_range(**time_range_kwargs)


@pytest.fixture()
def cftime_index(time_range_kwargs):
    return xr.cftime_range(**time_range_kwargs)


def da(index):
    return xr.DataArray(
        np.arange(100.0, 100.0 + index.size), coords=[index], dims=["time"]
    )


@pytest.mark.parametrize(
    "freq", ["3A-MAY", "5Q-JUN", "7M", "6480H", "302431T", "23144781S"]
)
def test_time_bnds(freq, datetime_index, cftime_index):
    da_datetime = da(datetime_index).resample(time=freq)
    da_cftime = da(cftime_index).resample(time=freq)

    cftime_bounds = time_bnds(da_cftime, freq=freq)
    cftime_starts, cftime_ends = zip(*cftime_bounds)
    cftime_starts = CFTimeIndex(cftime_starts).to_datetimeindex()
    cftime_ends = CFTimeIndex(cftime_ends).to_datetimeindex()

    # cftime resolution goes down to microsecond only, code below corrects
    # that to allow for comparison with pandas datetime
    cftime_ends += np.timedelta64(999, "ns")
    datetime_starts = da_datetime._full_index.to_period(freq).start_time
    datetime_ends = da_datetime._full_index.to_period(freq).end_time

    assert_array_equal(cftime_starts, datetime_starts)
    assert_array_equal(cftime_ends, datetime_ends)


def test_percentile_doye(tas_series):
    tas = tas_series(np.arange(365), start="1/1/2001")
    tas = xr.concat((tas, tas), "dim0")
    p1 = percentile_doy(tas, window=5, per=0.5)
    assert p1.sel(dayofyear=3, dim0=0).data == 2
    assert p1.attrs["units"] == "K"


class TestAdjustDoyCalendar:
    def test_360_to_366(self):
        source = xr.DataArray(
            np.arange(360), coords=[np.arange(1, 361)], dims="dayofyear"
        )
        time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")

        out = adjust_doy_calendar(source, target)

        assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
        assert out.sel(dayofyear=366) == source.sel(dayofyear=360)

    def test_infer_doy_max(self):
        fn = os.path.join(
            TESTS_DATA,
            "CanESM2_365day",
            "pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        )
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 365

        fn = os.path.join(
            TESTS_DATA,
            "HadGEM2-CC_360day",
            "pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        )
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 360

        fn = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 366
