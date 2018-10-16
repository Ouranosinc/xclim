#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Tests for `xclim` package.
#
# We want to tests multiple things here:
#  - that data results are correct
#  - that metadata is correct and complete
#  - that missing data are handled appropriately
#  - that various calendar formats and supported
#  - that non-valid input frequencies or holes in the time series are detected
#
#
# For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it,
# saving the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as
# a first line of defense.


# import cftime
import calendar
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim.indices as xci

xr.set_options(enable_cftimeindex=True)

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')
K2C = 273.15


class TestMaxNDayPrecipitationAmount:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'precipitation_flux',
                                   'cell_methods': 'time: sum (interval: 1 day)',
                                   'units': 'mm'})

    # test 2 day max precip
    def test_single_max(self):
        a = self.time_series(np.array([3, 4, 20, 20, 0, 6, 9, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, 2)
        assert rxnday == 40
        assert rxnday.time.dt.year == 2000

    # test whether sum over entire length is resolved
    def test_sumlength_max(self):
        a = self.time_series(np.array([3, 4, 20, 20, 0, 6, 9, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, len(a))
        assert rxnday == a.sum('time')
        assert rxnday.time.dt.year == 2000

    # test whether non-unique maxes are resolved
    def test_multi_max(self):
        a = self.time_series(np.array([3, 4, 20, 20, 0, 6, 15, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, 2)
        assert rxnday == 40
        assert len(rxnday) == 1
        assert rxnday.time.dt.year == 2000


class TestMax1DayPrecipitationAmount:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'precipitation_flux',
                                   'cell_methods': 'time: sum (interval: 1 day)',
                                   'units': 'mm'})

    # test max precip
    def test_single_max(self):
        a = self.time_series(np.array([3, 4, 20, 0, 0]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000

    # test whether repeated maxes are resolved
    def test_multi_max(self):
        a = self.time_series(np.array([20, 4, 20, 20, 0]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000
        assert len(rx1day) == 1

    # test whether uniform maxes are resolved
    def test_uniform_max(self):
        a = self.time_series(np.array([20, 20, 20, 20, 20]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000
        assert len(rx1day) == 1

    # test nan behavior
    def test_nan_max(self):
        from xclim.precip import R1Max

        a = self.time_series(np.array([20, np.nan, 20, 20, 0]))
        r1max = R1Max()
        rx1day = r1max(a)
        assert np.isnan(rx1day)


class TestConsecutiveFrostDays:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: minimum within days',
                                   'units': 'K'})

    def test_one_freeze_day(self):
        a = self.time_series(np.array([3, 4, 5, -1, 3]) + K2C)
        cfd = xci.consecutive_frost_days(a)
        assert cfd == 1
        assert cfd.time.dt.year == 2000

    def test_no_freeze(self):
        a = self.time_series(np.array([3, 4, 5, 1, 3]) + K2C)
        cfd = xci.consecutive_frost_days(a)
        assert cfd == 0

    @pytest.mark.skip("This is probably badly defined anyway...")
    def test_all_year_freeze(self):
        a = self.time_series(np.zeros(365) + K2C - 10)
        cfd = xci.consecutive_frost_days(a)
        assert cfd == 365


class TestCoolingDegreeDays:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: mean within days',
                                   'units': 'K'})

    def test_no_cdd(self):
        a = self.time_series(np.array([10, 15, -5, 18]) + K2C)
        cdd = xci.cooling_degree_days(a)
        assert cdd == 0

    def test_cdd(self):
        a = self.time_series(np.array([20, 25, -15, 19]) + K2C)
        cdd = xci.cooling_degree_days(a)
        assert cdd == 10

    def test_attrs(self):
        a = self.time_series(np.array([20, 25, -15, 19]) + K2C)
        cdd = xci.cooling_degree_days(a)
        assert cdd.standard_name == 'cooling_degree_days'
        assert cdd.long_name == 'cooling degree days'
        assert cdd.units == 'K*day'
        assert cdd.description


class TestPrcpTotal:
    # build test data for different calendar
    time_std = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    da_std = xr.DataArray(time_std.year, coords=[time_std], dims='time')

    # calendar 365_day and 360_day not tested for now since xarray.resample
    # does not support other calendars than standard
    #
    # units = 'days since 2000-01-01 00:00'
    # time_365 = cftime.num2date(np.arange(0, 10 * 365), units, '365_day')
    # time_360 = cftime.num2date(np.arange(0, 10 * 360), units, '360_day')
    # da_365 = xr.DataArray(np.arange(time_365.size), coords=[time_365], dims='time')
    # da_360 = xr.DataArray(np.arange(time_360.size), coords=[time_360], dims='time')

    def test_yearly(self):
        da_std = self.da_std
        out_std = xci.prcp_tot(da_std, units='mm')
        # l_years = np.unique(da_std.time.dt.year) TODO: Unused local variables are a PEP8 violation
        target = [(365 + calendar.isleap(y)) * y for y in np.unique(da_std.time.dt.year)]
        assert (np.allclose(target, out_std.values))


class TestTxMin:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})


class TestTxMean:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})

    def test_attrs(self):
        a = self.time_series(np.array([20, 21, 22, 23, 24]) + K2C)
        txm = xci.tx_mean(a, freq='YS')
        assert txm == 22 + K2C


class TestTxMax:

    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})

    def test_simple(self):
        a = self.time_series(np.array([20, 25, -15, 19]) + K2C)
        txm = xci.tx_max(a, freq='YS')
        assert txm == 25 + K2C


class TestTxMaxTxMinIndices:

    def tmax_tmin_time_series(self, values):
        coords = pd.date_range('1/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        tas_plus10 = xr.DataArray(values + 10, coords=[coords, ], dims='time',
                                  attrs={'standard_name': 'air_temperature',
                                         'cell_methods': 'time: maximum within days',
                                         'units': 'K'})
        tas_minus10 = xr.DataArray(values - 10, coords=[coords, ], dims='time',
                                   attrs={'standard_name': 'air_temperature',
                                          'cell_methods': 'time: minimum within days',
                                          'units': 'K'})
        return tas_plus10, tas_minus10

    def test_static_daily_temperature_range(self):
        temp_values = np.random.uniform(-30, 30, size=1095)
        tasmax, tasmin = self.tmax_tmin_time_series(temp_values + K2C)
        dtr = xci.daily_temperature_range(tasmax, tasmin, freq="YS")
        np.testing.assert_array_equal([dtr.mean()], [20])

    def test_all_freeze_thaw_cycles(self):
        temp_values = np.zeros(365)
        tasmax, tasmin = self.tmax_tmin_time_series(temp_values + K2C)
        ft = xci.daily_freezethaw_cycles(tasmax, tasmin, freq="YS")
        np.testing.assert_array_equal([np.sum(ft)], [365])

    def test_random_freeze_thaw_cycles(self):
        runs = np.array([])
        for i in range(10):
            temp_values = np.random.uniform(-30, 30, 365)
            tasmax, tasmin = self.tmax_tmin_time_series(temp_values + K2C)
            ft = xci.daily_freezethaw_cycles(tasmax, tasmin, freq="YS")
            runs = np.append(runs, ft)
        np.testing.assert_allclose(np.mean(runs), 120, atol=20)


# I'd like to parametrize some of these tests so we don't have to write individual tests for each indicator.
@pytest.mark.skip
class TestTG:
    def test_cmip3(self, cmip3_day_tas):  # This fails, xarray chokes on the time dimension. Unclear why.
        # rd = xci.TG(cmip3_day_tas)
        pass

    def compare_against_icclim(self, cmip3_day_tas):
        pass


@pytest.fixture(scope="session")
def cmip3_day_tas():
    # xr.set_options(enable_cftimeindex=False)
    ds = xr.open_dataset(os.path.join(TESTS_DATA, 'cmip3', 'tas.sresb1.giss_model_e_r.run1.atm.da.nc'))
    yield ds.tas
    ds.close()


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
