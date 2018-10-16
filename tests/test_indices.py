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


class Test_wet_days():
    # testing of wet_day and daily_intensity, both are related

    nc_file = 'testdata/NRCANdaily/nrcan_canada_daily_pr_1990.nc'

    def test_3d_data_with_nans(self):

        # test with 3d data
        pr = xr.open_dataset(self.nc_file).pr
        pr = pr * 86400.
        pr['units'] = 'mm'
        # put a nan somewhere
        pr.values[10, 1, 0] = np.nan

        # compute wet days with both skipna options
        pr_min = 5.
        wd = xci.wet_days(pr, pr_min=pr_min, freq='MS', skipna=False)
        wds = xci.wet_days(pr, pr_min=pr_min, freq='MS', skipna=True)

        # check some vector with and without a nan
        x1 = pr[:31, 0, 0].values
        x2 = pr[:31, 1, 0].values
        wd1 = ((x1 >= pr_min) * 1).sum()
        wd2 = ((x2 >= pr_min) * 1).sum()
        assert (wd1 == wd.values[0, 0, 0])
        assert (wd1 == wds.values[0, 0, 0])
        assert (np.isnan(wd.values[0, 1, 0]))
        assert (wd2 == wds.values[0, 1, 0])

        # make sure that vecotre with all nans gives nans wathever skipna
        assert (np.isnan(wd.values[0, -1, -1]))
        assert (np.isnan(wds.values[0, -1, -1]))

class Test_daily_intensity():
    # testing of wet_day and daily_intensity, both are related

    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily/nrcan_canada_daily_pr_1990.nc')

    def test_3d_data_with_nans(self):

        # test with 3d data
        pr = xr.open_dataset(self.nc_file).pr
        pr = pr * 86400.
        pr['units'] = 'mm'
        # put a nan somewhere
        pr.values[10, 1, 0] = np.nan

        # compute with both skipna options
        pr_min = 2.
        di = xci.daily_intensity(pr, pr_min=pr_min, freq='MS', skipna = False)
        dis = xci.daily_intensity(pr, pr_min=pr_min, freq='MS', skipna=True)

        x1 = pr[:31, 0, 0].values
        x2 = pr[:31, 1, 0].values
        x3 = pr[:31, -1, -1].values

        di1 = x1[x1 >= pr_min].mean()
        buffer = np.ma.masked_invalid(x2)
        di2 = buffer[buffer >= pr_min].mean()

        assert (np.allclose(di1, di.values[0, 0, 0]))
        assert (np.allclose(di1, dis.values[0, 0, 0]))
        assert (np.isnan(di.values[0, 1, 0]))
        assert (np.allclose(di2, dis.values[0, 1, 0]))
        assert (np.isnan(di.values[0, -1, -1]))
        assert (np.isnan(dis.values[0, -1, -1]))


class TestTxMin:
    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})


# I'd like to parametrize some of these tests so we don't have to write individual tests for each indicator.
@pytest.mark.skip('')
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
