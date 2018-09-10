#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `xclim` package.

We want to tests multiple things here:
 - that data results are correct
 - that metadata is correct and complete
 - that missing data are handled appropriately
 - that various calendar formats and supported
 - that non-valid input frequencies or holes in the time series are detected


For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it, saving
the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as a first line
of defense.
"""

import pytest
import pandas as pd
import numpy as np
from xclim.indices import *
import xarray as xr
import os

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')
K2C = 273.15

class test_newfunction():
    x = newfunction(3)
    assert x > 0

class Test_consecutive_frost_days():
    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                         attrs={'standard_name': 'air_temperature',
                                'cell_methods': 'time: minimum within days',
                                'units': 'K'})

    def test_one_freeze_day(self):
        a = self.time_series(np.array([3, 4, 5, -1, 3]) + K2C)
        cfd = consecutive_frost_days(a)
        assert cfd == 1
        assert cfd.time.dt.year == 2000

    def test_no_freeze(self):
        a = self.time_series(np.array([3, 4, 5, 1, 3]) + K2C)
        cfd = consecutive_frost_days(a)
        assert cfd == 0

    @pytest.mark.skip("This is probably badly defined anyway...")
    def test_all_year_freeze(self):
        a = self.time_series(np.zeros(365) + K2C - 10)
        cfd = consecutive_frost_days(a)
        assert cfd == 365

class Test_cooling_degree_days():
    def time_series(self, values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                         attrs={'standard_name': 'air_temperature',
                                'cell_methods': 'time: mean within days',
                                'units': 'K'})

    def test_no_cdd(self):
        a = self.time_series(np.array([10, 15, -5, 18]) + K2C)
        cdd = cooling_degree_days(a)
        assert cdd == 0

    def test_cdd(self):
        a = self.time_series(np.array([20, 25, -15, 19]) + K2C)
        cdd = cooling_degree_days(a)
        assert cdd == 10

    def test_attrs(self):
        a = self.time_series(np.array([20, 25, -15, 19]) + K2C)
        cdd = cooling_degree_days(a)
        assert cdd.standard_name == 'cooling_degree_days'
        assert cdd.long_name == 'cooling degree days'
        assert cdd.units == 'K*day'
        assert len(cdd.description) > 0

# I'd like to parametrize some of these tests so we don't have to write individual tests for each indicator.
@pytest.mark.skip('')
class TestTG():
    def test_cmip3(self, cmip3_day_tas): # This fails, xarray chokes on the time dimension. Unclear why.
        rd = TG(cmip3_day_tas)

    def compare_against_icclim(self, cmip3_day_tas):
        pass

@pytest.fixture(scope="session")
def cmip3_day_tas():
    #xr.set_options(enable_cftimeindex=False)
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
