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
import xclim
import xarray as xr
import os

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')

@pytest.fixture(scope="session")
def cmip3_day_tas():
    #xr.set_options(enable_cftimeindex=False)
    ds = xr.open_dataset(os.path.join(TESTS_DATA, 'cmip3', 'tas.sresb1.giss_model_e_r.run1.atm.da.nc'))
    yield ds.tas
    ds.close()


# I'd like to parametrize some of these tests so we don't have to write individual tests for each indicator.
class TestTG():
    def test_cmip3(self, cmip3_day_tas): # This fails, xarray chokes on the time dimension. Unclear why.
        rd = xclim.TG(cmip3_day_tas)

    def compare_against_icclim(self, cmip3_day_tas):
        pass

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
