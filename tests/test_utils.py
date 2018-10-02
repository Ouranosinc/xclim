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

import os
import cftime
import numpy as np
import pandas as pd
import xarray as xr

from xclim.utils import daily_downsampler
from xclim.indices import format_kwargs

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')


class Test_daily_downsampler():

    def test_std_calendar(self):

        # standard calendar
        # generate test DataArray
        time_std = pd.date_range('2000-01-01', '2000-12-31', freq='D')
        da_std = xr.DataArray(np.arange(time_std.size), coords=[time_std], dims='time')

        for freq in 'YS MS QS-DEC'.split():
            resampler = da_std.resample(time=freq)
            grouper = daily_downsampler(da_std, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')

            # assert the values of resampler and grouper are the same
            assert (np.allclose(x1.values, x2.values))

    def test_365_day(self):

        # 365_day calendar
        # generate test DataArray
        units = 'days since 2000-01-01 00:00'
        time_365 = cftime.num2date(np.arange(0, 1 * 365), units, '365_day')
        da_365 = xr.DataArray(np.arange(time_365.size), coords=[time_365], dims='time', name='data')
        units = 'days since 2001-01-01 00:00'
        time_std = cftime.num2date(np.arange(0, 1 * 365), units, 'standard')
        da_std = xr.DataArray(np.arange(time_std.size), coords=[time_std], dims='time', name='data')

        for freq in 'YS MS QS-DEC'.split():
            resampler = da_std.resample(time=freq)
            grouper = daily_downsampler(da_365, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_365.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')

            # assert the values of resampler of non leap year with standard calendar
            # is identical to grouper
            assert (np.allclose(x1.values, x2.values))

    def test_360_days(self):
        #
        # 360_day calendar
        #
        units = 'days since 2000-01-01 00:00'
        time_360 = cftime.num2date(np.arange(0, 360), units, '360_day')
        da_360 = xr.DataArray(np.arange(1, time_360.size + 1), coords=[time_360], dims='time', name='data')

        for freq in 'YS MS QS-DEC'.split():
            grouper = daily_downsampler(da_360, freq=freq)

            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_360.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')

            # assert grouper values == expected values
            target_year = 180.5
            target_month = [n * 30 + 15.5 for n in range(0, 12)]
            target_season = [30.5] + [(n - 1) * 30 + 15.5 for n in [4, 7, 10, 12]]
            target = {'YS': target_year, 'MS': target_month, 'QS-DEC': target_season}[freq]
            assert (np.allclose(x2.values, target))


def test_format_kwargs():
    attrs = dict(standard_name='tx_min', long_name='Minimum of daily maximum temperature',
                 cell_methods='time: minimum within {freq}')

    format_kwargs(attrs, {'freq': 'YS'})
    assert attrs['cell_methods'] == 'time: minimum within years'
