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


import os
import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import utils
from xclim.utils import daily_downsampler, Indicator, format_kwargs, parse_doc, walk_map, adjust_doy_calendar
from xclim.utils import units
from xclim.testing.common import tas_series, pr_series
from xclim import indices as ind

TAS_SERIES = tas_series
PR_SERIES = pr_series
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')


class TestDailyDownsampler:

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

    @pytest.mark.skip
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


class UniIndTemp(Indicator):
    identifier = 'tmin{thresh}'
    units = 'K'
    required_units = 'K'
    long_name = '{freq} mean surface temperature'
    standard_name = '{freq} mean temperature'
    cell_methods = 'time: mean within {freq}'

    @staticmethod
    def compute(da, thresh=0., freq='YS'):
        """Docstring"""
        return da.resample(time=freq).mean() - thresh


class UniIndPr(Indicator):
    identifier = 'prmax'
    units = 'kg m-2 s-1'
    required_units = 'kg m-2 s-1'
    context = 'hydro'

    @staticmethod
    def compute(da, freq):
        """Docstring"""
        return da.resample(time=freq).mean()


class TestIndicator:

    def test_attrs(self, tas_series):
        import datetime as dt
        a = tas_series(np.arange(360))
        ind = UniIndTemp()
        txm = ind(a, freq='YS')
        assert txm.cell_methods == 'time: mean within days time: mean within years'
        assert '{:%Y-%m-%d %H}'.format(dt.datetime.now()) in txm.attrs['history']
        assert txm.name == "tmin0"

    def test_temp_unit_conversion(self, tas_series):
        a = tas_series(np.arange(360))
        ind = UniIndTemp()
        txk = ind(a, freq='YS')

        ind.required_units = ('degC',)
        ind.units = 'degC'
        txc = ind(a, freq='YS')

        np.testing.assert_array_almost_equal(txk, txc+273.15)

    def test_pr_unit_conversion(self, pr_series):
        a = pr_series(np.arange(360))
        ind = UniIndPr()
        txk = ind(a, freq='YS')

        ind.required_units = ('mm/day',)
        ind.units = 'mm'
        txm = ind(a, freq='YS')

        np.testing.assert_array_almost_equal(txk, txm/86400)

    def test_json(self, pr_series):
        ind = UniIndPr()
        meta = ind.json()

        expected = {'identifier', 'units', 'long_name', 'standard_name', 'cell_methods', 'keywords', 'abstract',
                    'parameters', 'description', 'history', 'references', 'comment', 'notes'}

        assert set(meta.keys()).issubset(expected)

    def test_factory(self, pr_series):
        attrs = dict(identifier='test', units='days', required_units='mm/day', long_name='long name',
                     standard_name='standard name', context='hydro'
                     )
        cls = Indicator.factory(attrs)

        assert issubclass(cls, Indicator)
        da = pr_series(np.arange(365))
        cls(compute=ind.wetdays)(da)

    def test_signature(self):
        from inspect2 import signature
        ind = UniIndTemp()
        assert signature(ind.compute) == signature(ind.__call__)

    def test_doc(self):
        ind = UniIndTemp()
        assert ind.__call__.__doc__ == ind.compute.__doc__


class TestKwargs:

    def test_format_kwargs(self):
        attrs = dict(standard_name='tx_min', long_name='Minimum of daily maximum temperature',
                     cell_methods='time: minimum within {freq}')

        format_kwargs(attrs, {'freq': 'YS'})
        assert attrs['cell_methods'] == 'time: minimum within years'


class TestParseDoc:

    def test_simple(self):
        parse_doc(ind.tg_mean.__doc__)


class TestAdjustDoyCalendar:

    def test_360_to_366(self):
        source = xr.DataArray(np.arange(360), coords=[np.arange(1, 361), ], dims='dayofyear')
        time = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        target = xr.DataArray(np.arange(len(time)), coords=[time, ], dims='time')
        out = adjust_doy_calendar(source, target)

        assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
        assert out.sel(dayofyear=366) == source.sel(dayofyear=360)


class TestWalkMap:

    def test_simple(self):
        d = {'a': -1, 'b': {'c': -2}}
        o = walk_map(d, lambda x: 0)
        assert o['a'] == 0
        assert o['b']['c'] == 0


class TestUnits:

    def test_temperature(self):
        assert 4 * units.d == 4 * units.day
        Q_ = units.Quantity
        assert Q_(1, units.C) == Q_(1, units.degC)

    def test_hydro(self):
        with units.context('hydro'):
            q = 1 * units.kg / units.m**2 / units.s
            assert q.to('mm/day') == q.to('mm/d')

    def test_lat_lon(self):
        assert 100 * units.degreeN == 100 * units.degree

    def test_pcic(self):
        with units.context('hydro'):
            fu = units.parse_units("kilogram / d / meter ** 2")
            tu = units.parse_units("mm/day")
            np.isclose(1 * fu, 1 * tu)


class TestThresholdCount:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(365))
        out = utils.threshold_count(ts, '<', 50, 'Y')
        np.testing.assert_array_equal(out, [50, 0])
