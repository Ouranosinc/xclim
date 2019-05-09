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
import glob
import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask


from xclim import ensembles
from xclim import indices
from xclim import subset
from xclim import utils
from xclim.utils import daily_downsampler, Indicator, format_kwargs, parse_doc, walk_map
from xclim.utils import infer_doy_max, adjust_doy_calendar, percentile_doy
from xclim.utils import units, pint2cfunits, units2pint
from xclim.testing.common import tas_series, pr_series

TAS_SERIES = tas_series
PR_SERIES = pr_series
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')


class TestEnsembleStats:
    nc_files_simple = glob.glob(os.path.join(TESTS_DATA, 'EnsembleStats', '*1950-2100*.nc'))
    nc_files = glob.glob(os.path.join(TESTS_DATA, 'EnsembleStats', '*.nc'))

    def test_create_ensemble(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        assert len(ens.realization) == len(self.nc_files_simple)

    def test_create_unequal_times(self):
        ens = ensembles.create_ensemble(self.nc_files)
        assert len(ens.realization) == len(self.nc_files)
        assert ens.time.dt.year.min() == 1970
        assert ens.time.dt.year.max() == 2050

    def test_calc_perc(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(np.percentile(ens['tg_mean'][:, 0, 5, 5], 10), out1['tg_mean_p10'][0, 5, 5])
        np.testing.assert_array_equal(np.percentile(ens['tg_mean'][:, 0, 5, 5], 50), out1['tg_mean_p50'][0, 5, 5])
        np.testing.assert_array_equal(np.percentile(ens['tg_mean'][:, 0, 5, 5], 90), out1['tg_mean_p90'][0, 5, 5])
        assert np.all(out1['tg_mean_p90'] > out1['tg_mean_p50'])
        assert np.all(out1['tg_mean_p50'] > out1['tg_mean_p10'])
        out1 = ensembles.ensemble_percentiles(ens, values=(25, 75))
        assert np.all(out1['tg_mean_p75'] > out1['tg_mean_p25'])

    def test_calc_perc_blocks(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_percentiles(ens)
        out2 = ensembles.ensemble_percentiles(ens, values=(10, 50, 90), time_block=10)
        np.testing.assert_array_equal(out1['tg_mean_p10'], out2['tg_mean_p10'])
        np.testing.assert_array_equal(out1['tg_mean_p50'], out2['tg_mean_p50'])
        np.testing.assert_array_equal(out1['tg_mean_p90'], out2['tg_mean_p90'])

    def test_calc_perc_nans(self):
        ens = ensembles.create_ensemble(self.nc_files_simple).load()

        ens.tg_mean[2, 0, 5, 5] = np.nan
        ens.tg_mean[2, 7, 5, 5] = np.nan
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(np.percentile(ens['tg_mean'][:, 0, 5, 5], 10), np.nan)
        np.testing.assert_array_equal(np.percentile(ens['tg_mean'][:, 7, 5, 5], 10), np.nan)
        np.testing.assert_array_equal(np.nanpercentile(ens['tg_mean'][:, 0, 5, 5], 10), out1['tg_mean_p10'][0, 5, 5])
        np.testing.assert_array_equal(np.nanpercentile(ens['tg_mean'][:, 7, 5, 5], 10), out1['tg_mean_p10'][7, 5, 5])
        assert np.all(out1['tg_mean_p90'] > out1['tg_mean_p50'])
        assert np.all(out1['tg_mean_p50'] > out1['tg_mean_p10'])

    def test_calc_mean_std_min_max(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_mean_std_max_min(ens)
        np.testing.assert_array_equal(ens['tg_mean'][:, 0, 5, 5].mean(dim='realization'), out1.tg_mean_mean[0, 5, 5])
        np.testing.assert_array_equal(ens['tg_mean'][:, 0, 5, 5].std(dim='realization'), out1.tg_mean_stdev[0, 5, 5])
        np.testing.assert_array_equal(ens['tg_mean'][:, 0, 5, 5].max(dim='realization'), out1.tg_mean_max[0, 5, 5])
        np.testing.assert_array_equal(ens['tg_mean'][:, 0, 5, 5].min(dim='realization'), out1.tg_mean_min[0, 5, 5])


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
    long_name = '{freq} mean surface temperature'
    standard_name = '{freq} mean temperature'
    cell_methods = 'time: mean within {freq}'

    @staticmethod
    def compute(da, thresh=0., freq='YS'):
        """Docstring"""
        out = da
        out -= thresh
        return out.resample(time=freq).mean(keep_attrs=True)


class UniIndPr(Indicator):
    identifier = 'prmax'
    units = 'mm/s'
    context = 'hydro'

    @staticmethod
    def compute(da, freq):
        """Docstring"""
        return da.resample(time=freq).mean(keep_attrs=True)


class TestIndicator:

    def test_attrs(self, tas_series):
        import datetime as dt
        a = tas_series(np.arange(360.))
        ind = UniIndTemp()
        txm = ind(a, freq='YS')
        assert txm.cell_methods == 'time: mean within days time: mean within years'
        assert '{:%Y-%m-%d %H}'.format(dt.datetime.now()) in txm.attrs['history']
        assert txm.name == "tmin0"

    def test_temp_unit_conversion(self, tas_series):
        a = tas_series(np.arange(360.))
        ind = UniIndTemp()
        txk = ind(a, freq='YS')

        ind.units = 'degC'
        txc = ind(a, freq='YS')

        np.testing.assert_array_almost_equal(txk, txc + 273.15)

    def test_json(self, pr_series):
        ind = UniIndPr()
        meta = ind.json()

        expected = {'identifier', 'units', 'long_name', 'standard_name', 'cell_methods', 'keywords', 'abstract',
                    'parameters', 'description', 'history', 'references', 'comment', 'notes'}

        assert set(meta.keys()).issubset(expected)

    def test_factory(self, pr_series):
        attrs = dict(identifier='test', units='days', required_units='[length] / [time]',
                     long_name='long name',
                     standard_name='standard name', context='hydro'
                     )
        cls = Indicator.factory(attrs)

        assert issubclass(cls, Indicator)
        da = pr_series(np.arange(365))
        cls(compute=indices.wetdays)(da)

    def test_signature(self):
        from inspect2 import signature
        ind = UniIndTemp()
        assert signature(ind.compute) == signature(ind.__call__)

    def test_doc(self):
        ind = UniIndTemp()
        assert ind.__call__.__doc__ == ind.compute.__doc__

    def test_delayed(self):
        fn = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

        # Load dataset as a dask array
        ds = xr.open_dataset(fn, chunks={'time': 10}, cache=True)

        tx = UniIndTemp()
        txk = tx(ds.tasmax)

        # Check that the calculations are delayed
        assert isinstance(txk.data, dask.array.core.Array)

        # Same with unit conversion
        tx.required_units = ('C',)
        tx.units = 'C'
        txc = tx(ds.tasmax)

        assert isinstance(txc.data, dask.array.core.Array)


class TestKwargs:

    def test_format_kwargs(self):
        attrs = dict(standard_name='tx_min', long_name='Minimum of daily maximum temperature',
                     cell_methods='time: minimum within {freq}')

        format_kwargs(attrs, {'freq': 'YS'})
        assert attrs['cell_methods'] == 'time: minimum within years'


class TestParseDoc:

    def test_simple(self):
        parse_doc(indices.tg_mean.__doc__)


class TestPercentileDOY:

    def test_simple(self, tas_series):
        tas = tas_series(np.arange(365), start='1/1/2001')
        p1 = percentile_doy(tas, window=5, per=.5)
        assert p1.sel(dayofyear=3).data == 2
        assert p1.attrs['units'] == 'K'


class TestAdjustDoyCalendar:

    def test_360_to_366(self):
        source = xr.DataArray(np.arange(360), coords=[np.arange(1, 361), ], dims='dayofyear')
        time = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        target = xr.DataArray(np.arange(len(time)), coords=[time, ], dims='time')

        out = adjust_doy_calendar(source, target)

        assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
        assert out.sel(dayofyear=366) == source.sel(dayofyear=360)

    def test_infer_doy_max(self):
        fn = os.path.join(TESTS_DATA, 'CanESM2_365day',
                          'pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc')
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 365

        fn = os.path.join(TESTS_DATA, 'HadGEM2-CC_360day',
                          'pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc')
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 360

        fn = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 366


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
            q = 1 * units.kg / units.m ** 2 / units.s
            assert q.to('mm/day') == q.to('mm/d')

    def test_lat_lon(self):
        assert 100 * units.degreeN == 100 * units.degree

    def test_pcic(self):
        with units.context('hydro'):
            fu = units.parse_units("kilogram / d / meter ** 2")
            tu = units.parse_units("mm/day")
            np.isclose(1 * fu, 1 * tu)

    def test_dimensionality(self):
        with units.context('hydro'):
            fu = 1 * units.parse_units('kg / m**2 / s')
            tu = 1 * units.parse_units('mm / d')
            fu.to('mmday')
            tu.to('mmday')


class TestConvertUnitsTo:

    def test_deprecation(self, tas_series):
        with pytest.warns(FutureWarning):
            out = utils.convert_units_to(0, units.K)
            assert out == 273.15

            out = utils.convert_units_to(10, units.mm / units.day, context='hydro')
            assert out == 10

        with pytest.warns(FutureWarning):
            tas = tas_series(np.arange(365), start='1/1/2001')
            out = indices.tx_days_above(tas, 30)
            out1 = indices.tx_days_above(tas, '30 degC')
            out2 = indices.tx_days_above(tas, '303.15 K')
            np.testing.assert_array_equal(out, out1)
            np.testing.assert_array_equal(out, out2)


class TestUnitConversion:

    def test_pint2cfunits(self):
        u = units('mm/d')
        assert pint2cfunits(u.units) == 'mm d-1'

    def test_cfunits2pint(self, pr_series):
        u = units2pint(pr_series([1, 2]))
        assert (str(u)) == 'kilogram / meter ** 2 / second'
        assert pint2cfunits(u) == 'kg m-2 s-1'

        u = units2pint('m^3 s-1')
        assert str(u) == 'meter ** 3 / second'
        assert pint2cfunits(u) == 'm^3 s-1'

    def test_pint_multiply(self, pr_series):
        a = pr_series([1, 2, 3])
        out = utils.pint_multiply(a, 1 * units.days)
        assert out[0] == 1 * 60 * 60 * 24
        assert out.units == 'kg m-2'


class TestSubsetGridPoint:
    nc_poslons = os.path.join(TESTS_DATA, 'cmip3', 'tas.sresb1.giss_model_e_r.run1.atm.da.nc')
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_2dlonlat = os.path.join(TESTS_DATA, 'CRCM5', 'tasmax_bby_198406_se.nc')

    def test_dataset(self):
        da = xr.open_mfdataset([self.nc_file, self.nc_file.replace('tasmax', 'tasmin')])
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    def test_simple(self):
        da = xr.open_dataset(self.nc_file).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        da = xr.open_dataset(self.nc_poslons).tas
        da['lon'] -= 360
        yr_st = 2050
        yr_ed = 2059

        out = subset.subset_gridpoint(da, lon=lon, lat=lat, start_yr=yr_st, end_yr=yr_ed)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

    def test_irregular(self):
        da = xr.open_dataset(self.nc_2dlonlat).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

    def test_positive_lons(self):
        da = xr.open_dataset(self.nc_poslons).tas
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        out = subset.subset_gridpoint(da, lon=lon + 360, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

    def test_raise(self):
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_gridpoint(da, lon=-72.4, lat=46.1, start_yr=2056, end_yr=2055)


class TestSubsetBbox:
    nc_poslons = os.path.join(TESTS_DATA, 'cmip3', 'tas.sresb1.giss_model_e_r.run1.atm.da.nc')
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_2dlonlat = os.path.join(TESTS_DATA, 'CRCM5', 'tasmax_bby_198406_se.nc')
    lon = [-72.4, -60]
    lat = [42, 46.1]

    def test_dataset(self):
        da = xr.open_mfdataset([self.nc_file, self.nc_file.replace('tasmax', 'tasmin')])
        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert (np.all(out.lon >= np.min(self.lon)))
        assert (np.all(out.lon <= np.max(self.lon)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    def test_simple(self):
        da = xr.open_dataset(self.nc_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert (np.all(out.lon >= np.min(self.lon)))
        assert (np.all(out.lon <= np.max(self.lon)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))

        da = xr.open_dataset(self.nc_poslons).tas
        da['lon'] -= 360
        yr_st = 2050
        yr_ed = 2059

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat, start_yr=yr_st, end_yr=yr_ed)
        assert (np.all(out.lon >= np.min(self.lon)))
        assert (np.all(out.lon <= np.max(self.lon)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat, start_yr=yr_st)
        assert (np.all(out.lon >= np.min(self.lon)))
        assert (np.all(out.lon <= np.max(self.lon)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat, end_yr=yr_ed)
        assert (np.all(out.lon >= np.min(self.lon)))
        assert (np.all(out.lon <= np.max(self.lon)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())

    def test_irregular(self):
        da = xr.open_dataset(self.nc_2dlonlat).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

        # for irregular lat lon grids data matrix remains rectangular in native proj
        # but with data outside bbox assigned nans.  This means it can have lon and lats outside the bbox.
        # Check only non-nans gridcells using mask
        mask1 = ~np.isnan(out.sel(time=out.time[0]))

        assert (np.all(out.lon.values[mask1] >= np.min(self.lon)))
        assert (np.all(out.lon.values[mask1] <= np.max(self.lon)))
        assert (np.all(out.lat.values[mask1] >= np.min(self.lat)))
        assert (np.all(out.lat.values[mask1] <= np.max(self.lat)))

    def test_positive_lons(self):
        da = xr.open_dataset(self.nc_poslons).tas

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert (np.all(out.lon >= np.min(np.asarray(self.lon) + 360)))
        assert (np.all(out.lon <= np.max(np.asarray(self.lon) + 360)))
        assert (np.all(out.lat >= np.min(self.lat)))
        assert (np.all(out.lat <= np.max(self.lat)))

        out = subset.subset_bbox(da, lon_bnds=np.array(self.lon) + 360, lat_bnds=self.lat)
        assert (np.all(out.lon >= np.min(np.asarray(self.lon) + 360)))

    def test_raise(self):
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat, start_yr=2056, end_yr=2055)


class TestThresholdCount:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(365))
        out = utils.threshold_count(ts, '<', 50, 'Y')
        np.testing.assert_array_equal(out, [50, 0])
