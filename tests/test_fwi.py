import pytest
import numpy as np
import xarray as xr
from xclim.indices.fwi import fine_fuel_moisture_code, duff_moisture_code, drought_code
from xclim.indices.fwi import initial_spread_index, build_up_index, fire_weather_index
from xclim.indices import fire_weather_index as xfwi
import os
from pathlib import Path

TESTS_HOME = Path(os.path.dirname(__file__))
TESTS_DATA = TESTS_HOME / 'testdata' / 'fwi'


class TestFireWeatherIndex:
    """Note that some of the lines in the code are not exercised by the test data."""

    def get_data(self):
        import io
        import pandas as pd
        f = io.StringIO(testdata)
        return pd.read_table(f, sep=' ', header=0)

    def test_xfwi(self):
        import datetime as dt
        # Convert test data in DataArrays.
        data = self.get_data()
        t = [dt.datetime(2015, m, d) for (m, d) in zip(data.pop('mth'), data.pop('day'))]
        data['time'] = t
        data.set_index('time', inplace=True)

        ds = xr.Dataset(data)
        ds.temp.attrs['units'] = 'C'
        ds.pr.attrs['units'] = 'mm/day'
        ds.ws.attrs['units'] = 'km/h'
        ds.rh.attrs['units'] = 'pct'

        fwi = xfwi(ds.temp, ds.pr, ds.ws, ds.rh, 85.0, 6.0, 15.0)
        np.testing.assert_array_almost_equal(fwi, data['fwi'], 1)

    def test_gfwed(self):
        import datetime as dt

        init_fn = TESTS_DATA / 'FWI.MERRA2.Daily.Default.19850730.nc'
        if not init_fn.exists():
            pytest.skip()
            # https://data.giss.nasa.gov/impacts/gfwed/
        i = xr.open_dataset(init_fn)

        var_fn = [TESTS_DATA / 'Wx.MERRA2.Daily.Default.19850731.nc',
                 TESTS_DATA / 'Prec.MERRA2.Daily.Default.19850731.nc']
        v = xr.open_mfdataset(var_fn).reset_index('time', drop=True)
        v['time'] = xr.IndexVariable('time', [dt.datetime(1985, 7, 31)])
        v.set_coords('time', inplace=True)

        out_fn = TESTS_DATA / 'FWI.MERRA2.Daily.Default.19850731.nc'
        out = xr.open_dataset(out_fn)

        fwi = xfwi(v.MERRA2_t, v.MERRA2_prec, v.MERRA2_wdSpd, v.MERRA2_rh,
                   i.MERRA2_FFMC[0], i.MERRA2_DMC[0], i.MERRA2_DC[0])
        xr.testing.assert_allclose(fwi.sel(lat=slice(30, None)), out.MERRA2_FWI.sel(lat=slice(30, None)), atol=1)


    def test_fine_fuel_moisture_code(self):
        d = self.get_data()

        ffmc0 = 85.0

        ffmc = fine_fuel_moisture_code(d['temp'], d['pr'], d['ws'], d['rh'], ffmc0)

        np.testing.assert_array_almost_equal(ffmc, d['ffmc'], 1)

    def test_duff_moisture_code(self):
        d = self.get_data()

        dmc0 = 6.0

        dmc = duff_moisture_code(d['temp'], d['pr'], d['rh'], d['mth'].astype(int), dmc0)

        np.testing.assert_array_almost_equal(dmc, d['dmc'], 1)

    def test_drought_code(self):
        d = self.get_data()

        dc0 = 15.0

        dc = drought_code(d['temp'], d['pr'], d['mth'].astype(int), dc0)

        np.testing.assert_array_almost_equal(dc, d['dc'], 1)

    def test_initial_spread_index(self):
        # Note that using the rounded data as input creates rounding errors.
        d = self.get_data()
        ffmc0 = 85.0
        ffmc = fine_fuel_moisture_code(d['temp'], d['pr'], d['ws'], d['rh'], ffmc0)
        isi = initial_spread_index(d['ws'], ffmc)
        np.testing.assert_array_almost_equal(isi, d['isi'], 1)

    def test_build_up_index(self):
        d = self.get_data()

        dmc0 = 6.0
        dc0 = 15.0
        dmc = duff_moisture_code(d['temp'], d['pr'], d['rh'], d['mth'].astype(int), dmc0)
        dc = drought_code(d['temp'], d['pr'], d['mth'].astype(int), dc0)

        bui = build_up_index(dmc, dc)
        np.testing.assert_array_almost_equal(bui, d['bui'], 1)

    def test_fire_weather_index(self):
        d = self.get_data()

        dmc0 = 6.0
        dc0 = 15.0
        ffmc0 = 85.0
        ffmc = fine_fuel_moisture_code(d['temp'], d['pr'], d['ws'], d['rh'], ffmc0)
        dmc = duff_moisture_code(d['temp'], d['pr'], d['rh'], d['mth'].astype(int), dmc0)
        dc = drought_code(d['temp'], d['pr'], d['mth'].astype(int), dc0)
        isi = initial_spread_index(d['ws'].values, ffmc)
        bui = build_up_index(dmc, dc)
        fwi = fire_weather_index(isi, bui)
        np.testing.assert_array_almost_equal(fwi, d['fwi'], 1)


testdata = """mth day temp rh ws pr ffmc dmc dc isi bui fwi
4 13 17.0 42.0 25.0 0.0 87.7 8.5 19.0 10.9 8.5 10.1
4 14 20.0 21.0 25.0 2.4 86.2 10.4 23.6 8.8 10.4 9.3
4 15 8.5 40.0 17.0 0.0 87.0 11.8 26.1 6.5 11.7 7.6
4 16 6.5 25.0 6.0 0.0 88.8 13.2 28.2 4.9 13.1 6.2
4 17 13.0 34.0 24.0 0.0 89.1 15.4 31.5 12.6 15.3 14.8
4 18 6.0 40.0 22.0 0.4 88.7 16.5 33.5 10.7 16.4 13.5
4 19 5.5 52.0 6.0 0.0 87.4 17.2 35.4 4.0 17.1 5.9
4 20 8.5 46.0 16.0 0.0 87.4 18.5 37.9 6.6 18.4 9.7
4 21 9.5 54.0 20.0 0.0 86.8 19.7 40.6 7.4 19.6 11.0
4 22 7.0 93.0 14.0 9.0 29.9 10.1 29.5 0.0 10.9 0.0
4 23 6.5 71.0 17.0 1.0 49.4 10.7 31.6 0.4 11.6 0.2
4 24 6.0 59.0 17.0 0.0 67.3 11.4 33.7 1.3 12.3 0.9
4 25 13.0 52.0 4.0 0.0 77.8 13.0 37.0 1.1 13.9 0.8
4 26 15.5 40.0 11.0 0.0 85.5 15.4 40.7 3.9 15.9 5.5
4 27 23.0 25.0 9.0 0.0 91.5 19.8 45.8 8.4 19.8 12.2
4 28 19.0 46.0 16.0 0.0 89.9 22.5 50.2 9.5 22.4 14.3
4 29 18.0 41.0 20.0 0.0 90.0 25.2 54.4 11.7 25.1 17.7
4 30 14.5 51.0 16.0 0.0 88.4 27.0 57.9 7.7 27.0 13.3
5 1 14.5 69.0 11.0 0.0 85.7 28.3 63.0 4.0 28.2 8.0
5 2 15.5 42.0 8.0 0.0 87.4 30.8 68.2 4.4 30.8 9.1
5 3 21.0 37.0 8.0 0.0 89.4 34.5 74.3 5.9 34.4 12.3
5 4 23.0 32.0 16.0 0.0 91.0 38.8 80.9 11.1 38.7 21.0
5 5 23.0 32.0 14.0 0.0 91.2 43.1 87.4 10.3 43.0 21.1
5 6 27.0 33.0 12.0 0.0 91.7 48.1 94.7 9.9 47.9 21.7
5 7 28.0 17.0 27.0 0.0 95.2 54.5 102.1 34.5 54.3 52.6
5 8 23.5 54.0 20.0 0.0 89.7 57.4 108.8 11.3 57.2 25.9
5 9 16.0 50.0 22.0 12.2 62.2 29.9 91.8 1.4 33.0 3.0
5 10 11.5 58.0 20.0 0.0 76.7 31.3 96.3 2.3 34.5 5.4
5 11 16.0 54.0 16.0 0.0 83.5 33.4 101.6 3.8 36.7 8.9
5 12 21.5 37.0 9.0 0.0 88.7 37.1 107.8 5.6 39.9 12.8
5 13 14.0 61.0 22.0 0.2 86.7 38.7 112.8 8.1 41.6 17.3
5 14 15.0 30.0 27.0 0.0 89.6 41.7 117.9 15.9 44.2 28.8
5 15 20.0 23.0 11.0 0.0 92.1 45.9 123.9 10.1 47.7 21.9
5 16 14.0 95.0 3.0 16.4 21.3 20.1 97.0 0.0 26.5 0.0
5 17 20.0 53.0 4.0 2.8 51.0 18.3 103.0 0.2 25.3 0.2
5 18 19.5 30.0 16.0 0.0 82.3 22.1 108.9 3.3 29.3 6.8
5 19 25.5 51.0 20.0 6.0 75.4 16.4 106.4 2.1 23.7 3.8
5 20 10.0 38.0 24.0 0.0 84.3 18.2 110.6 6.4 25.8 11.3
5 21 19.0 27.0 16.0 0.0 90.3 22.1 116.4 10.0 29.9 17.2
5 22 26.0 46.0 11.0 4.2 77.6 18.7 117.7 1.6 26.8 2.9
5 23 30.0 38.0 22.0 0.0 90.2 23.8 125.5 13.4 32.3 22.0
5 24 25.5 67.0 19.0 12.6 65.3 13.1 108.5 1.4 20.2 1.9
5 25 12.0 53.0 28.0 11.8 55.4 7.7 91.6 1.2 12.8 0.8
5 26 21.0 38.0 8.0 0.0 80.8 11.3 97.8 1.9 17.6 2.6
5 27 13.0 70.0 20.0 3.8 61.7 8.4 97.9 1.2 13.8 0.9
5 28 9.0 78.0 24.0 1.4 64.5 9.0 101.9 1.7 14.7 2.0
5 29 11.0 54.0 16.0 0.0 77.6 10.5 106.3 2.0 16.8 2.8
5 30 15.5 39.0 9.0 0.0 85.4 13.1 111.5 3.5 20.3 5.8
5 31 18.0 36.0 5.0 0.0 88.5 16.3 117.1 4.4 24.2 7.9"""


def test_ufunc():
    """Test to experiment with ufuncs over arrays with non-identical dimensions."""
    v = xr.DataArray(np.resize(np.arange(10), (2, 10)),
                 dims=('x', 'time'),
                 coords=[('x', [0, 1]), ('time', np.arange(10))])
    i = xr.DataArray([10, 20], dims=('x'),
                     coords=[('x', [0, 1])])

    def func1d(values, init):
        if init < 15:
            return init + np.cumsum(values)
        else:
            return init - np.cumsum(values)

    xr.apply_ufunc(func1d,
                   v, i,
                   input_core_dims=(('time',), ()),
                   output_core_dims=(('time',),),
                   vectorize=True,
                   dask='parallelized',
                   output_dtypes=[np.float, ],
                   )
