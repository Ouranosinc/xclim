from __future__ import annotations

import numpy as np
import xarray as xr

from xclim import atmos


class TestWindSpeedIndicators:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_calm_windy_days(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcwind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas, calm_wind_thresh="0 m/s")
            calm = atmos.calm_days(sfcwind, thresh="5 m/s")
            windy = atmos.windy_days(sfcwind, thresh="5 m/s")
            c = sfcwind.resample(time="MS").count()
            np.testing.assert_array_equal(calm + windy, c)


class TestSfcWind:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_sfcWind_max(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWind_max = atmos.sfcWind_max(sfcWind)
            c = sfcWind.resample(time="YS").max()
            np.testing.assert_array_equal(sfcWind_max, c)

    def test_sfcWind_mean(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWind_mean = atmos.sfcWind_mean(sfcWind)
            c = sfcWind.resample(time="YS").mean()
            np.testing.assert_array_equal(sfcWind_mean, c)

    def test_sfcWind_min(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWind_min = atmos.sfcWind_min(sfcWind)
            c = sfcWind.resample(time="YS").min()
            np.testing.assert_array_equal(sfcWind_min, c)


class TestSfcWindMax:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_sfcWindmax_max(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWindmax_max = atmos.sfcWindmax_max(sfcWindmax)
            c = sfcWindmax.resample(time="YS").max()
            np.testing.assert_array_equal(sfcWindmax_max, c)

    def test_sfcWindmax_mean(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWindmax_mean = atmos.sfcWindmax_mean(sfcWindmax)
            c = sfcWindmax.resample(time="YS").mean()
            np.testing.assert_array_equal(sfcWindmax_mean, c)

    def test_sfcWindmax_min(self, nimbus):
        with xr.open_dataset(nimbus.fetch(self.test_data), engine="h5netcdf") as ds:
            sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWindmax_min = atmos.sfcWindmax_min(sfcWindmax)
            c = sfcWindmax.resample(time="YS").min()
            np.testing.assert_array_equal(sfcWindmax_min, c)
