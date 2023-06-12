from __future__ import annotations

import numpy as np

from xclim import atmos


class TestWindSpeedIndicators:
    def test_calm_windy_days(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcwind, _ = atmos.wind_speed_from_vector(
            ds.uas, ds.vas, calm_wind_thresh="0 m/s"
        )

        calm = atmos.calm_days(sfcwind, thresh="5 m/s")
        windy = atmos.windy_days(sfcwind, thresh="5 m/s")
        c = sfcwind.resample(time="MS").count()
        np.testing.assert_array_equal(calm + windy, c)


class TestSfcWindMax:
    def test_sfcWind_max(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWind_max = atmos.sfcWind_max(sfcWind)
        c = sfcWind.resample(time="YS").max()
        np.testing.assert_array_equal(sfcWind_max, c)


class TestSfcWindMean:
    def test_sfcWind_mean(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWind_mean = atmos.sfcWind_mean(sfcWind)
        c = sfcWind.resample(time="YS").mean()
        np.testing.assert_array_equal(sfcWind_mean, c)


class TestSfcWindMin:
    def test_sfcWind_min(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWind_min = atmos.sfcWind_min(sfcWind)
        c = sfcWind.resample(time="YS").min()
        np.testing.assert_array_equal(sfcWind_min, c)


class TestSfcWindmaxMax:
    def test_sfcWindmax_max(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWindmax_max = atmos.sfcWindmax_max(sfcWindmax)
        c = sfcWindmax.resample(time="YS").max()
        np.testing.assert_array_equal(sfcWindmax_max, c)


class TestSfcWindmaxMean:
    def test_sfcWindmax_mean(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWindmax_mean = atmos.sfcWindmax_mean(sfcWindmax)
        c = sfcWindmax.resample(time="YS").mean()
        np.testing.assert_array_equal(sfcWindmax_mean, c)


class TestSfcWindmaxMin:
    def test_sfcWindmax_mean(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWindmax_min = atmos.sfcWindmax_min(sfcWindmax)
        c = sfcWindmax.resample(time="YS").min()
        np.testing.assert_array_equal(sfcWindmax_min, c)
