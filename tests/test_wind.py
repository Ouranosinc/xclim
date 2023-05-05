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


class TestSfcWindMean:
    def test_sfcWind_mean(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWind, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWind_mean = atmos.sfcWind_mean(sfcWind)
        c = sfcWind.resample(time="YS").mean()
        np.testing.assert_array_equal(sfcWind_mean, c)


class TestSfcWindmaxMax:
    def test_sfcWindmax_max(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        sfcWindmax, _ = atmos.wind_speed_from_vector(ds.uas, ds.vas)

        sfcWindmax_max = atmos.sfcWindmax_max(sfcWindmax)
        c = sfcWindmax_max.resample(time="YS").max()
        np.testing.assert_array_equal(sfcWindmax_max, c)
