from __future__ import annotations

import numpy as np
import pytest

from xclim import atmos, convert


class TestWindSpeedIndicators:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_calm_windy_days(self, open_dataset):
        with open_dataset(self.test_data) as ds:
            sfcwind = convert.wind_speed_from_vector(ds.uas, ds.vas, calm_wind_thresh="0 m/s").sfcWind
            calm = atmos.calm_days(sfcwind, thresh="5 m/s")
            windy = atmos.windy_days(sfcwind, thresh="5 m/s")
            c = sfcwind.resample(time="MS").count()
            np.testing.assert_array_equal(calm.calm_days + windy.windy_days, c)


class TestSfcWind:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    @pytest.mark.parametrize(
        "metric",
        ["mean", "max", "min"],
    )
    def test_sfcWind(self, open_dataset, metric):
        with open_dataset(self.test_data) as ds:
            sfcWind = convert.wind_speed_from_vector(ds.uas, ds.vas).sfcWind
            sfcWind_calculated = getattr(atmos, f"sfcWind_{metric}")(sfcWind)

            resample = sfcWind.resample(time="YS")
            c = getattr(resample, metric)()
            np.testing.assert_array_equal(sfcWind_calculated[f"sfcWind_{metric}"], c)


class TestSfcWindMax:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    @pytest.mark.parametrize(
        "metric",
        ["mean", "max", "min"],
    )
    def test_sfcWindmax(self, open_dataset, metric):
        with open_dataset(self.test_data) as ds:
            sfcWind = convert.wind_speed_from_vector(ds.uas, ds.vas).sfcWind
            sfcWindmax_calculated = getattr(atmos, f"sfcWindmax_{metric}")(sfcWind)

            resample = sfcWind.resample(time="YS")
            c = getattr(resample, metric)()
            np.testing.assert_array_equal(sfcWindmax_calculated[f"sfcWindmax_{metric}"], c)
