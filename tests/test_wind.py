from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim import atmos, convert
from xclim.compute.helpers import resample_map


class TestWindSpeedIndicators:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_calm_windy_days(self, open_dataset):
        with open_dataset(self.test_data) as ds:
            sfcwind, _ = convert.wind_speed_from_vector(ds.uas, ds.vas, calm_wind_thresh="0 m/s")
            calm = atmos.calm_days(sfcwind, thresh="5 m/s")
            windy = atmos.windy_days(sfcwind, thresh="5 m/s")
            c = sfcwind.resample(time="MS").count()
            np.testing.assert_array_equal(calm + windy, c)


class TestSfcWind:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    @pytest.mark.parametrize(
        "metric",
        ["mean", "max", "min"],
    )
    def test_sfcWind(self, open_dataset, metric):
        with open_dataset(self.test_data) as ds:
            sfcWind, _ = convert.wind_speed_from_vector(ds.uas, ds.vas)
            sfcWind_calculated = getattr(atmos, f"sfcWind_{metric}")(sfcWind)

            resample = sfcWind.resample(time="YS")
            # resample.mean() would have float32/64 conversion errors
            func = getattr(xr.DataArray, metric)
            c = resample.map(func, dim="time")
            np.testing.assert_array_almost_equal(sfcWind_calculated, c)


class TestSfcWindMax:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    @pytest.mark.parametrize(
        "metric",
        ["mean", "max", "min"],
    )
    def test_sfcWindmax(self, open_dataset, metric):
        with open_dataset(self.test_data) as ds:
            sfcWind, _ = convert.wind_speed_from_vector(ds.uas, ds.vas)
            resample = sfcWind.resample(time="YS")
            func = getattr(xr.DataArray, metric)
            c = resample.map(func, dim="time")
            sfcWindmax_calculated = resample_map(sfcWind, "time", "YS", metric)
            np.testing.assert_array_equal(sfcWindmax_calculated, c)
