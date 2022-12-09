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
