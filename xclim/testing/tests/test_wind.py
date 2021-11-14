import numpy as np

from xclim import atmos
from xclim.testing import open_dataset


class TestWindSpeedIndicators:
    @classmethod
    def setup_class(self):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        self.sfcwind, _ = atmos.wind_speed_from_vector(
            ds.uas, ds.vas, calm_wind_thresh="0 m/s"
        )

    def test_calm_windy_days(self):
        calm = atmos.calm_days(self.sfcwind, thresh="5 m/s")
        windy = atmos.windy_days(self.sfcwind, thresh="5 m/s")

        c = self.sfcwind.resample(time="MS").count()
        np.testing.assert_array_equal(calm + windy, c)
