import numpy as np
import xarray as xr

from xclim import seaIce
from xclim.indices import sea_ice_area
from xclim.indices import sea_ice_extent


class TestSeaIceExtent:
    def values(self, areacella):
        s = xr.ones_like(areacella)
        s = s.where(s.lat > 0, 10)
        s = s.where(s.lat <= 0, 50)
        sic = xr.concat([s, s], dim="time")
        sic.attrs["units"] = "%"

        return areacella, sic

    def test_simple(self, areacella):
        area, sic = self.values(areacella)

        a = sea_ice_extent(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_indicator(self, areacella):
        area, sic = self.values(areacella)

        a = seaIce.sea_ice_extent(sic, area)
        assert a.units == "m^2"

    def test_dimensionless(self, areacella):
        area, sic = self.values(areacella)
        sic = sic / 100
        sic.attrs["units"] = ""

        a = sea_ice_extent(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_area_units(self, areacella):
        area, sic = self.values(areacella)

        # Convert area to km^2
        area /= 1e6
        area.attrs["units"] = "km^2"

        a = sea_ice_extent(sic, area)
        assert a.units == "km^2"

        expected = 4 * np.pi * area.r ** 2 / 2.0 / 1e6
        np.testing.assert_array_almost_equal(a / expected, 1, 3)


class TestSeaIceArea(TestSeaIceExtent):
    def test_simple(self, areacella):
        area, sic = self.values(areacella)

        a = sea_ice_area(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_indicator(self, areacella):
        area, sic = self.values(areacella)

        a = seaIce.sea_ice_area(sic, area)
        assert a.units == "m^2"

    def test_dimensionless(self, areacella):
        area, sic = self.values(areacella)
        sic /= 100
        sic.attrs["units"] = ""

        a = sea_ice_area(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_area_units(self, areacella):
        area, sic = self.values(areacella)

        # Convert area to km^2
        area /= 1e6
        area.attrs["units"] = "km^2"

        a = sea_ice_area(sic, area)
        assert a.units == "km^2"

        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0 / 1e6
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
