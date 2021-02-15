import numpy as np
import xarray as xr

from xclim import seaIce
from xclim.indices import sea_ice_area, sea_ice_extent


class TestSeaIceExtent:
    def values(self, areacello):
        s = xr.ones_like(areacello)
        s = s.where(s.lat > 0, 10)
        s = s.where(s.lat <= 0, 50)
        sic = xr.concat([s, s], dim="time")
        sic.attrs["units"] = "%"
        sic.attrs["standard_name"] = "sea_ice_area_fraction"

        return areacello, sic

    def test_simple(self, areacello):
        area, sic = self.values(areacello)

        a = sea_ice_extent(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_indicator(self, areacello):
        area, sic = self.values(areacello)

        a = seaIce.sea_ice_extent(sic, area)
        assert a.units == "m2"

    def test_dimensionless(self, areacello):
        area, sic = self.values(areacello)
        sic = sic / 100
        sic.attrs["units"] = ""

        a = sea_ice_extent(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_area_units(self, areacello):
        area, sic = self.values(areacello)

        # Convert area to km^2
        area /= 1e6
        area.attrs["units"] = "km^2"

        a = sea_ice_extent(sic, area)
        assert a.units == "km^2"

        expected = 4 * np.pi * area.r ** 2 / 2.0 / 1e6
        np.testing.assert_array_almost_equal(a / expected, 1, 3)


class TestSeaIceArea(TestSeaIceExtent):
    def test_simple(self, areacello):
        area, sic = self.values(areacello)

        a = sea_ice_area(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_indicator(self, areacello):
        area, sic = self.values(areacello)

        a = seaIce.sea_ice_area(sic, area)
        assert a.units == "m2"

    def test_dimensionless(self, areacello):
        area, sic = self.values(areacello)
        sic /= 100
        sic.attrs["units"] = ""

        a = sea_ice_area(sic, area)
        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
        assert a.units == "m^2"

    def test_area_units(self, areacello):
        area, sic = self.values(areacello)

        # Convert area to km^2
        area /= 1e6
        area.attrs["units"] = "km^2"

        a = sea_ice_area(sic, area)
        assert a.units == "km^2"

        expected = 4 * np.pi * area.r ** 2 / 2.0 / 2.0 / 1e6
        np.testing.assert_array_almost_equal(a / expected, 1, 3)
