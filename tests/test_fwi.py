import os

import pytest
import xarray as xr

from xclim.indices import drought_code
from xclim.indices import fire_weather_indexes
from xclim.indices.fwi import day_length
from xclim.indices.fwi import day_length_factor
from xclim.indices.fwi import fire_weather_ufunc

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


class TestFireWeatherIndex:
    """Note that some of the lines in the code are not exercised by the test data."""

    nc_gfwed = os.path.join(TESTS_DATA, "FWI", "FWITestData.nc")

    def test_fire_weather_indexes(self):
        ds = xr.open_dataset(self.nc_gfwed)
        fwis = fire_weather_indexes(
            ds.tas,
            ds.prbc,
            ds.sfcwind,
            ds.rh,
            ds.lat,
            snd=ds.snow_depth,
            ffmc0=ds.FFMC.sel(time="2017-03-02"),
            dmc0=ds.DMC.sel(time="2017-03-02"),
            dc0=ds.DC.sel(time="2017-03-02"),
            start_date="2017-03-03",
            start_up_mode="snow_depth",
        )
        for ind, name in zip(fwis, ["DC", "DMC", "FFMC", "ISI", "BUI", "FWI"]):
            xr.testing.assert_allclose(
                ind.sel(time=slice("2017-03-03", None)),
                ds[name].sel(time=slice("2017-03-03", None)),
                rtol=1e-4,
            )

    def test_drought_code(self):
        ds = xr.open_dataset(self.nc_gfwed)
        dc = drought_code(
            ds.tas,
            ds.prbc,
            ds.lat,
            snd=ds.snow_depth,
            dc0=ds.DC.sel(time="2017-03-02"),
            start_date="2017-03-03",
            start_up_mode="snow_depth",
        )
        xr.testing.assert_allclose(
            dc.sel(time=slice("2017-03-03", None)),
            ds.DC.sel(time=slice("2017-03-03", None)),
        )

    def test_day_length(self):
        assert day_length(44)[0] == 6.5

    def test_day_lengh_factor(self):
        assert day_length_factor(44)[0] == -1.6

    def test_fire_weather_ufunc_errors(self):
        ds = xr.open_dataset(self.nc_gfwed)

        # Test invalid combination
        with pytest.raises(TypeError):
            fire_weather_ufunc(
                tas=ds.tas,
                pr=ds.prbc,
                rh=ds.rh,
                ws=ds.sfcwind,
                lat=ds.lat,
                dc0=ds.DC.isel(time=0),
                indexes=["DC", "ISI"],
                start_up_mode="precip",
            )

        # Test missing arguments
        with pytest.raises(TypeError):
            fire_weather_ufunc(
                tas=ds.tas,
                pr=ds.prbc,  # lat=ds.lat,
                dc0=ds.DC.isel(time=0),
                indexes=["DC"],
                start_up_mode="precip",
            )

        with pytest.raises(TypeError):
            fire_weather_ufunc(
                tas=ds.tas,
                pr=ds.prbc,
                lat=ds.lat,
                dc0=ds.DC.isel(time=0),
                indexes=["DC"],
                start_up_mode="snow_depth",
            )
        # Test starting too early
        with pytest.raises(ValueError):
            fire_weather_ufunc(
                tas=ds.tas,
                pr=ds.prbc,
                lat=ds.lat,
                snd=ds.snow_depth,
                dc0=ds.DC.isel(time=0),
                indexes=["DC"],
                start_up_mode="snow_depth",
                start_date="2017-01-01",
            )

        # Test output is complete
        out = fire_weather_ufunc(
            tas=ds.tas,
            pr=ds.prbc,
            lat=ds.lat,
            snd=ds.snow_depth,
            dc0=ds.DC.sel(time="2017-03-02"),
            indexes=["DC"],
            start_up_mode="snow_depth",
            start_date="2017-03-03",
        )

        assert len(out.keys()) == 1

        out = fire_weather_ufunc(
            tas=ds.tas,
            pr=ds.prbc,
            rh=ds.rh,
            ws=ds.sfcwind,
            lat=ds.lat,
            snd=ds.snow_depth,
            dc0=ds.DC.sel(time="2017-03-02"),
            dmc0=ds.DMC.sel(time="2017-03-02"),
            ffmc0=ds.FFMC.sel(time="2017-03-02"),
            indexes=["DC", "DMC", "FFMC"],
            start_up_mode="snow_depth",
            start_date="2017-03-03",
        )

        assert len(out.keys()) == 3
