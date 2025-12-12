import numpy as np
import pytest

from xclim import atmos, set_options
from xclim.indices.helpers import make_hourly_temperature

K2C = 273.16


class TestLateFrostDays:
    def test_late_frost_days(self, atmosds):
        tasmin = atmosds.tasmin

        # Expected values
        exp = [2, 9, 72, 24, 0]

        lfd = atmos.late_frost_days(tasmin, date_bounds=("04-01", "06-30"))

        np.testing.assert_allclose(lfd.isel(time=0), exp, rtol=1e-03)


def test_high_precip_low_temp(pr_series, tasmin_series):
    pr = np.zeros(365)
    pr[1:3] = [1, 2]
    pr = pr_series(pr, start="1999-01-01")

    tas = np.zeros(365)
    tas[2:4] = [1, 1]
    tas += K2C
    tas = tasmin_series(tas, start="1999-01-01")

    out = atmos.high_precip_low_temp(pr, tas, pr_thresh="1 kg m-2 s-1", tas_thresh="1 C")
    np.testing.assert_array_equal(out, [1])


class TestDrynessIndex:
    def test_simple(self, atmosds):
        ds = atmosds.isel(location=3)

        pr = ds.pr
        evspsblpot = ds.evspsblpot

        di = atmos.dryness_index(pr, evspsblpot)
        np.testing.assert_allclose(di, np.array([13.355, 102.426, 65.576, 158.078]), rtol=1e-03)
        assert di.attrs["long_name"] == "Growing season humidity"

    def test_variable_initial_conditions(self, atmosds):
        ds = atmosds

        pr = ds.pr
        evspsblpot = ds.evspsblpot

        di = atmos.dryness_index(pr, evspsblpot)
        di_wet = atmos.dryness_index(pr, evspsblpot, wo="250 mm")
        di_dry = atmos.dryness_index(pr, evspsblpot, wo="100 mm")

        assert np.all(di_wet > di_dry)
        di_plus_50 = di + 50
        np.testing.assert_allclose(di_wet, di_plus_50, rtol=1e-03)
        di_minus_100 = di - 100
        np.testing.assert_allclose(di_dry, di_minus_100, rtol=1e-03)

        for value, array in {"200 mm": di, "250 mm": di_wet, "100 mm": di_dry}.items():
            assert array.attrs["long_name"] == "Growing season humidity"
            assert value in array.attrs["description"]


class TestChill:
    def test_chill_units(self, atmosds):
        tasmax = atmosds.tasmax
        tasmin = atmosds.tasmin
        tas = make_hourly_temperature(tasmin, tasmax)
        cu = atmos.chill_units(tas, date_bounds=("04-01", "06-30"))
        assert cu.attrs["units"] == "1"
        assert cu.name == "cu"
        assert cu.time.size == 4

        # Values are confirmed with chillR package although not an exact match
        # due to implementation details
        exp = [1546.5, 1344.0, 1162.0, 1457.5]
        np.testing.assert_allclose(cu.isel(location=0), exp, rtol=1e-03)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_chill_portions(self, atmosds, use_dask):
        pytest.importorskip("flox")
        tasmax = atmosds.tasmax
        tasmin = atmosds.tasmin
        tas = make_hourly_temperature(tasmin, tasmax)
        if use_dask:
            tas = tas.chunk(time=tas.time.size // 2, location=1)

        with set_options(resample_map_blocks=True):
            cp = atmos.chill_portions(tas, date_bounds=("09-01", "03-30"), freq="YS-JUL")

        assert cp.attrs["units"] == "1"
        assert cp.name == "cp"
        # Although its 4 years of data its 5 seasons starting in July
        assert cp.time.size == 5

        # Values are confirmed with chillR package although not an exact match
        # due to implementation details
        exp = [np.nan, 99.91534493, 92.5473925, 99.03177047, np.nan]
        np.testing.assert_allclose(cp.isel(location=0), exp, rtol=1e-03)


def test_water_cycle_intensity(pr_series, evspsbl_series):
    pr = pr_series(np.ones(31))
    evspsbl = evspsbl_series(np.ones(31))

    wci = atmos.water_cycle_intensity(pr=pr, evspsbl=evspsbl, freq="MS")
    np.testing.assert_allclose(wci, 2 * 60 * 60 * 24 * 31)


def test_simple(pr_hr_series, evspsblpot_hr_series):
    # 2 years of hourly data
    pr = np.ones(8760 * 2)
    pet = np.ones(8760 * 2) * 0.8

    # Year 1 different
    pr[1:8761] = 3
    pet[1:8761] = 1.5

    # Create a daily time index
    pr = pr_hr_series(pr)
    pet = evspsblpot_hr_series(pet)
    out = atmos.aridity_index(pr, pet)

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)
