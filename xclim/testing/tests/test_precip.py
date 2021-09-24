import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import atmos, core, set_options
from xclim.core.calendar import percentile_doy
from xclim.testing import open_dataset

K2C = 273.15


class TestRainOnFrozenGround:
    @pytest.mark.parametrize("chunks", [{"time": 366}, None])
    def test_3d_data_with_nans(self, chunks):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

        pr = ds.pr.copy()
        pr.values[1, 10] = np.nan

        if chunks:
            ds = ds.chunk(chunks)

        out = atmos.rain_on_frozen_ground_days(pr, ds.tas, freq="YS")
        np.testing.assert_array_equal(out.sel(location="Montréal"), [np.nan, 4, 5, 3])


class TestPrecipAccumulation:
    # TODO: replace by fixture
    nc_pr = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_pr).pr  # mm/s
        prMM = open_dataset(self.nc_pr).pr
        prMM *= 86400
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan

        out1 = atmos.precip_accumulation(pr, freq="MS")
        out2 = atmos.precip_accumulation(prMM, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.precip_accumulation(pr, freq="MS")

        np.testing.assert_array_almost_equal(out1, out2, 3)
        np.testing.assert_array_almost_equal(out1, out3, 5)

        # check some vector with and without a nan
        x1 = prMM[:31, 0, 0].values

        prTot = x1.sum()

        np.testing.assert_almost_equal(prTot, out1.values[0, 0, 0], 4)

        assert np.isnan(out1.values[0, 1, 0])

        assert np.isnan(out1.values[0, -1, -1])

    def test_with_different_phases(self):
        # test with different phases
        pr = open_dataset(self.nc_pr).pr  # mm/s
        tasmin = open_dataset(self.nc_tasmin).tasmin  # K

        out_tot = atmos.precip_accumulation(pr, freq="MS")
        out_sol = atmos.solid_precip_accumulation(pr, tas=tasmin, freq="MS")
        out_liq = atmos.liquid_precip_accumulation(pr, tas=tasmin, freq="MS")

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)

        assert "solid" in out_sol.long_name
        assert "liquid" in out_liq.long_name
        assert out_sol.standard_name == "lwe_thickness_of_snowfall_amount"

        # With a non-default threshold
        out_sol = atmos.solid_precip_accumulation(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )
        out_liq = atmos.liquid_precip_accumulation(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)


class TestWetDays:
    # TODO: replace by fixture
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_file).pr
        prMM = open_dataset(self.nc_file).pr
        prMM.values *= 86400.0
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan
        pr_min = "5 mm/d"
        out1 = atmos.wetdays(pr, thresh=pr_min, freq="MS")
        out2 = atmos.wetdays(prMM, thresh=pr_min, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.wetdays(pr, thresh=pr_min, freq="MS")

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)

        # check some vector with and without a nan
        x1 = prMM[:31, 0, 0].values

        wd1 = (x1 >= int(pr_min.split(" ")[0])).sum()

        assert wd1 == out1.values[0, 0, 0]

        assert np.isnan(out1.values[0, 1, 0])

        # make sure that vector with all nans gives nans whatever skipna
        assert np.isnan(out1.values[0, -1, -1])
        # assert (np.isnan(wds.values[0, -1, -1]))


class TestWetPrcptot:
    """Testing of prcptot with wet days"""

    def test_simple(self):
        pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr

        thresh = "1 mm/day"
        out = atmos.wet_precip_accumulation(pr, thresh=thresh)

        # Reference value
        t = core.units.convert_units_to(thresh, pr)
        pa = atmos.precip_accumulation(pr.where(pr >= t, 0))
        np.testing.assert_array_equal(out, pa)


class TestDailyIntensity:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_file).pr
        prMM = open_dataset(self.nc_file).pr
        prMM.values *= 86400.0
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan

        # compute with both skipna options
        pr_min = "2 mm/d"
        # dis = daily_pr_intensity(pr, pr_min=pr_min, freq='MS', skipna=True)

        out1 = atmos.daily_pr_intensity(pr, thresh=pr_min, freq="MS")
        out2 = atmos.daily_pr_intensity(prMM, thresh=pr_min, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.daily_pr_intensity(pr, thresh=pr_min, freq="MS")

        np.testing.assert_array_almost_equal(out1, out2, 3)
        np.testing.assert_array_almost_equal(out1, out3, 3)

        x1 = prMM[:31, 0, 0].values

        di1 = x1[x1 >= int(pr_min.split(" ")[0])].mean()
        # buffer = np.ma.masked_invalid(x2)
        # di2 = buffer[buffer >= pr_min].mean()

        assert np.allclose(di1, out1.values[0, 0, 0])
        # assert (np.allclose(di1, dis.values[0, 0, 0]))
        assert np.isnan(out1.values[0, 1, 0])
        # assert (np.allclose(di2, dis.values[0, 1, 0]))
        assert np.isnan(out1.values[0, -1, -1])
        # assert (np.isnan(dis.values[0, -1, -1]))


class TestMaxPrIntensity:
    def test_simple(self, pr_hr_series):
        pr1 = pr_hr_series(np.zeros(366 * 24))
        pr1[10:20] += np.arange(10)
        pr2 = pr_hr_series(np.ones(366 * 24))

        pr = xr.concat([pr1, pr2], dim="site")
        out = atmos.max_pr_intensity(pr, window=2, freq="Y")
        np.testing.assert_array_almost_equal(out.isel(time=0), [8.5 * 3600, 3600])


class TestMax1Day:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_file).pr
        prMM = open_dataset(self.nc_file).pr
        prMM.values *= 86400.0
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan

        out1 = atmos.max_1day_precipitation_amount(pr, freq="MS")
        out2 = atmos.max_1day_precipitation_amount(prMM, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.max_1day_precipitation_amount(pr, freq="MS")

        np.testing.assert_array_almost_equal(out1, out2, 3)
        np.testing.assert_array_almost_equal(out1, out3, 3)

        x1 = prMM[:31, 0, 0].values
        rx1 = x1.max()

        assert np.allclose(rx1, out1.values[0, 0, 0])
        # assert (np.allclose(di1, dis.values[0, 0, 0]))
        assert np.isnan(out1.values[0, 1, 0])
        # assert (np.allclose(di2, dis.values[0, 1, 0]))
        assert np.isnan(out1.values[0, -1, -1])
        # assert (np.isnan(dis.values[0, -1, -1]))


class TestMaxNDay:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    @pytest.mark.parametrize(
        "units,factor,chunks",
        [
            ("mm/day", 86400.0, None),
            ("kg m-2 s-1", 1, None),
            ("mm/s", 1, {"time": 73.0}),
        ],
    )
    def test_3d_data_with_nans(self, units, factor, chunks):
        # test with 3d data
        pr1 = open_dataset(self.nc_file).pr
        pr2 = open_dataset(self.nc_file, chunks=chunks).pr
        pr2.values *= factor
        pr2.attrs["units"] = units
        # put a nan somewhere
        pr2.values[10, 1, 0] = np.nan
        pr1.values[10, 1, 0] = np.nan
        wind = 3
        out1 = atmos.max_n_day_precipitation_amount(pr1, window=wind, freq="MS")
        out2 = atmos.max_n_day_precipitation_amount(pr2, window=wind, freq="MS")

        np.testing.assert_array_almost_equal(out1, out2, 3)

        x1 = pr1[:31, 0, 0].values * 86400
        df = pd.DataFrame({"pr": x1})
        rx3 = df.rolling(wind).sum().max()

        assert np.allclose(rx3, out1.values[0, 0, 0])
        assert np.isnan(out1.values[0, 1, 0])
        assert np.isnan(out1.values[0, -1, -1])


class TestMaxConsecWetDays:
    # TODO: replace by fixture
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_file).pr
        prMM = open_dataset(self.nc_file).pr
        prMM.values *= 86400.0
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan
        pr_min = "5 mm/d"
        out1 = atmos.maximum_consecutive_wet_days(pr, thresh=pr_min, freq="MS")
        out2 = atmos.maximum_consecutive_wet_days(prMM, thresh=pr_min, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.maximum_consecutive_wet_days(pr, thresh=pr_min, freq="MS")

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)

        # check some vector with and without a nan
        x1 = prMM[:31, 0, 0] * 0.0
        x1[5:10] = 10
        x1.attrs["units"] = "mm/day"
        cwd1 = atmos.maximum_consecutive_wet_days(x1, freq="MS")

        assert cwd1 == 5

        assert np.isnan(out1.values[0, 1, 0])

        # make sure that vector with all nans gives nans whatever skipna
        assert np.isnan(out1.values[0, -1, -1])
        # assert (np.isnan(wds.values[0, -1, -1]))


class TestMaxConsecDryDays:
    # TODO: replace by fixture
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        pr = open_dataset(self.nc_file).pr
        prMM = open_dataset(self.nc_file).pr
        prMM.values *= 86400.0
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan
        pr_min = "5 mm/d"
        out1 = atmos.maximum_consecutive_dry_days(pr, thresh=pr_min, freq="MS")
        out2 = atmos.maximum_consecutive_dry_days(prMM, thresh=pr_min, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.maximum_consecutive_dry_days(pr, thresh=pr_min, freq="MS")

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)

        # check some vector with and without a nan
        x1 = prMM[:31, 0, 0] * 0.0 + 50.0
        x1[5:10] = 0
        x1.attrs["units"] = "mm/day"
        cdd1 = atmos.maximum_consecutive_dry_days(x1, freq="MS")

        assert cdd1 == 5

        assert np.isnan(out1.values[0, 1, 0])

        # make sure that vector with all nans gives nans whatever skipna
        assert np.isnan(out1.values[0, -1, -1])
        # assert (np.isnan(wds.values[0, -1, -1]))


class TestSnowfallDate:
    tasmin_file = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"
    pr_file = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"

    def get_snowfall(self):
        dnr = xr.merge((open_dataset(self.pr_file), open_dataset(self.tasmin_file)))
        return atmos.snowfall_approximation(
            dnr.pr, tas=dnr.tasmin, thresh="-0.5 degC", method="binary"
        )

    def test_first_snowfall(self):
        with set_options(check_missing="skip"):
            fs = atmos.first_snowfall(prsn=self.get_snowfall(), thresh="0.5 mm/day")

        np.testing.assert_array_equal(
            fs[:, [0, 45, 82], [10, 105, 155]],
            np.array(
                [
                    [[1, 1, 1], [1, 1, 1], [11, np.nan, np.nan]],
                    [[254, 256, 277], [274, 292, 275], [300, np.nan, np.nan]],
                ]
            ),
        )

    def test_last_snowfall(self):
        with set_options(check_missing="skip"):
            ls = atmos.last_snowfall(prsn=self.get_snowfall(), thresh="0.5 mm/day")

        np.testing.assert_array_equal(
            ls[:, [0, 45, 82], [10, 105, 155]],
            np.array(
                [
                    [[155, 151, 129], [127, 157, 110], [106, np.nan, np.nan]],
                    [[365, 363, 363], [365.0, 364, 364], [362, np.nan, np.nan]],
                ]
            ),
        )


class TestDaysWithSnow:
    def test_simple(self, prsn_series):
        prsn = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").prsn
        out = atmos.days_with_snow(prsn, low="0 kg m-2 s-1")
        np.testing.assert_array_equal(out[1], [np.nan, 224, 263, 123, np.nan])


def test_days_over_precip_thresh():
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = percentile_doy(pr, window=5, per=80)

    out1 = atmos.days_over_precip_thresh(pr, per)
    np.testing.assert_array_equal(out1[1, :, 0], np.array([81, 61, 69, 78]))

    out2 = atmos.days_over_precip_thresh(pr, per, thresh="2 mm/d")
    np.testing.assert_array_equal(out2[1, :, 0], np.array([81, 61, 66, 78]))

    assert "only days with at least 2 mm/d are counted." in out2.description


def test_fraction_over_precip_thresh():
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = percentile_doy(pr, window=5, per=80)

    out = atmos.fraction_over_precip_thresh(pr, per)
    np.testing.assert_allclose(
        out[1, :, 0], np.array([0.809, 0.770, 0.748, 0.807]), atol=0.001
    )

    out = atmos.fraction_over_precip_thresh(pr, per, thresh="0.002 m/d")
    np.testing.assert_allclose(
        out[1, :, 0], np.array([0.831, 0.803, 0.774, 0.833]), atol=0.001
    )

    assert "only days with at least 0.002 m/d are included" in out.description


def test_liquid_precip_ratio():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    out = atmos.liquid_precip_ratio(pr=ds.pr, tas=ds.tas, thresh="0 degC", freq="YS")
    np.testing.assert_allclose(
        out[:, 0], np.array([0.919, 0.805, 0.525, 0.740, 0.993]), atol=1e3
    )

    with set_options(cf_compliance="raise"):
        # Test if tasmax is allowed
        out = atmos.liquid_precip_ratio(
            pr=ds.pr, tas=ds.tasmax, thresh="33 degF", freq="YS"
        )
        np.testing.assert_allclose(
            out[:, 0], np.array([0.975, 0.921, 0.547, 0.794, 0.999]), atol=1e3
        )
        assert "where temperature is above 33 degf." in out.description


def test_dry_spell():
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr

    events = atmos.dry_spell_frequency(pr, thresh="3 mm", window=7, freq="YS")
    total_d = atmos.dry_spell_total_length(pr, thresh="3 mm", window=7, freq="YS")

    np.testing.assert_allclose(events[0:2, 0], [5, 8], rtol=1e-1)
    np.testing.assert_allclose(total_d[0:2, 0], [50, 67], rtol=1e-1)

    assert (
        "The annual number of dry periods of 7 days and more, during which the accumulated "
        "precipitation on a window of 7 days is under 3 mm."
    ) in events.description
    assert (
        "The annual number of days in dry periods of 7 days and more"
        in total_d.description
    )
