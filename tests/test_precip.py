from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim.indices as xci
from xclim import atmos, core, set_options
from xclim.core.calendar import build_climatology_bounds, percentile_doy
from xclim.core.units import convert_units_to

K2C = 273.15


class TestRainOnFrozenGround:
    @pytest.mark.parametrize("chunks", [{"time": 366}, None])
    def test_3d_data_with_nans(self, open_dataset, chunks):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

        pr = ds.pr.copy()
        pr.values[1, 10] = np.nan

        if chunks:
            ds = ds.chunk(chunks)

        out = atmos.rain_on_frozen_ground_days(pr, ds.tas, freq="YS")
        np.testing.assert_array_equal(out.sel(location="Montréal"), [np.nan, 4, 5, 3])


class TestRainSeason:
    # @pytest.mark.parametrize("chunks", [{"time": 366}, None])
    def test_3d_data_with_nans(self, open_dataset):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

        pr = ds.pr.isel(location=0).copy()
        pr[{"time": [0, 10, 100]}] = np.nan
        out = {}
        out["start"], out["end"], out["length"] = atmos.rain_season(
            pr,
            freq="AS-JAN",
            window_dry_end=5,
            date_min_start="01-01",
            date_min_end="01-01",
        )
        out_arr = np.array([out[var].values for var in ["start", "end", "length"]])
        out_exp = np.array(
            [
                [np.nan, 12.0, 6.0, 27.0],
                [np.nan, np.nan, 141.0, np.nan],
                [np.nan, 354.0, 135.0, 339.0],
            ]
        )
        np.testing.assert_array_equal(out_arr, out_exp)


class TestPrecipAccumulation:
    nc_pr = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
    nc_tasmin = Path("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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

        pr_tot = x1.sum()

        np.testing.assert_almost_equal(pr_tot, out1.values[0, 0, 0], 4)

        assert np.isnan(out1.values[0, 1, 0])
        assert np.isnan(out1.values[0, -1, -1])

    def test_with_different_phases(self, open_dataset):
        # test with different phases
        pr = open_dataset(self.nc_pr).pr  # mm/s
        tasmin = open_dataset(self.nc_tasmin).tasmin  # K

        out_tot = atmos.precip_accumulation(pr, freq="MS")
        out_sol = atmos.solid_precip_accumulation(pr, tas=tasmin, freq="MS")
        out_liq = atmos.liquid_precip_accumulation(pr, tas=tasmin, freq="MS")

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)

        assert "solid" in out_sol.description
        assert "liquid" in out_liq.description
        assert out_sol.standard_name == "lwe_thickness_of_snowfall_amount"

        # With a non-default threshold
        out_sol = atmos.solid_precip_accumulation(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )
        out_liq = atmos.liquid_precip_accumulation(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)


class TestPrecipAverage:
    nc_pr = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
    nc_tasmin = Path("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
        # test with 3d data
        pr = open_dataset(self.nc_pr).pr  # mm/s
        prMM = open_dataset(self.nc_pr).pr
        prMM *= 86400
        prMM.attrs["units"] = "mm/day"
        # put a nan somewhere
        prMM.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan

        out1 = atmos.precip_average(pr, freq="MS")
        out2 = atmos.precip_average(prMM, freq="MS")

        # test kg m-2 s-1
        pr.attrs["units"] = "kg m-2 s-1"
        out3 = atmos.precip_average(pr, freq="MS")

        np.testing.assert_array_almost_equal(out1, out2, 3)
        np.testing.assert_array_almost_equal(out1, out3, 5)

        # check some vector with and without a nan
        x1 = prMM[:31, 0, 0].values

        pr_mean = x1.mean()

        np.testing.assert_almost_equal(pr_mean, out1.values[0, 0, 0], 4)

        assert np.isnan(out1.values[0, 1, 0])

        assert np.isnan(out1.values[0, -1, -1])

    def test_with_different_phases(self, open_dataset):
        # test with different phases
        pr = open_dataset(self.nc_pr).pr  # mm/s
        tasmin = open_dataset(self.nc_tasmin).tasmin  # K

        out_tot = atmos.precip_average(pr, freq="MS")
        out_sol = atmos.solid_precip_average(pr, tas=tasmin, freq="MS")
        out_liq = atmos.liquid_precip_average(pr, tas=tasmin, freq="MS")

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)

        assert "solid" in out_sol.description
        assert "liquid" in out_liq.description
        assert out_sol.standard_name == "lwe_average_of_snowfall_amount"

        # With a non-default threshold
        out_sol = atmos.solid_precip_average(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )
        out_liq = atmos.liquid_precip_average(
            pr, tas=tasmin, thresh="40 degF", freq="MS"
        )

        np.testing.assert_array_almost_equal(out_liq + out_sol, out_tot, 4)


class TestStandardizedPrecip:
    nc_ds = Path("sdba", "CanESM2_1950-2100.nc")

    @pytest.mark.slow
    def test_3d_data_with_nans(self, open_dataset):
        # test with data
        ds = open_dataset(self.nc_ds)
        pr = ds.pr.sel(time=slice("2000"))  # kg m-2 s-1
        prMM = convert_units_to(pr, "mm/day", context="hydro")  # noqa
        # put a nan somewhere
        prMM.values[10] = np.nan
        pr.values[10] = np.nan

        out1 = atmos.standardized_precipitation_index(
            pr,
            pr_cal=pr,
            freq="MS",
            window=1,
            dist="gamma",
            method="APP",
        )
        out2 = atmos.standardized_precipitation_index(
            prMM,
            pr_cal=prMM,
            freq="MS",
            window=1,
            dist="gamma",
            method="APP",
        )
        np.testing.assert_array_almost_equal(out1, out2, 3)

        # preparing water_budget for SPEI test
        with xr.set_options(keep_attrs=True):
            tasmax = ds.tasmax
            tas = tasmax - 2.5
            tasmin = tasmax - 5
            wb = xci.water_budget(pr, None, tasmin, tasmax, tas, None)
            wbMM = convert_units_to(wb, "mm/day", context="hydro")  # noqa

        out3 = atmos.standardized_precipitation_evapotranspiration_index(
            wb,
            wb_cal=wb,
            freq="MS",
            window=1,
            dist="gamma",
            method="APP",
            # method="ML",
        )
        out4 = atmos.standardized_precipitation_evapotranspiration_index(
            wbMM,
            wb_cal=wbMM,
            freq="MS",
            window=1,
            dist="gamma",
            method="APP",
            # method="ML",
        )

        np.testing.assert_array_almost_equal(out3, out4, 3)


class TestWetDays:
    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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

    def test_simple(self, atmosds):
        pr = atmosds.pr

        thresh = "1 mm/day"
        out = atmos.wet_precip_accumulation(pr, thresh=thresh)

        # Reference value
        t = core.units.convert_units_to(thresh, pr, context="hydro")
        pa = atmos.precip_accumulation(pr.where(pr >= t, 0))
        np.testing.assert_array_equal(out, pa)


class TestDailyIntensity:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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
        out = atmos.max_pr_intensity(pr, window=2, freq="YS")
        np.testing.assert_array_almost_equal(out.isel(time=0), [8.5 * 3600, 3600])


class TestMax1Day:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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
        assert np.isnan(out1.values[0, 1, 0])
        assert np.isnan(out1.values[0, -1, -1])


class TestMaxNDay:
    # testing of wet_day and daily_pr_intensity, both are related

    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    @pytest.mark.parametrize(
        "units,factor,chunks",
        [
            ("mm/day", 86400.0, None),
            ("kg m-2 s-1", 1, None),
            ("mm/s", 1, {"time": 73.0}),
        ],
    )
    def test_3d_data_with_nans(self, open_dataset, units, factor, chunks):
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
    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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


class TestMaxConsecDryDays:
    nc_file = Path("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_3d_data_with_nans(self, open_dataset):
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


class TestSnowfallDate:
    tasmin_file = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"
    pr_file = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"

    @classmethod
    def get_snowfall(cls, open_dataset):
        dnr = xr.merge((open_dataset(cls.pr_file), open_dataset(cls.tasmin_file)))
        return atmos.snowfall_approximation(
            dnr.pr, tas=dnr.tasmin, thresh="-0.5 degC", method="binary"
        )

    def test_first_snowfall(self, open_dataset):
        with set_options(check_missing="skip"):
            fs = atmos.first_snowfall(
                prsn=self.get_snowfall(open_dataset), thresh="0.5 mm/day"
            )

        np.testing.assert_array_equal(
            fs[:, [0, 45, 82], [10, 105, 155]],
            np.array(
                [
                    [[1, 1, 1], [1, 1, 1], [11, np.nan, np.nan]],
                    [[254, 256, 277], [274, 292, 275], [300, np.nan, np.nan]],
                ]
            ),
        )

    def test_last_snowfall(self, open_dataset):
        with set_options(check_missing="skip"):
            ls = atmos.last_snowfall(
                prsn=self.get_snowfall(open_dataset), thresh="0.5 mm/day"
            )

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
    def test_simple(self, open_dataset, prsn_series):
        prsn = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").prsn
        out = atmos.days_with_snow(prsn, low="0 kg m-2 s-1", high="1e6 kg m-2 s-1")
        np.testing.assert_array_equal(out[1], [np.nan, 162, 159, 126, np.nan])


def test_days_over_precip_doy_thresh(open_dataset):
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = percentile_doy(pr, window=5, per=80)

    out1 = atmos.days_over_precip_doy_thresh(pr, per)
    np.testing.assert_array_equal(out1[1, :, 0], np.array([81, 60, 68, 78]))

    out2 = atmos.days_over_precip_doy_thresh(pr, per, thresh="2 mm/d")
    np.testing.assert_array_equal(out2[1, :, 0], np.array([80, 59, 66, 78]))

    assert "only days with at least 2 mm/d are counted." in out2.description
    assert "[80]th percentile" in out2.attrs["description"]
    assert "['1990-01-01', '1993-12-31'] period" in out2.attrs["description"]
    assert "5 day(s)" in out2.attrs["description"]


def test_days_over_precip_thresh(open_dataset):
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = pr.quantile(0.8, "time", keep_attrs=True)
    per.attrs["climatology_bounds"] = build_climatology_bounds(pr)

    out = atmos.days_over_precip_thresh(pr, per)

    np.testing.assert_allclose(
        out[1, :], np.array([80.0, 64.0, 65.0, 83.0]), atol=0.001
    )
    assert "80.0th percentile" in out.attrs["description"]
    assert "['1990-01-01', '1993-12-31'] period" in out.attrs["description"]


def test_days_over_precip_thresh__seasonal_indexer(open_dataset):
    # GIVEN
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = pr.quantile(0.8, "time", keep_attrs=True)
    # WHEN
    out = atmos.days_over_precip_thresh(
        pr, per, freq="AS", date_bounds=("01-10", "12-31")
    )
    # THEN
    np.testing.assert_almost_equal(out[0], np.array([81.0, 66.0, 66.0, 75.0]))


def test_fraction_over_precip_doy_thresh(open_dataset):
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = percentile_doy(pr, window=5, per=80)

    out = atmos.fraction_over_precip_doy_thresh(pr, per)
    np.testing.assert_allclose(
        out[1, :, 0], np.array([0.803, 0.747, 0.745, 0.806]), atol=0.001
    )

    out = atmos.fraction_over_precip_doy_thresh(pr, per, thresh="0.002 m/d")
    np.testing.assert_allclose(
        out[1, :, 0], np.array([0.822, 0.780, 0.771, 0.829]), atol=0.001
    )

    assert "only days with at least 0.002 m/d are included" in out.description
    assert "[80]th percentile" in out.attrs["description"]
    assert "['1990-01-01', '1993-12-31'] period" in out.attrs["description"]
    assert "5 day(s)" in out.attrs["description"]


def test_fraction_over_precip_thresh(open_dataset):
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    per = pr.quantile(0.8, "time", keep_attrs=True)
    per.attrs["climatology_bounds"] = build_climatology_bounds(pr)

    out = atmos.fraction_over_precip_thresh(pr, per)

    np.testing.assert_allclose(
        out[1, :], np.array([0.839, 0.812, 0.776, 0.864]), atol=0.001
    )

    assert "80.0th percentile" in out.attrs["description"]
    assert "['1990-01-01', '1993-12-31'] period" in out.attrs["description"]


def test_liquid_precip_ratio(open_dataset):
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


def test_dry_spell(atmosds):
    pr = atmosds.pr

    events = atmos.dry_spell_frequency(pr, thresh="3 mm", window=7, freq="YS")
    total_d_sum = atmos.dry_spell_total_length(
        pr, thresh="3 mm", window=7, op="sum", freq="YS"
    )
    total_d_max = atmos.dry_spell_total_length(
        pr, thresh="3 mm", window=7, op="max", freq="YS"
    )
    max_d_sum = atmos.dry_spell_max_length(
        pr, thresh="3 mm", window=7, op="sum", freq="YS"
    )
    max_d_max = atmos.dry_spell_max_length(
        pr, thresh="3 mm", window=7, op="max", freq="YS"
    )
    total_d_sum = total_d_sum.sel(location="Halifax", drop=True).isel(time=slice(0, 2))
    total_d_max = total_d_max.sel(location="Halifax", drop=True).isel(time=slice(0, 2))
    max_d_sum = max_d_sum.sel(location="Halifax", drop=True).isel(time=slice(0, 2))
    max_d_max = max_d_max.sel(location="Halifax", drop=True).isel(time=slice(0, 2))

    np.testing.assert_allclose(events[0:2, 0], [5, 7], rtol=1e-1)
    np.testing.assert_allclose(total_d_sum, [50, 53], rtol=1e-1)
    np.testing.assert_allclose(total_d_max, [68, 97], rtol=1e-1)
    np.testing.assert_allclose(max_d_sum, [14, 10], rtol=1e-1)
    np.testing.assert_allclose(max_d_max, [14, 14], rtol=1e-1)

    assert (
        "The annual number of dry periods of 7 day(s) or more, "
        "during which the total precipitation on a window of 7 day(s) is below 3 mm."
    ) in events.description
    assert (
        "The annual number of days in dry periods of 7 day(s) or more"
        in total_d_sum.description
    )
    assert (
        "The annual number of days in dry periods of 7 day(s) or more"
        in total_d_max.description
    )
    assert (
        "The maximum annual number of consecutive days in a dry period of 7 day(s) or more"
        in max_d_sum.description
    )
    assert (
        "The maximum annual number of consecutive days in a dry period of 7 day(s) or more"
        in max_d_max.description
    )


def test_dry_spell_total_length_indexer(pr_series):
    pr = pr_series(
        [np.NaN] + [1] * 4 + [0] * 10 + [1] * 350, start="1900-01-01", units="mm/d"
    )
    out = atmos.dry_spell_total_length(
        pr,
        window=7,
        op="sum",
        thresh="3 mm",
        freq="MS",
    )
    np.testing.assert_allclose(out, [np.NaN] + [0] * 11)

    out = atmos.dry_spell_total_length(
        pr, window=7, op="sum", thresh="3 mm", freq="MS", date_bounds=("01-10", "12-31")
    )
    np.testing.assert_allclose(out, [9] + [0] * 11)


def test_dry_spell_max_length_indexer(pr_series):
    pr = pr_series(
        [np.NaN] + [1] * 4 + [0] * 10 + [1] * 350, start="1900-01-01", units="mm/d"
    )
    out = atmos.dry_spell_max_length(
        pr,
        window=7,
        op="sum",
        thresh="3 mm",
        freq="MS",
    )
    np.testing.assert_allclose(out, [np.NaN] + [0] * 11)

    out = atmos.dry_spell_total_length(
        pr, window=7, op="sum", thresh="3 mm", freq="MS", date_bounds=("01-10", "12-31")
    )
    np.testing.assert_allclose(out, [9] + [0] * 11)


def test_dry_spell_frequency_op(open_dataset):
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr
    test_sum = atmos.dry_spell_frequency(
        pr, thresh="3 mm", window=7, freq="MS", op="sum"
    )
    test_max = atmos.dry_spell_frequency(
        pr, thresh="3 mm", window=7, freq="MS", op="max"
    )

    np.testing.assert_allclose(
        test_sum[0, :14], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], rtol=1e-1
    )
    np.testing.assert_allclose(
        test_max[0, :14], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 2, 1], rtol=1e-1
    )
    assert (
        "The monthly number of dry periods of 7 day(s) or more, "
        "during which the total precipitation on a window of 7 day(s) is below 3 mm."
    ) in test_sum.description
    assert (
        "The monthly number of dry periods of 7 day(s) or more, "
        "during which the maximal precipitation on a window of 7 day(s) is below 3 mm."
    ) in test_max.description


class TestSnowfallMeteoSwiss:
    tasmin_file = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"
    pr_file = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"

    @classmethod
    def get_snowfall(cls, open_dataset):
        dnr = xr.merge((open_dataset(cls.pr_file), open_dataset(cls.tasmin_file)))
        return atmos.snowfall_approximation(
            dnr.pr, tas=dnr.tasmin, thresh="-0.5 degC", method="binary"
        )

    def test_snowfall_frequency(self, open_dataset):
        prsn = self.get_snowfall(open_dataset)
        with set_options(check_missing="skip"):
            sf = atmos.snowfall_frequency(prsn=prsn, thresh="1 mm/day")
        expected = np.array(
            [
                [
                    [27.624, 29.834, 25.414],
                    [22.652, 25.414, 22.652],
                    [12.155, 0.0, 0.0],
                ],
                [
                    [23.913, 23.370, 20.652],
                    [17.391, 15.761, 13.043],
                    [4.891, 0.0, 0.0],
                ],
            ]
        )
        np.testing.assert_allclose(
            sf[:, [0, 45, 82], [10, 105, 155]],
            expected,
            rtol=1e-3,
        )

    def test_snowfall_intensity(self, open_dataset):
        prsn = self.get_snowfall(open_dataset)
        with set_options(check_missing="skip"):
            si = atmos.snowfall_intensity(prsn=prsn, thresh="1 mm/day")
        expected = np.array(
            [
                [
                    [3.585, 3.839, 4.446],
                    [5.148, 5.884, 5.764],
                    [5.910, 0.0, 0.0],
                ],
                [
                    [5.093, 5.284, 5.539],
                    [5.946, 6.840, 9.702],
                    [9.522, 0.0, 0.0],
                ],
            ]
        )
        np.testing.assert_allclose(
            si[:, [0, 45, 82], [10, 105, 155]],
            expected,
            rtol=1e-3,
        )
