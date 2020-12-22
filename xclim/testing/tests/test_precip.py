import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import atmos
from xclim.testing import open_dataset

K2C = 273.15


class TestRainOnFrozenGround:
    nc_pr = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    @pytest.mark.parametrize("prunits,prfac", [("kg m-2 s-1", 1), ("mm/day", 86400)])
    @pytest.mark.parametrize(
        "tasunits,tasoffset,chunks", [(None, None, {"time": 73.0}), ("C", K2C, None)]
    )
    def test_3d_data_with_nans(self, prunits, prfac, tasunits, tasoffset, chunks):
        pr = open_dataset(self.nc_pr).pr
        pr2 = pr.copy()
        pr2.values *= prfac
        pr2.attrs["units"] = prunits

        tasmax = open_dataset(self.nc_tasmax, chunks=chunks).tasmax
        tasmin = open_dataset(self.nc_tasmin, chunks=chunks).tasmin
        tas = 0.5 * (tasmax + tasmin)
        tas.attrs = tasmax.attrs
        tas2 = tas.copy()
        if chunks:
            tas2 = tas2.chunk(chunks)
        if tasunits:
            tas2.values -= tasoffset
            tas2.attrs["units"] = tasunits

        pr2.values[10, 1, 0] = np.nan
        pr.values[10, 1, 0] = np.nan

        outref = atmos.rain_on_frozen_ground_days(pr, tas, freq="MS")
        out21 = atmos.rain_on_frozen_ground_days(pr2, tas, freq="MS")
        out22 = atmos.rain_on_frozen_ground_days(pr2, tas2, freq="MS")
        out12 = atmos.rain_on_frozen_ground_days(pr, tas2, freq="MS")

        np.testing.assert_array_equal(outref, out21)
        np.testing.assert_array_equal(outref, out22)
        np.testing.assert_array_equal(outref, out12)

        assert np.isnan(out22.values[0, 1, 0])
        assert np.isnan(out22.values[0, -1, -1])


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
