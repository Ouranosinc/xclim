from xclim import precip
import xarray as xr
import numpy as np
import os

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')


class TestWetDays():

    # TODO: replace by fixture
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_3d_data_with_nans(self):

        # test with 3d data
        pr = xr.open_dataset(self.nc_file).pr
        pr.values *= 86400.
        pr.attrs['units'] = 'mm/day'
        # put a nan somewhere
        pr.values[10, 1, 0] = np.nan

        wet_days = precip.WetDays()
        pr_min = 5
        wd = wet_days(pr, thresh=pr_min, freq='MS')

        # wds = wet_days(pr, pr_min=pr_min, freq='MS', skipna=False)

        # check some vector with and without a nan
        x1 = pr[:31, 0, 0].values
        # x2 = pr[:31, 1, 0].values
        wd1 = ((x1 >= pr_min) * 1).sum()
        # wd2 = ((x2 >= pr_min) * 1).sum()
        assert (wd1 == wd.values[0, 0, 0])
        # assert (wd1 == wds.values[0, 0, 0])
        assert (np.isnan(wd.values[0, 1, 0]))
        # assert (wd2 == wds.values[0, 1, 0])

        # make sure that vector with all nans gives nans whatever skipna
        assert (np.isnan(wd.values[0, -1, -1]))
        # assert (np.isnan(wds.values[0, -1, -1]))


class TestDailyIntensity():
    # testing of wet_day and daily_intensity, both are related

    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_3d_data_with_nans(self):

        # test with 3d data
        pr = xr.open_dataset(self.nc_file).pr
        pr = pr * 86400.
        pr.attrs['units'] = 'mm/day'
        # put a nan somewhere
        pr.values[10, 1, 0] = np.nan

        # compute with both skipna options
        daily_intensity = precip.DailyIntensity()
        pr_min = 2.
        # dis = daily_intensity(pr, pr_min=pr_min, freq='MS', skipna=True)

        di = daily_intensity(pr, thresh=pr_min, freq='MS')

        x1 = pr[:31, 0, 0].values
        # x2 = pr[:31, 1, 0].values
        # x3 = pr[:31, -1, -1].values

        di1 = x1[x1 >= pr_min].mean()
        # buffer = np.ma.masked_invalid(x2)
        # di2 = buffer[buffer >= pr_min].mean()

        assert (np.allclose(di1, di.values[0, 0, 0]))
        # assert (np.allclose(di1, dis.values[0, 0, 0]))
        assert (np.isnan(di.values[0, 1, 0]))
        # assert (np.allclose(di2, dis.values[0, 1, 0]))
        assert (np.isnan(di.values[0, -1, -1]))
        # assert (np.isnan(dis.values[0, -1, -1]))
