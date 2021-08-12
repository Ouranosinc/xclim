import numpy as np

from xclim import atmos
from xclim.core.calendar import percentile_doy

K2C = 273


class TestColdAndDry:
    def test_simple(seldf, tas_series, pr_series):
        # GIVEN
        raw_temp = np.full(365 * 4, 20) + K2C
        raw_temp[10:20] -= 10
        ts = tas_series(raw_temp)
        ts_per = percentile_doy(ts, 5, 25).sel(percentiles=25)
        raw_prec = np.full(365 * 4, 10)
        raw_prec[10:20] = 0
        pr = pr_series(raw_prec)
        pr_per = percentile_doy(pr, 5, 25).sel(percentiles=25)
        # WHEN
        result = atmos.cold_and_dry_days(ts, ts_per, pr, pr_per, "MS")
        # THEN january has 10 cold and dry days
        assert result.data[0] == 10


class TestWarmAndDry:
    def test_simple(seldf, tas_series, pr_series):
        # GIVEN
        raw_temp = np.full(365 * 4, 20) + K2C
        raw_temp[10:30] += 10
        ts = tas_series(raw_temp)
        ts_per = percentile_doy(ts, 5, 75).sel(percentiles=75)
        raw_prec = np.full(365 * 4, 10)
        raw_prec[10:30] = 0
        pr = pr_series(raw_prec)
        pr_per = percentile_doy(pr, 5, 25).sel(percentiles=25)
        # WHEN
        result = atmos.warm_and_dry_days(ts, ts_per, pr, pr_per, "MS")
        # THEN january has 20 warm and dry days
        assert result.data[0] == 20


class TestWarmAndWet:
    def test_simple(seldf, tas_series, pr_series):
        # GIVEN
        raw_temp = np.full(365 * 4, 20) + K2C
        raw_temp[10:30] += 10
        ts = tas_series(raw_temp)
        ts_per = percentile_doy(ts, 5, 75).sel(percentiles=75)
        raw_prec = np.full(365 * 4, 10)
        raw_prec[10:30] += 20
        pr = pr_series(raw_prec)
        pr_per = percentile_doy(pr, 5, 75).sel(percentiles=75)
        # WHEN
        result = atmos.warm_and_wet_days(ts, ts_per, pr, pr_per, "MS")
        # THEN january has 20 warm and wet days
        assert result.data[0] == 20


class TestColdAndWet:
    def test_simple(seldf, tas_series, pr_series):
        # GIVEN
        raw_temp = np.full(365 * 4, 20) + K2C
        raw_temp[10:25] -= 20
        ts = tas_series(raw_temp)
        ts_per = percentile_doy(ts, 5, 75).sel(percentiles=75)
        raw_prec = np.full(365 * 4, 10)
        raw_prec[15:30] += 20
        pr = pr_series(raw_prec)
        pr_per = percentile_doy(pr, 5, 75).sel(percentiles=75)
        # WHEN
        result = atmos.cold_and_wet_days(ts, ts_per, pr, pr_per, "MS")
        # THEN january has 10 cold and wet days
        assert result.data[0] == 10
