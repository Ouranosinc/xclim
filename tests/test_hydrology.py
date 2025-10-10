from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xclim import indices as xci
from xclim import land
from xclim.core.units import convert_units_to


class TestBaseFlowIndex:
    def test_simple(self, q_series):
        a = np.zeros(365) + 10
        a[10:17] = 1
        q = q_series(a)
        out = xci.base_flow_index(q)
        np.testing.assert_array_equal(out, 1.0 / a.mean())


class TestRBIndex:
    def test_simple(self, q_series):
        a = np.zeros(365)
        a[10] = 10
        q = q_series(a)
        out = xci.rb_flashiness_index(q)
        np.testing.assert_array_equal(out, 2)


class TestStandardizedStreamflow:
    @pytest.mark.slow
    def test_3d_data_with_nans(self, open_dataset):
        # test with data
        ds = open_dataset("Raven/q_sim.nc")
        q = ds.q_obs.sel(time=slice("2008")).rename("q")
        qMM = convert_units_to(q, "mm**3/s", context="hydro")  # noqa
        # put a nan somewhere
        qMM.values[10] = np.nan
        q.values[10] = np.nan

        out1 = land.standardized_streamflow_index(
            q,
            freq="MS",
            window=1,
            dist="genextreme",
            method="APP",
            fitkwargs={"floc": 0},
        )
        out2 = land.standardized_streamflow_index(
            qMM,
            freq="MS",
            window=1,
            dist="genextreme",
            method="APP",
            fitkwargs={"floc": 0},
        )
        np.testing.assert_array_almost_equal(out1, out2, 3)

    @pytest.mark.slow
    def test_3d_data_with_nans_value(self, open_dataset):
        # test with data
        ds = open_dataset("Raven/q_sim.nc")
        q = ds.q_obs.sel(time=slice("2008", "2018")).rename("q")
        q[{"time": 10}] = np.nan

        out1 = land.standardized_streamflow_index(
            q,
            freq=None,
            window=1,
            dist="genextreme",
            method="APP",
            fitkwargs={"floc": 0},
        )
        assert np.isnan(out1[{"time": 10}])


class TestSnwMax:
    def test_simple(self, snw_series):
        a = np.zeros(366)
        a[10:20] = np.arange(0, 10)
        snw = snw_series(a, start="1999-01-01")
        out = xci.snw_max(snw, freq="YS")
        np.testing.assert_array_equal(out, [9, 0])
        assert out.units == "kg m-2"


class TestSnwMaxDoy:
    def test_simple(self, snw_series):
        a = np.zeros(366)
        a[10] = 10
        snw = snw_series(a, start="1999-01-01")
        out = xci.snw_max_doy(snw, freq="YS")
        np.testing.assert_array_equal(out, [11, np.nan])
        assert out.attrs["units"] == "1"


class TestSnowMeltWEMax:
    def test_simple(self, snw_series):
        a = np.zeros(365)
        a[10:20] = np.arange(0, 10)
        a[20:25] = np.arange(10, 0, -2)
        snw = snw_series(a, start="1999-07-01")
        out = xci.snow_melt_we_max(snw)
        np.testing.assert_array_equal(out, 6)
        assert out.units == "kg m-2"


class TestMeltandPrecipMax:
    def test_simple(self, snw_series, pr_series):
        a = np.zeros(365)

        # 1 km / m2 of melt on day 11.
        a[10] = 1
        snw = snw_series(a, start="1999-07-01")

        # 1 kg/ m2 /d of rain on day 11
        b = np.zeros(365)
        b[11] = 1.0 / 60**2 / 24
        pr = pr_series(b, start="1999-07-01")

        out = xci.melt_and_precip_max(snw, pr)
        np.testing.assert_array_equal(out, 2)
        assert out.units == "kg m-2"


class TestFlowindex:
    def test_simple(self, q_series):
        a = np.ones(365 * 2) * 10
        a[10:50] = 50
        q = q_series(a)
        out = xci.flow_index(q, 0.95)
        np.testing.assert_array_equal(out, 5)


class TestHighflowfrequency:
    def test_simple(self, q_series):
        a = np.zeros(365 * 2)
        a[50:60] = 10
        a[200:210] = 20
        q = q_series(a)
        out = xci.high_flow_frequency(q, 9, freq="YS")
        np.testing.assert_array_equal(out, [20, 0])


class TestLowflowfrequency:
    def test_simple(self, q_series):
        a = np.ones(365 * 2) * 10
        a[50:60] = 1
        a[200:210] = 1
        q = q_series(a)
        out = xci.low_flow_frequency(q, 0.2, freq="YS")

        np.testing.assert_array_equal(out, [20, 0])


class TestAntecedentPrecipitationIndex:
    def test_simple(self, pr_series):
        a = np.ones(50) * 10
        a[15:20] = 20
        a[35:40] = 0
        pr = pr_series(a, units="mm d-1")
        out = xci.antecedent_precipitation_index(pr)
        np.testing.assert_allclose(out.max(), [101.65], atol=1e-2)
        np.testing.assert_allclose(out.min(), [13.83], atol=1e-2)

    def test_nan_present(self, pr_series):
        a = np.ones(50) * 10
        a[25] = np.nan
        pr = pr_series(a, units="mm d-1")
        window = 7
        out = xci.antecedent_precipitation_index(pr, window=window, p_exp=0.935)
        np.testing.assert_array_equal(out[25], [np.nan])

    def test_nan_start_window(self, pr_series):
        a = np.ones(50) * 10
        pr = pr_series(a, units="mm d-1")
        window = 7
        out = xci.antecedent_precipitation_index(pr, window=window, p_exp=0.935)
        np.testing.assert_array_equal(out[: window - 1], np.nan)

    def test_manual_calc(self, pr_series):
        a = np.ones(10) * 10
        pr = pr_series(a, units="mm d-1")
        window = 7
        p_exp = 0.935
        out = xci.antecedent_precipitation_index(pr, window=window, p_exp=p_exp)

        out_manual = np.zeros(out.shape) * np.nan
        for idx in range(pr.shape[0] - window + 1):
            idxend = window + idx
            weights = list(reversed([p_exp ** (ii + 1 - 1) for ii in range(window)]))
            weighted_sum = (pr[idx:idxend] * weights).sum()
            out_manual[idxend - 1] = weighted_sum
        np.testing.assert_allclose(out, out_manual, atol=1e-7)


class TestRR:
    def test_simple(self, q_series, area_series, pr_series):
        # 1 years of daily data
        q = np.ones(365, dtype=float) * 10
        pr = np.ones(365, dtype=float) * 20

        # 30 days with low flows, ratio should stay the same
        q[300:330] = 5
        pr[270:300] = 10
        a = 1000
        a = area_series(a)

        q = q_series(q)
        new_start = "2000-07-01"
        q_shifted = q.assign_coords(time=pd.date_range(new_start, periods=q.sizes["time"], freq="D"))

        pr = pr_series(pr, units="mm/hr")

        out = xci.runoff_ratio(q_shifted, a, pr, freq="YS")
        # verify RR
        np.testing.assert_allclose(out.values, 0.0018, atol=1e-15)


class TestDaysWithSnowpack:
    def test_simple(self, swe_series):
        # 2 years of daily data
        a = np.zeros(365 * 2)

        # Year 1: 15 days of SWE = 20 mm
        a[50:65] = 20
        # Year 2: 5 days of SWE = 5 mm
        a[400:405] = 5

        # Create a daily time index
        swe = swe_series(a)

        out = xci.days_with_snowpack(swe, thresh=".01 m", freq="YS")

        # Year 1: 15 days >= 10 → expect 15, Year 2: only 5 days but all < 10 → expect 0
        np.testing.assert_array_equal(out.values, [15, 0])


class TestAnnualAridityIndex:
    def test_simple(self, pr_hr_series, evspsblpot_hr_series):
        # 2 years of hourly data
        pr = np.ones(8760 * 2)
        pet = np.ones(8760 * 2) * 0.8

        # Year 1 different
        pr[1:8761] = 3
        pet[1:8761] = 1.5

        # Create a daily time index
        pr = pr_hr_series(pr)
        pet = evspsblpot_hr_series(pet)

        out = xci.aridity_index(pr, pet)
        np.testing.assert_allclose(out, [2.0, 1.25], rtol=1e-3, atol=0)


class TestLagSnowpackFlowPeaks:
    def test_simple(self, swe_series, q_series):
        # 1 years of daily data (2 values due to freq resampling)
        a = np.zeros(365)

        # Year 1: 1 day of SWE = 20 mm
        a[50:51] = 20
        # Year 2: 1 day of SWE = 5 mm
        a[300:301] = 5

        # Create a daily time index
        swe = swe_series(a)

        b = np.zeros(365)
        # Year 1: 35 days of high flows directly after max swe
        b[50:85] = 20
        # Year 2: 35 days of high flows 10 days after max swe
        b[310:345] = 5

        # Create a daily time index
        q = q_series(b)

        out = xci.lag_snowpack_flow_peaks(swe, q)
        np.testing.assert_allclose(out, [17.0, 27.0], atol=1e-14)


class TestSenSlope:
    def test_simple(self, q_series):
        # 5 years of increasing data with slope of 1
        q = np.arange(1, 1826)

        # 5 years of increasing data with slope of 2
        qsim = np.arange(1, 1826) * 2

        # Create a daily time index
        q = q_series(q)
        qsim = q_series(qsim)

        out = xci.sen_slope(q, qsim)

        # verify Sen_slopes
        Sen_slope_obs = out["Sen_slope_obs"]
        np.testing.assert_allclose(Sen_slope_obs.values, [360.0, 365.0, 365.0, 365.0, 360.0], atol=1e-15)

        Sen_slope_sim = out["Sen_slope_sim"]
        np.testing.assert_allclose(Sen_slope_sim.values, [720.0, 730.0, 730.0, 730.0, 720.0], atol=1e-15)

        # verify p-values
        p_value_obs = out["p_value_obs"]
        np.testing.assert_allclose(
            p_value_obs.values, [0.008535, 0.027486, 0.027486, 0.027486, 0.008535], rtol=1e-06, atol=1e-06
        )

        p_value_sim = out["p_value_sim"]
        np.testing.assert_allclose(
            p_value_sim.values, [0.008535, 0.027486, 0.027486, 0.027486, 0.008535], rtol=1e-06, atol=1e-06
        )

        # verify ratio
        ratio = out["ratio"]
        np.testing.assert_allclose(ratio.values, [0.5, 0.5, 0.5, 0.5, 0.5], atol=1e-15)
