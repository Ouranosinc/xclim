from __future__ import annotations

import numpy as np
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
