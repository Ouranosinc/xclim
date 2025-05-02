from __future__ import annotations

from pathlib import Path

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
        nc_ds = Path("Raven", "q_sim.nc")
        # test with data
        ds = open_dataset(nc_ds)
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
        nc_ds = Path("Raven", "q_sim.nc")
        # test with data
        ds = open_dataset(nc_ds)
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
