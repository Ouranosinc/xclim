import numpy as np

from xclim import indices as xci


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
        b[11] = 1.0 / 60 ** 2 / 24
        pr = pr_series(b, start="1999-07-01")

        out = xci.melt_and_precip_max(snw, pr)
        np.testing.assert_array_equal(out, 2)
        assert out.units == "kg m-2"
