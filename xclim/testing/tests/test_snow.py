import numpy as np

from xclim import land


class TestSnowCoverDuration:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")
        out = land.snow_cover_duration(snd, freq="M")
        assert out.units == "days"
        np.testing.assert_array_equal(out, [31, 28, 31, np.nan])


class TestContinuousSnowCoverStartEnd:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        a[100:200] = 0.03
        snd = snd_series(a, start="2001-07-01")
        out = land.continuous_snow_cover_start(snd)
        assert out.units == ""
        np.testing.assert_array_equal(out, snd.time.dt.dayofyear[100])

        out = land.continuous_snow_cover_end(snd)
        assert out.units == ""
        np.testing.assert_array_equal(out, snd.time.dt.dayofyear[200])
