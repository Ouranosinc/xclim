import numpy as np
import pytest

from xclim import land
from xclim.core.utils import ValidationError


class TestSnowDepth:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")
        out = land.snow_depth(snd, freq="M")
        assert out.units == "cm"
        np.testing.assert_array_equal(out, [100, 100, 100, np.nan])


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


class TestSndMaxDoy:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        a[200] = 1
        snd = snd_series(a, start="2001-07-01")
        out = land.snd_max_doy(snd)
        np.testing.assert_array_equal(out, snd.time.dt.dayofyear[200])

    def test_units(self, tas_series):
        """Check that unit declaration works."""
        tas = tas_series(np.random.rand(365), start="1999-07-01")
        with pytest.raises(ValidationError):
            land.snd_max_doy(tas)

    def test_no_snow(self, snd_series):
        snd = snd_series(np.zeros(365), start="2001-07-01")
        out = land.snd_max_doy(snd)
        np.testing.assert_array_equal(out, np.nan)
