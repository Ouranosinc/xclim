import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
