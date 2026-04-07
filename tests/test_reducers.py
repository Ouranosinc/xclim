"""Tests for reducers."""

from __future__ import annotations

import numpy as np
import pytest

from xclim.indices import reducers


class TestDoyMinMax:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_doyminmax(self, q_series, use_dask):
        a = np.ones(365)
        a[9] = 2
        a[19] = -2
        a[39] = 4
        a[49] = -4
        q = q_series(a)
        if use_dask:
            q = q.chunk({"time": 200})
        dmx = reducers.doymax(q)
        dmn = reducers.doymin(q)
        assert dmx.values == [40]
        assert dmn.values == [50]

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_doyminmax_novariance(self, q_series, use_dask):
        q = q_series(np.ones(365))
        if use_dask:
            q = q.chunk({"time": 200})
        dmx = reducers.doymax(q).load()
        dmn = reducers.doymin(q).load()
        assert dmx.isnull().all()
        assert dmn.isnull().all()

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_doyminmax_allna(self, q_series, use_dask):
        q = q_series(np.ones(365)) * np.nan
        if use_dask:
            q = q.chunk({"time": 200})
        dmx = reducers.doymax(q).load()
        dmn = reducers.doymin(q).load()
        assert dmx.isnull().all()
        assert dmn.isnull().all()
