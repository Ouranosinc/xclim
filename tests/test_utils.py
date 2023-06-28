#!/usr/bin/env python
# Test for utils
from __future__ import annotations

from inspect import signature

import numpy as np
import xarray as xr

from xclim.core.utils import (
    ensure_chunk_size,
    nan_calc_percentiles,
    walk_map,
    wrapped_partial,
)


def test_walk_map():
    d = {"a": -1, "b": {"c": -2}}
    o = walk_map(d, lambda x: 0)
    assert o["a"] == 0
    assert o["b"]["c"] == 0


def test_wrapped_partial():
    def func(a, b=1, c=1):
        """Docstring"""
        return (a, b, c)

    newf = wrapped_partial(func, b=2)
    assert list(signature(newf).parameters.keys()) == ["a", "c"]
    assert newf(1) == (1, 2, 1)

    newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    assert list(signature(newf).parameters.keys()) == ["a", "c"]
    assert newf(1) == (1, 2, 2)
    assert newf.__doc__ == func.__doc__

    def func(a, b=1, c=1, **kws):
        """Docstring"""
        return (a, b, c)

    newf = wrapped_partial(func, suggested=dict(c=2), a=2, b=2)
    assert list(signature(newf).parameters.keys()) == ["c", "kws"]
    assert newf() == (2, 2, 2)


def test_ensure_chunk_size():
    da = xr.DataArray(np.zeros((20, 21, 20)), dims=("x", "y", "z"))

    out = ensure_chunk_size(da, x=10, y=-1)

    assert da is out

    dac = da.chunk({"x": (1,) * 20, "y": (10, 10, 1), "z": (10, 10)})

    out = ensure_chunk_size(dac, x=3, y=5, z=-1)

    assert out.chunks[0] == (3, 3, 3, 3, 3, 5)
    assert out.chunks[1] == (10, 11)
    assert out.chunks[2] == (20,)


class TestNanCalcPercentiles:
    def test_calc_perc_type7(self):
        # Exemple array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0])
        res = nan_calc_percentiles(arr, percentiles=[40.0], alpha=1, beta=1)
        # The expected is from R `quantile(arr, probs=c(0.4), type=7)`
        assert res[()] == 29

    def test_calc_perc_type8(self):
        # Example array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray(
            [[15.0, 20.0, 35.0, 40.0, 50.0], [15.0, 20.0, 35.0, 40.0, 50.0]]
        )
        res = nan_calc_percentiles(
            arr,
            percentiles=[40.0],
            alpha=1.0 / 3.0,
            beta=1.0 / 3.0,
        )
        # The expected is from R `quantile(arr, probs=c(0.4), type=8)`
        assert np.all(res[0][0] == 27)
        assert np.all(res[0][1] == 27)

    def test_calc_perc_2d(self):
        # Exemple array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray(
            [[15.0, 20.0, 35.0, 40.0, 50.0], [15.0, 20.0, 35.0, 40.0, 50.0]]
        )
        res = nan_calc_percentiles(arr, percentiles=[40.0])
        # The expected is from R ` quantile(c(15.0, 20.0, 35.0, 40.0, 50.0), probs=0.4)`
        assert np.all(res[0][0] == 29)
        assert np.all(res[0][1] == 29)

    def test_calc_perc_nan(self):
        arr = np.asarray([np.NAN])
        res = nan_calc_percentiles(arr, percentiles=[50.0])
        assert np.isnan(res)

    def test_calc_perc_empty(self):
        arr = np.asarray([])
        res = nan_calc_percentiles(arr)
        assert np.isnan(res)

    def test_calc_perc_partial_nan(self):
        arr = np.asarray([np.NaN, 41.0, 41.0, 43.0, 43.0])
        res = nan_calc_percentiles(arr, percentiles=[50.0], alpha=1 / 3.0, beta=1 / 3.0)
        # The expected is from R `quantile(arr, 0.5, type=8, na.rm = TRUE)`
        # Note that scipy mquantiles would give a different result here
        assert res[()] == 42.0
