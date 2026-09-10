#!/usr/bin/env python
# Test for utils
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim.core.utils import _chunk_like, ensure_chunk_size, nan_calc_percentiles, sel_with_nans
from xclim.testing.helpers import test_timeseries as _test_timeseries


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
        # Example array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0])
        res = nan_calc_percentiles(arr, percentiles=[40.0], alpha=1, beta=1)
        # The expected is from R `quantile(arr, probs=c(0.4), type=7)`
        assert res[()] == 29

    def test_calc_perc_type8(self):
        # Example array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray([[15.0, 20.0, 35.0, 40.0, 50.0], [15.0, 20.0, 35.0, 40.0, 50.0]])
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
        # Example array from: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
        arr = np.asarray([[15.0, 20.0, 35.0, 40.0, 50.0], [15.0, 20.0, 35.0, 40.0, 50.0]])
        res = nan_calc_percentiles(arr, percentiles=[40.0])
        # The expected is from R ` quantile(c(15.0, 20.0, 35.0, 40.0, 50.0), probs=0.4)`
        assert np.all(res[0][0] == 29)
        assert np.all(res[0][1] == 29)

    def test_calc_perc_nan(self):
        arr = np.asarray([np.nan])
        res = nan_calc_percentiles(arr, percentiles=[50.0])
        assert np.isnan(res)

    def test_calc_perc_empty(self):
        arr = np.asarray([])
        res = nan_calc_percentiles(arr)
        assert np.isnan(res)

    def test_calc_perc_partial_nan(self):
        arr = np.asarray([np.nan, 41.0, 41.0, 43.0, 43.0])
        res = nan_calc_percentiles(arr, percentiles=[50.0], alpha=1 / 3.0, beta=1 / 3.0)
        # The expected is from R `quantile(arr, 0.5, type=8, na.rm = TRUE)`
        # Note that scipy mquantiles would give a different result here
        assert res[()] == 42.0


def test_chunk_like():
    da = _test_timeseries(
        np.zeros(
            100,
        ),
        "tas",
    )
    da = xr.concat([da] * 10, xr.DataArray(np.arange(10), dims=("lat",), name="lat"))

    assert isinstance(da.lat.variable, xr.core.variable.IndexVariable)
    t, la = _chunk_like(da.time, da.lat, chunks={"time": 10, "lat": 1})
    assert t.chunks[0] == tuple([10] * 10)
    assert la.chunks[0] == tuple([1] * 10)


@pytest.mark.parametrize(["uses_dask", "lazy"], [(True, True), (True, False), (False, False)])
def test_sel_with_nans(uses_dask, lazy):
    tas = xr.DataArray(
        np.linspace(0, 1, 125).reshape((5, 5, 5)),
        coords={"time": np.arange(5), "lat": np.arange(5), "lon": np.arange(5)},
    )
    time = xr.DataArray([-1, 0, 1, 2, 3, 4, 5, 3, 2, 1, 0], dims=("time"))
    if uses_dask:
        tas = tas.chunk(time=1, lat=2, lon=3)
        time = time.chunk(time=2)

    tas_sel = sel_with_nans(tas, "time", time, fill=-1, lazy=lazy).compute()
    time = time.compute()
    assert (tas_sel.isel(time=[0, 6]) == -1).all()

    assert (tas.isel(time=[0, 1, 2, 3, 4, 3, 2, 1, 0]) == tas_sel.where(time.isin(tas.time), drop=True)).all()
