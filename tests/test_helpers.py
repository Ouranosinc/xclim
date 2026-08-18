from __future__ import annotations

import pytest
import xarray as xr

from xclim.core.options import set_options
from xclim.core.utils import uses_dask
from xclim.indices import helpers
from xclim.testing.helpers import assert_lazy


@pytest.mark.parametrize(["in_chunks", "exp_chunks"], [(60, 6 * (2,)), (30, 12 * (1,)), (-1, (12,))])
def test_resample_map(tas_series, in_chunks, exp_chunks):
    pytest.importorskip("flox")
    tas = tas_series(365 * [1]).chunk(time=in_chunks)
    with assert_lazy:
        out = helpers.resample_map(tas, "time", "MS", lambda da: da.mean("time"), map_blocks=True)
    assert out.chunks[0] == exp_chunks
    out.load()  # Trigger compute to see if it actually works


def test_resample_map_dataset(tas_series, pr_series):
    pytest.importorskip("flox")
    tas = tas_series(3 * 365 * [1], start="2000-01-01").chunk(time=365)
    pr = pr_series(3 * 365 * [1], start="2000-01-01").chunk(time=365)
    ds = xr.Dataset({"pr": pr, "tas": tas})
    with set_options(resample_map_blocks=True):
        with assert_lazy:
            out = helpers.resample_map(
                ds,
                "time",
                "YS",
                lambda da: da.mean("time"),
            )
    assert out.chunks["time"] == (1, 1, 1)
    out.load()


def test_resample_map_passthrough(tas_series):
    tas = tas_series(365 * [1])
    with assert_lazy:
        out = helpers.resample_map(tas, "time", "MS", lambda da: da.mean("time"))
    assert not uses_dask(out)
