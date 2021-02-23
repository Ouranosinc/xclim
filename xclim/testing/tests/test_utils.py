#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test for utils
import os
from inspect import signature

import numpy as np
import xarray as xr

from xclim.core.indicator import Daily
from xclim.core.utils import ensure_chunk_size, walk_map, wrapped_partial
from xclim.testing import open_dataset


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


def test_wrapped_indicator(tas_series):
    def indice(tas, tas2=None, thresh=0, freq="YS"):
        if tas2 is None:
            out = tas < thresh
        else:
            out = tas < tas2
        out = out.resample(time="YS").sum()
        out.attrs["units"] = "days"
        return out

    ind1 = Daily(
        realm="atmos",
        identifier="test_ind1",
        nvar=1,
        units="days",
        compute=wrapped_partial(indice, tas2=None),
    )

    ind2 = Daily(
        realm="atmos",
        identifier="test_ind2",
        nvar=2,
        units="days",
        compute=wrapped_partial(indice, thresh=None),
    )

    tas = tas_series(np.arange(366), start="2000-01-01")
    tas2 = tas_series(1 + np.arange(366), start="2000-01-01")

    assert ind2(tas, tas2) == 366
    assert ind1(tas, thresh=1111) == 366


def test_ensure_chunk_size():
    da = xr.DataArray(np.zeros((20, 21, 20)), dims=("x", "y", "z"))

    out = ensure_chunk_size(da, x=10, y=-1)

    assert da is out

    dac = da.chunk({"x": (1,) * 20, "y": (10, 10, 1), "z": (10, 10)})

    out = ensure_chunk_size(dac, x=3, y=5, z=-1)

    assert out.chunks[0] == (3, 3, 3, 3, 3, 5)
    assert out.chunks[1] == (10, 11)
    assert out.chunks[2] == (20,)


def test_open_testdata():
    ds = open_dataset(
        os.path.join("cmip5", "tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712")
    )
    assert ds.lon.size == 128
