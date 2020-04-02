#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test for utils
from inspect import signature

import numpy as np

from xclim.core.indicator import Indicator
from xclim.core.utils import walk_map
from xclim.core.utils import wrapped_partial


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

    ind1 = Indicator(_nvar=1, units="days", compute=wrapped_partial(indice, tas2=None))
    ind2 = Indicator(
        _nvar=2, units="days", compute=wrapped_partial(indice, thresh=None)
    )

    tas = tas_series(np.arange(366), start="2000-01-01")
    tas2 = tas_series(1 + np.arange(366), start="2000-01-01")

    assert ind2(tas, tas2) == 366
    assert ind1(tas, thresh=1111) == 366
