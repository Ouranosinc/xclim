#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test for utils
import os

import numpy as np
import pytest
import xarray as xr

from xclim.core.utils import walk_map

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")
K2C = 273.15


def test_walk_map():
    d = {"a": -1, "b": {"c": -2}}
    o = walk_map(d, lambda x: 0)
    assert o["a"] == 0
    assert o["b"]["c"] == 0
