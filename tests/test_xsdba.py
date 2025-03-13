"""Test xsdba integration."""

from __future__ import annotations

import numpy as np
import pytest

import xclim


def test_simple(timeseries):
    pytest.importorskip("xsdba")

    ref = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True).tas

    sim = timeseries(
        np.concatenate([np.ones(365 * 2) * 2, np.ones(365) * 3]),
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    ).tas

    ADJ = xclim.sdba.EmpiricalQuantileMapping.train(ref=ref, hist=sim.sel(time=slice("2001", "2003")))
    ADJ.adjust(sim=sim)
