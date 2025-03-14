"""Test xsdba integration."""

# sdba may or may not be imported, which fails some QA tests
# pylint: disable=E0606

from __future__ import annotations

import importlib.util as _util

import numpy as np
import pytest

xsdba_installed = _util.find_spec("xsdba")
if xsdba_installed:
    from xclim import sdba


@pytest.mark.skipif(not xsdba_installed, reason="`xsdba` is not installed")
def test_simple(timeseries):
    ref = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True).tas

    sim = timeseries(
        np.concatenate([np.ones(365 * 2) * 2, np.ones(365) * 3]),
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    ).tas

    ADJ = sdba.EmpiricalQuantileMapping.train(ref=ref, hist=sim.sel(time=slice("2001", "2003")))
    ADJ.adjust(sim=sim)


@pytest.mark.skipif(xsdba_installed, reason="Import failure of `sdba` only tested if `xsdba` is not installed")
def test_import_failure():
    error_msg = (
        "No module named 'xsdba'. `sdba` was split from `xclim` in its "
        "own submodule `xsdba`. Use conda or pip to install `xsdba`."
    )
    with pytest.raises(ModuleNotFoundError) as e:
        pass
    assert e.value.args[0] == error_msg
