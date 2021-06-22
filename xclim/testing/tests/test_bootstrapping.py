import numpy as np
import pytest
import xarray as xr
from dask import array as dsk
from xarray.core.dataarray import DataArray

from xclim.core import bootstrapping
from xclim.core.calendar import percentile_doy
from xclim.core.utils import ValidationError
from xclim.indices import tg90p


def ar1(alpha, n):
    """Return random AR1 DataArray."""

    # White noise
    wn = np.random.randn(n - 1) * np.sqrt(1 - alpha ** 2)

    # Autoregressive series of order 1
    out = np.empty(n)
    out[0] = np.random.randn()
    for i, w in enumerate(wn):
        out[i + 1] = alpha * out[i] + w

    return out


def test_bootstrap(tas_series):
    """Just a smoke test for now."""
    n = int(60 * 365.25)
    alpha = 0.8
    tas = tas_series(ar1(alpha, n), start="2000-01-01")
    per = percentile_doy(tas.sel(time=slice("2000-01-01", "2029-12-31")), per=90)
    tg90p(tas, per, bootstrap=True)
