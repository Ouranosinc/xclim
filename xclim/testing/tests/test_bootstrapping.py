import numpy as np
import pytest
import xarray as xr
from dask import array as dsk
from xarray.core.dataarray import DataArray

from xclim.core import bootstrapping
from xclim.core.utils import ValidationError

test_da = DataArray(
    ["2021-05-05", "2021-06-05"],
    dims=("time",),
    name="time",
)


class TestBootstrapping:
    def test_bootstrapping(self):
        bootstrapping._bootstrap_period(test_da)
        # TODO see how to make the U.T work
        pass
