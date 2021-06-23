import logging
import time

import dask
import numpy as np
import xarray as xr
from distributed import Client
from icclim import icclim
from icclim.util import read
from xarray.core.dataarray import DataArray

import xclim
import xclim as xc
import xclim.core.calendar as calendar
from xclim import __version__, atmos, indices
from xclim.core import bootstrapping
from xclim.core.calendar import percentile_doy
from xclim.core.indicator import Daily, Indicator, registry
from xclim.core.units import units
from xclim.core.utils import InputKind, MissingVariableError, ValidationError
from xclim.indices import tg_mean
from xclim.indices.generic import select_time
from xclim.testing import open_dataset

#############
# TODO Delete this file once U.T are ok
#############

# TODO list
# - Fix the issue with the result, it is not exactly equal to what icclim 4.x provides
#       See what rclimdex gives
# - add logs and delete prints
# - add unit tests
# - make it run in parallel with dask


def netcdf_processing():

    time_start = time.perf_counter()
    ds = xr.open_dataset("climpact.sampledata.gridded.1991-2010.nc")
    ds_per = ds.sel(time=slice("1991-01-01", "2000-12-31"))
    t90 = percentile_doy(ds_per.tmax, window=5, per=90)
    result = xc.atmos.tx90p(
        tasmax=ds.tmax,
        t90=t90,
        # window=3,
        freq="MS",
        bootstrap=True,
    )
    result.to_netcdf("australia/new_xclim_tx90-base-period-91-00.nc")
    time_elapsed = time.perf_counter() - time_start
    print(time_elapsed, " secs")


if __name__ == "__main__":
    netcdf_processing()
