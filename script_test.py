from icclim import icclim
from icclim.util import read
import logging
import time
import xclim as xc
import xarray as xr
from xclim.core.calendar import percentile_doy

import xclim
from xclim import __version__, atmos
from xclim.core.formatting import (
    AttrFormatter,
    default_formatter,
    merge_attributes,
    parse_doc,
    update_history,
)
from xclim.core.indicator import Daily, Indicator, registry
from xclim.core.units import units
from xclim.core.utils import InputKind, MissingVariableError
from xclim.indices import tg_mean
from xclim.indices.generic import select_time
from xclim.testing import open_dataset

from xclim.indicators import icclim
from xarray.core.dataarray import DataArray
from xclim.core import bootstrapping
from xclim.core.utils import ValidationError

import dask
from distributed import Client
from xclim.indicators import icclim

xlog = logging.getLogger("xclim")
xlog.disabled = False
xlog.setLevel(logging.DEBUG)


#############
# TODO Delete this file once U.T are ok
#############

# TODO list
# - Fix the issue with the result, it is not exactly equal to what icclim 4.x provides
# - add logs
# - add unit tests
# - make it run in parallel with dask

def netcdf_processing():

    ds = xr.open_dataset("tasmax_day_MIROC6_ssp585_r1i1p1f1_gn_20150101-20241231.nc")
    # t90 = percentile_doy(ds.tasmax, window=1, per=90).sel(percentiles=90)
    # out = xc.atmos.tg90p(ds.tasmax, t90, freq="MS")

    # for i in range(0, 5):
    #     print("run_", i + 1)
    #     time_start = time.perf_counter()
    #     ds = xr.open_dataset(
    #         "tasmax_day_MIROC6_ssp585_r1i1p1f1_gn_20150101-20241231.nc")

    # t90 = percentile_doy(ds.tasmax, window=1, per=90).sel(percentiles=90)
    # out = icclim.TX90p(ds.tasmax, t90, freq="MS")
    # out = xc.atmos.tx90p(ds.tasmax, t90, freq="MS")
    # out = xc.atmos.tx_days_above(ds.tasmax, freq="MS")

    #     # pr_min = "1 mm/d"
    #     # ds = xr.open_dataset(
    #     #     "pr_day_CanESM5_ssp585_r10i1p1f1_gn_20150101-21001231.nc", chunks={"lat": 2})
    #     # out = xc.atmos.maximum_consecutive_dry_days(ds.pr, thresh=pr_min, freq="MS")

    # out.to_netcdf('tx90_results_xclim.nc')
    #     time_elapsed = (time.perf_counter() - time_start)
    #     print(time_elapsed, ' secs')
    #     break

    time_start = time.perf_counter()
    ds = xr.open_dataset(
        "tasmax_day_MIROC6_ssp585_r1i1p1f1_gn_20150101-20241231.nc")
    # ds = ds.sel(time=slice("2015-01-01", "2015-12-31"))
    # t90 = percentile_doy(ds.tasmax, window=1, per=90).sel(percentiles=90)
    config = bootstrapping.BootstrapConfig(
        operator='>',
        percentile=90,
        window=1,
        in_base_slice=slice("2015-01-01", "2017-12-31"),
        out_of_base_slice=slice("2018-01-01", "2024-12-31")
    )
    result = bootstrapping.compute_bootstrapped_exceedance_rate(ds.tasmax, config)
    result.to_netcdf('bootstrap_results_xclim_2.nc')
    time_elapsed = (time.perf_counter() - time_start)
    print(time_elapsed, ' secs')


if __name__ == "__main__":
    netcdf_processing()
