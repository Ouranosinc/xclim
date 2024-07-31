"""Module for loading testing data."""

from __future__ import annotations

import os
import re
import time
import warnings
from datetime import datetime as dt
from pathlib import Path
from shutil import copytree
from sys import platform

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import Callback
from filelock import FileLock
from packaging.version import Version

import xclim
from xclim import __version__ as __xclim_version__
from xclim.core import calendar
from xclim.core.utils import VARIABLES
from xclim.indices import (
    longwave_upwelling_radiation_from_net_downwelling,
    shortwave_upwelling_radiation_from_net_downwelling,
)
from xclim.testing.utils import _default_cache_dir  # noqa
from xclim.testing.utils import get_file as _get_file
from xclim.testing.utils import get_local_testdata as _get_local_testdata
from xclim.testing.utils import open_dataset as _open_dataset

TESTDATA_BRANCH = os.getenv("XCLIM_TESTDATA_BRANCH", "main")
"""Sets the branch of Ouranosinc/xclim-testdata to use when fetching testing datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_BRANCH="my_testing_branch" pytest

"""

PREFETCH_TESTING_DATA = os.getenv("XCLIM_PREFETCH_TESTING_DATA", False)
"""Indicates whether the testing data should be downloaded when running tests.

Notes
-----
When running tests multiple times, this flag allows developers to significantly speed up the pytest suite
by preventing sha256sum checks for all downloaded files. Proceed with caution.

This can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_PREFETCH_TESTING_DATA=1

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_PREFETCH_TESTING_DATA=1 pytest

"""

__all__ = [
    "PREFETCH_TESTING_DATA",
    "TESTDATA_BRANCH",
    "add_example_file_paths",
    "assert_lazy",
    "generate_atmos",
    "populate_testing_data",
    "test_timeseries",
]


def testing_setup_warnings():
    """Warn users about potential incompatibilities between xclim and xclim-testdata versions."""
    if re.match(r"^\d+\.\d+\.\d+$", __xclim_version__) and TESTDATA_BRANCH == "main":
        # This does not need to be emitted on GitHub Workflows and ReadTheDocs
        if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
            warnings.warn(
                f'`xclim` {__xclim_version__} is running tests against the "main" branch of `Ouranosinc/xclim-testdata`. '
                "It is possible that changes in xclim-testdata may be incompatible with test assertions in this version. "
                "Please be sure to check https://github.com/Ouranosinc/xclim-testdata for more information.",
                UserWarning,
            )

    if re.match(r"^v\d+\.\d+\.\d+", TESTDATA_BRANCH):
        # Find the date of last modification of xclim source files to generate a calendar version
        install_date = dt.strptime(
            time.ctime(os.path.getmtime(xclim.__file__)),
            "%a %b %d %H:%M:%S %Y",
        )
        install_calendar_version = (
            f"{install_date.year}.{install_date.month}.{install_date.day}"
        )

        if Version(TESTDATA_BRANCH) > Version(install_calendar_version):
            warnings.warn(
                f"Installation date of `xclim` ({install_date.ctime()}) "
                f"predates the last release of `xclim-testdata` ({TESTDATA_BRANCH}). "
                "It is very likely that the testing data is incompatible with this build of `xclim`.",
                UserWarning,
            )


def generate_atmos(cache_dir: Path) -> dict[str, xr.DataArray]:
    """Create the `atmosds` synthetic testing dataset."""
    with _open_dataset(
        "ERA5/daily_surface_cancities_1990-1993.nc",
        cache_dir=cache_dir,
        branch=TESTDATA_BRANCH,
        engine="h5netcdf",
    ) as ds:
        tn10 = calendar.percentile_doy(ds.tasmin, per=10)
        t10 = calendar.percentile_doy(ds.tas, per=10)
        t90 = calendar.percentile_doy(ds.tas, per=90)
        tx90 = calendar.percentile_doy(ds.tasmax, per=90)

        rsus = shortwave_upwelling_radiation_from_net_downwelling(ds.rss, ds.rsds)
        rlus = longwave_upwelling_radiation_from_net_downwelling(ds.rls, ds.rlds)

        ds = ds.assign(
            rsus=rsus,
            rlus=rlus,
            tn10=tn10,
            t10=t10,
            t90=t90,
            tx90=tx90,
        )

        # Create a file in session scoped temporary directory
        atmos_file = cache_dir.joinpath("atmosds.nc")
        ds.to_netcdf(atmos_file, engine="h5netcdf")

    # Give access to dataset variables by name in namespace
    namespace = dict()
    with _open_dataset(
        atmos_file, branch=TESTDATA_BRANCH, cache_dir=cache_dir, engine="h5netcdf"
    ) as ds:
        for variable in ds.data_vars:
            namespace[f"{variable}_dataset"] = ds.get(variable)
    return namespace


def populate_testing_data(
    temp_folder: Path | None = None,
    branch: str = TESTDATA_BRANCH,
    _local_cache: Path = _default_cache_dir,
):
    """Perform `_get_file` or `get_local_dataset` calls to GitHub to download or copy relevant testing data."""
    if _local_cache.joinpath(".data_written").exists():
        # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
        return

    data_entries = [
        "CanESM2_365day/pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        "ERA5/daily_surface_cancities_1990-1993.nc",
        "EnsembleReduce/TestEnsReduceCriteria.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc",
        "FWI/GFWED_sample_2017.nc",
        "FWI/cffdrs_test_fwi.nc",
        "FWI/cffdrs_test_wDC.nc",
        "HadGEM2-CC_360day/pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        "NRCANdaily/nrcan_canada_daily_pr_1990.nc",
        "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc",
        "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc",
        "Raven/q_sim.nc",
        "SpatialAnalogs/CanESM2_ScenGen_Chibougamau_2041-2070.nc",
        "SpatialAnalogs/NRCAN_SECan_1981-2010.nc",
        "SpatialAnalogs/dissimilarity.nc",
        "SpatialAnalogs/indicators.nc",
        "cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc",
        "cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc",
        "sdba/CanESM2_1950-2100.nc",
        "sdba/ahccd_1950-2013.nc",
        "sdba/nrcan_1950-2013.nc",
        "uncertainty_partitioning/cmip5_pr_global_mon.nc",
        "uncertainty_partitioning/seattle_avg_tas.csv",
    ]

    data = dict()
    for filepattern in data_entries:
        if temp_folder is None:
            try:
                data[filepattern] = _get_file(
                    filepattern, branch=branch, cache_dir=_local_cache
                )
            except FileNotFoundError:
                warnings.warn(
                    "File {filepattern} was not found. Consider verifying the file exists."
                )
                continue
        elif temp_folder:
            try:
                data[filepattern] = _get_local_testdata(
                    filepattern,
                    temp_folder=temp_folder,
                    branch=branch,
                    _local_cache=_local_cache,
                )
            except FileNotFoundError:
                warnings.warn("File {filepattern} was not found.")
                continue
    return


def gather_testing_data(threadsafe_data_dir: Path, worker_id: str):
    """Gather testing data across workers."""
    if (
        not _default_cache_dir.joinpath(TESTDATA_BRANCH).exists()
        or PREFETCH_TESTING_DATA
    ):
        if PREFETCH_TESTING_DATA:
            print("`XCLIM_PREFETCH_TESTING_DATA` set. Prefetching testing data...")
        if platform == "win32":
            raise OSError(
                "UNIX-style file-locking is not supported on Windows. "
                "Consider running `$ xclim prefetch_testing_data` to download testing data."
            )
        elif worker_id in ["master"]:
            populate_testing_data(branch=TESTDATA_BRANCH)
        else:
            _default_cache_dir.mkdir(exist_ok=True, parents=True)
            lockfile = _default_cache_dir.joinpath(".lock")
            test_data_being_written = FileLock(lockfile)
            with test_data_being_written:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=TESTDATA_BRANCH)
                _default_cache_dir.joinpath(".data_written").touch()
            with test_data_being_written.acquire():
                if lockfile.exists():
                    lockfile.unlink()
    copytree(_default_cache_dir, threadsafe_data_dir)


def add_example_file_paths() -> dict[str, str | list[xr.DataArray]]:
    """Create a dictionary of relevant datasets to be patched into the xdoctest namespace."""
    namespace: dict = dict()
    namespace["path_to_ensemble_file"] = "EnsembleReduce/TestEnsReduceCriteria.nc"
    namespace["path_to_pr_file"] = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"
    namespace["path_to_sfcWind_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    namespace["path_to_tas_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    namespace["path_to_tasmax_file"] = "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"
    namespace["path_to_tasmin_file"] = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"

    # For core.utils.load_module example
    namespace["path_to_example_py"] = (
        Path(__file__).parent.parent.parent.parent / "docs" / "notebooks" / "example.py"
    )

    time = xr.cftime_range("1990-01-01", "2049-12-31", freq="D")
    namespace["temperature_datasets"] = [
        xr.DataArray(
            12 * np.random.random_sample(time.size) + 273,
            coords={"time": time},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
        xr.DataArray(
            12 * np.random.random_sample(time.size) + 273,
            coords={"time": time},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
    ]

    return namespace


def add_doctest_filepaths():
    """Add filepaths to the xdoctest namespace."""
    namespace: dict = dict()
    namespace["np"] = np
    namespace["xclim"] = xclim
    namespace["tas"] = test_timeseries(
        np.random.rand(365) * 20 + 253.15, variable="tas"
    )
    namespace["pr"] = test_timeseries(np.random.rand(365) * 5, variable="pr")

    return namespace


def test_timeseries(
    values,
    variable,
    start: str = "2000-07-01",
    units: str | None = None,
    freq: str = "D",
    as_dataset: bool = False,
    cftime: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Create a generic timeseries object based on pre-defined dictionaries of existing variables."""
    if cftime:
        coords = xr.cftime_range(start, periods=len(values), freq=freq)
    else:
        coords = pd.date_range(start, periods=len(values), freq=freq)

    if variable in VARIABLES:
        attrs = {
            a: VARIABLES[variable].get(a, "")
            for a in ["description", "standard_name", "cell_methods"]
        }
        attrs["units"] = VARIABLES[variable]["canonical_units"]

    else:
        warnings.warn(f"Variable {variable} not recognised. Attrs will not be filled.")
        attrs = {}

    if units is not None:
        attrs["units"] = units

    da = xr.DataArray(values, coords=[coords], dims="time", name=variable, attrs=attrs)

    if as_dataset:
        return da.to_dataset()
    else:
        return da


def _raise_on_compute(dsk: dict):
    """Raise an AssertionError mentioning the number triggered tasks."""
    raise AssertionError(
        f"Not lazy. Computation was triggered with a graph of {len(dsk)} tasks."
    )


assert_lazy = Callback(start=_raise_on_compute)
"""Context manager that raises an AssertionError if any dask computation is triggered."""
