"""Module for loading testing data."""
from __future__ import annotations

import os
import warnings
from importlib.resources import open_text
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from yaml import safe_load

from xclim.core import calendar
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
    "generate_atmos",
    "populate_testing_data",
    "test_timeseries",
]


def generate_atmos(cache_dir: str | Path):
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
        ds.to_netcdf(atmos_file)


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
        "ERA5/daily_surface_cancities_1990-1993.nc",
        "EnsembleReduce/TestEnsReduceCriteria.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc",
        "NRCANdaily/nrcan_canada_daily_pr_1990.nc",
        "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc",
        "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc",
        "cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc",
        "sdba/CanESM2_1950-2100.nc",
    ]

    data = dict()
    for filepattern in data_entries:
        if temp_folder is None:
            try:
                data[filepattern] = _get_file(
                    filepattern, branch=branch, cache_dir=_local_cache
                )
            except FileNotFoundError:
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
                continue
    return


def add_example_file_paths(cache_dir: Path) -> dict[str]:
    """Create a dictionary of relevant datasets to be patched into the xdoctest namespace."""
    ns = dict()
    ns["path_to_ensemble_file"] = "EnsembleReduce/TestEnsReduceCriteria.nc"
    ns["path_to_pr_file"] = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"
    ns["path_to_sfcWind_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    ns["path_to_tas_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    ns["path_to_tasmax_file"] = "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"
    ns["path_to_tasmin_file"] = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"

    # For core.utils.load_module example
    ns["path_to_example_py"] = (
        Path(__file__).parent.parent.parent.parent / "docs" / "notebooks" / "example.py"
    )

    time = xr.cftime_range("1990-01-01", "2049-12-31", freq="D")
    ns["temperature_datasets"] = [
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

    # Give access to this file within xdoctest namespace
    atmos_file = cache_dir.joinpath("atmosds.nc")

    # Give access to dataset variables by name in xdoctest namespace
    with _open_dataset(
        atmos_file, branch=TESTDATA_BRANCH, cache_dir=cache_dir, engine="h5netcdf"
    ) as ds:
        for variable in ds.data_vars:
            ns[f"{variable}_dataset"] = ds.get(variable)

    return ns


def test_timeseries(
    values, variable, start="7/1/2000", units=None, freq="D", as_dataset=False
):
    """Create a generic timeseries object based on pre-defined dictionaries of existing variables."""
    coords = pd.date_range(start, periods=len(values), freq=freq)
    data_on_var = safe_load(open_text("xclim.data", "variables.yml"))["variables"]
    if variable in data_on_var:
        attrs = {
            a: data_on_var[variable].get(a, "")
            for a in ["description", "standard_name", "cell_methods"]
        }
        attrs["units"] = data_on_var[variable]["canonical_units"]

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
