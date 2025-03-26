"""Module for loading testing data."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from dask.callbacks import Callback

import xclim
import xclim.testing.utils as xtu
from xclim.core import VARIABLES
from xclim.core.calendar import percentile_doy
from xclim.indices import (
    longwave_upwelling_radiation_from_net_downwelling,
    shortwave_upwelling_radiation_from_net_downwelling,
)

logger = logging.getLogger("xclim")

__all__ = [
    "add_doctest_filepaths",
    "add_ensemble_dataset_objects",
    "add_example_file_paths",
    "assert_lazy",
    "generate_atmos",
    "test_timeseries",
]


def generate_atmos(
    branch: str | os.PathLike[str] | Path,
    cache_dir: str | os.PathLike[str] | Path,
) -> dict[str, xr.DataArray]:
    """
    Create the `atmosds` synthetic testing dataset.

    Parameters
    ----------
    branch : str or os.PathLike[str] or Path
        The branch to use for the testing dataset.
    cache_dir : str or os.PathLike[str] or Path
        The directory to store the testing dataset.

    Returns
    -------
    dict[str, xr.DataArray]
        A dictionary of xarray DataArrays.
    """
    with xtu.open_dataset(
        "ERA5/daily_surface_cancities_1990-1993.nc",
        branch=branch,
        cache_dir=cache_dir,
        engine="h5netcdf",
    ) as ds:
        rsus = shortwave_upwelling_radiation_from_net_downwelling(ds.rss, ds.rsds)
        rlus = longwave_upwelling_radiation_from_net_downwelling(ds.rls, ds.rlds)
        tn10 = percentile_doy(ds.tasmin, per=10)
        t10 = percentile_doy(ds.tas, per=10)
        t90 = percentile_doy(ds.tas, per=90)
        tx90 = percentile_doy(ds.tasmax, per=90)

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
    with xtu.open_dataset(atmos_file, branch=branch, cache_dir=cache_dir, engine="h5netcdf") as ds:
        namespace = {f"{var}_dataset": ds[var] for var in ds.data_vars}
    return namespace


def add_ensemble_dataset_objects() -> dict[str, str]:
    """
    Create a dictionary of xclim ensemble-related datasets to be patched into the xdoctest namespace.

    Returns
    -------
    dict[str, str]
        A dictionary of xclim ensemble-related datasets.
    """
    namespace = {
        "nc_files_simple": [
            "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
        ],
        "nc_files_extra": [
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc"
        ],
    }
    namespace["nc_files"] = namespace["nc_files_simple"] + namespace["nc_files_extra"]
    return namespace


def add_example_file_paths() -> dict[str, str | list[xr.DataArray]]:
    """
    Create a dictionary of doctest-relevant datasets to be patched into the xdoctest namespace.

    Returns
    -------
    dict of str or dict of list of xr.DataArray
        A dictionary of doctest-relevant datasets.
    """
    namespace = {
        "path_to_ensemble_file": "EnsembleReduce/TestEnsReduceCriteria.nc",
        "path_to_gwl_file": "Raven/gwl_obs.nc",
        "path_to_pr_file": "NRCANdaily/nrcan_canada_daily_pr_1990.nc",
        "path_to_q_file": "Raven/q_sim.nc",
        "path_to_sfcWind_file": "ERA5/daily_surface_cancities_1990-1993.nc",
        "path_to_tas_file": "ERA5/daily_surface_cancities_1990-1993.nc",
        "path_to_tasmax_file": "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc",
        "path_to_tasmin_file": "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc",
        "path_to_example_py": (Path(__file__).parent.parent.parent.parent / "docs" / "notebooks" / "example.py"),
    }

    # For core.utils.load_module example
    sixty_years = xr.date_range("1990-01-01", "2049-12-31", freq="D")
    namespace["temperature_datasets"] = [
        xr.DataArray(
            12 * np.random.random_sample(sixty_years.size) + 273,
            coords={"time": sixty_years},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
        xr.DataArray(
            12 * np.random.random_sample(sixty_years.size) + 273,
            coords={"time": sixty_years},
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


def add_doctest_filepaths() -> dict[str, Any]:
    """
    Overload some libraries directly into the xdoctest namespace.

    Returns
    -------
    dict[str, Any]
        A dictionary of xdoctest namespace objects.
    """
    namespace = {
        "np": np,
        "xclim": xclim,
        "tas": test_timeseries(np.random.rand(365) * 20 + 253.15, variable="tas"),
        "pr": test_timeseries(np.random.rand(365) * 5, variable="pr"),
    }
    return namespace


def test_timeseries(
    values,
    variable,
    start: str = "2000-07-01",
    units: str | None = None,
    freq: str = "D",
    as_dataset: bool = False,
    cftime: bool | None = None,
    calendar: str | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Create a generic timeseries object based on pre-defined dictionaries of existing variables.

    Parameters
    ----------
    values : np.ndarray
        The values of the DataArray.
    variable : str
        The name of the DataArray.
    start : str
        The start date of the time dimension. Default is "2000-07-01".
    units : str or None
        The units of the DataArray. Default is None.
    freq : str
        The frequency of the time dimension. Default is daily/"D".
    as_dataset : bool
        Whether to return a Dataset or a DataArray. Default is False.
    cftime : bool
        Whether to use cftime or not. Default is None, which uses cftime only for non-standard calendars.
    calendar : str or None
        Whether to use a calendar. If a calendar is provided, cftime is used.

    Returns
    -------
    xr.DataArray or xr.Dataset
        A DataArray or Dataset with time, lon and lat dimensions.
    """
    coords = xr.date_range(start, periods=len(values), freq=freq, calendar=calendar or "standard", use_cftime=cftime)

    if variable in VARIABLES:
        attrs = {a: VARIABLES[variable].get(a, "") for a in ["description", "standard_name", "cell_methods"]}
        attrs["units"] = VARIABLES[variable]["canonical_units"]

    else:
        warnings.warn(f"Variable {variable} not recognised. Attrs will not be filled.")
        attrs = {}

    if units is not None:
        attrs["units"] = units

    da = xr.DataArray(values, coords=[coords], dims="time", name=variable, attrs=attrs)

    if as_dataset:
        return da.to_dataset()
    return da


def _raise_on_compute(dsk: dict):
    """
    Raise an AssertionError mentioning the number triggered tasks.

    Parameters
    ----------
    dsk : dict
        The dask graph.

    Raises
    ------
    AssertionError
        If the dask computation is triggered.
    """
    raise AssertionError(f"Not lazy. Computation was triggered with a graph of {len(dsk)} tasks.")


assert_lazy = Callback(start=_raise_on_compute)
"""Context manager that raises an AssertionError if any dask computation is triggered."""
