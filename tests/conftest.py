# noqa: D104
from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core import indicator
from xclim.core.calendar import max_doy
from xclim.testing.helpers import (
    add_ensemble_dataset_objects,
    generate_atmos,
    test_timeseries,
)
from xclim.testing.utils import (
    TESTDATA_BRANCH,
    TESTDATA_CACHE_DIR,
    TESTDATA_REPO_URL,
    default_testdata_cache,
    gather_testing_data,
    testing_setup_warnings,
)
from xclim.testing.utils import nimbus as _nimbus
from xclim.testing.utils import open_dataset as _open_dataset


@pytest.fixture
def random() -> np.random.Generator:
    return np.random.default_rng(seed=list(map(ord, "𝕽𝔞𝖓𝔡𝖔𝔪")))


@pytest.fixture
def lat_series():
    def _lat_series(values):
        return xr.DataArray(
            values,
            dims=("lat",),
            coords={"lat": values},
            attrs={"standard_name": "latitude", "units": "degrees_north"},
            name="lat",
        )

    return _lat_series


@pytest.fixture
def timeseries():
    return test_timeseries


@pytest.fixture
def tas_series():
    """Return mean temperature time series."""
    _tas_series = partial(test_timeseries, variable="tas")
    return _tas_series


@pytest.fixture
def tasmax_series():
    """Return maximum temperature time series."""
    _tasmax_series = partial(test_timeseries, variable="tasmax")
    return _tasmax_series


@pytest.fixture
def tasmin_series():
    """Return minimum temperature times series."""
    _tasmin_series = partial(test_timeseries, variable="tasmin")
    return _tasmin_series


@pytest.fixture
def pr_series():
    """Return precipitation time series."""
    _pr_series = partial(test_timeseries, variable="pr")
    return _pr_series


@pytest.fixture
def evspsbl_series():
    """Return precipitation time series."""
    return partial(test_timeseries, variable="evspsbl")


@pytest.fixture
def prc_series():
    """Return convective precipitation time series."""
    _prc_series = partial(test_timeseries, variable="prc")
    return _prc_series


@pytest.fixture
def prsn_series():
    """Return snowfall series time series."""
    _prsn_series = partial(test_timeseries, variable="prsn")
    return _prsn_series


@pytest.fixture
def prsnd_series():
    """Return snowfall rate series time series."""
    _prsnd_series = partial(test_timeseries, variable="prsnd")
    return _prsnd_series


@pytest.fixture
def pr_hr_series():
    """Return precipitation hourly time series."""
    _pr_hr_series = partial(test_timeseries, start="1/1/2000", variable="pr", units="kg m-2 s-1", freq="h")
    return _pr_hr_series


@pytest.fixture
def pr_ndseries():
    def _pr_series(values, start="1/1/2000", units="kg m-2 s-1"):
        nt, nx, ny = np.atleast_3d(values).shape
        time_range = pd.date_range(start, periods=nt, freq="D")
        x = np.arange(nx)
        y = np.arange(ny)
        return xr.DataArray(
            values,
            coords=[time_range, x, y],
            dims=("time", "x", "y"),
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

    return _pr_series


@pytest.fixture
def q_series():
    def _q_series(values, start="1/1/2000", units="m3 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="q",
            attrs={
                "standard_name": "water_volume_transport_in_river_channel",
                "units": units,
            },
        )

    return _q_series


@pytest.fixture
def ndq_series(random):
    nx, ny, nt = 2, 3, 5000
    x = np.arange(0, nx)
    y = np.arange(0, ny)

    cx = xr.IndexVariable("x", x)
    cy = xr.IndexVariable("y", y)
    dates = pd.date_range("1900-01-01", periods=nt, freq="D")

    time_range = xr.IndexVariable("time", dates, attrs={"units": "days since 1900-01-01", "calendar": "standard"})

    return xr.DataArray(
        random.lognormal(10, 1, (nt, nx, ny)),
        dims=("time", "x", "y"),
        coords={"time": time_range, "x": cx, "y": cy},
        attrs={
            "units": "m3 s-1",
            "standard_name": "water_volume_transport_in_river_channel",
        },
    )


@pytest.fixture
def evspsblpot_series():
    """Return evapotranspiration time series."""
    _evspsblpot_series = partial(test_timeseries, variable="evspsblpot")
    return _evspsblpot_series


@pytest.fixture
def per_doy():
    def _per_doy(values, calendar="standard", units="kg m-2 s-1"):
        n = max_doy[calendar]
        if len(values) != n:
            raise ValueError("Values must be same length as number of days in calendar.")
        coords = xr.IndexVariable("dayofyear", np.arange(1, n + 1))
        return xr.DataArray(values, coords=[coords], attrs={"calendar": calendar, "units": units})

    return _per_doy


@pytest.fixture
def areacella() -> xr.DataArray:
    """Return a rectangular grid of grid cell area."""
    r = 6100000
    lon_bnds = np.arange(-180, 181, 1)
    lat_bnds = np.arange(-90, 91, 1)
    d_lon = np.diff(lon_bnds)
    d_lat = np.diff(lat_bnds)
    lon = np.convolve(lon_bnds, [0.5, 0.5], "valid")
    lat = np.convolve(lat_bnds, [0.5, 0.5], "valid")
    area = r * np.radians(d_lat)[:, np.newaxis] * r * np.cos(np.radians(lat)[:, np.newaxis]) * np.radians(d_lon)
    return xr.DataArray(
        data=area,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        attrs={"r": r, "units": "m2", "standard_name": "cell_area"},
    )


areacello = areacella


@pytest.fixture
def hurs_series():
    """Return relative humidity time series."""
    _hurs_series = partial(test_timeseries, variable="hurs")
    return _hurs_series


@pytest.fixture
def sfcWind_series():  # noqa
    """Return surface wind speed time series."""
    _sfcWind_series = partial(test_timeseries, variable="sfcWind", units="km h-1")
    return _sfcWind_series


@pytest.fixture
def sfcWindmax_series():  # noqa
    """Return maximum surface wind speed time series."""
    _sfcWindmax_series = partial(test_timeseries, variable="sfcWindmax", units="km h-1")
    return _sfcWindmax_series


@pytest.fixture
def huss_series():
    """Return specific humidity time series."""
    _huss_series = partial(test_timeseries, variable="huss")
    return _huss_series


@pytest.fixture
def snd_series():
    """Return snow depth time series."""
    _snd_series = partial(test_timeseries, variable="snd")
    return _snd_series


@pytest.fixture
def snw_series():
    """Return surface snow amount time series."""
    _snw_series = partial(test_timeseries, variable="snw", units="kg m-2")
    return _snw_series


@pytest.fixture
def ps_series():
    """Return atmospheric pressure time series."""
    _ps_series = partial(test_timeseries, variable="ps")
    return _ps_series


@pytest.fixture
def rsds_series():
    """Return surface downwelling shortwave radiation time series."""
    _rsds_series = partial(test_timeseries, variable="rsds")
    return _rsds_series


@pytest.fixture
def rsus_series():
    """Return surface upwelling shortwave radiation time series."""
    _rsus_series = partial(test_timeseries, variable="rsus")
    return _rsus_series


@pytest.fixture
def rlds_series():
    """Return surface downwelling longwave radiation time series."""
    _rlds_series = partial(test_timeseries, variable="rlds")
    return _rlds_series


@pytest.fixture
def rlus_series():
    """Return surface upwelling longwave radiation time series."""
    _rlus_series = partial(test_timeseries, variable="rlus")
    return _rlus_series


@pytest.fixture(scope="session")
def threadsafe_data_dir(tmp_path_factory):
    return Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def nimbus(threadsafe_data_dir, worker_id):
    return _nimbus(
        repo=TESTDATA_REPO_URL,
        branch=TESTDATA_BRANCH,
        cache_dir=(TESTDATA_CACHE_DIR if worker_id == "master" else threadsafe_data_dir),
    )


@pytest.fixture(scope="session")
def open_dataset(nimbus):
    def _open_session_scoped_file(file: str | os.PathLike, **xr_kwargs):
        xr_kwargs.setdefault("cache", True)
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file,
            branch=TESTDATA_BRANCH,
            repo=TESTDATA_REPO_URL,
            cache_dir=nimbus.path,
            **xr_kwargs,
        )

    return _open_session_scoped_file


@pytest.fixture(scope="session")
def official_indicators():
    # Remove unofficial indicators (as those created during the tests, and those from YAML-built modules)
    registry_cp = indicator.registry.copy()
    for cls in indicator.registry.values():
        if cls.identifier.upper() != cls._registry_id:
            registry_cp.pop(cls._registry_id)
    return registry_cp


@pytest.fixture
def lafferty_sriver_ds(nimbus) -> xr.Dataset:
    """
    Get data from Lafferty & Sriver unit test.

    Notes
    -----
    https://github.com/david0811/lafferty-sriver_2023_npjCliAtm/tree/main/unit_test
    """
    fn = nimbus.fetch(
        "uncertainty_partitioning/seattle_avg_tas.csv",
    )

    df = pd.read_csv(fn, parse_dates=["time"]).rename(columns={"ssp": "scenario", "ensemble": "downscaling"})

    # Make xarray dataset
    return xr.Dataset.from_dataframe(df.set_index(["scenario", "model", "downscaling", "time"]))


@pytest.fixture
def atmosds(nimbus) -> xr.Dataset:
    """Get synthetic atmospheric dataset."""
    return _open_dataset(
        "atmosds.nc",
        cache_dir=nimbus.path,
        engine="h5netcdf",
    ).load()


@pytest.fixture(scope="session")
def ensemble_dataset_objects() -> dict[str, str]:
    return add_ensemble_dataset_objects()


@pytest.fixture(autouse=True, scope="session")
def gather_session_data(request, nimbus, worker_id):
    """
    Gather testing data on pytest run.

    When running pytest with multiple workers, one worker will copy data remotely to default cache dir while
    other workers wait using lockfile. Once the lock is released, all workers will then copy data to their local
    threadsafe_data_dir. As this fixture is scoped to the session, it will only run once per pytest run.

    Due to the lack of UNIX sockets on Windows, the lockfile mechanism is not supported, requiring users on
    Windows to run `$ xclim prefetch_testing_data` before running any tests for the first time to populate the
    default cache dir.

    Additionally, this fixture is also used to generate the `atmosds` synthetic testing dataset.
    """
    testing_setup_warnings()
    gather_testing_data(worker_cache_dir=nimbus.path, worker_id=worker_id)
    generate_atmos(branch=TESTDATA_BRANCH, cache_dir=nimbus.path)

    def remove_data_written_flag():
        """Cleanup cache folder once we are finished."""
        flag = default_testdata_cache.joinpath(".data_written")
        if flag.exists():
            try:
                flag.unlink()
            except FileNotFoundError:
                logging.info("Teardown race condition occurred: .data_written flag already removed. Lucky!")
                pass

    request.addfinalizer(remove_data_written_flag)
