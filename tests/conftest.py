# noqa: D104
from __future__ import annotations

import os
import re
import shutil
import time
import warnings
from datetime import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from filelock import FileLock
from pkg_resources import parse_version, working_set

import xclim
from xclim import __version__ as __xclim_version__
from xclim.core import indicator
from xclim.core.calendar import max_doy
from xclim.testing import helpers
from xclim.testing.helpers import test_timeseries
from xclim.testing.utils import _default_cache_dir  # noqa
from xclim.testing.utils import open_dataset as _open_dataset

if not __xclim_version__.endswith("-beta") and helpers.TESTDATA_BRANCH == "main":
    # This does not need to be emitted on GitHub Workflows and ReadTheDocs
    if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
        warnings.warn(
            f'`xclim` {__xclim_version__} is running tests against the "main" branch of `Ouranosinc/xclim-testdata`. '
            "It is possible that changes in xclim-testdata may be incompatible with test assertions in this version. "
            "Please be sure to check https://github.com/Ouranosinc/xclim-testdata for more information.",
            UserWarning,
        )

if re.match(r"^v\d+\.\d+\.\d+", helpers.TESTDATA_BRANCH):
    # Find the date of last modification of xclim source files to generate a calendar version
    install_date = dt.strptime(
        time.ctime(os.path.getmtime(working_set.by_key["xclim"].location)),
        "%a %b %d %H:%M:%S %Y",
    )
    install_calendar_version = (
        f"{install_date.year}.{install_date.month}.{install_date.day}"
    )

    if parse_version(helpers.TESTDATA_BRANCH) > parse_version(install_calendar_version):
        warnings.warn(
            f"Installation date of `xclim` ({install_date.ctime()}) "
            f"predates the last release of `xclim-testdata` ({helpers.TESTDATA_BRANCH}). "
            "It is very likely that the testing data is incompatible with this build of `xclim`.",
            UserWarning,
        )


@pytest.fixture
def tmp_netcdf_filename(tmpdir) -> Path:
    yield Path(tmpdir).joinpath("testfile.nc")


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


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
def prc_series():
    """Return convective precipitation time series."""
    _prc_series = partial(test_timeseries, variable="prc")
    return _prc_series


@pytest.fixture
def bootstrap_series():
    def _bootstrap_series(values, start="7/1/2000", units="kg m-2 s-1", cf_time=False):
        if cf_time:
            coords = xr.cftime_range(start, periods=len(values), freq="D")
        else:
            coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

    return _bootstrap_series


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
    _pr_hr_series = partial(
        test_timeseries, start="1/1/2000", variable="pr", units="kg m-2 s-1", freq="1H"
    )
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
def ndq_series():
    nx, ny, nt = 2, 3, 5000
    x = np.arange(0, nx)
    y = np.arange(0, ny)

    cx = xr.IndexVariable("x", x)
    cy = xr.IndexVariable("y", y)
    dates = pd.date_range("1900-01-01", periods=nt, freq="D")

    time_range = xr.IndexVariable(
        "time", dates, attrs={"units": "days since 1900-01-01", "calendar": "standard"}
    )

    return xr.DataArray(
        np.random.lognormal(10, 1, (nt, nx, ny)),
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
            raise ValueError(
                "Values must be same length as number of days in calendar."
            )
        coords = xr.IndexVariable("dayofyear", np.arange(1, n + 1))
        return xr.DataArray(
            values, coords=[coords], attrs={"calendar": calendar, "units": units}
        )

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
    area = (
        r
        * np.radians(d_lat)[:, np.newaxis]
        * r
        * np.cos(np.radians(lat)[:, np.newaxis])
        * np.radians(d_lon)
    )
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
def cmip3_day_tas(threadsafe_data_dir):
    # xr.set_options(enable_cftimeindex=False)
    ds = _open_dataset(
        "cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc",
        cache_dir=threadsafe_data_dir,
        branch=helpers.TESTDATA_BRANCH,
        engine="h5netcdf",
    )
    yield ds.tas
    ds.close()


@pytest.fixture(scope="session")
def open_dataset(threadsafe_data_dir):
    def _open_session_scoped_file(
        file: str | os.PathLike, branch: str = helpers.TESTDATA_BRANCH, **xr_kwargs
    ):
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file, cache_dir=threadsafe_data_dir, branch=branch, **xr_kwargs
        )

    return _open_session_scoped_file


@pytest.fixture(autouse=True, scope="session")
def add_imports(xdoctest_namespace, threadsafe_data_dir) -> None:
    """Add these imports into the doctests scope."""
    ns = xdoctest_namespace
    ns["np"] = np
    ns["xr"] = xclim.testing  # xr.open_dataset(...) -> xclim.testing.open_dataset(...)
    ns["xclim"] = xclim
    ns["open_dataset"] = partial(
        _open_dataset, cache_dir=threadsafe_data_dir, branch=helpers.TESTDATA_BRANCH
    )  # Needed for modules where xarray is imported as `xr`


@pytest.fixture(autouse=True, scope="function")
def add_example_dataarray(xdoctest_namespace, tas_series) -> None:
    ns = xdoctest_namespace
    ns["tas"] = tas_series(np.random.rand(365) * 20 + 253.15)


@pytest.fixture(autouse=True, scope="session")
def is_matplotlib_installed(xdoctest_namespace) -> None:
    def _is_matplotlib_installed():
        try:
            import matplotlib  # noqa

            return
        except ImportError:
            return pytest.skip("This doctest requires matplotlib to be installed.")

    ns = xdoctest_namespace
    ns["is_matplotlib_installed"] = _is_matplotlib_installed


@pytest.fixture
def official_indicators():
    # Remove unofficial indicators (as those created during the tests, and those from YAML-built modules)
    registry_cp = indicator.registry.copy()
    for cls in indicator.registry.values():
        if cls.identifier.upper() != cls._registry_id:
            registry_cp.pop(cls._registry_id)
    return registry_cp


@pytest.fixture(scope="function")
def atmosds(threadsafe_data_dir) -> xr.Dataset:
    return _open_dataset(
        threadsafe_data_dir.joinpath("atmosds.nc"),
        cache_dir=threadsafe_data_dir,
        branch=helpers.TESTDATA_BRANCH,
        engine="h5netcdf",
    ).load()


@pytest.fixture(scope="function")
def ensemble_dataset_objects() -> dict:
    edo = dict()
    edo["nc_files_simple"] = [
        "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
    ]
    edo["nc_files_extra"] = [
        "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc"
    ]
    edo["nc_files"] = edo["nc_files_simple"] + edo["nc_files_extra"]
    return edo


@pytest.fixture(scope="session", autouse=True)
def gather_session_data(threadsafe_data_dir, worker_id, xdoctest_namespace):
    """Gather testing data on pytest run.

    When running pytest with multiple workers, one worker will copy data remotely to _default_cache_dir while
    other workers wait using lockfile. Once the lock is released, all workers will copy data to their local
    threadsafe_data_dir."""

    if (
        not _default_cache_dir.joinpath(helpers.TESTDATA_BRANCH).exists()
        or helpers.PREFETCH_TESTING_DATA
    ):
        if worker_id in "master":
            helpers.populate_testing_data(branch=helpers.TESTDATA_BRANCH)
        else:
            _default_cache_dir.mkdir(exist_ok=True)
            test_data_being_written = FileLock(_default_cache_dir.joinpath(".lock"))
            with test_data_being_written as fl:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                helpers.populate_testing_data(branch=helpers.TESTDATA_BRANCH)
                _default_cache_dir.joinpath(".data_written").touch()
            fl.acquire()
        shutil.copytree(_default_cache_dir, threadsafe_data_dir)
    helpers.generate_atmos(threadsafe_data_dir)
    xdoctest_namespace.update(helpers.add_example_file_paths(threadsafe_data_dir))


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing file once we are finished.

    This flag prevents remote data from being downloaded multiple times in the same pytest run.
    """

    def remove_data_written_flag():
        flag = _default_cache_dir.joinpath(".data_written")
        if flag.exists():
            flag.unlink()

    request.addfinalizer(remove_data_written_flag)


@pytest.fixture
def timeseries():
    return test_timeseries
