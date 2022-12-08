# noqa: D104
from __future__ import annotations

import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from filelock import FileLock

import xclim
from xclim.core.calendar import max_doy
from xclim.testing.tests.data import (
    add_example_file_paths,
    generate_atmos,
    populate_testing_data,
)
from xclim.testing.utils import _default_cache_dir
from xclim.testing.utils import open_dataset as _open_dataset

MAIN_TESTDATA_BRANCH = os.getenv("MAIN_TESTDATA_BRANCH", "main")
SKIP_TEST_DATA = os.getenv("SKIP_TEST_DATA")


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
    def _tas_series(values, start="7/1/2000", units="K"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

    return _tas_series


@pytest.fixture
def tasmax_series():
    def _tasmax_series(values, start="7/1/2000", units="K"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmax",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: maximum within days",
                "units": units,
            },
        )

    return _tasmax_series


@pytest.fixture
def tasmin_series():
    def _tasmin_series(values, start="7/1/2000", units="K"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmin",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: minimum within days",
                "units": units,
            },
        )

    return _tasmin_series


@pytest.fixture
def pr_series():
    def _pr_series(values, start="7/1/2000", units="kg m-2 s-1"):
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

    return _pr_series


@pytest.fixture
def prc_series():
    def _prc_series(values, start="7/1/2000", units="kg m-2 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "convective_precipitation_flux",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

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
    def _prsn_series(values, start="7/1/2000", units="kg m-2 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "solid_precipitation_flux",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

    return _prsn_series


@pytest.fixture
def pr_hr_series():
    """Return hourly time series."""

    def _pr_hr_series(values, start="1/1/2000", units="kg m-2 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="1H")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: mean within hours",
                "units": units,
            },
        )

    return _pr_hr_series


@pytest.fixture
def pr_ndseries():
    def _pr_series(values, start="1/1/2000", units="kg m-2 s-1"):
        nt, nx, ny = np.atleast_3d(values).shape
        time = pd.date_range(start, periods=nt, freq="D")
        x = np.arange(nx)
        y = np.arange(ny)
        return xr.DataArray(
            values,
            coords=[time, x, y],
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

    time = xr.IndexVariable(
        "time", dates, attrs={"units": "days since 1900-01-01", "calendar": "standard"}
    )

    return xr.DataArray(
        np.random.lognormal(10, 1, (nt, nx, ny)),
        dims=("time", "x", "y"),
        coords={"time": time, "x": cx, "y": cy},
        attrs={
            "units": "m3 s-1",
            "standard_name": "water_volume_transport_in_river_channel",
        },
    )


@pytest.fixture
def evspsblpot_series():
    def _evspsblpot_series(values, start="7/1/2000", units="kg m-2 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="evspsblpot",
            attrs={
                "standard_name": "water_evapotranspiration_flux",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )

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
    def _hurs_series(values, start="7/1/2000", units="%"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="hurs",
            attrs={
                "standard_name": "relative humidity",
                "units": units,
            },
        )

    return _hurs_series


@pytest.fixture
def sfcWind_series():  # noqa
    def _sfcWind_series(values, start="7/1/2000", units="km h-1"):  # noqa
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="sfcWind",
            attrs={
                "standard_name": "wind_speed",
                "units": units,
            },
        )

    return _sfcWind_series


@pytest.fixture
def huss_series():
    def _huss_series(values, start="7/1/2000", units=""):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="huss",
            attrs={
                "standard_name": "specific_humidity",
                "units": units,
            },
        )

    return _huss_series


@pytest.fixture
def snd_series():
    def _snd_series(values, start="7/1/2000", units="m"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="snd",
            attrs={
                "standard_name": "surface_snow_thickness",
                "units": units,
            },
        )

    return _snd_series


@pytest.fixture
def snw_series():
    def _snw_series(values, start="7/1/2000", units="kg/m2"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="snw",
            attrs={
                "standard_name": "surface_snow_amount",
                "units": units,
            },
        )

    return _snw_series


@pytest.fixture
def ps_series():
    def _ps_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="ps",
            attrs={"standard_name": "air_pressure", "units": "Pa"},
        )

    return _ps_series


@pytest.fixture
def rsds_series():
    def _rsds_series(values, start="7/1/2000", units="W m-2"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rsds",
            attrs={
                "standard_name": "surface_downwelling_shortwave_flux_in_air",
                "units": units,
            },
        )

    return _rsds_series


@pytest.fixture
def rsus_series():
    def _rsus_series(values, start="7/1/2000", units="W m-2"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rsus",
            attrs={
                "standard_name": "surface_upwelling_shortwave_flux_in_air",
                "units": units,
            },
        )

    return _rsus_series


@pytest.fixture
def rlds_series():
    def _rlds_series(values, start="7/1/2000", units="W m-2"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rlds",
            attrs={
                "standard_name": "surface_downwelling_longwave_flux_in_air",
                "units": units,
            },
        )

    return _rlds_series


@pytest.fixture
def rlus_series():
    def _rlus_series(values, start="7/1/2000", units="W m-2"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rlus",
            attrs={
                "standard_name": "surface_upwelling_longwave_flux_in_air",
                "units": units,
            },
        )

    return _rlus_series


@pytest.fixture(scope="session")
def cmip3_day_tas(threadsafe_data_dir):
    # xr.set_options(enable_cftimeindex=False)
    ds = _open_dataset(
        "cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc",
        cache_dir=threadsafe_data_dir,
        branch=MAIN_TESTDATA_BRANCH,
    )
    yield ds.tas
    ds.close()


@pytest.fixture(scope="session")
def open_dataset(threadsafe_data_dir):
    def _open_session_scoped_file(
        file: str | os.PathLike, branch: str = MAIN_TESTDATA_BRANCH, **xr_kwargs
    ):
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
        _open_dataset, cache_dir=threadsafe_data_dir, branch=MAIN_TESTDATA_BRANCH
    )  # Needed for modules where xarray is imported as `xr`


@pytest.fixture(autouse=True)
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
    registry_cp = xclim.core.indicator.registry.copy()
    for cls in xclim.core.indicator.registry.values():
        if cls.identifier.upper() != cls._registry_id:
            registry_cp.pop(cls._registry_id)
    return registry_cp


@pytest.fixture(scope="function")
def atmosds(threadsafe_data_dir) -> xr.Dataset:
    return _open_dataset(
        threadsafe_data_dir.joinpath("atmosds.nc"),
        cache_dir=threadsafe_data_dir,
        branch=MAIN_TESTDATA_BRANCH,
    )


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
    if worker_id == "master":
        if not SKIP_TEST_DATA:
            populate_testing_data(branch=MAIN_TESTDATA_BRANCH)
    else:
        if not SKIP_TEST_DATA:
            _default_cache_dir.mkdir(exist_ok=True)
            test_data_being_written = FileLock(_default_cache_dir.joinpath(".lock"))
            with test_data_being_written as fl:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=MAIN_TESTDATA_BRANCH)
                _default_cache_dir.joinpath(".data_written").touch()
            fl.acquire()
        shutil.copytree(_default_cache_dir, threadsafe_data_dir)
    generate_atmos(threadsafe_data_dir)
    xdoctest_namespace.update(add_example_file_paths(threadsafe_data_dir))


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
