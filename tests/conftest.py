from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim
from xclim.core.calendar import max_doy

TD = Path(__file__).parent / "testdata"


@pytest.fixture
def tmp_netcdf_filename(tmpdir):
    return Path(tmpdir).joinpath("testfile.nc")


@pytest.fixture
def tas_series():
    def _tas_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": "K",
            },
        )

    return _tas_series


@pytest.fixture
def tasmax_series():
    def _tasmax_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmax",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: maximum within days",
                "units": "K",
            },
        )

    return _tasmax_series


@pytest.fixture
def tasmin_series():
    def _tasmin_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmin",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: minimum within days",
                "units": "K",
            },
        )

    return _tasmin_series


@pytest.fixture
def pr_series():
    def _pr_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: sum over day",
                "units": "kg m-2 s-1",
            },
        )

    return _pr_series


@pytest.fixture
def pr_ndseries():
    def _pr_series(values, start="1/1/2000"):
        nt, nx, ny = np.atleast_3d(values).shape
        time = pd.date_range(start, periods=nt, freq=pd.DateOffset(days=1))
        x = np.arange(nx)
        y = np.arange(ny)
        return xr.DataArray(
            values,
            coords=[time, x, y],
            dims=("time", "x", "y"),
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: sum over day",
                "units": "kg m-2 s-1",
            },
        )

    return _pr_series


@pytest.fixture
def q_series():
    def _q_series(values, start="1/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="q",
            attrs={"standard_name": "streamflow", "units": "m3 s-1"},
        )

    return _q_series


@pytest.fixture
def ndq_series():
    nx, ny, nt = 2, 3, 5000
    x = np.arange(0, nx)
    y = np.arange(0, ny)

    cx = xr.IndexVariable("x", x)
    cy = xr.IndexVariable("y", y)
    dates = pd.date_range("1900-01-01", periods=nt, freq=pd.DateOffset(days=1))

    time = xr.IndexVariable(
        "time", dates, attrs={"units": "days since 1900-01-01", "calendar": "standard"}
    )

    return xr.DataArray(
        np.random.lognormal(10, 1, (nt, nx, ny)),
        dims=("time", "x", "y"),
        coords={"time": time, "x": cx, "y": cy},
        attrs={"units": "m^3 s-1", "standard_name": "streamflow"},
    )


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
def areacella():
    """Return a rectangular grid of grid cell area. """
    r = 6100000
    lon_bnds = np.arange(-180, 181, 1)
    lat_bnds = np.arange(-90, 91, 1)
    dlon = np.diff(lon_bnds)
    dlat = np.diff(lat_bnds)
    lon = np.convolve(lon_bnds, [0.5, 0.5], "valid")
    lat = np.convolve(lat_bnds, [0.5, 0.5], "valid")
    area = (
        r
        * np.radians(dlat)[:, np.newaxis]
        * r
        * np.cos(np.radians(lat)[:, np.newaxis])
        * np.radians(dlon)
    )
    return xr.DataArray(
        data=area,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        attrs={"r": r, "units": "m^2"},
    )


@pytest.fixture
def rh_series():
    def _rh_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rh",
            attrs={"standard_name": "relative humidity", "units": "%",},
        )

    return _rh_series


@pytest.fixture
def ws_series():
    def _ws_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="ws",
            attrs={"standard_name": "wind speed", "units": "km h-1",},
        )

    return _ws_series


@pytest.fixture
def huss_series():
    def _huss_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="huss",
            attrs={"standard_name": "specific_humidity", "units": "",},
        )

    return _huss_series


@pytest.fixture
def ps_series():
    def _ps_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="ps",
            attrs={"standard_name": "air_pressure", "units": "Pa"},
        )

    return _ps_series


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """Add these imports into the doctests scope."""
    ns = doctest_namespace
    ns["np"] = np
    ns["xr"] = xr
    ns["xclim"] = xclim


@pytest.fixture(autouse=True)
def add_example_file_paths(doctest_namespace):
    """Add these datasets in the doctests scope."""
    ns = doctest_namespace
    ns["path_to_pr_file"] = str(TD / "NRCANdaily" / "nrcan_canada_daily_pr_1990.nc")

    ns["path_to_tasmax_file"] = str(
        TD / "NRCANdaily" / "nrcan_canada_daily_tasmax_1990.nc"
    )

    ns["path_to_tasmin_file"] = str(
        TD / "NRCANdaily" / "nrcan_canada_daily_tasmin_1990.nc"
    )

    ns["path_to_tas_file"] = str(
        TD / "cmip3" / "tas.sresb1.giss_model_e_r.run1.atm.da.nc"
    )

    ns["path_to_multi_shape_file"] = str(TD / "cmip5" / "multi_regions.json")

    ns["path_to_shape_file"] = str(TD / "cmip5" / "southern_qc_geojson.json")

    ns["temperature_datasets"] = [
        str(
            TD
            / "EnsembleStats"
            / "BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc"
        ),
        str(
            TD
            / "EnsembleStats"
            / "BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc"
        ),
    ]
    ns["precipitation_datasets"] = [
        str(TD / "NRCANdaily" / "nrcan_canada_daily_pr_1990.nc"),
        str(
            TD
            / "CanESM2_365day"
            / "pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc"
        ),
    ]


@pytest.fixture(autouse=True)
def add_example_dataarray(doctest_namespace, tas_series):
    ns = doctest_namespace
    ns["tas"] = tas_series(np.random.rand(365) * 20 + 253.15)
