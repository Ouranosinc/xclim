from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim
import xclim.testing
from xclim.core.calendar import max_doy
from xclim.testing.tests import TD


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
    def _pr_series(values, start="7/1/2000", units="kg m-2 s-1"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
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
def prsn_series():
    def _prsn_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "solid_precipitation_flux",
                "cell_methods": "time: mean within days",
                "units": "kg m-2 s-1",
            },
        )

    return _prsn_series


@pytest.fixture
def pr_hr_series():
    """Return hourly time series."""

    def _pr_hr_series(values, start="1/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(hours=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: mean within hours",
                "units": "kg m-2 s-1",
            },
        )

    return _pr_hr_series


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
                "cell_methods": "time: mean within days",
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
            attrs={
                "standard_name": "water_volume_transport_in_river_channel",
                "units": "m3 s-1",
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
    dates = pd.date_range("1900-01-01", periods=nt, freq=pd.DateOffset(days=1))

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
    """Return a rectangular grid of grid cell area."""
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
        attrs={"r": r, "units": "m2", "standard_name": "cell_area"},
    )


areacello = areacella


@pytest.fixture
def hurs_series():
    def _hurs_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="hurs",
            attrs={
                "standard_name": "relative humidity",
                "units": "%",
            },
        )

    return _hurs_series


@pytest.fixture
def sfcWind_series():
    def _sfcWind_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="sfcWind",
            attrs={
                "standard_name": "wind_speed",
                "units": "km h-1",
            },
        )

    return _sfcWind_series


@pytest.fixture
def huss_series():
    def _huss_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="huss",
            attrs={
                "standard_name": "specific_humidity",
                "units": "",
            },
        )

    return _huss_series


@pytest.fixture
def snd_series():
    def _snd_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="snd",
            attrs={
                "standard_name": "surface_snow_thickness",
                "units": "m",
            },
        )

    return _snd_series


@pytest.fixture
def snw_series():
    def _snw_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="snw",
            attrs={
                "standard_name": "surface_snow_amount",
                "units": "kg/m2",
            },
        )

    return _snw_series


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
def add_imports(xdoctest_namespace):
    """Add these imports into the doctests scope."""
    ns = xdoctest_namespace
    ns["np"] = np
    ns["xr"] = xclim.testing
    ns["xclim"] = xclim
    ns["open_dataset"] = xclim.testing.open_dataset


@pytest.fixture(autouse=True)
def add_example_file_paths(xdoctest_namespace, tas_series):
    """Add these datasets in the doctests scope."""
    ns = xdoctest_namespace

    nrcan = Path("NRCANdaily")
    era5 = Path("ERA5")

    ns["path_to_pr_file"] = str(nrcan / "nrcan_canada_daily_pr_1990.nc")

    ns["path_to_tasmax_file"] = str(nrcan / "nrcan_canada_daily_tasmax_1990.nc")

    ns["path_to_tasmin_file"] = str(nrcan / "nrcan_canada_daily_tasmin_1990.nc")

    ns["path_to_tas_file"] = str(era5 / "daily_surface_cancities_1990-1993.nc")

    ns["path_to_multi_shape_file"] = str(TD / "multi_regions.json")

    ns["path_to_shape_file"] = str(TD / "southern_qc_geojson.json")

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

    ns["path_to_ensemble_file"] = str(
        Path("EnsembleReduce").joinpath("TestEnsReduceCriteria.nc")
    )


@pytest.fixture(autouse=True)
def add_example_dataarray(xdoctest_namespace, tas_series):
    ns = xdoctest_namespace
    ns["tas"] = tas_series(np.random.rand(365) * 20 + 253.15)


@pytest.fixture(autouse=True)
def is_matplotlib_installed(xdoctest_namespace):
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


@pytest.fixture(scope="session")
def atmosds():
    ds = xclim.testing.open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    sfcWind, sfcWindfromdir = xclim.atmos.wind_speed_from_vector(ds=ds)
    sfcWind.attrs.update(cell_methods="time: mean within days")
    huss = xclim.atmos.specific_humidity(ds=ds)
    snw = ds.swe * 1000
    # Liquid water equivalent snow thickness [m] to snow thickness in [m] : lwe [m] * 1000 kg/m³ / 300 kg/m³
    snd = snw / 300
    snw.attrs.update(
        standard_name="surface_snow_amount",
        units="kg m-2",
        cell_methods="time: mean within days",
    )
    snd.attrs.update(
        standard_name="surface_snow_thickness",
        units="m",
        cell_methods="time: mean within days",
    )

    psl = ds.ps
    psl.attrs.update(standard_name="air_pressure_at_sea_level")

    tn10 = xclim.core.calendar.percentile_doy(ds.tasmin, per=10)
    t10 = xclim.core.calendar.percentile_doy(ds.tas, per=10)
    t90 = xclim.core.calendar.percentile_doy(ds.tas, per=90)
    tx90 = xclim.core.calendar.percentile_doy(ds.tasmax, per=90)

    ds = ds.assign(
        sfcWind=sfcWind,
        sfcWindfromdir=sfcWindfromdir,
        huss=huss,
        psl=psl,
        snw=snw,
        snd=snd,
        tn10=tn10,
        t10=t10,
        t90=t90,
        tx90=tx90,
    )
    return ds
