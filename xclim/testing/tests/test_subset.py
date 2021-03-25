import os

import numpy as np
import pytest
import xarray as xr

from xclim.testing import open_dataset

try:
    import clisops.core.subset as subset
    import geopandas as gpd
except ImportError:
    subset = False
    gpd = False

pytestmark = pytest.mark.slow
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "data")


class TestSubsetRaises:
    @pytest.mark.skipif(
        subset is False, reason="`clisops` subset utilities are not installed."
    )
    def test_raises_deprecation_warning(self):
        with pytest.deprecated_call():
            from xclim import subset

            assert subset.__doc__

    @pytest.mark.skipif(
        hasattr(subset, "__all__"), reason="Necessary dependencies are installed."
    )
    def test_raise_import_error(self):
        with pytest.raises(ImportError):
            from xclim import subset

            assert subset.__doc__


@pytest.mark.skipif(
    subset is False, reason="`clisops` subset utilities are not installed."
)
class TestSubsetTime:
    nc_poslons = os.path.join("cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc")

    def test_simple(self):
        da = open_dataset(self.nc_poslons).tas
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_time(da, start_date=yr_st, end_date=yr_ed)
        out1 = subset.subset_time(da, start_date=f"{yr_st}-01", end_date=f"{yr_ed}-12")
        out2 = subset.subset_time(
            da, start_date=f"{yr_st}-01-01", end_date=f"{yr_ed}-12-31"
        )
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out, out2)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))

    def test_time_dates_outofbounds(self):
        da = open_dataset(self.nc_poslons).tas
        yr_st = "1776"
        yr_ed = "2077"

        with pytest.warns(None) as record:
            out = subset.subset_time(
                da, start_date=f"{yr_st}-01", end_date=f"{yr_ed}-01"
            )
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

        assert (
            '"start_date" not found within input date time range. Defaulting to minimum time step in xarray object.'
            in [str(q.message) for q in record]
        )
        assert (
            '"end_date" not found within input date time range. Defaulting to maximum time step in xarray object.'
            in [str(q.message) for q in record]
        )

    def test_warnings(self):
        da = open_dataset(self.nc_poslons).tas

        with pytest.raises(ValueError):
            subset.subset_time(da, start_date="2059", end_date="2050")

        with pytest.raises(TypeError):
            subset.subset_time(da, start_yr=2050, end_yr=2059)

        with pytest.warns(None) as record:
            subset.subset_time(
                da,
                start_date=2050,  # noqa
                end_date=2055,  # noqa
            )
        assert (
            'start_date and end_date require dates in (type: str) using formats of "%Y", "%Y-%m" or "%Y-%m-%d".'
            in [str(q.message) for q in record]
        )

    def test_time_start_only(self):
        da = open_dataset(self.nc_poslons).tas
        yr_st = "2050"

        # start date only
        with pytest.warns(None):
            out = subset.subset_time(da, start_date=f"{yr_st}-01")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

        with pytest.warns(None):
            out = subset.subset_time(da, start_date=f"{yr_st}-07")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

        with pytest.warns(None):
            out = subset.subset_time(da, start_date=f"{yr_st}-07-15")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 15)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

    def test_time_end_only(self):
        da = open_dataset(self.nc_poslons).tas
        yr_ed = "2059"

        # end date only
        with pytest.warns(None):
            out = subset.subset_time(da, end_date=f"{yr_ed}-01")
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 1)
        np.testing.assert_array_equal(out.time.max().dt.day, 31)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

        with pytest.warns(None):
            out = subset.subset_time(da, end_date=f"{yr_ed}-06-15")
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 15)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

    def test_time_incomplete_years(self):
        da = open_dataset(self.nc_poslons).tas
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_time(
            da, start_date=f"{yr_st}-07-01", end_date=f"{yr_ed}-06-30"
        )
        out1 = subset.subset_time(da, start_date=f"{yr_st}-07", end_date=f"{yr_ed}-06")
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 1)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 30)


@pytest.mark.skipif(
    subset is False, reason="`clisops` subset utilities are not installed."
)
class TestSubsetGridPoint:
    nc_poslons = os.path.join("cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc")
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_2dlonlat = os.path.join("CRCM5", "tasmax_bby_198406_se.nc")

    def test_time_simple(self):
        da = open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))
        lon = -72.4
        lat = 46.1
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_gridpoint(
            da, lon=lon, lat=lat, start_date=yr_st, end_date=yr_ed
        )
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))

    @pytest.mark.skip(
        "We don't have a way of performing open_mfdataset access at the moment."
    )
    def test_dataset(self):
        da = xr.open_mfdataset(
            [self.nc_file, self.nc_file.replace("tasmax", "tasmin")],
            combine="by_coords",
        )
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    @pytest.mark.parametrize(
        "lon,lat", [([-72.4], [46.1]), ([-67.4, -67.3], [43.1, 46.1])]
    )
    @pytest.mark.parametrize("add_distance", [True, False])
    def test_simple(self, lat, lon, add_distance):
        da = open_dataset(self.nc_file).tasmax

        out = subset.subset_gridpoint(da, lon=lon, lat=lat, add_distance=add_distance)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        assert ("site" in out.dims) ^ (len(lat) == 1)
        assert ("distance" in out.coords) ^ (not add_distance)

    def test_irregular(self):
        da = open_dataset(self.nc_2dlonlat).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        assert "site" not in out.dims

        lon = [-72.4, -67.1]
        lat = [46.1, 48.2]
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        assert "site" in out.dims

        # dask for lon lat
        da.lon.chunk({"rlon": 10})
        da.lat.chunk({"rlon": 10})
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        # test_irregular transposed:
        da1 = open_dataset(self.nc_2dlonlat).tasmax
        dims = list(da1.dims)
        dims.reverse()
        daT = xr.DataArray(np.transpose(da1.values), dims=dims)
        for d in daT.dims:
            args = dict()
            args[d] = da1[d]
            daT = daT.assign_coords(**args)
        daT = daT.assign_coords(lon=(["rlon", "rlat"], np.transpose(da1.lon.values)))
        daT = daT.assign_coords(lat=(["rlon", "rlat"], np.transpose(da1.lat.values)))

        out1 = subset.subset_gridpoint(daT, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out1.lon, lon, 1)
        np.testing.assert_almost_equal(out1.lat, lat, 1)
        np.testing.assert_array_equal(out, out1)

        # Dataset with tasmax, lon and lat as data variables (i.e. lon, lat not coords of tasmax)
        daT1 = xr.DataArray(np.transpose(da1.values), dims=dims)
        for d in daT1.dims:
            args = dict()
            args[d] = da1[d]
            daT1 = daT1.assign_coords(**args)
        dsT = xr.Dataset(data_vars=None, coords=daT1.coords)
        dsT["tasmax"] = daT1
        dsT["lon"] = xr.DataArray(np.transpose(da1.lon.values), dims=["rlon", "rlat"])
        dsT["lat"] = xr.DataArray(np.transpose(da1.lat.values), dims=["rlon", "rlat"])
        out2 = subset.subset_gridpoint(dsT, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out2.lon, lon, 1)
        np.testing.assert_almost_equal(out2.lat, lat, 1)
        np.testing.assert_array_equal(out, out2.tasmax)

        # Dataset with lon and lat as 1D arrays
        lon = -60
        lat = -45
        da = xr.DataArray(
            np.random.rand(5, 4),
            dims=("time", "site"),
            coords={"time": np.arange(5), "site": np.arange(4)},
        )
        ds = xr.Dataset(
            data_vars={
                "da": da,
                "lon": ("site", np.linspace(lon, lon + 10, 4)),
                "lat": ("site", np.linspace(lat, lat + 5, 4)),
            }
        )
        gp = subset.subset_gridpoint(ds, lon=lon, lat=lat)
        np.testing.assert_almost_equal(gp.lon, lon)
        np.testing.assert_almost_equal(gp.lat, lat)
        assert gp.site == 0

    def test_positive_lons(self):
        da = open_dataset(self.nc_poslons).tas
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        out = subset.subset_gridpoint(da, lon=lon + 360, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

    def test_raise(self):
        da = open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055-03-15", end_date="2055-03-14"
            )
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055", end_date="2052"
            )
        da = open_dataset(self.nc_2dlonlat).tasmax.drop_vars(names=["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_gridpoint(da, lon=-72.4, lat=46.1)

    def test_tolerance(self):
        da = open_dataset(self.nc_poslons).tas
        lon = -72.5
        lat = 46.2
        out = subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1)
        assert out.isnull().all()

        subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1e5)


@pytest.mark.skipif(
    subset is False, reason="`clisops` subset utilities are not installed."
)
class TestSubsetBbox:
    nc_poslons = os.path.join("cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc")
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_2dlonlat = os.path.join("CRCM5", "tasmax_bby_198406_se.nc")
    lon = [-75.4, -68]
    lat = [44.1, 47.1]
    lonGCM = [-70.0, -60.0]
    latGCM = [43.0, 59.0]

    @pytest.mark.skip(
        "We don't have a way of performing open_mfdataset access at the moment."
    )
    def test_dataset(self):
        da = xr.open_mfdataset(
            [self.nc_file, self.nc_file.replace("tasmax", "tasmin")],
            combine="by_coords",
        )
        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    def test_simple(self):
        da = open_dataset(self.nc_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat.values >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))

        da = open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))
        yr_st = 2050
        yr_ed = 2059

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date=str(yr_st),
            end_date=str(yr_ed),
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(
            da, lon_bnds=self.lon, lat_bnds=self.lat, start_date=str(yr_st)
        )

        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(
            da, lon_bnds=self.lon, lat_bnds=self.lat, end_date=str(yr_ed)
        )

        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())

    def test_irregular(self):
        da = open_dataset(self.nc_2dlonlat).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

        # for irregular lat lon grids data matrix remains rectangular in native proj
        # but with data outside bbox assigned nans.  This means it can have lon and lats outside the bbox.
        # Check only non-nans gridcells using mask
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon.values[mask1.values] >= np.min(self.lon))
        assert np.all(out.lon.values[mask1.values] <= np.max(self.lon))
        assert np.all(out.lat.values[mask1.values] >= np.min(self.lat))
        assert np.all(out.lat.values[mask1.values] <= np.max(self.lat))

    def test_irregular_dataset(self):
        da = open_dataset(self.nc_2dlonlat)
        out = subset.subset_bbox(da, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        variables = list(da.data_vars)
        variables.pop(variables.index("tasmax"))
        # only tasmax should be subsetted/masked others should remain untouched
        for v in variables:
            assert out[v].dims == da[v].dims
            np.testing.assert_array_equal(out[v], da[v])

        # ensure results are equal to previous test on DataArray only
        out1 = subset.subset_bbox(da.tasmax, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        np.testing.assert_array_equal(out1, out.tasmax)

        # additional test if dimensions have no coordinates
        da = da.drop_vars(["rlon", "rlat"])
        subset.subset_bbox(da.tasmax, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        # We don't test for equality with previous datasets.
        # Without coords, sel defaults to isel which doesn't include the last element.

    # test datasets with descending coords
    def test_inverted_coords(self):
        lon = np.linspace(-90, -60, 200)
        lat = np.linspace(40, 80, 100)
        da = xr.Dataset(
            data_vars=None, coords={"lon": np.flip(lon), "lat": np.flip(lat)}
        )
        da["data"] = xr.DataArray(
            np.random.rand(lon.size, lat.size), dims=["lon", "lat"]
        )

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(np.asarray(self.lon)))
        assert np.all(out.lon <= np.max(np.asarray(self.lon)))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))

    def test_badly_named_latlons(self):
        da = open_dataset(self.nc_file)
        extended_latlons = {"lat": "latitude", "lon": "longitude"}
        da_extended_names = da.rename(extended_latlons)
        out = subset.subset_bbox(
            da_extended_names, lon_bnds=self.lon, lat_bnds=self.lat
        )
        assert {"latitude", "longitude"}.issubset(out.dims)

        long_for_some_reason = {"lon": "long"}
        da_long = da.rename(long_for_some_reason)
        out = subset.subset_bbox(da_long, lon_bnds=self.lon, lat_bnds=self.lat)
        assert {"long"}.issubset(out.dims)

        lons_lats = {"lon": "lons", "lat": "lats"}
        da_lonslats = da.rename(lons_lats)
        out = subset.subset_bbox(da_lonslats, lon_bnds=self.lon, lat_bnds=self.lat)
        assert {"lons", "lats"}.issubset(out.dims)

    def test_single_bounds_rectilinear(self):
        da = open_dataset(self.nc_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        np.testing.assert_array_equal(out.lat, da.lat)
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lon.values >= np.min(self.lon))

        out = subset.subset_bbox(da, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        np.testing.assert_array_equal(out.lon, da.lon)
        assert np.all(out.lat <= np.max(self.lat))
        assert np.all(out.lat.values >= np.min(self.lat))

    def test_single_bounds_curvilinear(self):
        da = open_dataset(self.nc_2dlonlat).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert np.all(out.lon.values[mask1.values] <= np.max(self.lon))
        assert np.all(out.lon.values[mask1.values] >= np.min(self.lon))

        out = subset.subset_bbox(da, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert np.all(out.lat.values[mask1.values] <= np.max(self.lat))
        assert np.all(out.lat.values[mask1.values] >= np.min(self.lat))

    def test_positive_lons(self):
        da = open_dataset(self.nc_poslons).tas

        out = subset.subset_bbox(da, lon_bnds=self.lonGCM, lat_bnds=self.latGCM)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(np.asarray(self.lonGCM) + 360))
        assert np.all(out.lon <= np.max(np.asarray(self.lonGCM) + 360))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))

        out = subset.subset_bbox(
            da, lon_bnds=np.array(self.lonGCM) + 360, lat_bnds=self.latGCM
        )
        assert np.all(out.lon >= np.min(np.asarray(self.lonGCM) + 360))

    def test_time(self):
        da = open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date="2050",
            end_date="2059",
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.min().dt.year, 2050)
        np.testing.assert_array_equal(out.time.min().dt.month, 1)
        np.testing.assert_array_equal(out.time.min().dt.day, 1)
        np.testing.assert_array_equal(out.time.max().dt.year, 2059)
        np.testing.assert_array_equal(out.time.max().dt.month, 12)
        np.testing.assert_array_equal(out.time.max().dt.day, 31)

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date="2050-02-05",
            end_date="2059-07-15",
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.min().dt.year, 2050)
        np.testing.assert_array_equal(out.time.min().dt.month, 2)
        np.testing.assert_array_equal(out.time.min().dt.day, 5)
        np.testing.assert_array_equal(out.time.max().dt.year, 2059)
        np.testing.assert_array_equal(out.time.max().dt.month, 7)
        np.testing.assert_array_equal(out.time.max().dt.day, 15)

    def test_raise(self):
        da = open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_bbox(
                da,
                lon_bnds=self.lonGCM,
                lat_bnds=self.latGCM,
                start_date="2056",
                end_date="2055",
            )

        da = open_dataset(self.nc_2dlonlat).tasmax.drop_vars(names=["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

    def test_warnings(self):
        da = open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))

        with pytest.raises(TypeError):
            subset.subset_bbox(
                da, lon_bnds=self.lon, lat_bnds=self.lat, start_yr=2050, end_yr=2059
            )
        with pytest.warns(None) as record:
            subset.subset_bbox(
                da,
                lon_bnds=self.lon,
                lat_bnds=self.lat,
                start_date="2050",
                end_date="2055",
            )
        assert (
            '"start_yr" and "end_yr" (type: int) are being deprecated. Temporal subsets will soon exclusively'
            ' support "start_date" and "end_date" (type: str) using formats of "%Y", "%Y-%m" or "%Y-%m-%d".'
            not in [str(q.message) for q in record]
        )


@pytest.mark.skipif(
    subset is False, reason="`clisops` subset utilities are not installed."
)
class TestSubsetShape:
    nc_file = os.path.join("cmip5", "tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
    lons_2d_nc_file = os.path.join("CRCM5", "tasmax_bby_198406_se.nc")
    nc_file_neglons = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    meridian_geojson = os.path.join(TESTS_DATA, "meridian.json")
    meridian_multi_geojson = os.path.join(TESTS_DATA, "meridian_multi.json")
    poslons_geojson = os.path.join(TESTS_DATA, "poslons.json")
    eastern_canada_geojson = os.path.join(TESTS_DATA, "eastern_canada.json")
    southern_qc_geojson = os.path.join(TESTS_DATA, "southern_qc_geojson.json")
    small_geojson = os.path.join(TESTS_DATA, "small_geojson.json")
    multi_regions_geojson = os.path.join(TESTS_DATA, "multi_regions.json")

    @staticmethod
    def compare_vals(ds, sub, vari, flag_2d=False):
        # check subsetted values against original
        imask = np.where(~np.isnan(sub[vari].isel(time=0)))
        if len(imask[0]) > 70:
            ii = np.random.randint(0, len(imask[0]), 70)
        else:
            ii = np.arange(0, len(imask[0]))
        for i in zip(imask[0][ii], imask[1][ii]):

            if flag_2d:
                lat1 = sub.lat[i[0], i[1]]
                lon1 = sub.lon[i[0], i[1]]
                np.testing.assert_array_equal(
                    subset.subset_gridpoint(sub, lon=lon1, lat=lat1)[vari],
                    subset.subset_gridpoint(ds, lon=lon1, lat=lat1)[vari],
                )
            else:
                lat1 = sub.lat.isel(lat=i[0])
                lon1 = sub.lon.isel(lon=i[1])
                np.testing.assert_array_equal(
                    sub[vari].sel(lon=lon1, lat=lat1), ds[vari].sel(lon=lon1, lat=lat1)
                )

    def test_wraps(self, tmp_netcdf_filename):
        ds = open_dataset(self.nc_file)

        # Polygon crosses meridian, a warning should be raised
        with pytest.warns(UserWarning):
            sub = subset.subset_shape(ds, self.meridian_geojson)

        # No time subsetting should occur.
        assert len(sub.tas) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tas.isel(time=0))), 285.064, 3
        )
        self.compare_vals(ds, sub, "tas")

        poly = gpd.read_file(self.meridian_multi_geojson)
        subtas = subset.subset_shape(ds.tas, poly)
        np.testing.assert_array_almost_equal(
            float(np.mean(subtas.isel(time=0))), 281.092, 3
        )

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))
            subset.subset_shape(ds, self.meridian_multi_geojson)

    def test_no_wraps(self, tmp_netcdf_filename):
        ds = open_dataset(self.nc_file)

        with pytest.warns(None) as record:
            sub = subset.subset_shape(ds, self.poslons_geojson)

        self.compare_vals(ds, sub, "tas")

        # No time subsetting should occur.
        assert len(sub.tas) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tas.isel(time=0))), 276.732, 3
        )
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [str(q.message) for q in record]
        )

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))
            subset.subset_shape(ds, self.poslons_geojson)

    def test_all_neglons(self):
        ds = open_dataset(self.nc_file_neglons)

        with pytest.warns(None) as record:
            sub = subset.subset_shape(ds, self.southern_qc_geojson)

        self.compare_vals(ds, sub, "tasmax")

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tasmax.isel(time=0))), 269.254, 3
        )
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [q.message for q in record]
        )

    def test_rotated_pole_with_time(self):
        ds = open_dataset(self.lons_2d_nc_file)

        with pytest.warns(None) as record:
            sub = subset.subset_shape(
                ds,
                self.eastern_canada_geojson,
                start_date="1984-06-01",
                end_date="1984-06-15",
            )

        self.compare_vals(
            ds.sel(time=slice("1984-06-01", "1984-06-15")), sub, "tasmax", flag_2d=True
        )

        # Should only have 15 days of data.
        assert len(sub.tasmax) == 15
        # Average max temperature at surface for region on June 1st, 1984 (time=0)
        np.testing.assert_allclose(float(np.mean(sub.tasmax.isel(time=0))), 289.634968)
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [str(q.message) for q in record]
        )

    def test_small_poly_buffer(self, tmp_netcdf_filename):
        ds = open_dataset(self.nc_file)

        with pytest.raises(ValueError):
            subset.subset_shape(ds, self.small_geojson)

        with pytest.raises(ValueError):
            subset.subset_shape(ds, self.small_geojson, buffer=0.6)

        sub = subset.subset_shape(ds, self.small_geojson, buffer=5)
        self.compare_vals(ds, sub, "tas")
        assert len(sub.lon.values) == 3
        assert len(sub.lat.values) == 3

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))

    def test_mask_multiregions(self):
        ds = open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson)
        regions.set_index("id")
        mask = subset.create_mask(
            x_dim=ds.lon, y_dim=ds.lat, poly=regions, wrap_lons=True
        )
        vals, counts = np.unique(mask.values[mask.notnull()], return_counts=True)
        assert all(vals == [0, 1, 2])
        assert all(counts == [58, 250, 22])

    def test_subset_multiregions(self):
        ds = open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson)
        regions.set_index("id")
        ds_sub = subset.subset_shape(ds, shape=regions)
        assert ds_sub.notnull().sum() == 58 + 250 + 22


@pytest.mark.skipif(
    subset is False, reason="`clisops` subset utilities are not installed."
)
class TestDistance:
    def test_values(self):
        # Check values are OK. Values taken from pyproj test.
        boston_lat = 42.0 + (15.0 / 60.0)
        boston_lon = -71.0 - (7.0 / 60.0)
        portland_lat = 45.0 + (31.0 / 60.0)
        portland_lon = -123.0 - (41.0 / 60.0)

        da = xr.DataArray(
            0, coords={"lon": [boston_lon], "lat": [boston_lat]}, dims=["lon", "lat"]
        )
        d = subset.distance(da, lon=portland_lon, lat=portland_lat)
        np.testing.assert_almost_equal(d, 4164074.239, decimal=3)

    def test_broadcasting(self):
        # Check output dimensions match lons and lats.
        lon = np.linspace(-180, 180, 20)
        lat = np.linspace(-90, 90, 30)
        da = xr.Dataset(data_vars=None, coords={"lon": lon, "lat": lat})
        da["data"] = xr.DataArray(
            np.random.rand(lon.size, lat.size), dims=["lon", "lat"]
        )

        d = subset.distance(da, lon=-34, lat=56).squeeze("site")
        assert d.dims == da.data.dims
        assert d.shape == da.data.shape
        assert d.units == "m"

        # Example of how to get the actual 2D indices.
        k = d.argmin()
        i, j = np.unravel_index(k, da.data.shape)
        assert d[i, j] == d.min()
