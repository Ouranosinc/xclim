import os

import numpy as np
import pytest
import xarray as xr

from xclim import subset

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


class TestSubsetGridPoint:
    nc_poslons = os.path.join(
        TESTS_DATA, "cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc"
    )
    nc_file = os.path.join(
        TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc"
    )
    nc_2dlonlat = os.path.join(TESTS_DATA, "CRCM5", "tasmax_bby_198406_se.nc")

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

    def test_simple(self):
        da = xr.open_dataset(self.nc_file).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
        yr_st = 2050
        yr_ed = 2059

        out = subset.subset_gridpoint(
            da, lon=lon, lat=lat, start_date=str(yr_st), end_date=str(yr_ed)
        )
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        # test time only
        out = subset.subset_gridpoint(da, start_date=str(yr_st), end_date=str(yr_ed))
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

    def test_time_simple(self):
        da = xr.open_dataset(self.nc_file).tasmax
        lon = -72.4
        lat = 46.1

        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
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

        # test different but equivalent strings
        out = subset.subset_gridpoint(da, start_date=yr_st, end_date=yr_ed)
        out1 = subset.subset_gridpoint(
            da,
            start_date="{yr_st}-01".format(yr_st=yr_st),
            end_date="{yr_ed}-12".format(yr_ed=yr_ed),
        )
        out2 = subset.subset_gridpoint(
            da,
            start_date="{yr_st}-01-01".format(yr_st=yr_st),
            end_date="{yr_ed}-12-31".format(yr_ed=yr_ed),
        )
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out, out2)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))

    def test_time_dates_outofbounds(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
        yr_st = "1950"
        yr_ed = "2099"

        out = subset.subset_gridpoint(
            da,
            start_date="{yr_st}-01".format(yr_st=yr_st),
            end_date="{yr_ed}-01".format(yr_ed=yr_ed),
        )
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

    def test_time_start_only(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
        yr_st = "2050"

        # start date only
        out = subset.subset_gridpoint(da, start_date="{yr_st}-01".format(yr_st=yr_st))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

        out = subset.subset_gridpoint(da, start_date="{yr_st}-07".format(yr_st=yr_st))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

        out = subset.subset_gridpoint(
            da, start_date="{yr_st}-07-15".format(yr_st=yr_st)
        )
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 15)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

    def test_time_end_only(self):

        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360

        yr_ed = "2059"

        # end date only
        out = subset.subset_gridpoint(da, end_date="{yr_ed}-01".format(yr_ed=yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 1)
        np.testing.assert_array_equal(out.time.max().dt.day, 31)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

        out = subset.subset_gridpoint(da, end_date="{yr_ed}-06-15".format(yr_ed=yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 15)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

    def test_time_incomplete_years(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_gridpoint(
            da,
            start_date="{yr_st}-07-01".format(yr_st=yr_st),
            end_date="{yr_ed}-06-30".format(yr_ed=yr_ed),
        )
        out1 = subset.subset_gridpoint(
            da,
            start_date="{yr_st}-07".format(yr_st=yr_st),
            end_date="{yr_ed}-06".format(yr_ed=yr_ed),
        )
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 1)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 30)

    def test_irregular(self):

        da = xr.open_dataset(self.nc_2dlonlat).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        # test_irregular transposed:
        da1 = xr.open_dataset(self.nc_2dlonlat).tasmax
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

    def test_positive_lons(self):
        da = xr.open_dataset(self.nc_poslons).tas
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        out = subset.subset_gridpoint(da, lon=lon + 360, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

    def test_raise(self):
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055-03-15", end_date="2055-03-14"
            )
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055", end_date="2052"
            )
        da = xr.open_dataset(self.nc_2dlonlat).tasmax.drop(["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_gridpoint(da, lon=-72.4, lat=46.1)

    def test_tolerance(self):
        da = xr.open_dataset(self.nc_poslons).tas
        lon = -72.5
        lat = 46.2
        with pytest.raises(ValueError):
            subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1)

        subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1e5)


class TestSubsetBbox:
    nc_poslons = os.path.join(
        TESTS_DATA, "cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc"
    )
    nc_file = os.path.join(
        TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc"
    )
    nc_2dlonlat = os.path.join(TESTS_DATA, "CRCM5", "tasmax_bby_198406_se.nc")
    lon = [-75.4, -68]
    lat = [44.1, 47.1]
    lonGCM = [-70.0, -60.0]
    latGCM = [43.0, 59.0]

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
        da = xr.open_dataset(self.nc_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat.values >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))

        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360
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
        da = xr.open_dataset(self.nc_2dlonlat).tasmax

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

    def test_irregular_datset(self):
        da = xr.open_dataset(self.nc_2dlonlat)
        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        variables = list(da.data_vars)
        variables.pop(variables.index("tasmax"))
        # only tasmax should be subsetted/masked others should remain untouched
        for v in variables:
            assert out[v].dims == da[v].dims
            np.testing.assert_array_equal(out[v], da[v])

        # ensure results are equal to previous test on DataArray only
        out1 = subset.subset_bbox(da.tasmax, lon_bnds=self.lon, lat_bnds=self.lat)
        np.testing.assert_array_equal(out1, out.tasmax)

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

    def test_single_bounds_rectilinear(self):
        da = xr.open_dataset(self.nc_file).tasmax

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
        da = xr.open_dataset(self.nc_2dlonlat).tasmax

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
        da = xr.open_dataset(self.nc_poslons).tas

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
        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360

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
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_bbox(
                da,
                lon_bnds=self.lonGCM,
                lat_bnds=self.latGCM,
                start_date="2056",
                end_date="2055",
            )

        da = xr.open_dataset(self.nc_2dlonlat).tasmax.drop(["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

    def test_warnings(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da["lon"] -= 360

        with pytest.warns(FutureWarning):
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
            not in [q.message for q in record]
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

        d = subset.distance(da, -34, 56)
        assert d.dims == da.data.dims
        assert d.shape == da.data.shape
        assert d.units == "m"

        # Example of how to get the actual 2D indices.
        k = d.argmin()
        i, j = np.unravel_index(k, da.data.shape)
        assert d[i, j] == d.min()
