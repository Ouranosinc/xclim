import numpy as np
from pyproj import Geod


def subset_bbox(da, lon_bnds=None, lat_bnds=None, start_yr=None, end_yr=None):
    """Subset a datarray or dataset spatially (and temporally) using a lat lon bounding box and years selection.

    Return a subsetted data array for grid points falling within a spatial bounding box
    defined by longitude and latitudinal bounds and for years falling within provided year bounds.

    Parameters
    ----------
    arr : xarray.DataArray or xarray.Dataset
      Input data.
    lon_bnds : list of floats
      List of maximum and minimum longitudinal bounds. Optional. Defaults to all longitudes in original data-array.
    lat_bnds :  list of floats
      List maximum and minimum latitudinal bounds. Optional. Defaults to all latitudes in original data-array.
    start_yr : int
      First year of the subset. Defaults to first year of input.
    end_yr : int
      Last year of the subset. Defaults to last year of input.

    Returns
    -------
    xarray.DataArray or xarray.DataSet
      subsetted data array or dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xr.open_dataset('pr.day.nc')
    Subset lat lon and years
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1999)
    Subset data array lat, lon and single year
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1990)
    Subset dataarray single year keep entire lon, lat grid
    >>> prSub = subset.subset_bbox(ds.pr,start_yr=1990,end_yr=1990) # one year only entire grid
    Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_bbox(ds,lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1999)
    """
    # check if trying to subset lon and lat

    if not lat_bnds is None or not lon_bnds is None:
        if hasattr(da, "lon") and hasattr(da, "lat"):
            if lon_bnds is None:
                lon_bnds = [da.lon.min(), da.lon.max()]

            lon_bnds = _check_lons(da, np.asarray(lon_bnds))

            lon_cond = (da.lon >= lon_bnds.min()) & (da.lon <= lon_bnds.max())

            if lat_bnds is None:
                lat_bnds = [da.lat.min(), da.lat.max()]

            lat_bnds = np.asarray(lat_bnds)
            lat_cond = (da.lat >= lat_bnds.min()) & (da.lat <= lat_bnds.max())
            dims = list(da.dims)

            if "lon" in dims and "lat" in dims:
                da = da.sel(lon=lon_cond, lat=lat_cond)
            else:
                ind = np.where(lon_cond & lat_cond)
                dims_lonlat = da.lon.dims
                # reduce size using isel
                args = {}
                for d in dims_lonlat:
                    coords = da[d][ind[dims_lonlat.index(d)]]
                    args[d] = slice(coords.min(), coords.max())
                da = da.sel(**args)
                lon_cond = (da.lon >= lon_bnds.min()) & (da.lon <= lon_bnds.max())
                lat_cond = (da.lat >= lat_bnds.min()) & (da.lat <= lat_bnds.max())

                # mask irregular grid with new lat lon conditions
                da = da.where(lon_cond & lat_cond, drop=True)
        else:
            raise (
                Exception(
                    'subset_bbox() requires input data with "lon" and "lat" dimensions, coordinates or data variables.'
                )
            )

    if start_yr or end_yr:
        da = subset_time(da, start_yr=start_yr, end_yr=end_yr)

    return da


def subset_gridpoint(da, lon=None, lat=None, start_yr=None, end_yr=None):
    """Extract a nearest gridpoint from datarray based on lat lon coordinate.
    Time series can optionally be subsetted by year(s)

    Return a subsetted data array (or dataset) for the grid point falling nearest the input
    longitude and latitudecoordinates. Optionally subset the data array for years falling
    within provided year bounds

    Parameters
    ----------
    da : xarray.DataArray or xarray.DataSet
      Input data.
    lon : float
      Longitude coordinate.
    lat:  float
      Latitude coordinate.
    start_yr : int
      First year of the subset. Defaults to first year of input.
    end_yr : int
      Last year of the subset. Defaults to last year of input.

    Returns
    -------
    xarray.DataArray or xarray.DataSet
      Subsetted data array or dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xr.open_dataset('pr.day.nc')
    Subset lat lon point and multiple years
    >>> prSub = subset.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1999)
    Subset lat, lon point and single year
    >>> prSub = subset.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1990)
     Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_gridpoint(ds, lon=-75,lat=45,start_yr=1990,end_yr=1999)
    """

    # check if trying to subset lon and lat
    if not lat is None and not lon is None:
        # make sure input data has 'lon' and 'lat'(dims, coordinates, or data_vars)
        if hasattr(da, "lon") and hasattr(da, "lat"):
            # adjust negative/positive longitudes if necessary
            lon = _check_lons(da, lon)

            dims = list(da.dims)

            # if 'lon' and 'lat' are present as data dimensions use the .sel method.
            if "lat" in dims and "lon" in dims:
                da = da.sel(lat=lat, lon=lon, method="nearest")
            else:
                g = Geod(ellps="WGS84")  # WGS84 ellipsoid - decent globaly
                lon1 = da.lon.values
                lat1 = da.lat.values
                shp_orig = lon1.shape
                lon1 = np.reshape(lon1, lon1.size)
                lat1 = np.reshape(lat1, lat1.size)
                # calculate geodesic distance between grid points and point of interest
                az12, az21, dist = g.inv(
                    lon1,
                    lat1,
                    np.broadcast_to(lon, lon1.shape),
                    np.broadcast_to(lat, lat1.shape),
                )
                dist = dist.reshape(shp_orig)
                iy, ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                xydims = [x for x in da.lon.dims]
                args = dict()
                args[xydims[0]] = iy
                args[xydims[1]] = ix
                da = da.isel(**args)
        else:
            raise (
                Exception(
                    'subset_gridpoint() requires input data with "lon" and "lat" coordinates or data variables.'
                )
            )

    if start_yr or end_yr:
        da = subset_time(da, start_yr=start_yr, end_yr=end_yr)

    return da


def subset_time(da, start_yr=None, end_yr=None):
    """Subset input data based on start and end years

    Return a subsetted data array (or dataset) for years falling
    within provided year bounds

    Parameters
    ----------
    da : xarray.DataArray or xarray.DataSet
      Input data.
    start_yr : int
      First year of the subset. Defaults to first year of input.
    end_yr : int
      Last year of the subset. Defaults to last year of input.

    Returns
    -------
    xarray.DataArray or xarray.DataSet
      Subsetted data array or dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xr.open_dataset('pr.day.nc')
    Subset multiple years
    >>> prSub = subset.subset_time(ds.pr,start_yr=1990,end_yr=1999)
    Subset single year
    >>> prSub = subset.subset_time(ds.pr,start_yr=1990,end_yr=1990)
    Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_time(ds,start_yr=1990,end_yr=1999)
    """

    if not start_yr:
        start_yr = da.time.dt.year.min()
    if not end_yr:
        end_yr = da.time.dt.year.max()

    if start_yr > end_yr:
        raise ValueError("Start date is after end date.")

    year_bnds = np.asarray([start_yr, end_yr])

    if len(year_bnds) == 1:
        time_cond = da.time.dt.year == year_bnds
    else:
        time_cond = (da.time.dt.year >= year_bnds.min()) & (
            da.time.dt.year <= year_bnds.max()
        )
    return da.sel(time=time_cond)


def _check_lons(da, lon_bnds):
    if np.all(da.lon > 0) and np.any(lon_bnds < 0):
        if isinstance(lon_bnds, float):
            lon_bnds += 360
        else:
            lon_bnds[lon_bnds < 0] += 360
    if np.all(da.lon < 0) and np.any(lon_bnds > 0):
        if isinstance(lon_bnds, float):
            lon_bnds -= 360
        else:
            lon_bnds[lon_bnds < 0] -= 360
    return lon_bnds
