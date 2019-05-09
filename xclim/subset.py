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
    >>> from xclim import utils
    >>> ds = xr.open_dataset('pr.day.nc')
    Subset lat lon and years
    >>> prSub = utils.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1999)
    Subset data array lat, lon and single year
    >>> prSub = utils.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1990)
    Subset dataarray single year keep entire lon, lat grid
    >>> prSub = utils.subset_bbox(ds.pr,start_yr=1990,end_yr=1990) # one year only entire grid
    Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = utils.subset_bbox(ds,lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1999)
    """

    if lon_bnds is not None:
        lon_bnds = np.asarray(lon_bnds)
        if np.all(da.lon > 0) and np.any(lon_bnds < 0):
            lon_bnds[lon_bnds < 0] += 360
        if np.all(da.lon < 0) and np.any(lon_bnds > 0):
            lon_bnds[lon_bnds < 0] -= 360
        da = da.where((da.lon >= lon_bnds.min()) & (da.lon <= lon_bnds.max()), drop=True)

    if lat_bnds is not None:
        lat_bnds = np.asarray(lat_bnds)
        da = da.where((da.lat >= lat_bnds.min()) & (da.lat <= lat_bnds.max()), drop=True)

    if start_yr or end_yr:
        if not start_yr:
            start_yr = da.time.dt.year.min()
        if not end_yr:
            end_yr = da.time.dt.year.max()

        if start_yr > end_yr:
            raise ValueError("Start date is after end date.")

        year_bnds = np.asarray([start_yr, end_yr])
        da = da.where((da.time.dt.year >= year_bnds.min()) & (da.time.dt.year <= year_bnds.max()), drop=True)

    return da


def subset_gridpoint(da, lon, lat, start_yr=None, end_yr=None):
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
    >>> from xclim import utils
    >>> ds = xr.open_dataset('pr.day.nc')
    Subset lat lon point and multiple years
    >>> prSub = utils.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1999)
    Subset lat, lon point and single year
    >>> prSub = utils.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1990)
     Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = utils.subset_gridpoint(ds, lon=-75,lat=45,start_yr=1990,end_yr=1999)
    """

    g = Geod(ellps='WGS84')  # WGS84 ellipsoid - decent globaly
    # adjust negative/positive longitudes if necessary
    if np.all(da.lon > 0) and lon < 0:
        lon += 360
    if np.all(da.lon < 0) and lon > 0:
        lon -= 360

    if len(da.lon.shape) == 1 & len(da.lat.shape) == 1:
        # create a 2d grid of lon, lat values
        lon1, lat1 = np.meshgrid(np.asarray(da.lon.values), np.asarray(da.lat.values))

    else:
        lon1 = da.lon.values
        lat1 = da.lat.values
    shp_orig = lon1.shape
    lon1 = np.reshape(lon1, lon1.size)
    lat1 = np.reshape(lat1, lat1.size)
    # calculate geodesic distance between grid points and point of interest
    az12, az21, dist = g.inv(lon1, lat1, np.broadcast_to(lon, lon1.shape), np.broadcast_to(lat, lat1.shape))
    dist = dist.reshape(shp_orig)

    iy, ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    xydims = [x for x in da.dims if 'time' not in x]

    args = dict()
    args[xydims[0]] = iy
    args[xydims[1]] = ix
    out = da.isel(**args)
    if start_yr or end_yr:
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
            time_cond = (da.time.dt.year >= year_bnds.min()) & (da.time.dt.year <= year_bnds.max())
        out = out.where(time_cond, drop=True)

    return out
