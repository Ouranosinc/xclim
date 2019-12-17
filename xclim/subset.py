import warnings
from functools import wraps
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import xarray
from pyproj import Geod

__all__ = ["subset_bbox", "subset_gridpoint", "subset_time"]


def check_date_signature(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        A decorator to reformat the deprecated `start_yr` and `end_yr` calls to subset functions and return
         `start_date` and `end_date` to kwargs. Deprecation warnings are raised for deprecated usage.
        """

        _DEPRECATION_MESSAGE = (
            '"start_yr" and "end_yr" (type: int) are being deprecated. Temporal subsets will soon exclusively'
            ' support "start_date" and "end_date" (type: str) using formats of "%Y", "%Y-%m" or "%Y-%m-%d".'
        )

        if "start_yr" in kwargs:
            warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=3)
            if kwargs["start_yr"] is not None:
                kwargs["start_date"] = str(kwargs.pop("start_yr"))
            elif kwargs["start_yr"] is None:
                kwargs["start_date"] = None
        elif "start_date" not in kwargs:
            kwargs["start_date"] = None

        if "end_yr" in kwargs:
            if kwargs["end_yr"] is not None:
                warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=3)
                kwargs["end_date"] = str(kwargs.pop("end_yr"))
            elif kwargs["end_yr"] is None:
                kwargs["end_date"] = None
        elif "end_date" not in kwargs:
            kwargs["end_date"] = None

        return func(*args, **kwargs)

    return func_checker


def check_start_end_dates(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        A decorator to verify that start and end dates are valid in a time subsetting function.
        """
        da = args[0]
        if "start_date" not in kwargs:
            # use string for first year only - .sel() will include all time steps
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        if "end_date" not in kwargs:
            # use string for last year only - .sel() will include all time steps
            kwargs["end_date"] = da.time.max().dt.strftime("%Y").values

        try:
            da.time.sel(time=kwargs["start_date"])
        except KeyError:
            warnings.warn(
                '"start_date" not found within input date time range. Defaulting to minimum time step in '
                "xarray object.",
                Warning,
                stacklevel=2,
            )
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        try:
            da.time.sel(time=kwargs["end_date"])
        except KeyError:
            warnings.warn(
                '"end_date" not found within input date time range. Defaulting to maximum time step in '
                "xarray object.",
                Warning,
                stacklevel=2,
            )
            kwargs["end_date"] = da.time.max().dt.strftime("%Y").values

        if (
            da.time.sel(time=kwargs["start_date"]).min()
            > da.time.sel(time=kwargs["end_date"]).max()
        ):
            raise ValueError("Start date is after end date.")

        return func(*args, **kwargs)

    return func_checker


def check_lons(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        A decorator to reformat user-specified "lon" or "lon_bnds" values based on the lon dimensions of a supplied
         xarray DataSet or DataArray. Examines an xarray object longitude dimensions and depending on extent
         (either -180 to +180 or 0 to +360), will reformat user-specified lon values to be synonymous with
         xarray object boundaries.
         Returns a numpy array of reformatted `lon` or `lon_bnds` in kwargs with min() and max() values.
        """
        if "lon_bnds" in kwargs:
            lon = "lon_bnds"
        elif "lon" in kwargs:
            lon = "lon"
        else:
            return func(*args, **kwargs)

        if isinstance(args[0], (xarray.DataArray, xarray.Dataset)):
            if kwargs[lon] is None:
                kwargs[lon] = np.asarray(args[0].lon.min(), args[0].lon.max())
            else:
                kwargs[lon] = np.asarray(kwargs[lon])
            if np.all(args[0].lon >= 0) and np.any(kwargs[lon] < 0):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] += 360
                else:
                    kwargs[lon][kwargs[lon] < 0] += 360
            if np.all(args[0].lon <= 0) and np.any(kwargs[lon] > 0):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] -= 360
                else:
                    kwargs[lon][kwargs[lon] < 0] -= 360
        return func(*args, **kwargs)

    return func_checker


@check_lons
@check_date_signature
def subset_bbox(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon_bnds: Union[np.array, Tuple[Optional[float], Optional[float]]] = None,
    lat_bnds: Union[np.array, Tuple[Optional[float], Optional[float]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a datarray or dataset spatially (and temporally) using a lat lon bounding box and date selection.

    Return a subsetted data array for grid points falling within a spatial bounding box
    defined by longitude and latitudinal bounds and for dates falling within provided bounds.

    In the case of a lat-lon rectilinear grid, this simply returns the

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    lon_bnds : Union[np.array, Tuple[Optional[float], Optional[float]]]
      List of minimum and maximum longitudinal bounds. Optional. Defaults to all longitudes in original data-array.
    lat_bnds : Union[np.array, Tuple[Optional[float], Optional[float]]]
      List of minimum and maximum latitudinal bounds. Optional. Defaults to all latitudes in original data-array.
    start_date : Optional[str]
      Start date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to first day of input data-array.
    end_date : Optional[str]
      End date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to last day of input data-array.
    start_yr : int
      Deprecated
        First year of the subset. Defaults to first year of input data-array.
    end_yr : int
      Deprecated
        Last year of the subset. Defaults to last year of input data-array.


    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xarray.open_dataset('pr.day.nc')
    Subset lat lon and years
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr='1990',end_yr='1999')
    Subset data array lat, lon and single year
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr='1990',end_yr='1990')
    Subset dataarray single year keep entire lon, lat grid
    >>> prSub = subset.subset_bbox(ds.pr,start_yr='1990',end_yr='1990') # one year only entire grid
    Subset multiple variables in a single dataset
    >>> ds = xarray.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_bbox(ds,lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr='1990',end_yr='1999')
     # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> prSub = subset.subset_time(ds.pr,lon_bnds=[-75,-70],lat_bnds=[40,45],start_date='1990-03',end_date='1999-08')
    # Subset with specific start_dates and end_dates
    >>> prSub = subset.subset_time(ds.pr,lon_bnds=[-75,-70],lat_bnds=[40,45],start_date='1990-03-13',end_date='1990-08-17')
    """
    # start_date, end_date = _check_times(
    #     start_date=start_date, end_date=end_date, start_yr=start_yr, end_yr=end_yr
    # )

    # Rectilinear case (lat and lon are the 1D dimensions)
    if ("lat" in da.dims) or ("lon" in da.dims):

        if "lat" in da.dims and lat_bnds is not None:
            lat_bnds = _check_desc_coords(coord=da.lat, bounds=lat_bnds, dim="lat")
            da = da.sel(lat=slice(*lat_bnds))

        if "lon" in da.dims and lon_bnds is not None:
            lon_bnds = _check_desc_coords(coord=da.lon, bounds=lon_bnds, dim="lon")
            da = da.sel(lon=slice(*lon_bnds))

    # Curvilinear case (lat and lon are coordinates, not dimensions)
    elif (("lat" in da.coords) and ("lon" in da.coords)) or (
        ("lat" in da.data_vars) and ("lon" in da.data_vars)
    ):

        # Define a bounding box along the dimensions
        # This is an optimization, a simple `where` would work but take longer for large hi-res grids.
        if lat_bnds is not None:
            lat_b = assign_bounds(lat_bnds, da.lat)
            lat_cond = in_bounds(lat_b, da.lat)
        else:
            lat_cond = True

        if lon_bnds is not None:
            lon_b = assign_bounds(lon_bnds, da.lon)
            lon_cond = in_bounds(lon_b, da.lon)
        else:
            lon_cond = True

        # Crop original array using slice, which is faster than `where`.
        ind = np.where(lon_cond & lat_cond)
        args = {}
        for i, d in enumerate(da.lat.dims):
            coords = da[d][ind[i]]
            args[d] = slice(coords.min(), coords.max())
        da = da.sel(**args)

        # Recompute condition on cropped coordinates
        if lat_bnds is not None:
            lat_cond = in_bounds(lat_b, da.lat)

        if lon_bnds is not None:
            lon_cond = in_bounds(lon_b, da.lon)

        # Mask coordinates outside the bounding box
        if isinstance(da, xarray.Dataset):
            # If da is a xr.DataSet Mask only variables that have the
            # same 2d coordinates as da.lat (or da.lon)
            for var in da.data_vars:
                if set(da.lat.dims).issubset(da[var].dims):
                    da[var] = da[var].where(lon_cond & lat_cond, drop=True)
        else:

            da = da.where(lon_cond & lat_cond, drop=True)

    else:
        raise (
            Exception(
                'subset_bbox() requires input data with "lon" and "lat" dimensions, coordinates or variables'
            )
        )

    if start_date or end_date:
        da = subset_time(da, start_date=start_date, end_date=end_date)

    return da


def assign_bounds(
    bounds: Tuple[Optional[float], Optional[float]], coord: xarray.Coordinate
) -> tuple:
    """Replace unset boundaries by the minimum and maximum coordinates.

    Parameters
    ----------
    bounds : Tuple[Optional[float], Optional[float]]
      Boundaries.
    coord : xarray.Coordinate
      Grid coordinates.

    Returns
    -------
    tuple
      Lower and upper grid boundaries.

    """
    if bounds[0] > bounds[1]:
        bounds = np.flip(bounds)
    bn, bx = bounds
    bn = bn if bn is not None else coord.min()
    bx = bx if bx is not None else coord.max()
    return bn, bx


def in_bounds(bounds: Tuple[float, float], coord: xarray.Coordinate) -> bool:
    """Check which coordinates are within the boundaries."""
    bn, bx = bounds
    return (coord >= bn) & (coord <= bx)


def _check_desc_coords(coord, bounds, dim):
    """If dataset coordinates are descending reverse bounds"""
    if np.all(coord.diff(dim=dim) < 0):
        bounds = np.flip(bounds)
    return bounds


@check_lons
@check_date_signature
def subset_gridpoint(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Extract a nearest gridpoint from datarray based on lat lon coordinate.

    Return a subsetted data array (or Dataset) for the grid point falling nearest the input longitude and latitude
    coordinates. Optionally subset the data array for years falling within provided date bounds.
    Time series can optionally be subsetted by dates.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    lon : Optional[float]
      Longitude coordinate.
    lat : Optional[float]
      Latitude coordinate.
    start_date : Optional[str]
      Start date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to first day of input data-array.
    end_date : Optional[str]
      End date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to last day of input data-array.
    start_yr : int
      Deprecated
        First year of the subset. Defaults to first year of input data-array.
    end_yr : int
      Deprecated
        Last year of the subset. Defaults to last year of input data-array.
    tolerance : Optional[float]
      Raise error if the distance to the nearest gridpoint is larger than tolerance in meters.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xarray.open_dataset('pr.day.nc')
    Subset lat lon point and multiple years
    >>> prSub = subset.subset_gridpoint(ds.pr, lon=-75,lat=45,start_date='1990',end_date='1999')
    Subset lat, lon point and single year
    >>> prSub = subset.subset_gridpoint(ds.pr, lon=-75,lat=45,start_date='1990',end_date='1999')
     Subset multiple variables in a single dataset
    >>> ds = xarray.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_gridpoint(ds, lon=-75,lat=45,start_date='1990',end_date='1999')
    # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> prSub = subset.subset_time(ds.pr,lon=-75, lat=45, start_date='1990-03',end_date='1999-08')
    # Subset with specific start_dates and end_dates
    >>> prSub = subset.subset_time(ds.pr,lon=-75,lat=45, start_date='1990-03-13',end_date='1990-08-17')
    """

    # check if trying to subset lon and lat
    if lat is not None and lon is not None:
        # make sure input data has 'lon' and 'lat'(dims, coordinates, or data_vars)
        if hasattr(da, "lon") and hasattr(da, "lat"):
            dims = list(da.dims)

            # if 'lon' and 'lat' are present as data dimensions use the .sel method.
            if "lat" in dims and "lon" in dims:
                da = da.sel(lat=lat, lon=lon, method="nearest")

                if tolerance is not None:
                    # Calculate the geodesic distance between grid points and the point of interest.
                    dist = distance(da, lon, lat)

            else:
                # Calculate the geodesic distance between grid points and the point of interest.
                dist = distance(da, lon, lat)

                # Find the indices for the closest point
                iy, ix = np.unravel_index(dist.argmin(), dist.shape)

                # Select data from closest point
                xydims = [x for x in dist.dims]
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

    if tolerance is not None:
        if dist.min() > tolerance:
            raise ValueError(
                "Distance to closest point ({}) is larger than tolerance ({})".format(
                    dist, tolerance
                )
            )

    if start_date or end_date:
        da = subset_time(da, start_date=start_date, end_date=end_date)

    return da


@check_start_end_dates
def subset_time(
    da: Union[xarray.DataArray, xarray.Dataset],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input data based on start and end years.

    Return a subsetted data array (or dataset) for dates falling within the provided bounds.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    start_date : Optional[str]
      Start date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to first day of input data-array.
    end_date : Optional[str]
      End date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to last day of input data-array.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    >>> from xclim import subset
    >>> ds = xarray.open_dataset('pr.day.nc')
    # Subset complete years
    >>> prSub = subset.subset_time(ds.pr,start_date='1990',end_date='1999')
    # Subset single complete year
    >>> prSub = subset.subset_time(ds.pr,start_date='1990',end_date='1990')
    # Subset multiple variables in a single dataset
    >>> ds = xarray.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_time(ds,start_date='1990',end_date='1999')
    # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> prSub = subset.subset_time(ds.pr,start_date='1990-03',end_date='1999-08')
    # Subset with specific start_dates and end_dates
    >>> prSub = subset.subset_time(ds.pr,start_date='1990-03-13',end_date='1990-08-17')

    Notes
    -----
    TODO add notes about different calendar types. Avoid "%Y-%m-31". If you want complete month use only "%Y-%m".
    """

    return da.sel(time=slice(start_date, end_date))


def distance(da, lon, lat):
    """Return distance to point in meters.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    lon : Optional[float]
      Longitude coordinate.
    lat : Optional[float]
      Latitude coordinate.

    Returns
    -------
    xarray.DataArray
      Distance in meters to point.

    Note
    ----
    To get the indices from closest point, use
    >>> d = distance(da)
    >>> k = d.argmin()
    >>> i, j = numpy.unravel_index(k, d.shape)

    """
    g = Geod(ellps="WGS84")  # WGS84 ellipsoid - decent globaly

    def func(lons, lats):
        return g.inv(*np.broadcast_arrays(lons, lats, lon, lat))[2]

    out = xarray.apply_ufunc(func, da.lon, da.lat)
    out.attrs["units"] = "m"
    return out
