import copy
import warnings
from functools import wraps
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import fiona.crs as fiocrs
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray
from pyproj import Geod
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from shapely.ops import split

__all__ = [
    "create_mask",
    "subset_bbox",
    "subset_gridpoint",
    "subset_shape",
    "subset_time",
]


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

        if isinstance(kwargs["start_date"], int) or isinstance(kwargs["end_date"], int):
            warnings.warn(
                "start_date and end_date require dates in (type: str) "
                'using formats of "%Y", "%Y-%m" or "%Y-%m-%d".',
                UserWarning,
                stacklevel=2,
            )
            kwargs["start_date"] = str(kwargs["start_date"])
            kwargs["end_date"] = str(kwargs["end_date"])

        try:
            da.time.sel(time=kwargs["start_date"])
        except KeyError:
            warnings.warn(
                '"start_date" not found within input date time range. Defaulting to minimum time step in '
                "xarray object.",
                UserWarning,
                stacklevel=2,
            )
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        try:
            da.time.sel(time=kwargs["end_date"])
        except KeyError:
            warnings.warn(
                '"end_date" not found within input date time range. Defaulting to maximum time step in '
                "xarray object.",
                UserWarning,
                stacklevel=2,
            )
            kwargs["end_date"] = da.time.max().dt.strftime("%Y").values

        if (
            da.time.sel(time=kwargs["start_date"]).min()
            > da.time.sel(time=kwargs["end_date"]).max()
        ):
            raise ValueError(
                f'Start date ("{kwargs["start_date"]}") is after end date ("{kwargs["end_date"]}").'
            )

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
            if np.all(args[0].lon >= 0) and np.all(kwargs[lon] < 0):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] += 360
                else:
                    kwargs[lon][kwargs[lon] < 0] += 360
            elif np.all(args[0].lon >= 0) and np.any(kwargs[lon] < 0):
                raise NotImplementedError(
                    f"Input longitude bounds ({kwargs[lon]}) cross the 0 degree meridian but"
                    " dataset longitudes are all positive."
                )
            if np.all(args[0].lon <= 0) and np.any(kwargs[lon] > 0):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] -= 360
                else:
                    kwargs[lon][kwargs[lon] < 0] -= 360

        return func(*args, **kwargs)

    return func_checker


def wrap_lons_and_split_at_greenwich(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        A decorator to split and reproject polygon vectors in a GeoDataFram whose values cross the Greenwich Meridian.
         Begins by examining whether the geometry bounds the supplied cross longitude = 0 and if so, proceeds to split
         the polygons at the meridian into new polygons and erase a small buffer to prevent invalid geometries when
         transforming the lons from WGS84 to WGS84 +lon_wrap=180 (longitudes from 0 to 360).

         Returns a GeoDataFrame with the new features in a wrap_lon WGS84 projection if needed.
        """
        try:
            poly = kwargs["poly"]
            x_dim = kwargs["x_dim"]
            wrap_lons = kwargs["wrap_lons"]
        except KeyError:
            return func(*args, **kwargs)

        if wrap_lons:
            if (np.min(x_dim) < 0 and np.max(x_dim) >= 360) or (
                np.min(x_dim) < -180 and np.max >= 180
            ):
                warnings.warn(
                    "Dataset doesn't seem to be using lons between 0 and 360 degrees or between -180 and 180 degrees."
                    " Tread with caution.",
                    UserWarning,
                    stacklevel=4,
                )
            split_flag = False
            if (poly.geometry.total_bounds[0] < 0) and (
                poly.geometry.total_bounds[2] > 0
            ):
                split_flag = True
                warnings.warn(
                    "Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."
                    " This feature is experimental. Output might not be accurate.",
                    UserWarning,
                    stacklevel=4,
                )

                # Create a meridian line at Greenwich, split polygons at this line and erase a buffer line
                union = Polygon(cascaded_union(poly.geometry))
                meridian = LineString([Point(0, 90), Point(0, -90)])
                buffered = meridian.buffer(0.000000001)
                split_polygons = split(union, meridian)
                # TODO: This doesn't seem to be thread safe in Travis CI on macOS. Merits testing with a local machine.
                buffered_split_polygons = [
                    feat for feat in split_polygons.difference(buffered)
                ]

                # Load split features into a new GeoDataFrame with WGS84 CRS
                split_gdf = gpd.GeoDataFrame(
                    list(range(len(buffered_split_polygons))),
                    geometry=buffered_split_polygons,
                    crs={"init": "epsg:4326"},
                )
                # split_gdf.crs = CRS.from_epsg(4326)
                split_gdf.columns = ["index", "geometry"]

                poly = split_gdf

            # Reproject features in WGS84 CSR to use 0 to 360 as longitudinal values
            poly = poly.to_crs(
                "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
            )
            crs1 = poly.crs
            if split_flag:
                warnings.warn(
                    "Rebuffering split polygons to ensure edge inclusion in selection",
                    UserWarning,
                    stacklevel=4,
                )
                poly = gpd.GeoDataFrame(poly.buffer(0.000000001), columns=["geometry"])
                poly.crs = crs1
            kwargs["poly"] = poly

        return func(*args, **kwargs)

    return func_checker


@wrap_lons_and_split_at_greenwich
def create_mask(
    *,
    x_dim: xarray.DataArray = None,
    y_dim: xarray.DataArray = None,
    poly: gpd.GeoDataFrame = None,
    wrap_lons: bool = False,
):
    """
    Parameters
    ----------
    x_dim : xarray.DataArray
      X or longitudinal dimension of xarray object.
    y_dim : xarray.DataArray
      Y or latitudinal dimension of xarray object.
    poly : gpd.GeoDataFrame
      GeoDataFrame used to create the xarray.DataArray mask.
    wrap_lons : bool
      Shift vector longitudes by -180,180 degrees to 0,360 degrees; Default = False

    Returns
    -------
    xarray.DataArray
    """
    if len(x_dim.shape) == 1 & len(y_dim.shape) == 1:
        # create a 2d grid of lon, lat values
        lon1, lat1 = np.meshgrid(
            np.asarray(x_dim.values), np.asarray(y_dim.values), indexing="ij"
        )
        dims_out = x_dim.dims + y_dim.dims
        coords_out = dict()
        coords_out[dims_out[0]] = x_dim.values
        coords_out[dims_out[1]] = y_dim.values
    else:
        lon1 = x_dim.values
        lat1 = y_dim.values
        dims_out = x_dim.dims
        coords_out = x_dim.coords

    # create pandas Dataframe from NetCDF lat and lon points
    df = pd.DataFrame(
        {"id": np.arange(0, lon1.size), "lon": lon1.flatten(), "lat": lat1.flatten()}
    )
    df["Coordinates"] = list(zip(df.lon, df.lat))
    df["Coordinates"] = df["Coordinates"].apply(Point)

    # create geodataframe (spatially referenced with shifted longitude values if needed).
    if wrap_lons:
        shifted = fiocrs.from_string(
            "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
        )
        gdf_points = gpd.GeoDataFrame(df, geometry="Coordinates", crs=shifted)
    else:
        gdf_points = gpd.GeoDataFrame(
            df, geometry="Coordinates", crs=fiocrs.from_epsg(4326)
        )

    # spatial join geodata points with region polygons and remove duplicates
    point_in_poly = gpd.tools.sjoin(gdf_points, poly, how="left", op="intersects")
    point_in_poly = point_in_poly.loc[~point_in_poly.index.duplicated(keep="first")]

    # extract polygon ids for points
    mask = point_in_poly["index_right"]
    mask_2d = np.array(mask).reshape(lat1.shape[0], lat1.shape[1])
    mask_2d = xarray.DataArray(mask_2d, dims=dims_out, coords=coords_out)
    return mask_2d


def subset_shape(
    ds: Union[xarray.DataArray, xarray.Dataset],
    shape: Union[str, Path],
    raster_crs: Optional[Union[str, int]] = None,
    shape_crs: Optional[Union[str, int]] = None,
    buffer: Optional[Union[int, float]] = None,
    wrap_lons: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a DataArray or Dataset spatially (and temporally) using a vector shape and date selection.

    Return a subsetted data array for grid points falling within the area of a Polygon and/or MultiPolygon shape,
      or grid points along the path of a LineString and/or MultiLineString.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
      Input values.
    shape : Union[str, Path]
      Path to shape file. Supports formats compatible with geopandas.
    raster_crs : Optional[Union[str, int]]
      EPSG number or PROJ4 string.
    shape_crs : Optional[Union[str, int]]
      EPSG number or PROJ4 string.
    buffer : Optional[Union[int, float]]
      Buffer the shape in order to select a larger region stemming from it. Units are based on the shape degrees/metres.
    wrap_lons: Optional[bool]
      Manually set whether vector longitudes should extend from 0 to 360 degrees.
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
    Tuple[Union[xarray.DataArray, xarray.Dataset], xarray.DataArray]

    Examples
    --------
    >>> from xclim import subset
    >>> import xarray as xr
    >>> pr = xarray.open_dataset('pr.day.nc').pr
    Subset data array by shape and multiple years
    >>> prSub = subset.subset_shape(pr, shape="/path/to/polygon.shp", start_yr='1990', end_yr='1999')
    Subset data array by shape and single year
    >>> prSub = subset.subset_shape(pr, shape="/path/to/polygon.shp", start_yr='1990', end_yr='1990')
    Subset multiple variables in a single dataset
    >>> ds = xarray.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_shape(ds, shape="/path/to/polygon.shp", start_yr='1990', end_yr='1999')
     # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> prSub = subset.subset_shape(ds.pr, shape="/path/to/polygon.shp", start_date='1990-03', end_date='1999-08')
    # Subset with specific start_dates and end_dates
    >>> prSub = \
            subset.subset_shape(ds.pr, shape="/path/to/polygon.shp", start_date='1990-03-13', end_date='1990-08-17')
    """
    # TODO : edge case using polygon splitting decorator touches original ds when subsetting?
    ds_copy = copy.deepcopy(ds)
    poly = gpd.GeoDataFrame.from_file(shape)

    if buffer is not None:
        poly.geometry = poly.buffer(buffer)

    # Get the shape's bounding box
    bounds = poly.bounds
    lon_bnds = (float(bounds.minx.values), float(bounds.maxx.values))
    lat_bnds = (float(bounds.miny.values), float(bounds.maxy.values))

    # If polygon doesn't cross prime meridian, subset bbox first to reduce processing time
    # Only case not implemented is when lon_bnds cross the 0 deg meridian but dataset grid has all positive lons
    try:
        ds_copy = subset_bbox(ds_copy, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    except NotImplementedError:
        pass

    if ds_copy.lon.size == 0 or ds_copy.lat.size == 0:
        raise ValueError(
            "No gridcell centroids found within provided polygon bounding box. "
            'Try using the "buffer" option to create an expanded area'
        )

    if start_date or end_date:
        ds_copy = subset_time(ds_copy, start_date=start_date, end_date=end_date)

    # Determine whether CRS types are the same between shape and raster
    if shape_crs is not None:
        try:
            shape_crs = fiocrs.from_epsg(shape_crs)
        except ValueError:
            try:
                shape_crs = fiocrs.from_string(shape_crs)
            except ValueError:
                raise
    else:
        shape_crs = poly.crs

    if raster_crs is not None:
        try:
            raster_crs = fiocrs.from_epsg(raster_crs)
        except ValueError:
            try:
                raster_crs = fiocrs.from_string(raster_crs)
            except ValueError:
                raise
    else:
        if np.min(ds_copy.lon) >= 0 and np.max(ds_copy.lon) <= 360:
            # PROJ4 definition for WGS84 with Prime Meridian at -180 deg lon.
            raster_crs = fiocrs.from_string(
                "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
            )
            wrap_lons = True
        else:
            raster_crs = fiocrs.from_epsg(4326)
            wrap_lons = False

    if (shape_crs != raster_crs) or (
        fiocrs.from_epsg(4326) not in [shape_crs, raster_crs]
    ):
        warnings.warn(
            "CRS definitions are not similar or both not using WGS84. Caveat emptor.",
            UserWarning,
            stacklevel=3,
        )

    mask_2d = create_mask(
        x_dim=ds_copy.lon, y_dim=ds_copy.lat, poly=poly, wrap_lons=wrap_lons
    )

    if np.all(np.isnan(mask_2d)):
        raise ValueError(
            "No gridcell centroids found within provided polygon. "
            'Try using the "buffer" option to create an expanded areas or verify polygon '
        )

    # loop through variables
    for v in ds_copy.data_vars:
        if set.issubset(set(mask_2d.dims), set(ds_copy[v].dims)):
            ds_copy[v] = ds_copy[v].where((~np.isnan(mask_2d)), drop=True)

    # Remove coordinates where all values are outside of region mask
    if "lon" in ds_copy.dims:
        ds_copy = ds_copy.dropna(dim="lon", how="all")
        ds_copy = ds_copy.dropna(dim="lat", how="all")
    else:  # curvilinear case
        for d in ds_copy.lon.dims:
            ds_copy = ds_copy.dropna(dim=d, how="all")

    # Add a CRS definition as a coordinate for reference purposes
    if wrap_lons:
        ds_copy.coords["crs"] = 0
        ds_copy.coords["crs"].attrs = dict(
            spatial_ref="+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
        )

    return ds_copy


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

    TODO: returns the what?
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
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45], start_yr='1990', end_yr='1999')
    Subset data array lat, lon and single year
    >>> prSub = subset.subset_bbox(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45], start_yr='1990', end_yr='1990')
    Subset dataarray single year keep entire lon, lat grid
    >>> prSub = subset.subset_bbox(ds.pr, start_yr='1990', end_yr='1990') # one year only entire grid
    Subset multiple variables in a single dataset
    >>> ds = xarray.open_mfdataset(['pr.day.nc','tas.day.nc'])
    >>> dsSub = subset.subset_bbox(ds, lon_bnds=[-75, -70], lat_bnds=[40, 45], start_yr='1990', end_yr='1999')
     # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> prSub = \
        subset.subset_time(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45],start_date='1990-03', end_date='1999-08')
    # Subset with specific start_dates and end_dates
    >>> prSub = subset.subset_time(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45],\
                                    start_date='1990-03-13', end_date='1990-08-17')
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
            lat_b = None
            lat_cond = True

        if lon_bnds is not None:
            lon_b = assign_bounds(lon_bnds, da.lon)
            lon_cond = in_bounds(lon_b, da.lon)
        else:
            lon_b = None
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
                    f'{subset_gridpoint.__name__} requires input data with "lon" and "lat" coordinates or data variables.'
                )
            )

    if tolerance is not None:
        if dist.min() > tolerance:
            raise ValueError(
                f"Distance to closest point ({dist}) is larger than tolerance ({tolerance})"
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
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xclim.subset
    >>> da = xr.open_dataset("/path/to/file.nc").variable
    >>> d = xclim.subset.distance(da)
    >>> k = d.argmin()
    >>> i, j = np.unravel_index(k, d.shape)

    """
    g = Geod(ellps="WGS84")  # WGS84 ellipsoid - decent globaly

    def func(lons, lats):
        return g.inv(*np.broadcast_arrays(lons, lats, lon, lat))[2]

    out = xarray.apply_ufunc(func, da.lon, da.lat)
    out.attrs["units"] = "m"
    return out
