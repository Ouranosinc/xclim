"""Subset module."""
import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray
from pyproj import Geod
from pyproj.crs import CRS
from shapely import vectorized
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import cascaded_union, split

__all__ = [
    "create_mask",
    "create_mask_vectorize",
    "distance",
    "subset_bbox",
    "subset_gridpoint",
    "subset_shape",
    "subset_time",
]


def check_start_end_dates(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Verify that start and end dates are valid in a time subsetting function."""
        da = args[0]
        if "start_date" not in kwargs or kwargs["start_date"] is None:
            # use string for first year only - .sel() will include all time steps
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        if "end_date" not in kwargs or kwargs["end_date"] is None:
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
        Reformat user-specified "lon" or "lon_bnds" values based on the lon dimensions of a supplied Dataset or DataArray.

        Examines an xarray object longitude dimensions and depending on extent (either -180 to +180 or 0 to +360),
        will reformat user-specified lon values to be synonymous with xarray object boundaries.
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


def check_latlon_dimnames(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        Examine the names of the latitude and longitude dimensions and rename them temporarily.

        Checks here ensure that the names supplied via the xarray object dims are changed to be synonymous with subset
        algorithm dimensions, conversions are saved and are then undone to the processed file.
        """
        if range(len(args)) == 0:
            return func(*args, **kwargs)

        formatted_args = list()
        conversion = dict()
        for argument in args:
            if isinstance(argument, (xarray.DataArray, xarray.Dataset)):
                dims = argument.dims
            else:
                logging.info(f"No file or no dimensions found in arg `{argument}`.")
                formatted_args.append(argument)
                continue

            if not {"lon", "lat"}.issubset(dims):
                if {"long"}.issubset(dims):
                    conversion["long"] = "lon"
                elif {"latitude", "longitude"}.issubset(dims):
                    conversion["latitude"] = "lat"
                    conversion["longitude"] = "lon"
                elif {"lats", "lons"}.issubset(dims):
                    conversion["lats"] = "lat"
                    conversion["lons"] = "lon"
                if not conversion and not {"rlon", "rlat"}.issubset(dims):
                    warnings.warn(
                        f"lat and lon-like dimensions are not found among arg `{argument}` dimensions: {list(dims)}."
                    )
                argument = argument.rename(conversion)

            formatted_args.append(argument)

        final = func(*formatted_args, **kwargs)

        for k, v in conversion.items():
            final = final.rename({v: k})

        return final

    return func_checker


def convert_lat_lon_to_da(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        Transform input lat, lon to DataArrays.

        Input can be int, float or any iterable.
        Expects a DataArray as first argument and checks is dim "site" already exists,
        uses "_site" in that case.

        If the input are not already DataArrays, the new lon and lat objects are 1D DataArrays
        with dimension "site".
        """
        lat = kwargs.pop("lat", None)
        lon = kwargs.pop("lon", None)
        if not isinstance(lat, (type(None), xarray.DataArray)) or not isinstance(
            lon, (type(None), xarray.DataArray)
        ):
            try:
                if len(lat) != len(lon):
                    raise ValueError("'lat' and 'lon' must have the same length")
            except TypeError:  # They have no len : not iterables
                lat = [lat]
                lon = [lon]
            ptdim = xarray.core.utils.get_temp_dimname(args[0].dims, "site")
            if ptdim != "site" and len(lat) > 1:
                warnings.warn(
                    f"Dimension 'site' already on input, output will use {ptdim} instead."
                )
            lon = xarray.DataArray(lon, dims=(ptdim,))
            lat = xarray.DataArray(lat, dims=(ptdim,))
        return func(*args, lat=lat, lon=lon, **kwargs)

    return func_checker


def wrap_lons_and_split_at_greenwich(func):
    @wraps(func)
    def func_checker(*args, **kwargs):
        """
        Split and reproject polygon vectors in a GeoDataFrame whose values cross the Greenwich Meridian.

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
                # TODO: This should raise an exception, right?
                warnings.warn(
                    "DataArray doesn't seem to be using lons between 0 and 360 degrees or between -180 and 180 degrees."
                    " Tread with caution.",
                    UserWarning,
                    stacklevel=4,
                )
            split_flag = False
            for index, feature in poly.iterrows():
                if (feature.geometry.bounds[0] < 0) and (
                    feature.geometry.bounds[2] > 0
                ):
                    split_flag = True
                    warnings.warn(
                        "Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."
                        " This feature is experimental. Output might not be accurate.",
                        UserWarning,
                        stacklevel=4,
                    )

                    # Create a meridian line at Greenwich, split polygons at this line and erase a buffer line
                    if isinstance(feature.geometry, MultiPolygon):
                        union = MultiPolygon(cascaded_union(feature.geometry))
                    else:
                        union = Polygon(cascaded_union(feature.geometry))
                    meridian = LineString([Point(0, 90), Point(0, -90)])
                    buffered = meridian.buffer(0.000000001)
                    split_polygons = split(union, meridian)
                    buffered_split_polygons = [
                        feat.difference(buffered) for feat in split_polygons
                    ]

                    # Cannot assign iterable with `at` (pydata/pandas#26333) so a small hack:
                    # Load split features into a new GeoDataFrame with WGS84 CRS
                    split_gdf = gpd.GeoDataFrame(
                        geometry=[cascaded_union(buffered_split_polygons)],
                        crs=CRS(4326),
                    )
                    poly.at[[index], "geometry"] = split_gdf.geometry.values

            # Reproject features in WGS84 CSR to use 0 to 360 as longitudinal values
            wrapped_lons = CRS.from_string(
                "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
            )

            poly = poly.to_crs(crs=wrapped_lons)
            if split_flag:
                warnings.warn(
                    "Rebuffering split polygons to ensure edge inclusion in selection.",
                    UserWarning,
                    stacklevel=4,
                )
                poly = gpd.GeoDataFrame(poly.buffer(0.000000001), columns=["geometry"])
                poly.crs = wrapped_lons

            kwargs["poly"] = poly

        return func(*args, **kwargs)

    return func_checker


@wrap_lons_and_split_at_greenwich
def create_mask_vectorize(
    *,
    x_dim: xarray.DataArray = None,
    y_dim: xarray.DataArray = None,
    poly: gpd.GeoDataFrame = None,
    wrap_lons: bool = False,
    check_overlap: bool = False,
):
    """Create a mask with values corresponding to the features in a GeoDataFrame using vectorize methods.

    The returned mask's points have the value of the first geometry of `poly` they fall in.

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
    check_overlap: bool
      Perform a check to verify if shapes contain overlapping geometries.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import create_mask_vectorize  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_tasmin_file)  # doctest: +SKIP
    >>> polys = gpd.read_file(path_to_multi_shape_file)  # doctest: +SKIP
    ...
    # Get a mask from all polygons in the shape file
    >>> mask = create_mask_vectorize(x_dim=ds.lon, y_dim=ds.lat, poly=polys)  # doctest: +SKIP
    >>> ds = ds.assign_coords(regions=mask)  # doctest: +SKIP
    ...
    # Operations can be applied to each regions with  `groupby`. Ex:
    >>> ds = ds.groupby('regions').mean()  # doctest: +SKIP
    ...
    # Extra step to retrieve the names of those polygons stored in another column (here "id")
    >>> region_names = xr.DataArray(polys.id, dims=('regions',))  # doctest: +SKIP
    >>> ds = ds.assign_coords(regions_names=region_names)  # doctest: +SKIP
    """
    if check_overlap:
        _check_has_overlaps(polygons=poly)
    if wrap_lons:
        warnings.warn("Wrapping longitudes at 180 degrees.")

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

    # try vectorize
    mask = np.zeros(lat1.shape) + np.nan
    for pp in poly.index:
        for vv in poly[poly.index == pp].geometry.values:
            b1 = vectorized.contains(vv, lon1.flatten(), lat1.flatten()).reshape(
                lat1.shape
            )
            mask[b1] = pp

    mask = xarray.DataArray(mask, dims=dims_out, coords=coords_out)

    return mask


@wrap_lons_and_split_at_greenwich
def create_mask(
    *,
    x_dim: xarray.DataArray = None,
    y_dim: xarray.DataArray = None,
    poly: gpd.GeoDataFrame = None,
    wrap_lons: bool = False,
    check_overlap: bool = False,
):
    """Create a mask with values corresponding to the features in a GeoDataFrame using spatial join methods.

    The returned mask's points have the value of the first geometry of `poly` they fall in.

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
    check_overlap: bool
      Perform a check to verify if shapes contain overlapping geometries.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> from xclim.subset import create_mask  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_tasmin_file)  # doctest: +SKIP
    >>> polys = gpd.read_file(path_to_multi_shape_file)  # doctest: +SKIP
    ...
    # Get a mask from all polygons in the shape file
    >>> mask = create_mask(x_dim=ds.lon, y_dim=ds.lat, poly=polys)  # doctest: +SKIP
    >>> ds = ds.assign_coords(regions=mask)  # doctest: +SKIP
    ...
    # Operations can be applied to each regions with  `groupby`. Ex:
    >>> ds = ds.groupby('regions').mean()  # doctest: +SKIP
    ...
    # Extra step to retrieve the names of those polygons stored in the "id" column
    >>> region_names = xr.DataArray(polys.id, dims=('regions',))  # doctest: +SKIP
    >>> ds = ds.assign_coords(regions_names=region_names)  # doctest: +SKIP
    """
    wgs84 = CRS(4326)

    if check_overlap:
        _check_has_overlaps(polygons=poly)
    if wrap_lons:
        warnings.warn("Wrapping longitudes at 180 degrees.")

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

    # create GeoDataFrame (spatially referenced with shifted longitude values if needed).
    if wrap_lons:
        wgs84 = CRS.from_string(
            "+proj=longlat +datum=WGS84 +no_defs +type=crs +lon_wrap=180"
        )
    gdf_points = gpd.GeoDataFrame(df, geometry="Coordinates", crs=wgs84)

    # spatial join geodata points with region polygons and remove duplicates
    point_in_poly = gpd.tools.sjoin(gdf_points, poly, how="left", op="intersects")
    point_in_poly = point_in_poly.loc[~point_in_poly.index.duplicated(keep="first")]

    # extract polygon ids for points
    mask = point_in_poly["index_right"]
    mask_2d = np.array(mask).reshape(lat1.shape[0], lat1.shape[1])
    mask_2d = xarray.DataArray(mask_2d, dims=dims_out, coords=coords_out)
    return mask_2d


@check_latlon_dimnames
def subset_shape(
    ds: Union[xarray.DataArray, xarray.Dataset],
    shape: Union[str, Path, gpd.GeoDataFrame],
    vectorize: bool = True,
    raster_crs: Optional[Union[str, int]] = None,
    shape_crs: Optional[Union[str, int]] = None,
    buffer: Optional[Union[int, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a DataArray or Dataset spatially (and temporally) using a vector shape and date selection.

    Return a subset of a DataArray or Dataset for grid points falling within the area of a Polygon and/or
    MultiPolygon shape, or grid points along the path of a LineString and/or MultiLineString.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
      Input values.
    shape : Union[str, Path, gpd.GeoDataFrame]
      Path to shape file, or directly a geodataframe. Supports formats compatible with geopandas.
    vectorize: bool
      Whether to use the spatialjoin or vectorize backend.
    raster_crs : Optional[Union[str, int]]
      EPSG number or PROJ4 string.
    shape_crs : Optional[Union[str, int]]
      EPSG number or PROJ4 string.
    buffer : Optional[Union[int, float]]
      Buffer the shape in order to select a larger region stemming from it. Units are based on the shape degrees/metres.
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
      A subset of `ds`

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import subset_shape  # doctest: +SKIP
    >>> pr = xr.open_dataset(path_to_pr_file).pr  # doctest: +SKIP
    ...
    # Subset data array by shape
    >>> prSub = subset_shape(pr, shape=path_to_shape_file)  # doctest: +SKIP
    ...
    # Subset data array by shape and single year
    >>> prSub = subset_shape(pr, shape=path_to_shape_file, start_date='1990-01-01', end_date='1990-12-31')  # doctest: +SKIP
    ...
    # Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset([path_to_tasmin_file, path_to_tasmax_file])  # doctest: +SKIP
    >>> dsSub = subset_shape(ds, shape=path_to_shape_file)  # doctest: +SKIP
    """
    wgs84 = CRS(4326)
    # PROJ4 definition for WGS84 with longitudes ranged between -180/+180.
    wgs84_wrapped = CRS.from_string(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs lon_wrap=180"
    )

    if isinstance(ds, xarray.DataArray):
        ds_copy = ds._to_temp_dataset()
    else:
        ds_copy = ds.copy()

    if isinstance(shape, gpd.GeoDataFrame):
        poly = shape.copy()
    else:
        poly = gpd.GeoDataFrame.from_file(shape)

    if buffer is not None:
        poly.geometry = poly.buffer(buffer)

    # Get the shape's bounding box.
    minx, miny, maxx, maxy = poly.total_bounds
    lon_bnds = (minx, maxx)
    lat_bnds = (miny, maxy)

    # If polygon doesn't cross prime meridian, subset bbox first to reduce processing time
    # Only case not implemented is when lon_bnds cross the 0 deg meridian but dataset grid has all positive lons
    try:
        ds_copy = subset_bbox(ds_copy, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    except NotImplementedError:
        pass

    if ds_copy.lon.size == 0 or ds_copy.lat.size == 0:
        raise ValueError(
            "No grid cell centroids found within provided polygon bounding box. "
            'Try using the "buffer" option to create an expanded area.'
        )

    if start_date or end_date:
        ds_copy = subset_time(ds_copy, start_date=start_date, end_date=end_date)

    # Determine whether CRS types are the same between shape and raster
    if shape_crs is not None:
        try:
            shape_crs = CRS.from_user_input(shape_crs)
        except ValueError:
            raise
    else:
        shape_crs = CRS(poly.crs)

    wrap_lons = False
    if raster_crs is not None:
        try:
            raster_crs = CRS.from_user_input(raster_crs)
        except ValueError:
            raise
    else:
        if np.min(lat_bnds) < -90 or np.max(lat_bnds) > 90:
            raise ValueError("Latitudes exceed domain of WGS84 coordinate system.")
        if np.min(lon_bnds) < -180 or np.max(lon_bnds) > 180:
            raise ValueError("Longitudes exceed domain of WGS84 coordinate system.")

        try:
            # Extract CF-compliant CRS_WKT from crs variable.
            raster_crs = CRS.from_cf(ds_copy.crs.attrs)
        except AttributeError:
            if np.min(ds_copy.lon) >= 0 and np.max(ds_copy.lon) <= 360:
                wrap_lons = True
                raster_crs = wgs84_wrapped
            else:
                raster_crs = wgs84
    _check_crs_compatibility(shape_crs=shape_crs, raster_crs=raster_crs)

    # Create mask using the vectorize or spatial join methods.
    if vectorize:
        mask_2d = create_mask_vectorize(
            x_dim=ds_copy.lon, y_dim=ds_copy.lat, poly=poly, wrap_lons=wrap_lons
        )
    else:
        mask_2d = create_mask(
            x_dim=ds_copy.lon, y_dim=ds_copy.lat, poly=poly, wrap_lons=wrap_lons
        )

    if np.all(mask_2d.isnull()):
        raise ValueError(
            f"No grid cell centroids found within provided polygon bounds ({poly.bounds}). "
            'Try using the "buffer" option to create an expanded areas or verify polygon.'
        )

    # loop through variables
    for v in ds_copy.data_vars:
        if set.issubset(set(mask_2d.dims), set(ds_copy[v].dims)):
            ds_copy[v] = ds_copy[v].where(mask_2d.notnull())

    # Remove coordinates where all values are outside of region mask
    for dim in mask_2d.dims:
        mask_2d = mask_2d.dropna(dim, how="all")
    ds_copy = ds_copy.sel({dim: mask_2d[dim] for dim in mask_2d.dims})

    # Add a CRS definition using CF conventions and as a global attribute in CRS_WKT for reference purposes
    ds_copy.attrs["crs"] = raster_crs.to_string()
    ds_copy["crs"] = 1
    ds_copy["crs"].attrs.update(raster_crs.to_cf())

    for v in ds_copy.variables:
        if {"lat", "lon"}.issubset(set(ds_copy[v].dims)):
            ds_copy[v].attrs["grid_mapping"] = "crs"

    if isinstance(ds, xarray.DataArray):
        return ds._from_temp_dataset(ds_copy)
    return ds_copy


@check_latlon_dimnames
@check_lons
def subset_bbox(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon_bnds: Union[np.array, Tuple[Optional[float], Optional[float]]] = None,
    lat_bnds: Union[np.array, Tuple[Optional[float], Optional[float]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a DataArray or Dataset spatially (and temporally) using a lat lon bounding box and date selection.

    Return a subset of a DataArray or Dataset for grid points falling within a spatial bounding box
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

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import subset_bbox  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_pr_file)  # doctest: +SKIP
    ...
    # Subset lat lon
    >>> prSub = subset_bbox(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45])  # doctest: +SKIP
    """
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
        args = dict()
        for i, d in enumerate(da.lat.dims):
            coords = da[d][ind[i]]
            args[d] = slice(coords.min().values, coords.max().values)
        # If the dims of lat and lon do not have coords, sel defaults to isel,
        # and then the last element is not returned.
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
                f'{subset_bbox.__name__} requires input data with "lon" and "lat" dimensions, coordinates, or variables.'
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
    """If Dataset coordinates are descending reverse bounds."""
    if np.all(coord.diff(dim=dim) < 0):
        bounds = np.flip(bounds)
    return bounds


def _check_has_overlaps(polygons: gpd.GeoDataFrame):
    non_overlapping = []
    for n, p in enumerate(polygons["geometry"][:-1], 1):
        if not any(p.overlaps(g) for g in polygons["geometry"][n:]):
            non_overlapping.append(p)
    if len(polygons) != len(non_overlapping):
        warnings.warn(
            "List of shapes contains overlap between features. Results will vary on feature order.",
            UserWarning,
            stacklevel=5,
        )


def _check_has_overlaps_old(polygons: gpd.GeoDataFrame):
    for i, (inda, pola) in enumerate(polygons.iterrows()):
        for (indb, polb) in polygons.iloc[i + 1 :].iterrows():
            if pola.geometry.intersects(polb.geometry):
                warnings.warn(
                    f"List of shapes contains overlap between {inda} and {indb}. Points will be assigned to {inda}.",
                    UserWarning,
                    stacklevel=5,
                )


def _check_crs_compatibility(shape_crs: CRS, raster_crs: CRS):
    """If CRS definitions are not WGS84 or incompatible, raise operation warnings."""
    wgs84 = CRS(4326)
    if not shape_crs.equals(raster_crs):
        if (
            "lon_wrap" in raster_crs.to_string()
            and "lon_wrap" not in shape_crs.to_string()
        ):
            warnings.warn(
                "CRS definitions are similar but raster lon values must be wrapped.",
                UserWarning,
                stacklevel=3,
            )
        elif not shape_crs.equals(wgs84) and not raster_crs.equals(wgs84):
            warnings.warn(
                "CRS definitions are not similar or both not using WGS84 datum. Tread with caution.",
                UserWarning,
                stacklevel=3,
            )


@check_latlon_dimnames
@check_lons
@convert_lat_lon_to_da
def subset_gridpoint(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon: Optional[Union[float, Sequence[float], xarray.DataArray]] = None,
    lat: Optional[Union[float, Sequence[float], xarray.DataArray]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tolerance: Optional[float] = None,
    add_distance: bool = False,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Extract one or more nearest gridpoint(s) from datarray based on lat lon coordinate(s).

    Return a subsetted data array (or Dataset) for the grid point(s) falling nearest the input longitude and latitude
    coordinates. Optionally subset the data array for years falling within provided date bounds.
    Time series can optionally be subsetted by dates.
    If 1D sequences of coordinates are given, the gridpoints will be concatenated along the new dimension "site".

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    lon : Optional[Union[float, Sequence[float], xarray.DataArray]]
      Longitude coordinate(s). Must be of the same length as lat.
    lat : Optional[Union[float, Sequence[float], xarray.DataArray]]
      Latitude coordinate(s). Must be of the same length as lon.
    start_date : Optional[str]
      Start date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to first day of input data-array.
    end_date : Optional[str]
      End date of the subset.
      Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
      Defaults to last day of input data-array.
    tolerance : Optional[float]
      Masks values if the distance to the nearest gridpoint is larger than tolerance in meters.
    add_distance: bool

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import subset_gridpoint  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_pr_file)  # doctest: +SKIP
    ...
    # Subset lat lon point
    >>> prSub = subset_gridpoint(ds.pr, lon=-75,lat=45)  # doctest: +SKIP
    ...
    # Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset([path_to_tasmax_file, path_to_tasmin_file])  # doctest: +SKIP
    >>> dsSub = subset_gridpoint(ds, lon=-75, lat=45)  # doctest: +SKIP
    """
    if lat is None or lon is None:
        raise ValueError("Insufficient coordinates provided to locate grid point(s).")

    ptdim = lat.dims[0]

    # make sure input data has 'lon' and 'lat'(dims, coordinates, or data_vars)
    if hasattr(da, "lon") and hasattr(da, "lat"):
        dims = list(da.dims)

        # if 'lon' and 'lat' are present as data dimensions use the .sel method.
        if "lat" in dims and "lon" in dims:
            da = da.sel(lat=lat, lon=lon, method="nearest")

            if tolerance is not None or add_distance:
                # Calculate the geodesic distance between grid points and the point of interest.
                dist = distance(da, lon=lon, lat=lat)
            else:
                dist = None

        else:
            # Calculate the geodesic distance between grid points and the point of interest.
            dist = distance(da, lon=lon, lat=lat)
            pts = []
            dists = []
            for site in dist[ptdim]:
                # Find the indices for the closest point
                inds = np.unravel_index(
                    dist.sel({ptdim: site}).argmin(), dist.sel({ptdim: site}).shape
                )

                # Select data from closest point
                args = {xydim: ind for xydim, ind in zip(dist.dims, inds)}
                pts.append(da.isel(**args))
                dists.append(dist.isel(**args))
            da = xarray.concat(pts, dim=ptdim)
            dist = xarray.concat(dists, dim=ptdim)
    else:
        raise (
            Exception(
                f'{subset_gridpoint.__name__} requires input data with "lon" and "lat" coordinates or data variables.'
            )
        )

    if tolerance is not None and dist is not None:
        da = da.where(dist < tolerance)

    if add_distance:
        da = da.assign_coords(distance=dist)

    if len(lat) == 1:
        da = da.squeeze(ptdim)

    if start_date or end_date:
        da = subset_time(da, start_date=start_date, end_date=end_date)

    return da


@check_start_end_dates
def subset_time(
    da: Union[xarray.DataArray, xarray.Dataset],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input DataArray or Dataset based on start and end years.

    Return a subset of a DataArray or Dataset for dates falling within the provided bounds.

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
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import subset_time  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_pr_file)  # doctest: +SKIP
    ...
    # Subset complete years
    >>> prSub = subset_time(ds.pr,start_date='1990',end_date='1999')  # doctest: +SKIP
    ...
    # Subset single complete year
    >>> prSub = subset_time(ds.pr,start_date='1990',end_date='1990')  # doctest: +SKIP
    ...
    # Subset multiple variables in a single dataset
    >>> ds = xr.open_mfdataset([path_to_tasmax_file, path_to_tasmin_file])  # doctest: +SKIP
    >>> dsSub = subset_time(ds,start_date='1990',end_date='1999')  # doctest: +SKIP
    ...
    # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
    >>> txSub = subset_time(ds.tasmax,start_date='1990-03',end_date='1999-08')  # doctest: +SKIP
    ...
    # Subset with specific start_dates and end_dates
    >>> tnSub = subset_time(ds.tasmin,start_date='1990-03-13',end_date='1990-08-17')  # doctest: +SKIP

    Notes
    -----
    TODO add notes about different calendar types. Avoid "%Y-%m-31". If you want complete month use only "%Y-%m".
    """
    return da.sel(time=slice(start_date, end_date))


@convert_lat_lon_to_da
def distance(
    da: Union[xarray.DataArray, xarray.Dataset],
    *,
    lon: Union[float, Sequence[float], xarray.DataArray],
    lat: Union[float, Sequence[float], xarray.DataArray],
):
    """Return distance to a point in meters.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    lon : Union[float, Sequence[float], xarray.DataArray]
      Longitude coordinate.
    lat : Union[float, Sequence[float], xarray.DataArray]
      Latitude coordinate.

    Returns
    -------
    xarray.DataArray
      Distance in meters to point.

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> from xclim.subset import distance  # doctest: +SKIP
    ...
    To get the indices from closest point, use:
    >>> da = xr.open_dataset(path_to_pr_file).pr  # doctest: +SKIP
    >>> d = distance(da, lon=-75, lat=45)  # doctest: +SKIP
    >>> k = d.argmin()  # doctest: +SKIP
    >>> i, j, _ = np.unravel_index(k, d.shape)  # doctest: +SKIP
    """
    ptdim = lat.dims[0]

    g = Geod(ellps="WGS84")  # WGS84 ellipsoid - decent globally

    def func(lons, lats, lon, lat):
        return g.inv(lons, lats, lon, lat)[2]

    out = xarray.apply_ufunc(
        func,
        *xarray.broadcast(da.lon.load(), da.lat.load(), lon, lat),
        input_core_dims=[[ptdim]] * 4,
        output_core_dims=[[ptdim]],
    )
    out.attrs["units"] = "m"
    return out
