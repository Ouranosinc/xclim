# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import numpy as np
import xarray as xr
import six
import pint
import pandas as pd
from . import checks
from inspect2 import signature, _empty
import abc
from collections import defaultdict
import datetime as dt
from pyproj import Geod

from boltons.funcutils import wraps

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

units.define(pint.unit.UnitDefinition('percent', '%', (),
                                      pint.converters.ScaleConverter(0.01)))

# Define commonly encountered units not defined by pint
units.define('degrees_north = degree = degrees_N = degreesN = degree_north = degree_N '
             '= degreeN')
units.define('degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE')
units.define("degC = kelvin; offset: 273.15 = celsius = C")  # add 'C' as an abbrev for celsius (default Coulomb)
units.define("d = day")
hydro = pint.Context('hydro')
hydro.add_transformation('[mass] / [length]**2', '[length]', lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3))
hydro.add_transformation('[mass] / [length]**2 / [time]', '[length] / [time]',
                         lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3))
hydro.add_transformation('[length] / [time]', '[mass] / [length]**2 / [time]',
                         lambda ureg, x: x * (1000 * ureg.kg / ureg.m ** 3))
units.add_context(hydro)
units.enable_contexts(hydro)

# These are the changes that could be included in a units definition file.

# degrees_north = degree = degrees_N = degreesN = degree_north = degree_N = degreeN
# degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE
# degC = kelvin; offset: 273.15 = celsius = C
# day = 24 * hour = d
# @context hydro
#     [mass] / [length]**2 -> [length]: value / 1000 / kg / m ** 3
#     [mass] / [length]**2 / [time] -> [length] / [time] : value / 1000 / kg * m ** 3
#     [length] / [time] -> [mass] / [length]**2 / [time] : value * 1000 * kg / m ** 3
# @end
binary_ops = {'>': 'gt', '<': 'lt', '>=': 'ge', '<=': 'le'}


def calc_ens_perc(arr, p):
    dims = arr.dims
    # make sure time is the second dimension
    if dims.index('time') != 1:
        dims1 = [dims[dims.index('sim')], dims[dims.index('time')]]
        for d in dims:
            if not d in dims1:
                dims1.append(d)
        arr = arr.transpose(*dims1)
        dims = dims1

    nan_count = np.isnan(arr).sum(axis=dims.index('sim'))
    out = np.percentile(arr, p, axis=dims.index('sim'))

    if np.any((nan_count > 0) & (nan_count < arr.shape[dims.index('sim')])):
        # only use nanpercentile where we need it (slow performace compared to standard) :
        nan_index = np.where((nan_count > 0) & (nan_count < arr.shape[dims.index('sim')]))
        for t in nan_index[0]:
            if np.any((nan_count.sel(time=arr.time[t]) > 0) & (
                nan_count.sel(time=arr.time[t]) < arr.shape[dims.index('sim')])):  # nan_count.sel(time=t):
                # nans present but not for all simulations - use nanpercentile
                out[t,] = np.nanpercentile(arr.sel(time=arr.time[t]), p, axis=dims.index('sim'))

    return out


def create_ensemble(ncfiles, dim='sim'):
    """Create an xarray datset of ensemble of climate simulation from a list of netcdf files. Input data is
    concatenated along a newly created data dimension (default=='sim')

                Returns a xarray dataset object containing input data from the list of netcdf files concatenated along
                a new dimension (default:'sim'). In the case where input files have unequal time dimensions output
                ensemble dataset is created for overlapping time-steps common to all input files

                Notes
                -----
                Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions)
                If input data contains multiple cftime calendar types they must be at monthly or coarser frequency


                Parameters
                ----------

                ncfiles : list of netcdf file paths (list)

                dim (optional) : name of added dataset dimension (string) .


                Returns
                -------
                xarray dataset containing concadated data from all input files

                Examples
                --------

                >>> from xclim import utils
                >>> import glob
                >>> ncfiles = glob.glob('/*.nc')
                >>> ens = utils.create_ensemble(ncfiles)
                >>> print(ens)

             """
    ds1 = []
    start_end_flag = True
    print('finding common time-steps')
    for n in ncfiles:
        ds = xr.open_dataset(n, decode_times=False)
        ds['time'] = xr.decode_cf(ds).time
        # get times - use common
        time1 = pd.to_datetime({'year': ds.time.dt.year, 'month': ds.time.dt.month, 'day': ds.time.dt.day})
        if start_end_flag:
            start1 = time1.values[0]
            end1 = time1.values[-1]
            start_end_flag = False
        if time1.values.min() > start1:
            start1 = time1.values.min()
        if time1.values.max() < end1:
            end1 = time1.values.max()

    for n in ncfiles:
        print('accessing file ', ncfiles.index(n) + 1, ' of ', len(ncfiles))
        ds = xr.open_dataset(n, decode_times=False, chunks={'time': 10})

        ds['time'] = xr.decode_cf(ds).time
        ds['time'].values = pd.to_datetime({'year': ds.time.dt.year, 'month': ds.time.dt.month, 'day': ds.time.dt.day})

        ds = ds.where((ds.time >= start1) & (ds.time <= end1), drop=True)

        ds1.append(ds.drop('time'))
    print('concatenating files : adding dimension {dim}')
    ens = xr.concat(ds1, dim=dim)
    # assign time coords
    ens = ens.assign_coords(time=ds.time.values)
    return ens


def ensemble_statistics(ens, stats={'type': 'perc', 'values': [10, 50, 90], 'time_block': []}, ):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

            Returns a dataset containing ensemble statistics for input climate simulations.
            Alternatively calculate ensemble percentiles (default) or ensemble mean and standard deviation


            Parameters
            ----------

            ens : Ensemble dataset (see xclim.utils.create_ensemble)

            stats (optional) : Dictionary specifying type of statistics to perform.
            One of :

                stats = {'type':'perc', 'values':[perc1, perc2, ... percN]})
                - (Default) Calculate percentiles defined in 'values'. Defaults to [10, 50, 90]
                - For large ensembles - optionally specify a 'time_block' width (number of time steps) for iterative
                 percentile calculation. If empty the function will try to automatically determine a manageable size

                or

                stats = {'type':'mean_std_min_max'} - Calculate ensemble mean, standard-deviation, minimum and maximum


            Returns
            -------
            xarray dataset with containing data variables of requested ensemble statistics

            Examples
            --------

            >>> from xclim import utils
            >>> import glob
            >>> ncfiles = glob.glob('/*tas*.nc')
            Create ensemble dataset
            >>> ens = utils.create_ensemble(ncfiles)
            Calculate default ensemble percentiles
            >>> ens_percs = utils.ensemble_statistics(ens)
            >>> print(ens_percs['tas_p10'])
            Calculate ensemble mean and standard-deviation
            >>> ens_means_std = utils.ensemble_statistics(ens, stats = {'type':'mean_std_min_max'} )
            >>> print(ens_mean_std['tas_mean'])
            Calculate non-default percentiles (25th and 75th)
            >>> ens_percs = utils.ensemble_statistics(ens, stats = {'type':'perc','values':[25, 75]})
            >>> print(ens_percs['tas_p25'])

            """
    dsOut = ens.drop(ens.data_vars)
    for v in ens.data_vars:
        dims = ens[v].dims
        outdims = [x for x in dims if 'sim' not in x]

        if stats['type'] == 'perc':
            # Percentile calculation requires load to memory : automate size for large ensemble objects
            if stats['time_block']:
                time_block = stats['time_block']
            else:
                time_block = round(2E8 / (ens[v].size / ens[v].shape[dims.index('time')]), -1)  # 2E8

            if time_block > len(ens[v].time):

                print('loading ensemble data to memory')
                arr = ens[v].load()  # percentile calc requires loading the array
                coords = {}
                for c in outdims:
                    coords[c] = arr[c]
                for p in stats['values']:
                    outvar = v + '_p' + str(p)

                    out1 = calc_ens_perc(arr, p)

                    dsOut[outvar] = xr.DataArray(out1, dims=outdims, coords=coords)
                    dsOut[outvar].attrs = ens[v].attrs
                    if 'description' in dsOut[outvar].attrs.keys():
                        dsOut[outvar].attrs['description'] = dsOut[outvar].attrs['description'] + ' : ' + str(p) + \
                                                             'th percentile of ensemble'
                    else:
                        dsOut[outvar].attrs['description'] = str(p) + \
                                                             'th percentile of ensemble'
            else:
                # loop through blocks
                Warning('large ensemble size detected : statistics will be calculated in blocks of ', int(time_block),
                        ' time-steps')
                blocks = list(range(0, len(ens.time) + 1, int(time_block)))
                if blocks[-1] != len(ens[v].time):
                    blocks.append(len(ens[v].time))
                arr_p_all = {}
                for t in range(0, len(blocks) - 1):
                    print('Calculating block ', t + 1, ' of ', len(blocks) - 1)
                    time_sel = slice(blocks[t], blocks[t + 1])
                    arr = ens[v].isel(time=time_sel).load()  # percentile calc requires loading the array
                    coords = {}
                    for c in outdims:
                        coords[c] = arr[c]
                    for p in stats['values']:
                        outvar = v + '_p' + str(p)

                        out1 = calc_ens_perc(arr, p)

                        if t == 0:
                            arr_p_all[str(p)] = xr.DataArray(out1, dims=outdims, coords=coords)
                        else:
                            arr_p_all[str(p)] = xr.concat([arr_p_all[str(p)],
                                                           xr.DataArray(out1, dims=outdims, coords=coords)], dim='time')
                for p in stats['values']:
                    outvar = v + '_p' + str(p)
                    dsOut[outvar] = arr_p_all[str(p)]
                    dsOut[outvar].attrs = ens[v].attrs
                    if 'description' in dsOut[outvar].attrs.keys():
                        dsOut[outvar].attrs['description'] = dsOut[outvar].attrs['description'] + ' : ' + str(p) + \
                                                             'th percentile of ensemble'
                    else:
                        dsOut[outvar].attrs['description'] = str(p) + \
                                                             'th percentile of ensemble'

        elif stats['type'] == 'mean_std_min_max':
            dsOut[v + '_mean'] = ens[v].mean(dim='sim')
            dsOut[v + '_stdev'] = ens[v].std(dim='sim')
            dsOut[v + '_max'] = ens[v].max(dim='sim')
            dsOut[v + '_min'] = ens[v].min(dim='sim')
            for vv in dsOut.data_vars:
                dsOut[vv].attrs = ens[v].attrs

                if 'description' in dsOut[vv].attrs.keys():
                    vv.split()
                    dsOut[vv].attrs['description'] = dsOut[vv].attrs['description'] + ' : ' + vv.split('_')[-1] + \
                                                     ' of ensemble'

        else:
            raise (Exception(
                "unknown statistics type specified : please specify the statistics type as one of 'perc' or 'mean_std'"))

        return dsOut


def threshold_count(da, op, thresh, freq):
    """Count number of days above or below threshold.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    op : {>, <, >=, <=, gt, lt, ge, le }
      Logical operator, e.g. arr > thresh.
    thresh : float
      Threshold value.
    freq : str
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xarray.DataArray
      The number of days meeting the constraints for each period.
    """
    from xarray.core.ops import get_op

    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError("Operation `{}` not recognized.".format(op))

    func = getattr(da, '_binary_op')(get_op(op))
    c = func(da, thresh) * 1
    return c.resample(time=freq).sum(dim='time')


def percentile_doy(arr, window=5, per=.1):
    """Percentile value for each day of the year

    Return the climatological percentile over a moving window around each day of the year.

    Parameters
    ----------
    arr : xarray.DataArray
      Input data.
    window : int
      Number of days around each day of the year to include in the calculation.
    per : float
      Percentile between [0,1]

    Returns
    -------
    xarray.DataArray
      The percentiles indexed by the day of the year.
    """

    # TODO: Support percentile array, store percentile in attributes (or coordinates ?)
    rr = arr.rolling(min_periods=1, center=True, time=window).construct('window')

    # Create empty percentile array
    g = rr.groupby('time.dayofyear')
    c = g.count(dim=('time', 'window'))

    p = xr.full_like(c, np.nan).astype(float).load()

    for doy, ind in rr.groupby('time.dayofyear'):
        p.loc[{'dayofyear': doy}] = ind.compute().quantile(per, dim=('time', 'window'))

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = adjust_doy_calendar(p.loc[p.dayofyear < 366], arr)

    return p


def subset_bbox(da, lon_bnds='', lat_bnds='', start_yr='', end_yr=''):
    """Subset a datarray or dataset spatially (and temporally) using a lat lon bounding box and years selection

        Return a subsetted data array for grid points falling within a spatial bounding box
        defined by longitude and latitudinal bounds and for years falling within provided year bounds

        Parameters
        ----------
        arr : xarray.DataArray or xarray.Dataset
          Input data.
        lon_bnds (optional) : list of floats
          List of maximum and minimum longitudinal bounds. Defaults to all longitudes in original data-array
        lat_bnds (optional) :  list of floats
          List maximum and minimum latitudinal bounds.  Defaults to all latitudes in original data-array
        year_bnds (optional) :  list of floats or float
          List of maximum and minimum year bounds. If single float returns single year only.
          Defaults to all years in original data-array


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
        Subset mutliple variables in a single dataset
        >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
        >>> dsSub = utils.subset_bbox(ds,lon_bnds=[-75,-70],lat_bnds=[40,45],start_yr=1990,end_yr=1999)
        """

    if lon_bnds:

        lon_bnds = np.asarray(lon_bnds)

        # adjust for files with all postive longitudes if necessary
        if np.all(da.lon > 0) and np.any(lon_bnds < 0):
            lon_bnds[lon_bnds < 0] += 360

        lon_cond = (da.lon >= lon_bnds.min()) & (da.lon <= lon_bnds.max())
    else:
        lon_cond = (da.lon >= da.lon.min()) & (da.lon <= da.lon.max())

    if lat_bnds:
        lat_bnds = np.asarray(lat_bnds)
        lat_cond = (da.lat >= lat_bnds.min()) & (da.lat <= lat_bnds.max())
    else:
        lat_cond = (da.lat >= da.lat.min()) & (da.lat <= da.lat.max())

    if start_yr or end_yr:
        if not start_yr:
            start_yr = da.time.dt.year.min()
        if not end_yr:
            end_yr = da.time.dt.year.max()

        year_bnds = np.asarray([start_yr, end_yr])
        if len(year_bnds) == 1:
            time_cond = da.time.dt.year == year_bnds
        else:
            time_cond = (da.time.dt.year >= year_bnds.min()) & (da.time.dt.year <= year_bnds.max())
    else:
        time_cond = (da.time.dt.year >= da.time.dt.year.min()) & (da.time.dt.year <= da.time.dt.year.max())

    return da.where(lon_cond & lat_cond & time_cond, drop=True)


def subset_gridpoint(da, lon, lat, start_yr='', end_yr=''):
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
              longitude coordinate.
            lat:  float
              latitude coordinate.
            year_bnds (optional) :  list of floats or float
              List of maximum and minimum year bounds. If single float returns one year only.
              Defaults to all years in original data-array


            Returns
            -------
            xarray.DataArray or xarray.DataSet
              subsetted data array or dataset

        Examples
        --------

        >>> from xclim import utils
        >>> ds = xr.open_dataset('pr.day.nc')
        Subset lat lon point and multiple years
        >>> prSub = utils.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1999)
        Subset lat, lon point and single year
        >>> prSub = utils.subset_gridpoint(ds.pr, lon=-75,lat=45,start_yr=1990,end_yr=1990)
         Subset mutliple variables in a single dataset
        >>> ds = xr.open_mfdataset(['pr.day.nc','tas.day.nc'])
        >>> dsSub = utils.subset_gridpoint(ds, lon=-75,lat=45,start_yr=1990,end_yr=1999)
            """

    g = Geod(ellps='WGS84')  # WGS84 ellipsoid - decent globaly
    # adjust for files with all postive longitudes if necessary
    if np.all(da.lon > 0) and lon < 0:
        lon += 360

    if len(da.lon.shape) == 1 & len(da.lat.shape) == 1:
        # create a 2d grid of lon, lat values
        lon1, lat1 = np.meshgrid(np.asarray(da.lon.values), np.asarray(da.lat.values))

    else:
        lon1 = da.lon.values
        lat1 = da.lat.values
    shp_orig = lon1.shape
    lon1 = np.reshape(lon1, (lon1.size))
    lat1 = np.reshape(lat1, (lat1.size))
    # calculate geodesic distance between grid points and point of interest
    az12, az21, dist = g.inv(lon1, lat1, np.broadcast_to(lon, lon1.shape), np.broadcast_to(lat, lat1.shape))
    dist = dist.reshape(shp_orig)

    iy, ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    xydims = [x for x in da.dims if 'time' not in x]
    args = {}
    args[xydims[0]] = iy
    args[xydims[1]] = ix
    out = da.isel(**args)
    if start_yr or end_yr:
        if not start_yr:
            start_yr = da.time.dt.year.min()
        if not end_yr:
            end_yr = da.time.dt.year.max()
        year_bnds = np.asarray([start_yr, end_yr])

        if len(year_bnds) == 1:
            time_cond = da.time.dt.year == year_bnds
        else:
            time_cond = (da.time.dt.year >= year_bnds.min()) & (da.time.dt.year <= year_bnds.max())
        out = out.where(time_cond, drop=True)

    return out


# TODO: I'd like this function to use calendar instead of target (ie target calendar.)
# Depends on https://github.com/pydata/xarray/issues/2436
def adjust_doy_calendar(source, target):
    r"""Interpolate from one set of dayofyear range to another

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xarray.DataArray
      Array with `dayofyear` coordinates.
    target : xarray.DataArray
      Array with `time` coordinates the source should be mapped to.

    Returns
    -------
    xarray.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    if 'dayofyear' not in source.coords.keys():
        raise AttributeError("source should have dayofyear coordinates.")

    # Interpolation of source to target dayofyear range
    # When https://github.com/pydata/xarray/issues/2436 will be fixed, we might want to use calendar instead.
    doy_max_source = source.dayofyear.values.max()
    doy_max_target = target.time.dt.dayofyear.values.max()
    if doy_max_target not in [360, 365, 366]:
        raise ValueError("The target array's calendar is not recognized")

    # Interpolate to fill na values
    buffer = source.interpolate_na(dim='dayofyear')

    # Interpolate to target dayofyear range
    buffer.coords['dayofyear'] = np.linspace(start=1, stop=doy_max_target,
                                             num=doy_max_source)
    return buffer.interp(dayofyear=range(1, doy_max_target + 1))


def get_daily_events(da, da_value, operator):
    r"""
    function that returns a 0/1 mask when a condition is True or False

    the function returns 1 where operator(da, da_value) is True
                         0 where operator(da, da_value) is False
                         nan where da is nan

    Parameters
    ----------
    da : xarray.DataArray
    da_value : float
    operator : string


    Returns
    -------
    xarray.DataArray

    """
    events = operator(da, da_value) * 1
    events = events.where(~np.isnan(da))
    events = events.rename('events')
    return events


def daily_downsampler(da, freq='YS'):
    r"""Daily climate data downsampler

    Parameters
    ----------
    da : xarray.DataArray
    freq : string

    Returns
    -------
    xarray.DataArray


    Note
    ----

        Usage Example

            grouper = daily_downsampler(da_std, freq='YS')
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')
    """

    # generate tags from da.time and freq
    if isinstance(da.time.values[0], np.datetime64):
        years = ['{:04d}'.format(y) for y in da.time.dt.year.values]
        months = ['{:02d}'.format(m) for m in da.time.dt.month.values]
    else:
        # cannot use year, month, season attributes, not available for all calendars ...
        years = ['{:04d}'.format(v.year) for v in da.time.values]
        months = ['{:02d}'.format(v.month) for v in da.time.values]
    seasons = ['DJF DJF MAM MAM MAM JJA JJA JJA SON SON SON DJF'.split()[int(m) - 1] for m in months]

    n_t = da.time.size
    if freq == 'YS':
        # year start frequency
        l_tags = years
    elif freq == 'MS':
        # month start frequency
        l_tags = [years[i] + months[i] for i in range(n_t)]
    elif freq == 'QS-DEC':
        # DJF, MAM, JJA, SON seasons
        # construct tags from list of season+year, increasing year for December
        ys = []
        for i in range(n_t):
            m = months[i]
            s = seasons[i]
            y = years[i]
            if m == '12':
                y = str(int(y) + 1)
            ys.append(y + s)
        l_tags = ys
    else:
        raise RuntimeError('freqency {:s} not implemented'.format(freq))

    # add tags to buffer DataArray
    buffer = da.copy()
    buffer.coords['tags'] = ('time', l_tags)

    # return groupby according to tags
    return buffer.groupby('tags')


def walk_map(d, func):
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
      Input dictionary, possibly nested.
    func : function
      Function to apply to dictionary values.

    Returns
    -------
    dict
      Dictionary whose values are the output of the given function.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, (dict, defaultdict)):
            out[k] = walk_map(v, func)
        else:
            out[k] = func(v)
    return out


# This class needs to be subclassed by individual indicator classes defining metadata information, compute and
# missing functions. It can handle indicators with any number of forcing fields.
class Indicator(object):
    r"""Climate indicator based on xarray
    """
    # Unique ID for registry. May use tags {<tag>} that will be formatted at runtime.
    identifier = ''

    # CF-Convention metadata to be attributed to the output variable. May use tags {<tag>} formatted at runtime.
    standard_name = ''  # The set of permissible standard names is contained in the standard name table.
    long_name = ''  # Parsed.
    units = ''  # Representative units of the physical quantity.
    cell_methods = ''  # List of blank-separated words of the form "name: method"
    description = ''  # The description is meant to clarify the qualifiers of the fundamental quantities, such as which
    #   surface a quantity is defined on or what the flux sign conventions are.

    # The units expected by the function. Used to convert input units to the required_units.
    required_units = ''

    # The `pint` unit context. Use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    context = None

    # Additional information that can be used by third party libraries or to describe the file content.
    title = ''  # A succinct description of what is in the dataset. Default parsed from compute.__doc__
    abstract = ''  # Parsed
    keywords = ''  # Comma separated list of keywords
    references = ''  # Published or web-based references that describe the data or methods used to produce it. Parsed.
    comment = ''  # Miscellaneous information about the data or methods used to produce it.
    notes = ''  # Mathematical formulation. Parsed.

    # Tag mappings between keyword arguments and long-form text.
    _attrs_mapping = {'cell_methods': {'YS': 'years', 'MS': 'months'},  # I don't think this is necessary.
                      'long_name': {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal'},
                      'description': {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal'}}

    def __init__(self, **kwds):

        for key, val in kwds.items():
            setattr(self, key, val)

        # Sanity checks
        required = ['compute', 'required_units']
        for key in required:
            if not getattr(self, key):
                raise ValueError("{} needs to be defined during instantiation.".format(key))

        # Infer number of variables from `required_units`.
        if isinstance(self.required_units, six.string_types):
            self.required_units = (self.required_units,)

        self._nvar = len(self.required_units)

        # Extract information from the `compute` function.
        # The signature
        self._sig = signature(self.compute)

        # The input parameter names
        self._parameters = tuple(self._sig.parameters.keys())

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__.__func__)

        # Fill in missing metadata from the doc
        meta = parse_doc(self.compute.__doc__)
        for key in ['abstract', 'title', 'long_name', 'notes', 'references']:
            setattr(self, key, getattr(self, key) or meta.get(key, ''))

    def __call__(self, *args, **kwds):
        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        for i in range(self._nvar):
            p = self._parameters[i]
            for attr in ['history', 'cell_methods']:
                attrs[attr] += "{}: ".format(p) if self._nvar > 1 else ""
                attrs[attr] += getattr(ba.arguments[p], attr, '')
                if attrs[attr]:
                    attrs[attr] += "\n" if attr == 'history' else " "

        # Update attributes
        out_attrs = self.json(ba.arguments)
        formatted_id = out_attrs.pop('identifier')
        attrs['history'] += '[{:%Y-%m-%d %H:%M:%S}] {}{}'.format(dt.datetime.now(), formatted_id, ba.signature)
        attrs['cell_methods'] += out_attrs.pop('cell_methods')
        attrs.update(out_attrs)

        # Assume the first arguments are always the DataArray.
        das = tuple((ba.arguments.pop(self._parameters[i]) for i in range(self._nvar)))

        # Pre-computation validation checks
        for da in das:
            self.validate(da)
        self.cfprobe(*das)

        # Convert units if necessary
        das = tuple((self.convert_units(da, ru, self.context) for (da, ru) in zip(das, self.required_units)))

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(*das, **ba.arguments)
        out.attrs.update(attrs)

        # Bind call arguments to the `missing` function, whose signature might be different from `compute`.
        mba = signature(self.missing).bind(*das, **ba.arguments)

        # Mask results that do not meet criteria defined by the `missing` method.
        mask = self.missing(*mba.args, **mba.kwargs)
        ma_out = out.where(~mask)

        return ma_out.rename(formatted_id)

    @property
    def cf_attrs(self):
        """CF-Convention attributes of the output value."""
        names = ['standard_name', 'long_name', 'units', 'cell_methods', 'description', 'comment',
                 'references']
        return {k: getattr(self, k, '') for k in names}

    def json(self, args=None):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ['identifier', 'abstract', 'keywords']
        out = {key: getattr(self, key) for key in names}
        out.update(self.cf_attrs)
        out = self.format(out, args)

        out['notes'] = self.notes

        out['parameters'] = str({key: {'default': p.default if p.default != _empty else None, 'desc': ''}
                                 for (key, p) in self._sig.parameters.items()})

        if six.PY2:
            out = walk_map(out, lambda x: x.decode('utf8') if isinstance(x, six.string_types) else x)

        return out

    def cfprobe(self, *das):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        return True

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""

    def convert_units(self, da, req_units, context=None):
        """Return DataArray converted to unit."""
        fu = units.parse_units(da.attrs['units'].replace('-', '**-'))
        tu = units.parse_units(req_units.replace('-', '**-'))
        if fu != tu:
            if self.context:
                with units.context(self.context):
                    return units.convert(da, fu, tu)
            else:
                return units.convert(da, fu, tu)
        else:
            return da

    def format(self, attrs, args=None):
        """Format attributes including {} tags with arguments."""
        if args is None:
            return attrs

        out = {}
        for key, val in attrs.items():
            mba = {}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, six.string_types) and v in self._attrs_mapping.get(key, {}).keys():
                    mba[k] = '{' + v + '}'
                else:
                    mba[k] = int(v) if (isinstance(v, float) and v % 1 == 0) else v

            out[key] = val.format(**mba).format(**self._attrs_mapping.get(key, {}))

        return out

    @staticmethod
    def missing(*args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get('freq')
        miss = (checks.missing_any(da, freq) for da in args)
        return reduce(np.logical_or, miss)

    def validate(self, da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)

    @classmethod
    def factory(cls, attrs):
        """Create a subclass from the attributes dictionary."""
        name = attrs['identifier'].capitalize()
        return type(name, (cls,), attrs)


def parse_doc(doc):
    """Crude regex parsing."""
    import re
    if doc is None:
        return {}

    out = {}

    sections = re.split(r'(\w+)\n\s+-{4,50}', doc)  # obj.__doc__.split('\n\n')
    intro = sections.pop(0)
    if intro:
        content = list(map(str.strip, intro.strip().split('\n\n')))
        if len(content) == 1:
            out['title'] = content[0]
        elif len(content) == 2:
            out['title'], out['abstract'] = content

    for i in range(0, len(sections), 2):
        header, content = sections[i:i + 2]

        if header in ['Notes', 'References']:
            out[header.lower()] = content.replace('\n    ', '\n')
        elif header == 'Parameters':
            pass
        elif header == 'Returns':
            match = re.search(r'xarray\.DataArray\s*(.*)', content)
            if match:
                out['long_name'] = match.groups()[0]

    return out


def format_kwargs(attrs, params):
    """Modify attribute with argument values.

    Parameters
    ----------
    attrs : dict
      Attributes to be assigned to function output. The values of the attributes in braces will be replaced the
      the corresponding args values.
    params : dict
      A BoundArguments.arguments dictionary storing a function's arguments.
    """
    attrs_mapping = {'cell_methods': {'YS': 'years', 'MS': 'months'},
                     'long_name': {'YS': 'Annual', 'MS': 'Monthly'}}

    for key, val in attrs.items():
        mba = {}
        # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
        for k, v in params.items():
            if isinstance(v, six.string_types) and v in attrs_mapping.get(key, {}).keys():
                mba[k] = '{' + v + '}'
            else:
                mba[k] = v

        attrs[key] = val.format(**mba).format(**attrs_mapping.get(key, {}))
