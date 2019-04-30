# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import numpy as np
import six
import re
import pint
import pandas as pd
import xarray as xr
from . import checks
from inspect import signature
import functools
import abc
from collections import defaultdict
import datetime as dt
import calendar
from pyproj import Geod
import warnings
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

# Default context.
null = pint.Context('none')
units.add_context(null)

# Precipitation units. This is an artificial unit that we're using to verify that a given unit can be converted into
# a precipitation unit. Ideally this could be checked through the `dimensionality`, but I can't get it to work.
units.define("[precipitation] = [mass] / [length] ** 2 / [time]")
units.define("mmday = 1000 kg / meter ** 2 / day")

units.define("[discharge] = [length] ** 3 / [time]")
units.define("cms = meter ** 3 / second")

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

# Maximum day of year in each calendar.
calendars = {'standard': 366,
             'gregorian': 366,
             'proleptic_gregorian': 366,
             'julian': 366,
             'no_leap': 365,
             '365_day': 365,
             'all_leap': 366,
             '366_day': 366,
             'uniform30day': 360,
             '360_day': 360}


def units2pint(value):
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : xr.DataArray or string
      Input data array or expression.

    Returns
    -------
    pint.Unit
      Units of the data array.

    """

    def _transform(s):
        """Convert a CF-unit string to a pint expression."""
        return re.subn(r'\^?(-?\d)', r'**\g<1>', s)[0]

    if isinstance(value, str):
        unit = value
    elif isinstance(value, xr.DataArray):
        unit = value.attrs['units']
    elif isinstance(value, units.Quantity):
        return value.units
    else:
        raise NotImplementedError("Value of type {} not supported.".format(type(value)))

    try:  # Pint compatible
        return units.parse_expression(unit).units
    except (pint.UndefinedUnitError, pint.DimensionalityError):  # Convert from CF-units to pint-compatible
        return units.parse_expression(_transform(unit)).units


def pint2cfunits(value):
    """Return a CF-Convention unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
      Input unit.

    Returns
    -------
    out : str
      Units following CF-Convention.
    """
    # Print units using abbreviations (millimeter -> mm)
    s = "{:~}".format(value)

    # Search and replace patterns
    pat = r'(?P<inverse>/ )?(?P<unit>\w+)(?: \*\* (?P<pow>\d))?'

    def repl(m):
        i, u, p = m.groups()
        p = p or (1 if i else '')
        neg = '-' if i else ('^' if p else '')

        return "{}{}{}".format(u, neg, p)

    out, n = re.subn(pat, repl, s)
    return out


def pint_multiply(da, q, out_units=None):
    """Multiply xarray.DataArray by pint.Quantity.

    Parameters
    ----------
    da : xr.DataArray
      Input array.
    q : pint.Quantity
      Multiplicating factor.
    out_units : str
      Units the output array should be converted into.
    """
    a = 1 * units2pint(da)
    f = a * q.to_base_units()
    if out_units:
        f = f.to(out_units)
    out = da * f.magnitude
    out.attrs['units'] = pint2cfunits(f.units)
    return out


def convert_units_to(source, target, context=None):
    """
    Convert a mathematical expression into a value with the same units as a DataArray.

    Parameters
    ----------
    source : str, pint.Quantity or xr.DataArray
      The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str, pint.Unit or DataArray
      Target array of values to which units must conform.
    context : str


    Returns
    -------
    out
      The source value converted to target's units.
    """
    # Target units
    if isinstance(target, units.Unit):
        tu = target
    elif isinstance(target, (str, xr.DataArray)):
        tu = units2pint(target)
    else:
        raise NotImplementedError

    if isinstance(source, str):
        q = units.parse_expression(source)

        # Return magnitude of converted quantity. This is going to fail if units are not compatible.
        return q.to(tu).m

    if isinstance(source, units.Quantity):
        return source.to(tu).m

    if isinstance(source, xr.DataArray):
        fu = units2pint(source)

        if fu == tu:
            return source

        tu_u = pint2cfunits(tu)
        with units.context(context or 'none'):
            out = units.convert(source, fu, tu)
            out.attrs['units'] = tu_u
            return out

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(source, (float, int)):
        if context == 'hydro':
            fu = units.mm / units.day
        else:
            fu = units.degC
        warnings.warn("Future versions of XCLIM will require explicit unit specifications.", FutureWarning)
        return (source * fu).to(tu).m

    raise NotImplementedError("source of type {} is not supported.".format(type(source)))


def _check_units(val, dim):
    if dim is None or val is None:
        return

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(val, (int, float)):
        return

    expected = units.get_dimensionality(dim.replace('dimensionless', ''))
    val_dim = units2pint(val).dimensionality
    if val_dim == expected:
        return

    # Check if there is a transformation available
    start = pint.util.to_units_container(expected)
    end = pint.util.to_units_container(val_dim)
    graph = units._active_ctx.graph
    if pint.util.find_shortest_path(graph, start, end):
        return

    if dim == '[precipitation]':
        tu = 'mmday'
    elif dim == '[discharge]':
        tu = 'cms'
    else:
        raise NotImplementedError

    try:
        (1 * units2pint(val)).to(tu, 'hydro')
    except pint.UndefinedUnitError:
        raise AttributeError("Value's dimension {} does not match expected units {}.".format(val_dim, expected))


def declare_units(out_units, **units_by_name):
    """Create a decorator to check units of function arguments."""

    def dec(func):
        # Match the signature of the function to the arguments given to the decorator
        sig = signature(func)
        bound_units = sig.bind_partial(**units_by_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments so we can check units
            bound_args = sig.bind(*args, **kwargs)
            for name, val in bound_args.arguments.items():
                _check_units(val, bound_units.arguments.get(name, None))

            out = func(*args, **kwargs)

            # In the generic case, we use the default units that should have been propagated by the computation.
            if '[' in out_units:
                _check_units(out, out_units)

            # Otherwise, we specify explicitly the units.
            else:
                out.attrs['units'] = out_units
            return out

        return wrapper

    return dec


def create_ensemble(ncfiles, mf_flag=False):
    """Create an xarray datset of ensemble of climate simulation from a list of netcdf files. Input data is
    concatenated along a newly created data dimension ('realization')

    Returns a xarray dataset object containing input data from the list of netcdf files concatenated along
    a new dimension (name:'realization'). In the case where input files have unequal time dimensions output
    ensemble dataset is created for overlapping time-steps common to all input files

    Parameters
    ----------
    ncfiles : sequence
      List of netcdf file paths. If mf_flag is true ncfiles should be a list of lists where
    each sublist contains input .nc files of a multifile dataset

    mf_flag : Boolean . If true climate simulations are treated as multifile datasets before concatenation

    Returns
    -------
    xarray dataset containing concatenated data from all input files

    Notes
    -----
    Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions)
    If input data contains multiple cftime calendar types they must be at monthly or coarser frequency

    Examples
    --------
    >>> from xclim import utils
    >>> import glob
    >>> ncfiles = glob.glob('/*.nc')
    >>> ens = utils.create_ensemble(ncfiles)
    >>> print(ens)
    Using multifile datasets:
    simulation 1 is a list of .nc files (e.g. separated by time)
    >>> ncfiles = glob.glob('dir/*.nc')
    simulation 2 is also a list of .nc files
    >>> ens = utils.create_ensemble(ncfiles)
    """
    dim = 'realization'
    ds1 = []
    start_end_flag = True
    print('finding common time-steps')
    for n in ncfiles:
        if mf_flag:
            ds = xr.open_mfdataset(n, concat_dim='time', decode_times=False, chunks={'time': 10})
            ds['time'] = xr.open_mfdataset(n).time
        else:
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
        if mf_flag:
            ds = xr.open_mfdataset(n, concat_dim='time', decode_times=False, chunks={'time': 10})
            ds['time'] = xr.open_mfdataset(n).time
        else:
            ds = xr.open_dataset(n, decode_times=False, chunks={'time': 10})
            ds['time'] = xr.decode_cf(ds).time

        ds['time'].values = pd.to_datetime({'year': ds.time.dt.year, 'month': ds.time.dt.month, 'day': ds.time.dt.day})

        ds = ds.where((ds.time >= start1) & (ds.time <= end1), drop=True)

        ds1.append(ds.drop('time'))
    print('concatenating files : adding dimension ', dim, )
    ens = xr.concat(ds1, dim=dim)
    # assign time coords
    ens = ens.assign_coords(time=ds.time.values)
    return ens


def ensemble_percentiles(ens, values=(10, 50, 90), time_block=None):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble statistics for input climate simulations.
    Alternatively calculate ensemble percentiles (default) or ensemble mean and standard deviation

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.utils.create_ensemble)
    values : tuple of integers - percentile values to calculate  : default : (10, 50, 90)
    time_block : integer
      for large ensembles iteratively calculate percentiles in time-step blocks (n==time_block).
       If not defined the function tries to estimate an appropriate value

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
    Calculate non-default percentiles (25th and 75th)
    >>> ens_percs = utils.ensemble_statistics(ens, values=(25,75))
    >>> print(ens_percs['tas_p25'])
    Calculate by time blocks (n=10) if ensemble size is too large to load in memory
    >>> ens_percs = utils.ensemble_statistics(ens, time_block=10)
    >>> print(ens_percs['tas_p25'])

    """

    ds_out = ens.drop(ens.data_vars)
    dims = list(ens.dims)
    for v in ens.data_vars:
        # Percentile calculation requires load to memory : automate size for large ensemble objects
        if not time_block:
            time_block = round(2E8 / (ens[v].size / ens[v].shape[dims.index('time')]), -1)  # 2E8

        if time_block > len(ens[v].time):
            out = calc_percentiles_simple(ens, v, values)

        else:
            # loop through blocks
            Warning('large ensemble size detected : statistics will be calculated in blocks of ', int(time_block),
                    ' time-steps')
            out = calc_percentiles_blocks(ens, v, values, time_block)
        for vv in out.data_vars:
            ds_out[vv] = out[vv]
    return ds_out


def calc_percentiles_simple(ens, v, values):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if 'realization' not in x]

    print('loading ensemble data to memory')
    arr = ens[v].load()  # percentile calc requires loading the array
    coords = {}
    for c in outdims:
        coords[c] = arr[c]
    for p in values:
        outvar = v + '_p' + str(p)

        out1 = calc_perc(arr, p)

        ds_out[outvar] = xr.DataArray(out1, dims=outdims, coords=coords)
        ds_out[outvar].attrs = ens[v].attrs
        if 'description' in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs['description'] = ds_out[outvar].attrs['description'] + ' : ' + str(p) + \
                                                  'th percentile of ensemble'
        else:
            ds_out[outvar].attrs['description'] = str(p) + \
                                                  'th percentile of ensemble'
    return ds_out


def calc_percentiles_blocks(ens, v, values, time_block):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if 'realization' not in x]

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
        for p in values:

            out1 = calc_perc(arr, p)

            if t == 0:
                arr_p_all[str(p)] = xr.DataArray(out1, dims=outdims, coords=coords)
            else:
                arr_p_all[str(p)] = xr.concat([arr_p_all[str(p)],
                                               xr.DataArray(out1, dims=outdims, coords=coords)], dim='time')
    for p in values:
        outvar = v + '_p' + str(p)
        ds_out[outvar] = arr_p_all[str(p)]
        ds_out[outvar].attrs = ens[v].attrs
        if 'description' in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs['description'] = ds_out[outvar].attrs['description'] + ' : ' + str(p) + \
                                                  'th percentile of ensemble'
        else:
            ds_out[outvar].attrs['description'] = str(p) + \
                                                  'th percentile of ensemble'
    return ds_out


def calc_perc(arr, p):
    dims = arr.dims
    # make sure time is the second dimension
    if dims.index('time') != 1:
        dims1 = [dims[dims.index('realization')], dims[dims.index('time')]]
        for d in dims:
            if d not in dims1:
                dims1.append(d)
        arr = arr.transpose(*dims1)
        dims = dims1

    nan_count = np.isnan(arr).sum(axis=dims.index('realization'))
    out = np.percentile(arr, p, axis=dims.index('realization'))
    if np.any((nan_count > 0) & (nan_count < arr.shape[dims.index('realization')])):
        arr1 = arr.values.reshape(arr.shape[dims.index('realization')],
                                  int(arr.size / arr.shape[dims.index('realization')]))
        # only use nanpercentile where we need it (slow performace compared to standard) :
        nan_index = np.where((nan_count > 0) & (nan_count < arr.shape[dims.index('realization')]))
        t = np.ravel_multi_index(nan_index, nan_count.shape)
        out[np.unravel_index(t, nan_count.shape)] = np.nanpercentile(arr1[:, t], p, axis=dims.index('realization'))

    return out


def ensemble_mean_std_max_min(ens):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble mean, standard-deviation,
    minimum and maximum for input climate simulations.

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.utils.create_ensemble)

    Returns
    -------
    xarray dataset with containing data variables of ensemble statistics

    Examples
    --------
    >>> from xclim import utils
    >>> import glob
    >>> ncfiles = glob.glob('/*tas*.nc')
    Create ensemble dataset
    >>> ens = utils.create_ensemble(ncfiles)
    Calculate ensemble statistics
    >>> ens_means_std = utils.ensemble_mean_std_max_min(ens)
    >>> print(ens_mean_std['tas_mean'])
    """
    dsOut = ens.drop(ens.data_vars)
    for v in ens.data_vars:

        dsOut[v + '_mean'] = ens[v].mean(dim='realization')
        dsOut[v + '_stdev'] = ens[v].std(dim='realization')
        dsOut[v + '_max'] = ens[v].max(dim='realization')
        dsOut[v + '_min'] = ens[v].min(dim='realization')
        for vv in dsOut.data_vars:
            dsOut[vv].attrs = ens[v].attrs

            if 'description' in dsOut[vv].attrs.keys():
                vv.split()
                dsOut[vv].attrs['description'] = dsOut[vv].attrs['description'] + ' : ' + vv.split('_')[
                    -1] + ' of ensemble'

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
    # TODO: Support percentile array, store percentile in coordinates.
    #  This is supported by DataArray.quantile, but not by groupby.reduce.
    rr = arr.rolling(min_periods=1, center=True, time=window).construct('window')

    # Create empty percentile array
    g = rr.groupby('time.dayofyear')

    p = g.reduce(np.nanpercentile, dim=('time', 'window'), q=per * 100)

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = adjust_doy_calendar(p.loc[p.dayofyear < 366], arr)

    p.attrs.update(arr.attrs.copy())
    return p


def infer_doy_max(arr):
    """Return the largest doy allowed by calendar.

    Parameters
    ----------
    arr : xarray.DataArray
      Array with `time` coordinate.

    Returns
    -------
    int
      The largest day of the year found in calendar.
    """
    cal = arr.time.encoding.get('calendar', None)
    if cal in calendars:
        doy_max = calendars[cal]
    else:
        # If source is an array with no calendar information and whose length is not at least of full year,
        # then this inference could be wrong (
        doy_max = arr.time.dt.dayofyear.max().data
        if len(arr.time) < 360:
            raise ValueError("Cannot infer the calendar from a series less than a year long.")
        if doy_max not in [360, 365, 366]:
            raise ValueError("The target array's calendar is not recognized")

    return doy_max


def _interpolate_doy_calendar(source, doy_max):
    """Interpolate from one set of dayofyear range to another

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xarray.DataArray
      Array with `dayofyear` coordinates.
    doy_max : int
      Largest day of the year allowed by calendar.

    Returns
    -------
    xarray.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    if 'dayofyear' not in source.coords.keys():
        raise AttributeError("source should have dayofyear coordinates.")

    # Interpolation of source to target dayofyear range
    doy_max_source = source.dayofyear.max()

    # Interpolate to fill na values
    tmp = source.interpolate_na(dim='dayofyear')

    # Interpolate to target dayofyear range
    tmp.coords['dayofyear'] = np.linspace(start=1, stop=doy_max, num=doy_max_source)

    return tmp.interp(dayofyear=range(1, doy_max + 1))


def adjust_doy_calendar(source, target):
    """Interpolate from one set of dayofyear range to another calendar.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xarray.DataArray
      Array with `dayofyear` coordinates.
    target : xarray.DataArray
      Array with `time` coordinate.

    Returns
    -------
    xarray.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    doy_max_source = source.dayofyear.max()

    doy_max = infer_doy_max(target)
    if doy_max_source == doy_max:
        return source

    return _interpolate_doy_calendar(source, doy_max)


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

    if lon_bnds:

        lon_bnds = np.asarray(lon_bnds)
        if np.all(da.lon > 0) and np.any(lon_bnds < 0):
            lon_bnds[lon_bnds < 0] += 360
        da = da.where((da.lon >= lon_bnds.min()) & (da.lon <= lon_bnds.max()), drop=True)

    if lat_bnds:
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
    _nvar = 1

    # CF-Convention metadata to be attributed to the output variable. May use tags {<tag>} formatted at runtime.
    standard_name = ''  # The set of permissible standard names is contained in the standard name table.
    long_name = ''  # Parsed.
    units = ''  # Representative units of the physical quantity.
    cell_methods = ''  # List of blank-separated words of the form "name: method"
    description = ''  # The description is meant to clarify the qualifiers of the fundamental quantities, such as which
    #   surface a quantity is defined on or what the flux sign conventions are.

    # The `pint` unit context. Use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    context = 'none'

    # Additional information that can be used by third party libraries or to describe the file content.
    title = ''  # A succinct description of what is in the dataset. Default parsed from compute.__doc__
    abstract = ''  # Parsed
    keywords = ''  # Comma separated list of keywords
    references = ''  # Published or web-based references that describe the data or methods used to produce it. Parsed.
    comment = ''  # Miscellaneous information about the data or methods used to produce it.
    notes = ''  # Mathematical formulation. Parsed.

    # Tag mappings between keyword arguments and long-form text.
    months = {'m{}'.format(i): calendar.month_name[i].lower() for i in range(1, 13)}
    _attrs_mapping = {'cell_methods': {'YS': 'years', 'MS': 'months'},  # I don't think this is necessary.
                      'long_name': {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal', 'DJF': 'winter',
                                    'MAM': 'spring', 'JJA': 'summer', 'SON': 'fall'},
                      'description': {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal', 'DJF': 'winter',
                                      'MAM': 'spring', 'JJA': 'summer', 'SON': 'fall'},
                      'identifier': {'DJF': 'winter', 'MAM': 'spring', 'JJA': 'summer', 'SON': 'fall'}}

    for k, v in _attrs_mapping.items():
        v.update(months)

    # Whether or not the compute function is a partial.
    _partial = False

    # Can be used to override the compute docstring.
    doc_template = None

    def __init__(self, **kwds):

        for key, val in kwds.items():
            setattr(self, key, val)

        # Sanity checks
        required = ['compute', ]
        for key in required:
            if not getattr(self, key):
                raise ValueError("{} needs to be defined during instantiation.".format(key))

        # Extract information from the `compute` function.
        # The signature
        self._sig = signature(self.compute)

        # The input parameter names
        self._parameters = tuple(self._sig.parameters.keys())
        #        self._input_params = [p for p in self._sig.parameters.values() if p.default is p.empty]
        #        self._nvar = len(self._input_params)

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__.__func__)
        if self.doc_template is not None:
            self.__call__.__doc__ = self.doc_template.format(i=self)

        # Fill in missing metadata from the doc
        meta = parse_doc(self.compute.__doc__)
        for key in ['abstract', 'title', 'long_name', 'notes', 'references']:
            setattr(self, key, getattr(self, key) or meta.get(key, ''))

    def __call__(self, *args, **kwds):
        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        if self._partial:
            ba = self._sig.bind_partial(*args, **kwds)
            for key, val in self.compute.keywords.items():
                if key not in ba.arguments:
                    ba.arguments[key] = val
        else:
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

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(*das, **ba.kwargs)

        # Convert to output units
        out = convert_units_to(out, self.units, self.context)

        # Update netCDF attributes
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

        out['parameters'] = str({key: {'default': p.default if p.default != p.empty else None, 'desc': ''}
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

    def format(self, attrs, args=None):
        """Format attributes including {} tags with arguments."""
        if args is None:
            return attrs

        out = {}
        for key, val in attrs.items():
            mba = {'indexer': 'annual'}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, six.string_types) and v in self._attrs_mapping.get(key, {}).keys():
                    mba[k] = '{{{}}}'.format(v)
                elif isinstance(v, dict):
                    if v:
                        dk, dv = v.copy().popitem()
                        if dk == 'month':
                            dv = 'm{}'.format(dv)
                        mba[k] = '{{{}}}'.format(dv)
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


class Indicator2D(Indicator):
    _nvar = 2


def parse_doc(doc):
    """Crude regex parsing."""
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


def wrapped_partial(func, *args, **kwargs):
    from functools import partial, update_wrapper
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
