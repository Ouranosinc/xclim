# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import numpy as np
import xarray as xr
import pandas as pd
import time
import six
from functools import wraps
import pint
from . import checks
from inspect2 import signature
import dask

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

units.define(pint.unit.UnitDefinition('percent', '%', (),
                                      pint.converters.ScaleConverter(0.01)))

# Define commonly encountered units not defined by pint
units.define('degrees_north = degree = degrees_N = degreesN = degree_north = degree_N '
             '= degreeN')
units.define('degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE')
hydro = pint.Context('hydro')
hydro.add_transformation('[mass] / [length]**2', '[length]', lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3))
hydro.add_transformation('[mass] / [length]**2 / [time]', '[length] / [time]',
                         lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3))
hydro.add_transformation('[length] / [time]', '[mass] / [length]**2 / [time]',
                         lambda ureg, x: x * (1000 * ureg.kg / ureg.m ** 3))
units.add_context(hydro)
units.enable_contexts(hydro)


def get_daily_events(da, da_value, operator):
    r"""
    function that returns a 0/1 mask when a condtion is True or False

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


class UnivariateIndicator(object):
    r"""xclim indicator class"""


def get_ev_length(ev, verbose=True, method=2, timing=True, using_dask_loop=True):
    r"""Function computing event/non-event length

    Parameters
    ----------
    ev : xarray.DataArray
       array of 0/1 values indicating events/non-events with time dimension
    verbose : Logical
       make the program verbose
    method : int, optional
       choice of method to do the computing. See comments in code.
    timing : logical, optional
       if True print timing information
    using_dask_loop : logical, optional
       if True code uses dask to loop the array

    Returns
    -------
    xarray.DataArray
       array containing length of every event/non-event sequences

    Note
    ----

    with input = [0,0,1,1,1,0,0,1,1,1,1], output = [2,2,3,3,3,2,2,4,4,4,4]
         input = [0,1,1,2,2,0,0,0], output = [1,2,2,2,2,3,3,3]

    Has been tested with 1D and 3D DataArray

    # inspire/copy of :
    # https://stackoverflow.com/questions/45886518/identify-consecutive-same-values-in-pandas-dataframe-with-a-groupby
    """

    # make sure we have a time dimension
    assert ('time' in ev.dims)

    # echo option values to output
    if verbose:
        print('get_ev_length options : method={:}, using_dask_loop={:}'.format(method, using_dask_loop))

    # create mask of event change, 1 if no change and 0 otherwise
    # fill first value with 1
    start = ev.isel(time=0)
    if ev.ndim == 1:
        # special 1d case
        start.values = 1
    else:
        start.values[:] = 1
    # compute difference and apply mask
    ev_diff = (ev.diff(dim='time') != 0) * 1
    # add start
    ev_diff = xr.concat((start, ev_diff), dim='time')

    # make cumulative sum
    diff_cumsum = ev_diff.cumsum(dim='time')

    # define method of treating vectors of events
    # tests suggest method2 is faster, followed by method3
    # method1 is noticeably slower ...
    #
    def method1(v):
        # using the pd.Series.value_counts()
        s = pd.Series(v)
        d = s.map(s.value_counts())
        return d

    def method2(v):
        # using np.unique
        useless, ind = np.unique(v, return_index=True)
        i0 = 0
        d = np.zeros_like(v)
        for i in ind:
            ll = i - i0
            d[i0:i] = ll
            i0 = i
        d[i:] = d.size - i
        return d

    def method3(v):
        # variation of method2
        useless, ind = np.unique(v, return_index=True)
        ind = np.append(ind, v.size)
        dur = np.diff(ind)
        d = np.zeros_like(v)
        im = 0
        for idur, id in enumerate(ind[1:]):
            d[im:id] = dur[idur]
            im = id
        return d

    vec_method = {1: method1, 2: method2, 3: method3}[method]

    if timing: time0 = time.time()
    # treatment depends on number fo dimensions
    if ev.ndim == 1:
        ev_l = ev.copy()
        v = diff_cumsum.values
        ev_l.values = vec_method(v)
        return ev_l
    else:
        #
        # two way to loop the arrays. Using dask is noticeably faster.
        #
        if not using_dask_loop:
            # multidimension case
            #
            # reshape in 2D to simplify loop using stack
            #
            non_time_dims = [d for d in diff_cumsum.dims if d != 'time']
            mcumsum = diff_cumsum.stack(z=non_time_dims)
            nz = mcumsum.sizes['z']

            # prepare output
            ev_l = mcumsum.copy()

            # loop on stacked array
            for z in range(nz):
                v = mcumsum.isel(z=z).values
                ll = vec_method(v)
                ev_l.isel(z=z).values[:] = ll

            # go back to original shape and return event length
            ev_l = ev_l.unstack('z')
        else:

            # loop with apply_along_axis
            data = dask.array.apply_along_axis(vec_method, diff_cumsum.get_axis_num('time'),
                                               diff_cumsum).compute()
            diff_cumsum.values = data

            ev_l = diff_cumsum

            # reorder dimensions as ev
            ev, ev_l = xr.broadcast(ev, ev_l)

        if timing:
            print('timing for get_ev_length done in {:10.2f}s'.format(time.time() - time0))

        return ev_l


def get_ev_end(ev):
    r"""
    function flaging places when an event sequence ends

    :param ev: xarray DataArray
        array containing 1 for events and 0 for non-events
    :return: ev_end

    e.g. input = [0,0,1,1,1,0,0,1] returns [0,0,0,0,1,0,0,1]

    """

    # find when events finish and mask all other event points
    d = ev.diff(dim='time')
    ev_end = xr.where(d == -1, 1, 0)

    # shift end of events back for proper time alignment
    ev_end['time'] = ev.time[:-1]
    # deal with cases when last timestep is end of period
    ev_end = xr.concat((ev_end, ev.isel(time=-1)), 'time')
    return ev_end


def get_ev_start(ev):
    r"""
    function flaging places when an event sequence starts

    :param ev: xarray DataArray
        array containing 1 for events and 0 for non-events
    :return: ev_end

    e.g. input = [1,0,1,1,1,0,0,1] returns [1,0,1,0,0,0,0,1]

    """

    # find when events finish and mask all other event points
    d = ev.diff(dim='time')
    ev_start = xr.where(d == 1, 1, 0)

    # copy first timestep of ev to catch those start
    ev_start = xr.concat((ev.isel(time=0), ev_start), 'time')
    return ev_start


class Indicator(object):
    identifier = ''
    units = ''
    required_units = ''
    long_name = ''
    standard_name = ''
    description = ''
    keywords = ''
    cell_methods = ''

    _attrs_mapping = {'cell_methods': {'YS': 'years', 'MS': 'months'},
                      'long_name': {'YS': 'Annual', 'MS': 'Monthly'},
                      'standard_name': {'YS': 'Annual', 'MS': 'Monthly'}, }

    def compute(self, da, *args, **kwds):
        """Index computation method. To be subclassed"""
        raise NotImplementedError

    def convert_units(self, da):
        """Return DataArray with correct units, defined by `self.required_units`."""
        fu = units.parse_units(da.attrs['units'].replace('-', '**-'))
        tu = units.parse_units(self.required_units.replace('-', '**-'))
        if fu != tu:
            b = da.copy()
            b.values = (da.values * fu).to(tu, 'hydro')
            return b

        return da

    def cfprobe(self, da):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        pass

    def validate(self, da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)

    def decorate(self, da):
        """Modify output's attributes in place.

        If attribute's value contain formatting markup such {<name>}, they are replaced by call arguments.
        """

        attrs = {}
        for key, val in self.attrs.items():
            mba = {}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in self._ba.arguments.items():
                if isinstance(v, six.string_types) and v in self._attrs_mapping.get(key, {}).keys():
                    mba[k] = '{' + v + '}'
                else:
                    mba[k] = v

            attrs[key] = val.format(**mba).format(**self._attrs_mapping.get(key, {}))

        da.attrs.update(attrs)

    def missing(self, da, **kwds):
        """Return boolean DataArray."""
        return checks.missing_any(da, kwds['freq'])

    def __init__(self):
        # Extract DataArray arguments from compute signature.
        self.attrs = {'long_name': self.long_name,
                      'units': self.units,
                      'standard_name': self.standard_name,
                      'cell_methods': self.cell_methods,
                      }

    def __call__(self, *args, **kwds):
        # Bind call arguments
        self._ba = signature(self.compute).bind(*args, **kwds)
        self._ba.apply_defaults()

        self.validate(args[0])
        self.cfprobe(args[0])

        da = self.convert_units(args[0])

        out = self.compute(da, *args[1:], **kwds).rename(self.identifier.format(self._ba.arguments))

        self.decorate(out)

        # The missing method should be given the same `freq` as compute. It will be in args or kwds if given
        # explicitly, but if not, we pass the default from `compute`.
        return out.where(~self.missing(**self._ba.arguments))


def first_paragraph(txt):
    r"""Return the first paragraph of a text

    Parameters
    ----------
    txt : str
    """
    return txt.split('\n\n')[0]


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


def with_attrs(**func_attrs):
    r"""Set attributes in the decorated function at definition time,
    and assign these attributes to the function output at the
    execution time.

    Note
    ----
    Assumes the output has an attrs dictionary attribute (e.g. xarray.DataArray).
    """

    def attr_decorator(fn):
        # Use the docstring as the description attribute.
        func_attrs['description'] = first_paragraph(fn.__doc__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            # Bind the arguments
            ba = signature(fn).bind(*args, **kwargs)
            format_kwargs(func_attrs, ba.arguments)
            out.attrs.update(func_attrs)
            return out

        # Assign the attributes to the function itself
        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator
