# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import numpy as np
import xarray as xr
import six
import pint
from . import checks
from inspect2 import signature
import abc


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


def percentile_doy(arr, window=5, per=.1):
    """Percentile value for each day of the year

    Returns the climatological percentile over a moving window around each day of the year.

    Parameters
    ----------
    arr : xarray.DataArray
    window : int
    per : float
    """

    # TODO: Support percentile array, store percentile in attributes.
    rr = arr.rolling(min_periods=1, center=True, time=window).construct('window')

    # Create empty percentile array
    g = rr.groupby('time.dayofyear')
    c = g.count(dim=('time', 'window'))

    p = xr.full_like(c, np.nan).astype(float).load()

    for doy, ind in rr.groupby('time.dayofyear'):
        p.loc[{'dayofyear': doy}] = ind.compute().quantile(per, dim=('time', 'window'))

    return p


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


class Indicator(object):
    r"""Indicator class

    This class needs to be subclassed by individual indicator classes defining metadata information, compute and
    missing functions. It can handle indicators with any number of forcing fields.

    """
    # Unique ID for registry. May use tags {<tag>} that will be formatted at runtime.
    identifier = ''

    # CF-Convention metadata to be attributed to output. May use tags {<tag>} that will be formatted at runtime.
    standard_name = ''  # The set of permissible standard names is contained in the standard name table.
    long_name = ''  # Scraped from compute.__doc.__.
    units = ''  # Representative units of the physical quantity.
    cell_methods = ''  # List of blank-separated words of the form "name: method"
    description = ''  # The description is meant to clarify the qualifiers of the fundamental quantities, such as which
    #   surface a quantity is defined on or what the flux sign conventions are.

    # The units expected by the function. Used to convert input units to the required_units.
    required_units = ''

    # The `pint` unit context. Use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    context = None

    # Additional information made available to third party libraries.
    title = ''  # Scraped from compute.__doc.__
    abstract = ''  # Scraped from compute.__doc.__
    keywords = ''  # Comma separated list of keywords

    # Tag mappings between keyword arguments and long-form text.
    _attrs_mapping = {'cell_methods': {'YS': 'years', 'MS': 'months'},  # I don't think this is necessary.
                      'long_name': {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal'},
                      'description':  {'YS': 'Annual', 'MS': 'Monthly', 'QS-DEC': 'Seasonal'}}

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

        # The docstring
        self.__call__.__func__.__doc__ = self.compute.__doc__

        # Fill in missing metadata from the doc
        meta = parse_doc(self.compute)
        for key in ['abstract', 'title', 'long_name']:
            setattr(self, key, getattr(self, key) or meta.get(key, ''))

    def __call__(self, *args, **kwds):
        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assume the two first arguments are always the DataArray.
        das = tuple((ba.arguments.pop(self._parameters[i]) for i in range(self._nvar)))

        # Pre-computation validation checks
        for da in das:
            self.validate(da)
        self.cfprobe(*das)

        # Convert units if necessary
        das = tuple((self.convert_units(da, ru, self.context) for (da, ru) in zip(das, self.required_units)))

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(*das, **ba.arguments)

        # Set metadata attributes to the output according to class attributes.
        self.decorate(out, ba.arguments)

        # Bind call arguments to the `missing` function, whose signature might be different from `compute`.
        mba = signature(self.missing).bind(*das, **ba.arguments)

        # Mask results that do not meet criteria defined by the `missing` method.
        mask = self.missing(*mba.args, **mba.kwargs)
        ma_out = out.where(~mask)

        return ma_out.rename(self.identifier.format(ba.arguments))

    @property
    def cf_attrs(self):
        """CF-Convention attributes of the output value."""
        names = ['standard_name', 'long_name', 'units', 'cell_methods', 'description']
        return {k: getattr(self, k) for k in names}

    @property
    def json(self):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ['identifier', 'abstract', 'keywords', ]
        out = {key: getattr(self, key) for key in names}

        out['parameters'] = {key: {'default': p.default, 'desc': ''} for (key, p) in self._sig.parameters.items()}

        out.update(self.cf_attrs)

        return out

    def cfprobe(self, *das):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        pass

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""

    @staticmethod
    def convert_units(da, req_units, context=None):
        """Return DataArray converted to unit."""
        fu = units.parse_units(da.attrs['units'].replace('-', '**-'))
        tu = units.parse_units(req_units.replace('-', '**-'))
        if fu != tu:
            b = da.copy()
            if context:
                b.values = (da.values * fu).to(tu, context)
            else:
                b.values = (da.values * fu).to(tu)
            return b

        return da

    def decorate(self, da, args=None):
        """Modify output's attributes in place.

        If attribute's value contain formatting markup such {<name>}, they are replaced by call arguments.
        """
        if args is None:
            return

        attrs = {}
        for key, val in self.cf_attrs.items():
            mba = {}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, six.string_types) and v in self._attrs_mapping.get(key, {}).keys():
                    mba[k] = '{' + v + '}'
                else:
                    mba[k] = v

            attrs[key] = val.format(**mba).format(**self._attrs_mapping.get(key, {}))

        da.attrs.update(attrs)

    @staticmethod
    def missing(*args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get('freq')
        miss = (checks.missing_any(da, freq) for da in args)
        return reduce(xr.ufuncs.logical_or, miss)

    def validate(self, da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)

    @classmethod
    def factory(cls, attrs):
        """Create a subclass from the attributes dictionary."""
        name = attrs['identifier'].capitalize()
        return type(name, (cls,), attrs)


def parse_doc(obj):
    """Crude regex parsing."""
    import re
    if obj.__doc__ is None:
        return {}

    sections = obj.__doc__.split('\n\n')

    patterns = {'long_name': r'^\s+Return.\n\s+------.*\n\s+xarray\.DataArray\s*(.*)',
                'notes': r'^\s+Notes.\n\s+----.*\n(.*)'}

    out = {}
    for i, sec in enumerate(sections):
        if i == 0:
            out['title'] = sec.strip()
        elif i == 1:
            out['abstract'] = sec.strip()
        else:
            for key, pat in patterns.items():
                m = re.match(pat, sec)
                if m:
                    out[key] = m.groups()[0]

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
