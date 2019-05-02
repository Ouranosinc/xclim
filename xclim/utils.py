# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import abc
import calendar
import datetime as dt
import functools
import re
import warnings
from collections import defaultdict
from inspect import signature

import numpy as np
import pint
import six
import xarray as xr
from boltons.funcutils import wraps

from . import checks

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
