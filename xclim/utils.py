# -*- coding: utf-8 -*-
"""
xclim xarray.DataArray utilities module
"""
# import abc
import calendar
import functools
import re
import string
import warnings
from collections import defaultdict
from collections import OrderedDict
from datetime import timedelta
from inspect import signature
from types import FunctionType
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pint.converters
import pint.unit
import xarray as xr
from boltons.funcutils import wraps
from packaging import version
from xarray.coding.cftime_offsets import MonthBegin
from xarray.coding.cftime_offsets import MonthEnd
from xarray.coding.cftime_offsets import QuarterBegin
from xarray.coding.cftime_offsets import QuarterEnd
from xarray.coding.cftime_offsets import to_offset
from xarray.coding.cftime_offsets import YearBegin
from xarray.coding.cftime_offsets import YearEnd
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.resample import DataArrayResample


__all__ = [
    "units",
    "units2pint",
    "pint2cfunits",
    "pint_multiply",
    "convert_units_to",
    "declare_units",
    "units",
    "threshold_count",
    "percentile_doy",
    "infer_doy_max",
    "adjust_doy_calendar",
    "get_daily_events",
    "daily_downsampler",
    "walk_map",
    "default_formatter",
    "AttrFormatter",
    "parse_doc",
    "cfindex_start_time",
    "cfindex_end_time",
    "cftime_start_time",
    "cftime_end_time",
    "time_bnds",
    "wrapped_partial",
    "uas_vas_2_sfcwind",
    "sfcwind_2_uas_vas",
]


units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
units.define(
    pint.unit.UnitDefinition(
        "percent", "%", ("pct",), pint.converters.ScaleConverter(0.01)
    )
)

# Define commonly encountered units not defined by pint
if version.parse(pint.__version__) >= version.parse("0.10"):
    units.define("@alias degC = C = deg_C")
    units.define("@alias degK = deg_K")
    units.define("@alias day = d")
    units.define("@alias hour = h")  # Not the Planck constant...
    units.define(
        "@alias degree = degrees_north = degrees_N = degreesN = degree_north = degree_N = degreeN"
    )
    units.define(
        "@alias degree = degrees_east = degrees_E = degreesE = degree_east = degree_E = degreeE"
    )

else:
    units.define("degC = kelvin; offset: 273.15 = celsius = C = deg_C")
    units.define("d = day")
    units.define("h = hour")
    units.define(
        "degrees_north = degree = degrees_N = degreesN = degree_north = degree_N "
        "= degreeN"
    )
    units.define(
        "degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE"
    )

units.define("[speed] = [length] / [time]")

# Default context.
null = pint.Context("none")
units.add_context(null)

# Precipitation units. This is an artificial unit that we're using to verify that a given unit can be converted into
# a precipitation unit. Ideally this could be checked through the `dimensionality`, but I can't get it to work.
units.define("[precipitation] = [mass] / [length] ** 2 / [time]")
units.define("mmday = 1 kg / meter ** 2 / day")

units.define("[discharge] = [length] ** 3 / [time]")
units.define("cms = meter ** 3 / second")

hydro = pint.Context("hydro")
hydro.add_transformation(
    "[mass] / [length]**2",
    "[length]",
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3),
)
hydro.add_transformation(
    "[mass] / [length]**2 / [time]",
    "[length] / [time]",
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3),
)
hydro.add_transformation(
    "[length] / [time]",
    "[mass] / [length]**2 / [time]",
    lambda ureg, x: x * (1000 * ureg.kg / ureg.m ** 3),
)
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
binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le"}

# Maximum day of year in each calendar.
calendars = {
    "standard": 366,
    "gregorian": 366,
    "proleptic_gregorian": 366,
    "julian": 366,
    "no_leap": 365,
    "365_day": 365,
    "all_leap": 366,
    "366_day": 366,
    "uniform30day": 360,
    "360_day": 360,
}


def units2pint(value: Union[xr.DataArray, str]) -> pint.unit.UnitDefinition:
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : Union[xr.DataArray, str]
      Input data array or expression.

    Returns
    -------
    pint.unit.UnitDefinition
      Units of the data array.

    """

    def _transform(s):
        """Convert a CF-unit string to a pint expression."""
        if s == "%":
            return "percent"

        return re.subn(r"([a-zA-Z]+)\^?(-?\d)", r"\g<1>**\g<2>", s)[0]

    if isinstance(value, str):
        unit = value
    elif isinstance(value, xr.DataArray):
        unit = value.attrs["units"]
    elif isinstance(value, units.Quantity):
        return value.units
    else:
        raise NotImplementedError(f"Value of type `{type(value)}` not supported.")

    unit = unit.replace("%", "pct")
    try:  # Pint compatible
        return units.parse_expression(unit).units
    except (
        pint.UndefinedUnitError,
        pint.DimensionalityError,
        AttributeError,
    ):  # Convert from CF-units to pint-compatible
        return units.parse_expression(_transform(unit)).units


# Note: The pint library does not have a generic Unit or Quantity type at the moment. Using "Any" as a stand-in.
def pint2cfunits(value: Any) -> str:
    """Return a CF-Convention unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.UnitRegistry
      Input unit.

    Returns
    -------
    out : str
      Units following CF-Convention.
    """
    # Print units using abbreviations (millimeter -> mm)
    s = f"{value:~}"

    # Search and replace patterns
    pat = r"(?P<inverse>/ )?(?P<unit>\w+)(?: \*\* (?P<pow>\d))?"

    def repl(m):
        i, u, p = m.groups()
        p = p or (1 if i else "")
        neg = "-" if i else ("^" if p else "")

        return f"{u}{neg}{p}"

    out, n = re.subn(pat, repl, s)
    return out.replace("percent", "%")


def pint_multiply(da: xr.DataArray, q: Any, out_units: Optional[str] = None):
    """Multiply xarray.DataArray by pint.Quantity.

    Parameters
    ----------
    da : xr.DataArray
      Input array.
    q : pint.Quantity
      Multiplicative factor.
    out_units : Optional[str]
      Units the output array should be converted into.
    """
    a = 1 * units2pint(da)
    f = a * q.to_base_units()
    if out_units:
        f = f.to(out_units)
    out = da * f.magnitude
    out.attrs["units"] = pint2cfunits(f.units)
    return out


def convert_units_to(
    source: Union[str, xr.DataArray, Any],
    target: Union[str, xr.DataArray, Any],
    context: Optional[str] = None,
):
    """
    Convert a mathematical expression into a value with the same units as a DataArray.

    Parameters
    ----------
    source : Union[str, xr.DataArray, Any]
      The value to be converted, e.g. '4C' or '1 mm/d'.
    target : Union[str, xr.DataArray, Any]
      Target array of values to which units must conform.
    context : Optional[str]

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
        tu_u = pint2cfunits(tu)

        if fu == tu:
            # The units are the same, but the symbol may not be.
            source.attrs["units"] = tu_u
            return source

        with units.context(context or "none"):
            out = xr.DataArray(
                data=units.convert(source.values, fu, tu),
                coords=source.coords,
                attrs=source.attrs,
                name=source.name,
            )
            out.attrs["units"] = tu_u
            return out

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(source, (float, int)):
        if context == "hydro":
            fu = units.mm / units.day
        else:
            fu = units.degC
        warnings.warn(
            "Future versions of xclim will require explicit unit specifications.",
            FutureWarning,
            stacklevel=3,
        )
        return (source * fu).to(tu).m

    raise NotImplementedError(f"Source of type `{type(source)}` is not supported.")


def _check_units(val: Optional[Union[str, int, float]], dim: Optional[str]) -> None:
    if dim is None or val is None:
        return

    if str(val).startswith("UNSET "):
        warnings.warn(
            "This index calculation will soon require user-specified thresholds.",
            FutureWarning,
            stacklevel=4,
        )
        val = str(val).replace("UNSET ", "")

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(val, (int, float)):
        return

    expected = units.get_dimensionality(dim.replace("dimensionless", ""))
    val_dim = units2pint(val).dimensionality
    if val_dim == expected:
        return

    # Check if there is a transformation available
    start = pint.util.to_units_container(expected)
    end = pint.util.to_units_container(val_dim)
    graph = units._active_ctx.graph
    if pint.util.find_shortest_path(graph, start, end):
        return

    if dim == "[precipitation]":
        tu = "mmday"
    elif dim == "[discharge]":
        tu = "cms"
    elif dim == "[length]":
        tu = "m"
    else:
        raise NotImplementedError(f"Dimension `{dim}` is not supported.")

    try:
        (1 * units2pint(val)).to(tu, "hydro")
    except (pint.UndefinedUnitError, pint.DimensionalityError):
        raise AttributeError(
            f"Value's dimension `{val_dim}` does not match expected units `{expected}`."
        )


def declare_units(out_units, check_output: bool = True, **units_by_name):
    """Create a decorator to check units of function arguments.

    The decorator checks that input and output values have units that are compatible with expected dimensions.

    Examples
    --------
    In the following function definition:

    .. code::

       @declare_units("K", tas=["temperature"])
       def func(tas):
          ...

    the decorator will check that `tas` has units of temperature (C, K, F) and that the output is in Kelvins.

    """

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
            if check_output:
                if "units" in out.attrs:
                    # Check that output units dimensions match expectations, e.g. [temperature]
                    if "[" in out_units:
                        _check_units(out, out_units)
                    # Explicitly convert units if units are declared, e.g K
                    else:
                        out = convert_units_to(out, out_units)

                # Otherwise, we impose the units if given.
                elif "[" not in out_units:
                    out.attrs["units"] = out_units

                else:
                    raise ValueError(
                        "Output units are not propagated by computation nor specified by decorator."
                    )

            return out

        return wrapper

    return dec


def threshold_count(
    da: xr.DataArray, op: str, thresh: float, freq: str
) -> xr.DataArray:
    """Count number of days above or below threshold.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    op : {">", "<", ">=", "<=", "gt", "lt", "ge", "le"}
      Logical operator {>, <, >=, <=, gt, lt, ge, le}. e.g. arr > thresh.
    thresh : float
      Threshold value.
    freq : str
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
      The number of days meeting the constraints for each period.
    """
    from xarray.core.ops import get_op

    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError(f"Operation `{op}` not recognized.")

    func = getattr(da, "_binary_op")(get_op(op))
    c = func(da, thresh) * 1
    return c.resample(time=freq).sum(dim="time")


def percentile_doy(
    arr: xr.DataArray, window: int = 5, per: float = 0.1
) -> xr.DataArray:
    """Percentile value for each day of the year

    Return the climatological percentile over a moving window around each day of the year.

    Parameters
    ----------
    arr : xr.DataArray
      Input data.
    window : int
      Number of days around each day of the year to include in the calculation.
    per : float
      Percentile between [0,1]

    Returns
    -------
    xr.DataArray
      The percentiles indexed by the day of the year.
    """
    # TODO: Support percentile array, store percentile in coordinates.
    #  This is supported by DataArray.quantile, but not by groupby.reduce.
    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    # Create empty percentile array
    g = rr.groupby("time.dayofyear")

    p = g.reduce(np.nanpercentile, dim=("time", "window"), q=per * 100)

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = adjust_doy_calendar(p.sel(dayofyear=(p.dayofyear < 366)), arr)

    p.attrs.update(arr.attrs.copy())
    return p


def infer_doy_max(arr: xr.DataArray) -> int:
    """Return the largest doy allowed by calendar.

    Parameters
    ----------
    arr : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    int
      The largest day of the year found in calendar.
    """
    cal = arr.time.encoding.get("calendar", None)
    if cal in calendars:
        doy_max = calendars[cal]
    else:
        # If source is an array with no calendar information and whose length is not at least of full year,
        # then this inference could be wrong (
        doy_max = arr.time.dt.dayofyear.max().data
        if len(arr.time) < 360:
            raise ValueError(
                "Cannot infer the calendar from a series less than a year long."
            )
        if doy_max not in [360, 365, 366]:
            raise ValueError(f"The target array's calendar `{cal}` is not recognized.")

    return doy_max


def _interpolate_doy_calendar(source: xr.DataArray, doy_max: int) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
      Array with `dayofyear` coordinates.
    doy_max : int
      Largest day of the year allowed by calendar.

    Returns
    -------
    xr.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    if "dayofyear" not in source.coords.keys():
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Interpolation of source to target dayofyear range
    doy_max_source = int(source.dayofyear.max())

    # Interpolate to fill na values
    tmp = source.interpolate_na(dim="dayofyear")

    # Interpolate to target dayofyear range
    tmp.coords["dayofyear"] = np.linspace(start=1, stop=doy_max, num=doy_max_source)

    return tmp.interp(dayofyear=range(1, doy_max + 1))


def adjust_doy_calendar(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Interpolate from one set of dayofyear range to another calendar.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
      Array with `dayofyear` coordinate.
    target : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      Interpolated source array over coordinates spanning the target `dayofyear` range.

    """
    doy_max_source = source.dayofyear.max()

    doy_max = infer_doy_max(target)
    if doy_max_source == doy_max:
        return source

    return _interpolate_doy_calendar(source, doy_max)


def resample_doy(doy: xr.DataArray, arr: xr.DataArray) -> xr.DataArray:
    """Create a temporal DataArray where each day takes the value defined by the day-of-year.

    Parameters
    ----------
    doy : xr.DataArray
      Array with `dayofyear` coordinate.
    arr : xr.DataArray
      Array with `time` coordinate.

    Returns
    -------
    xr.DataArray
      An array with the same `time` dimension as `arr` whose values are filled according to the day-of-year value in
      `doy`.
    """
    if "dayofyear" not in doy.coords:
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Adjust calendar
    adoy = adjust_doy_calendar(doy, arr)

    # Create array with arr shape and coords
    out = xr.full_like(arr, np.nan)

    # Fill with values from `doy`
    d = out.time.dt.dayofyear.values
    out.data = adoy.sel(dayofyear=d)

    return out


def get_daily_events(da: xr.DataArray, da_value: float, operator: str) -> xr.DataArray:
    r"""
    function that returns a 0/1 mask when a condition is True or False

    the function returns 1 where operator(da, da_value) is True
                         0 where operator(da, da_value) is False
                         nan where da is nan

    Parameters
    ----------
    da : xr.DataArray
    da_value : float
    operator : str


    Returns
    -------
    xr.DataArray

    """
    events = operator(da, da_value) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


def daily_downsampler(da: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    r"""Daily climate data downsampler

    Parameters
    ----------
    da : xr.DataArray
    freq : str

    Returns
    -------
    xr.DataArray

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
        years = [f"{y:04d}" for y in da.time.dt.year.values]
        months = [f"{m:02d}" for m in da.time.dt.month.values]
    else:
        # cannot use year, month, season attributes, not available for all calendars ...
        years = [f"{v.year:04d}" for v in da.time.values]
        months = [f"{v.month:02d}" for v in da.time.values]
    seasons = [
        "DJF DJF MAM MAM MAM JJA JJA JJA SON SON SON DJF".split()[int(m) - 1]
        for m in months
    ]

    n_t = da.time.size
    if freq == "YS":
        # year start frequency
        l_tags = years
    elif freq == "MS":
        # month start frequency
        l_tags = [years[i] + months[i] for i in range(n_t)]
    elif freq == "QS-DEC":
        # DJF, MAM, JJA, SON seasons
        # construct tags from list of season+year, increasing year for December
        ys = []
        for i in range(n_t):
            m = months[i]
            s = seasons[i]
            y = years[i]
            if m == "12":
                y = str(int(y) + 1)
            ys.append(y + s)
        l_tags = ys
    else:
        raise RuntimeError(f"Frequency `{freq}` not implemented.")

    # add tags to buffer DataArray
    buffer = da.copy()
    buffer.coords["tags"] = ("time", l_tags)

    # return groupby according to tags
    return buffer.groupby("tags")


def walk_map(d: dict, func: FunctionType):
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
      Input dictionary, possibly nested.
    func : FunctionType
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


class AttrFormatter(string.Formatter):
    """A formatter for frequently used attribute values.

    See the doc of format_field() for more details.
    """

    def __init__(
        self, mapping: Mapping[str, Sequence[str]], modifiers: Sequence[str],
    ):
        """Initialize the formatter.

        Parameters
        ----------
        mapping : Mapping[str, Sequence[str]]
            A mapping from values to their possible variations.
        modifiers : Sequence[str]
            The list of modifiers, must be the as long as the longest value of `mapping`.
        """
        super().__init__()
        self.modifiers = modifiers
        self.mapping = mapping

    def format_field(self, value, format_spec):
        """Format a value given a formatting spec.

        If `format_spec` is in this Formatter's modifiers, the correspong variation
        of value is given. If `format_spec` is not specified but `value` is in the
        mapping, the first variation is returned.

        Example
        -------
        Let's say the string "The dog is {adj1}, the goose is {adj2}" is to be translated
        to french and that we know that possible values of `adj` are `nice` and `evil`.
        In french, the genre of the noun changes the adjective (cat = chat is masculine,
        and goose = oie is feminine) so we initialize the formatter as:

        >>> fmt = AttrFormatter({'nice': ['beau', 'belle'], 'evil' : ['méchant', 'méchante']},
                                ['m', 'f'])
        >>> fmt.format("Le chien est {adj1:m}, l'oie est {adj2:f}",
                       adj1='nice', adj2='evil')
        "Le chien est beau, l'oie est méchante"
        """
        if value in self.mapping and not format_spec:
            return self.mapping[value][0]

        if format_spec in self.modifiers:
            if value in self.mapping:
                return self.mapping[value][self.modifiers.index(format_spec)]
            raise ValueError(
                f"No known mapping for string '{value}' with modifier '{format_spec}'"
            )
        return super().format_field(value, format_spec)


# Tag mappings between keyword arguments and long-form text.
default_formatter = AttrFormatter(
    {
        "YS": ["annual", "years"],
        "MS": ["monthly", "months"],
        "QS-DEC": ["seasonal", "seasons"],
        "DJF": ["winter"],
        "MAM": ["spring"],
        "JJA": ["summer"],
        "SON": ["fall"],
        "norm": ["Normal"],
        "m1": ["january"],
        "m2": ["february"],
        "m3": ["march"],
        "m4": ["april"],
        "m5": ["may"],
        "m6": ["june"],
        "m7": ["july"],
        "m8": ["august"],
        "m9": ["september"],
        "m10": ["october"],
        "m11": ["november"],
        "m12": ["december"],
    },
    ["adj", "noun"],
)


def parse_doc(doc):
    """Crude regex parsing."""
    if doc is None:
        return {}

    out = {}

    sections = re.split(r"(\w+)\n\s+-{4,50}", doc)  # obj.__doc__.split('\n\n')
    intro = sections.pop(0)
    if intro:
        content = list(map(str.strip, intro.strip().split("\n\n")))
        if len(content) == 1:
            out["title"] = content[0]
        elif len(content) == 2:
            out["title"], out["abstract"] = content

    for i in range(0, len(sections), 2):
        header, content = sections[i : i + 2]

        if header in ["Notes", "References"]:
            out[header.lower()] = content.replace("\n    ", "\n")
        elif header == "Parameters":
            pass
        elif header == "Returns":
            match = re.search(r"xarray\.DataArray\s*(.*)", content)
            if match:
                out["long_name"] = match.groups()[0]

    return out


def cftime_start_time(date, freq):
    """
    Get the cftime.datetime for the start of a period. As we are not supplying
    actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used
    on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    datetime : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    cftime.datetime
        The starting datetime of the period inferred from date and freq.
    """

    freq = to_offset(freq)
    if isinstance(freq, (YearBegin, QuarterBegin, MonthBegin)):
        raise ValueError("Invalid frequency: " + freq.rule_code())
    if isinstance(freq, YearEnd):
        month = freq.month
        return date - YearEnd(n=1, month=month) + timedelta(days=1)
    if isinstance(freq, QuarterEnd):
        month = freq.month
        return date - QuarterEnd(n=1, month=month) + timedelta(days=1)
    if isinstance(freq, MonthEnd):
        return date - MonthEnd(n=1) + timedelta(days=1)
    return date


def cftime_end_time(date, freq):
    """
    Get the cftime.datetime for the end of a period. As we are not supplying
    actual period objects, assumptions regarding the period are made based on
    the given freq. IMPORTANT NOTE: this function cannot be used
    on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    datetime : cftime.datetime
        The original datetime object as a proxy representation for period.
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    cftime.datetime
        The ending datetime of the period inferred from date and freq.
    """
    freq = to_offset(freq)
    if isinstance(freq, (YearBegin, QuarterBegin, MonthBegin)):
        raise ValueError("Invalid frequency: " + freq.rule_code())
    if isinstance(freq, YearEnd):
        mod_freq = YearBegin(n=freq.n, month=freq.month)
    elif isinstance(freq, QuarterEnd):
        mod_freq = QuarterBegin(n=freq.n, month=freq.month)
    elif isinstance(freq, MonthEnd):
        mod_freq = MonthBegin(n=freq.n)
    else:
        mod_freq = freq
    return cftime_start_time(date + mod_freq, freq) - timedelta(microseconds=1)


def cfindex_start_time(cfindex, freq):
    """
    Get the start of a period for a pseudo-period index. As we are using
    datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function
    cannot be used on greater-than-day freq that start at the beginning of a
    month, e.g., 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    CFTimeIndex
        The starting datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_start_time(date, freq) for date in cfindex])


def cfindex_end_time(cfindex, freq):
    """
    Get the start of a period for a pseudo-period index. As we are using
    datetime indices to stand in for period indices, assumptions regarding the
    period are made based on the given freq. IMPORTANT NOTE: this function
    cannot be used on greater-than-day freq that start at the beginning of a
    month, e.g., 'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    cfindex : CFTimeIndex
        CFTimeIndex as a proxy representation for CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', 'H', or '3T'

    Returns
    _______
    CFTimeIndex
        The ending datetimes of periods inferred from dates and freq
    """
    return CFTimeIndex([cftime_end_time(date, freq) for date in cfindex])


def time_bnds(group, freq):
    """
    Find the time bounds for a pseudo-period index. As we are using datetime
    indices to stand in for period indices, assumptions regarding the period
    are made based on the given freq. IMPORTANT NOTE: this function cannot be
    used on greater-than-day freq that start at the beginning of a month, e.g.,
    'MS', 'QS', 'AS' -- this mirrors pandas behavior.

    Parameters
    __________
    group : CFTimeIndex or DataArrayResample
        Object which contains CFTimeIndex as a proxy representation for
        CFPeriodIndex
    freq : str
        String specifying the frequency/offset such as 'MS', '2D', or '3T'

    Returns
    _______
    start_time : cftime.datetime
        The start time of the period inferred from datetime and freq.

    Examples
    --------
    >>> index = xr.cftime_range(start='2000-01-01', periods=3,
                                freq='2QS', calendar='360_day')
    >>> time_bnds(index, '2Q')
    ((cftime.Datetime360Day(2000, 1, 1, 0, 0, 0, 0, 1, 1),
      cftime.Datetime360Day(2000, 3, 30, 23, 59, 59, 999999, 0, 91)),
     (cftime.Datetime360Day(2000, 7, 1, 0, 0, 0, 0, 6, 181),
      cftime.Datetime360Day(2000, 9, 30, 23, 59, 59, 999999, 5, 271)),
     (cftime.Datetime360Day(2001, 1, 1, 0, 0, 0, 0, 4, 1),
      cftime.Datetime360Day(2001, 3, 30, 23, 59, 59, 999999, 3, 91)))
    """
    if isinstance(group, CFTimeIndex):
        cfindex = group
    elif isinstance(group, DataArrayResample):
        if isinstance(group._full_index, CFTimeIndex):
            cfindex = group._full_index
        else:
            raise TypeError(
                "Index must be a CFTimeIndex, but got an instance of {}".format(
                    type(group).__name__
                )
            )
    else:
        raise TypeError(
            "Index must be a CFTimeIndex, but got an instance of {}".format(
                type(group).__name__
            )
        )

    return tuple(
        zip(cfindex_start_time(cfindex, freq), cfindex_end_time(cfindex, freq))
    )


def wrapped_partial(func: FunctionType, *args, **kwargs):
    from functools import partial, update_wrapper

    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def uas_vas_2_sfcwind(uas: xr.DataArray = None, vas: xr.DataArray = None):
    """Converts eastward and northward wind components to wind speed and direction.

    Parameters
    ----------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    Returns
    -------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Notes
    -----
    Northerly winds with a velocity less than 0.5 m/s are given a wind direction of 0°,
    while stronger winds are set to 360°.
    """
    # TODO: Add an attribute check to switch between sfcwind and wind

    # Converts the wind speed to m s-1
    uas = convert_units_to(uas, "m/s")
    vas = convert_units_to(vas, "m/s")

    # Wind speed is the hypotenuse of "uas" and "vas"
    wind = np.hypot(uas, vas)

    # Add attributes to wind. This is done by copying uas' attributes and overwriting a few of them
    wind.attrs = uas.attrs
    wind.name = "sfcWind"
    wind.attrs["standard_name"] = "wind_speed"
    wind.attrs["long_name"] = "Near-Surface Wind Speed"
    wind.attrs["units"] = "m s-1"

    # Calculate the angle
    # TODO: This creates decimal numbers such as 89.99992. Do we want to round?
    windfromdir_math = np.degrees(np.arctan2(vas, uas))

    # Convert the angle from the mathematical standard to the meteorological standard
    windfromdir = (270 - windfromdir_math) % 360.0

    # According to the meteorological standard, calm winds must have a direction of 0°
    # while northerly winds have a direction of 360°
    # On the Beaufort scale, calm winds are defined as < 0.5 m/s
    windfromdir = xr.where((windfromdir.round() == 0) & (wind >= 0.5), 360, windfromdir)
    windfromdir = xr.where(wind < 0.5, 0, windfromdir)

    # Add attributes to winddir. This is done by copying uas' attributes and overwriting a few of them
    windfromdir.attrs = uas.attrs
    windfromdir.name = "sfcWindfromdir"
    windfromdir.attrs["standard_name"] = "wind_from_direction"
    windfromdir.attrs["long_name"] = "Near-Surface Wind from Direction"
    windfromdir.attrs["units"] = "degree"

    return wind, windfromdir


def sfcwind_2_uas_vas(wind: xr.DataArray = None, windfromdir: xr.DataArray = None):
    """Converts wind speed and direction to eastward and northward wind components.

    Parameters
    ----------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Returns
    -------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    """
    # TODO: Add an attribute check to switch between sfcwind and wind

    # Converts the wind speed to m s-1
    wind = convert_units_to(wind, "m/s")

    # Converts the wind direction from the meteorological standard to the mathematical standard
    windfromdir_math = (-windfromdir + 270) % 360.0

    # TODO: This commented part should allow us to resample subdaily wind, but needs to be cleaned up and put elsewhere
    # if resample is not None:
    #     wind = wind.resample(time=resample).mean(dim='time', keep_attrs=True)
    #
    #     # nb_per_day is the number of values each day. This should be calculated
    #     windfromdir_math_per_day = windfromdir_math.reshape((len(wind.time), nb_per_day))
    #     # Averages the subdaily angles around a circle, i.e. mean([0, 360]) = 0, not 180
    #     windfromdir_math = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
    #                                       for angles in windfromdir_math_per_day])

    uas = wind * np.cos(np.radians(windfromdir_math))
    vas = wind * np.sin(np.radians(windfromdir_math))

    # Add attributes to uas and vas. This is done by copying wind' attributes and overwriting a few of them
    uas.attrs = wind.attrs
    uas.name = "uas"
    uas.attrs["standard_name"] = "eastward_wind"
    uas.attrs["long_name"] = "Near-Surface Eastward Wind"
    wind.attrs["units"] = "m s-1"

    vas.attrs = wind.attrs
    vas.name = "vas"
    vas.attrs["standard_name"] = "northward_wind"
    vas.attrs["long_name"] = "Near-Surface Northward Wind"
    wind.attrs["units"] = "m s-1"

    return uas, vas
