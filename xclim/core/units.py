# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Units handling submodule
========================

`Pint` is used to define the `units` `UnitRegistry` and `xclim.units.core` defines
most unit handling methods.
"""
import re
import warnings
from inspect import signature
from typing import Any, Callable, Optional, Tuple, Union

import pint.converters
import pint.unit
import xarray as xr
from boltons.funcutils import wraps
from packaging import version

from .calendar import parse_offset
from .options import datacheck
from .utils import ValidationError

__all__ = [
    "convert_units_to",
    "declare_units",
    "pint_multiply",
    "pint2cfunits",
    "str2pint",
    "units",
    "units2pint",
]


units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
units.define(
    pint.unit.UnitDefinition(
        "percent", "%", ("pct",), pint.converters.ScaleConverter(0.01)
    )
)
# In pint, the default symbol for year is "a" which is not CF-compliant (stands for "are")
units.define("year = 365.25 * day = yr")

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
units.define("mmday = 1000 kg / meter ** 2 / day")

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


def units2pint(value: Union[xr.DataArray, str, units.Quantity]) -> units.Unit:
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : Union[xr.DataArray, str. pint.Quantity]
      Input data array or string representing a unit (with no magnitude).

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
    if unit == "1":
        unit = ""

    try:  # Pint compatible
        return units.parse_units(unit)
    except (
        pint.UndefinedUnitError,
        pint.DimensionalityError,
        AttributeError,
        TypeError,
    ):  # Convert from CF-units to pint-compatible
        return units.parse_units(_transform(unit))


# Note: The pint library does not have a generic Unit or Quantity type at the moment. Using "Any" as a stand-in.
def pint2cfunits(value: Union[units.Unit, units.Quantity]) -> str:
    """Return a CF-compliant unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
      Input unit.

    Returns
    -------
    out : str
      Units following CF-Convention, using symbols.
    """
    if isinstance(value, pint.Quantity):
        value = value.units

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

    # Remove multiplications
    out = out.replace(" * ", " ")
    # Delta degrees:
    out = out.replace("Δ°", "delta_deg")
    return out.replace("percent", "%")


def ensure_cf_units(ustr: str) -> str:
    """Ensure the passed unit string is CF-compliant.

    The string will be parsed to pint then recasted to a string by xclim's `pint2cfunits`.
    """
    return pint2cfunits(units2pint(ustr))


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
    else:
        f = f.to_reduced_units()
    out = da * f.magnitude
    out.attrs["units"] = pint2cfunits(f.units)
    return out


def str2pint(val: str):
    """Convert a string to a pint.Quantity, splitting the magnitude and the units.

    Parameters
    ----------
    val : str
      A quantity in the form "[{magnitude} ]{units}", where magnitude is castable to a float and
      units is understood by `units2pint`.

    Returns
    -------
    pint.Quantity
      Magnitude is 1 if no magnitude was present in the string.
    """
    mstr, *ustr = val.split(" ", maxsplit=1)
    try:
        if ustr:
            return units.Quantity(float(mstr), units=units2pint(ustr[0]))
        return units.Quantity(float(mstr))
    except ValueError:
        return units.Quantity(1, units2pint(val))


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
        q = str2pint(source)
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
                data=units.convert(source.data, fu, tu),
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
        return units.Quantity(source, units=fu).to(tu).m

    raise NotImplementedError(f"Source of type `{type(source)}` is not supported.")


FREQ_UNITS = {
    "N": "ns",
    "L": "ms",
    "S": "s",
    "T": "min",
    "H": "h",
    "D": "d",
    "W": "week",
    "M": "month",
    "A": "yr",
}


def infer_sampling_units(da: xr.DataArray, deffreq: str = "D") -> Tuple[int, str]:
    """Infer a multiplicator and the units corresponding to one sampling period.

    If `xr.infer_freq` fails, returns `deffreq`.
    """
    freq = xr.infer_freq(da.time)
    if freq is None:
        freq = deffreq

    multi, base, _ = parse_offset(freq)
    try:
        return int(multi or "1"), FREQ_UNITS[base]
    except KeyError:
        raise ValueError(f"Sampling frequency {freq} has no corresponding units.")


def to_agg_units(
    out: xr.DataArray, orig: xr.DataArray, op: str, dim="time"
) -> xr.DataArray:
    """Set and convert units of an array after an aggregation operation along the sampling dimension (time).

    Parameters
    ----------
    out : xr.DataArray
      The output array of the aggregation operation, no units operation done yet.
    orig : xr.DataArray
      The original array before the aggregation operation,
      used to infer the sampling units and get the variable units.
    op : {'count', 'prod', 'delta_prod'}
      The type of aggregation operation performed. The special "delta_*" ops are used
      with temperature units needing conversion to their "delta" counterparts (e.g. degree days)
    dim : str
      The time dimension along which the aggregation was performed.

    Examples
    --------
    Take a daily array of temperature and count number of days above a threshold.
    `to_agg_units` will infer the units from the sampling rate along "time", so
    we ensure the final units are correct.

    >>> time = xr.cftime_range('2001-01-01', freq='D', periods=365)
    >>> tas = xr.DataArray(np.arange(365), dims=('time',), coords={'time': time}, attrs={'units': 'degC'})
    >>> cond = tas > 100  # Which days are boiling
    >>> Ndays = cond.sum('time')  # Number of boiling days
    >>> Ndays.attrs.get('units')
    None
    >>> Ndays = to_agg_units(Ndays, tas, op='count')
    >>> Ndays.units
    'd'

    Similarly, here we compute the total heating degree-days but we have weekly data:
    >>> time = xr.cftime_range('2001-01-01', freq='MS', periods=12)
    >>> tas = xr.DataArray(np.arange(12) + 10, dims=('time',), coords={'time': time}, attrs={'units': 'degC'})
    >>> degdays = (tas - 16).clip(0).sum('time')   # Integral of  temperature above a threshold
    >>> degdays = to_agg_units(degdays, tas, op='delta_prod')
    >>> degdays.units
    'month delta_degC'

    Which we can always convert to the more common "K days":

    >>> degdays = convert_units_to(degdays, "K days")
    >>> degdays.units
    'K d'
    """
    m, freq_u_raw = infer_sampling_units(orig[dim])
    freq_u = str2pint(freq_u_raw)
    orig_u = str2pint(orig.units)

    out = out * m
    if op == "count":
        out.attrs["units"] = freq_u_raw
    elif op == "prod":
        out.attrs["units"] = pint2cfunits(orig_u * freq_u)
    elif op == "delta_prod":
        out.attrs["units"] = pint2cfunits((orig_u - orig_u) * freq_u)
    else:
        raise ValueError(f"Aggregation op {op} not in [count, prod, delta_prod].")
    return out


def rate2amount(
    rate: xr.DataArray, dim: str = "time", out_units: str = None
) -> xr.DataArray:
    """Convert a rate variable to an amount by multiplying by the sampling period length.

    If the sampling period length cannot be inferred, the rate values
    are multiplied by the duration between their time coordinate and the next one. The last period
    is estimated with the duration of the one just before.

    Parameters
    ----------
    rate : xr.DataArray
      "Rate" variable, with units of "amount" per time. Ex: Precipitation in "mm / d".
    dim : str
      The time dimension.
    out_units : str, optional
      Optional output units to convert to.

    Examples
    --------
    The following converts a daily array of precipitation in mm/h to the daily amounts in mm.

    >>> time = xr.cftime_range('2001-01-01', freq='D', periods=365)
    >>> pr = xr.DataArray([1] * 365, dims=('time',), coords={'time': time}, attrs={'units': 'mm/h'})
    >>> pram = rate2amount(pr)
    >>> pram.units
    'mm'
    >>> float(pram[0])
    24.0

    Also works if the time axis is irregular : the rates are assumed constant for the whole period
    starting on the values timestamp to the next timestamp.

    >>> time = time[[0, 9, 30]]  # The time axis is Jan 1st, Jan 10th, Jan 31st
    >>> pr = xr.DataArray([1] * 3, dims=('time',), coords={'time': time}, attrs={'units': 'mm/h'})
    >>> pram = rate2amount(pr)
    >>> pram.values
    array([216., 504., 504.])

    Finally, we can force output units:

    >>> pram = rate2amount(pr, out_units='pc')  # Get rain amount in parsecs. Why not.
    >>> pram.values
    array([7.00008327e-18, 1.63335276e-17, 1.63335276e-17])
    """
    try:
        m, u = infer_sampling_units(rate.time, deffreq=None)
    except AttributeError:
        # In coherent time axis : xr.infer_freq returned None
        # Get sampling period lengths in nanoseconds. Last period as the same length as the one before.
        dt = (
            rate.time.diff(dim, label="lower")
            .reindex({dim: rate[dim]}, method="ffill")
            .astype(float)
        )
        dt = dt / 1e9  # Convert to seconds
        tu = (str2pint(rate.units) * str2pint("s")).to_reduced_units()
        amount = rate * dt * tu.m
        amount.attrs["units"] = pint2cfunits(tu)
        if out_units:
            amount = convert_units_to(amount, out_units)
    else:
        q = units.Quantity(m, u)
        amount = pint_multiply(rate, q, out_units=out_units)

    return amount


@datacheck
def check_units(val: Optional[Union[str, int, float]], dim: Optional[str]) -> None:
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
    if isinstance(val, str):
        val_dim = str2pint(val).dimensionality
    else:  # a DataArray
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
    elif dim == "[speed]":
        tu = "m s-1"
    else:
        raise NotImplementedError(f"Dimension `{dim}` is not supported.")

    try:
        units.Quantity(1, units=units2pint(val)).to(tu, "hydro")
    except (pint.UndefinedUnitError, pint.DimensionalityError):
        raise ValidationError(
            f"Value's dimension `{val_dim}` does not match expected units `{expected}`."
        )


def declare_units(
    **units_by_name,
) -> Callable:
    """Create a decorator to check units of function arguments.

    The decorator checks that input and output values have units that are compatible with expected dimensions.
    It also stores the input units as a 'in_units' attribute.

    Parameters
    ----------
    units_by_name : Mapping[str, str]
      Mapping from the input parameter names to their units or dimensionality ("[...]").

    Examples
    --------
    In the following function definition:

    .. code::

       @declare_units(tas=["temperature"])
       def func(tas):
          ...

    the decorator will check that `tas` has units of temperature (C, K, F).
    """

    def dec(func):
        # Match the signature of the function to the arguments given to the decorator
        sig = signature(func)
        bound_units = sig.bind_partial(**units_by_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments so we can check units
            bound_args = sig.bind(*args, **kwargs)
            for name, val in bound_args.arguments.items():
                check_units(val, bound_units.arguments.get(name, None))

            out = func(*args, **kwargs)

            # Perform very basic sanity check on the output.
            # Indice are responsible for unit management.
            # If this fails, it's a developer's error.
            if isinstance(out, tuple):
                for outd in out:
                    assert (
                        "units" in outd.attrs
                    ), "No units were assigned in one of the indice's outputs."
                    outd.attrs["units"] = ensure_cf_units(outd.attrs["units"])
            else:
                assert (
                    "units" in out.attrs
                ), "No units were assigned to the indice's output."
                out.attrs["units"] = ensure_cf_units(out.attrs["units"])

            return out

        setattr(wrapper, "in_units", units_by_name)
        return wrapper

    return dec
