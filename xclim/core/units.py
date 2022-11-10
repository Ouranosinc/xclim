# noqa: D205,D400
"""
Units handling submodule
========================

`Pint` is used to define the `units` `UnitRegistry` and `xclim.units.core` defines
most unit handling methods.
"""
from __future__ import annotations

import functools
import re
import warnings
from inspect import signature
from typing import Any, Callable

import pint
import xarray as xr
from boltons.funcutils import wraps

from .calendar import date_range, get_calendar, parse_offset
from .options import datacheck
from .utils import ValidationError

__all__ = [
    "amount2rate",
    "check_units",
    "convert_units_to",
    "declare_units",
    "infer_sampling_units",
    "pint_multiply",
    "pint2cfunits",
    "rate2amount",
    "str2pint",
    "to_agg_units",
    "units",
    "units2pint",
]


# shamelessly adapted from `cf-xarray` (which adopted it from MetPy and xclim itself)
units = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        functools.partial(
            re.compile(
                r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])"
            ).sub,
            "**",
        ),
        lambda string: string.replace("%", "percent"),
    ],
)

units.define("percent = 0.01 = % = pct")

# In pint, the default symbol for year is "a" which is not CF-compliant (stands for "are")
units.define("year = 365.25 * day = yr")

# Define commonly encountered units not defined by pint
units.define("@alias degC = C = deg_C = Celsius")
units.define("@alias degK = deg_K")
units.define("@alias day = d")
units.define("@alias hour = h")  # Not the Planck constant...
units.define(
    "degrees_north = 1 * degree = degrees_north = degrees_N = degreesN = degree_north = degree_N = degreeN"
)
units.define(
    "degrees_east = 1 * degree = degrees_east = degrees_E = degreesE = degree_east = degree_E = degreeE"
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
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m**3),
)
hydro.add_transformation(
    "[mass] / [length]**2 / [time]",
    "[length] / [time]",
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m**3),
)
hydro.add_transformation(
    "[length] / [time]",
    "[mass] / [length]**2 / [time]",
    lambda ureg, x: x * (1000 * ureg.kg / ureg.m**3),
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

# Radiation units
units.define("[radiation] = [power] / [length]**2")


def units2pint(value: xr.DataArray | str | units.Quantity) -> pint.Unit:
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : xr.DataArray or str or pint.Quantity
        Input data array or string representing a unit (with no magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.
    """
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

    # Catch user errors undetected by Pint
    degree_ex = ["deg", "degree", "degrees"]
    unit_ex = [
        "C",
        "K",
        "F",
        "Celsius",
        "Kelvin",
        "Fahrenheit",
        "celsius",
        "kelvin",
        "fahrenheit",
    ]
    possibilities = [f"{d} {u}" for d in degree_ex for u in unit_ex]
    if unit.strip() in possibilities:
        raise ValidationError(
            "Remove white space from temperature units, e.g. use `degC`."
        )

    return units.parse_units(unit)


# Note: The pint library does not have a generic Unit or Quantity type at the moment. Using "Any" as a stand-in.
def pint2cfunits(value: pint.Unit) -> str:
    """Return a CF-compliant unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
        Input unit.

    Returns
    -------
    str
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

    out, _ = re.subn(pat, repl, s)

    # Remove multiplications
    out = out.replace(" * ", " ")
    # Delta degrees:
    out = out.replace("Δ°", "delta_deg")
    # Percents
    return out.replace("percent", "%").replace("pct", "%")


def ensure_cf_units(ustr: str) -> str:
    """Ensure the passed unit string is CF-compliant.

    The string will be parsed to pint then recast to a string by xclim's `pint2cfunits`.
    """
    return pint2cfunits(units2pint(ustr))


def pint_multiply(da: xr.DataArray, q: Any, out_units: str | None = None):
    """Multiply xarray.DataArray by pint.Quantity.

    Parameters
    ----------
    da : xr.DataArray
        Input array.
    q : pint.Quantity
        Multiplicative factor.
    out_units : str, optional
        Units the output array should be converted into.
    """
    a = 1 * units2pint(da)  # noqa
    f = a * q.to_base_units()
    if out_units:
        f = f.to(out_units)
    else:
        f = f.to_reduced_units()
    out = da * f.magnitude
    out.attrs["units"] = pint2cfunits(f.units)
    return out


def str2pint(val: str) -> pint.Quantity:
    """Convert a string to a pint.Quantity, splitting the magnitude and the units.

    Parameters
    ----------
    val : str
        A quantity in the form "[{magnitude} ]{units}", where magnitude can be cast to a float and
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
    source: str | xr.DataArray | Any,
    target: str | xr.DataArray | Any,
    context: str | None = None,
) -> xr.DataArray | float | int | str | Any:
    """Convert a mathematical expression into a value with the same units as a DataArray.

    Parameters
    ----------
    source : str or xr.DataArray or Any
        The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str or xr.DataArray or Any
        Target array of values to which units must conform.
    context : str, optional
        The unit definition context. Default: None.

    Returns
    -------
    xr.DataArray or float or int or str or Any
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
                data=units.convert(source.data, fu, tu),  # noqa
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
}
"""
Resampling frequency units for :py:func:`infer_sampling_units`.

Mapping from offset base to CF-compliant unit. Only constant-length frequencies are included.
"""


def infer_sampling_units(
    da: xr.DataArray,
    deffreq: str | None = "D",
    dim: str = "time",
) -> tuple[int, str]:
    """Infer a multiplier and the units corresponding to one sampling period.

    Parameters
    ----------
    da : xr.DataArray
        A DataArray from which to take coordinate `dim`.
    deffreq : str, optional
        If no frequency is inferred from `da[dim]`, take this one.
    dim : str
        Dimension from which to infer the frequency.

    Raises
    ------
    ValueError
        If the frequency has no exact corresponding units.

    Returns
    -------
    int
        The magnitude (number of base periods per period)
    str
        Units as a string, understandable by pint.
    """
    dimmed = getattr(da, dim)
    freq = xr.infer_freq(dimmed)
    if freq is None:
        freq = deffreq

    multi, base, _, _ = parse_offset(freq)
    try:
        out = multi, FREQ_UNITS[base]
    except KeyError as err:
        raise ValueError(
            f"Sampling frequency {freq} has no corresponding units."
        ) from err
    if out == (7, "d"):
        # Special case for weekly frequency. xarray's CFTimeOffsets do not have "W".
        return 1, "week"
    return out


def to_agg_units(
    out: xr.DataArray, orig: xr.DataArray, op: str, dim: str = "time"
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

    Returns
    -------
    xr.DataArray

    Examples
    --------
    Take a daily array of temperature and count number of days above a threshold.
    `to_agg_units` will infer the units from the sampling rate along "time", so
    we ensure the final units are correct:

    >>> time = xr.cftime_range("2001-01-01", freq="D", periods=365)
    >>> tas = xr.DataArray(
    ...     np.arange(365),
    ...     dims=("time",),
    ...     coords={"time": time},
    ...     attrs={"units": "degC"},
    ... )
    >>> cond = tas > 100  # Which days are boiling
    >>> Ndays = cond.sum("time")  # Number of boiling days
    >>> Ndays.attrs.get("units")
    None
    >>> Ndays = to_agg_units(Ndays, tas, op="count")
    >>> Ndays.units
    'd'

    Similarly, here we compute the total heating degree-days, but we have weekly data:

    >>> time = xr.cftime_range("2001-01-01", freq="7D", periods=52)
    >>> tas = xr.DataArray(
    ...     np.arange(52) + 10,
    ...     dims=("time",),
    ...     coords={"time": time},
    ...     attrs={"units": "degC"},
    ... )
    >>> degdays = (
    ...     (tas - 16).clip(0).sum("time")
    ... )  # Integral of  temperature above a threshold
    >>> degdays = to_agg_units(degdays, tas, op="delta_prod")
    >>> degdays.units
    'week delta_degC'

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


def _rate_and_amount_converter(
    da: xr.DataArray, dim: str = "time", to: str = "amount", out_units: str = None
) -> xr.DataArray:
    """Private function performing the actual conversion for :py:func:`rate2amount` and :py:func:`amount2rate`."""
    m = 1
    u = None  # Default to assume a non-uniform axis
    label = "lower"
    time = da[dim]

    try:
        freq = xr.infer_freq(da[dim])
    except ValueError:
        freq = None
    if freq is not None:
        multi, base, start_anchor, _ = parse_offset(freq)
        if base in ["M", "Q", "A"]:
            start = time.indexes[dim][0]
            if not start_anchor:
                # Anchor is on the end of the period, substract 1 period.
                start = start - xr.coding.cftime_offsets.to_offset(freq)
                # In the diff below, assign to upper label!
                label = "upper"
            # We generate "time" with an extra element, so we do not need to repeat the last element below.
            time = xr.DataArray(
                date_range(
                    start, periods=len(time) + 1, freq=freq, calendar=get_calendar(time)
                ),
                dims=(dim,),
                name=dim,
                attrs=da[dim].attrs,
            )
        else:
            m, u = multi, FREQ_UNITS[base]

    # Freq is month, season or year, which are not constant units, or simply freq is not inferrable.
    if u is None:
        # Get sampling period lengths in nanoseconds
        # In the case with no freq, last period as the same length as the one before.
        # In the case with freq in M, Q, A, this has been dealt with above in `time`
        # and `label` has been updated accordingly.
        dt = (
            time.diff(dim, label=label)
            .reindex({dim: da[dim]}, method="ffill")
            .astype(float)
        )
        dt = dt / 1e9  # Convert to seconds

        if to == "amount":
            tu = (str2pint(da.units) * str2pint("s")).to_reduced_units()
            out = da * dt * tu.m
        elif to == "rate":
            tu = (str2pint(da.units) / str2pint("s")).to_reduced_units()
            out = (da / dt) * tu.m
        else:
            raise ValueError("Argument `to` must be one of 'amout' or 'rate'.")

        out.attrs["units"] = pint2cfunits(tu)

    else:
        q = units.Quantity(m, u)
        if to == "amount":
            out = pint_multiply(da, q)
        elif to == "rate":
            out = pint_multiply(da, 1 / q)
        else:
            raise ValueError("Argument `to` must be one of 'amout' or 'rate'.")

    if out_units:
        out = convert_units_to(out, out_units)

    return out


def rate2amount(
    rate: xr.DataArray, dim: str = "time", out_units: str = None
) -> xr.DataArray:
    """Convert a rate variable to an amount by multiplying by the sampling period length.

    If the sampling period length cannot be inferred, the rate values
    are multiplied by the duration between their time coordinate and the next one. The last period
    is estimated with the duration of the one just before.

    This is the inverse operation of :py:func:`amount2rate`.

    Parameters
    ----------
    rate : xr.DataArray
        "Rate" variable, with units of "amount" per time. Ex: Precipitation in "mm / d".
    dim : str
        The time dimension.
    out_units : str, optional
        Output units to convert to.

    Returns
    -------
    xr.DataArray

    Examples
    --------
    The following converts a daily array of precipitation in mm/h to the daily amounts in mm:

    >>> time = xr.cftime_range("2001-01-01", freq="D", periods=365)
    >>> pr = xr.DataArray(
    ...     [1] * 365, dims=("time",), coords={"time": time}, attrs={"units": "mm/h"}
    ... )
    >>> pram = rate2amount(pr)
    >>> pram.units
    'mm'
    >>> float(pram[0])
    24.0

    Also works if the time axis is irregular : the rates are assumed constant for the whole period
    starting on the values timestamp to the next timestamp:

    >>> time = time[[0, 9, 30]]  # The time axis is Jan 1st, Jan 10th, Jan 31st
    >>> pr = xr.DataArray(
    ...     [1] * 3, dims=("time",), coords={"time": time}, attrs={"units": "mm/h"}
    ... )
    >>> pram = rate2amount(pr)
    >>> pram.values
    array([216., 504., 504.])

    Finally, we can force output units:

    >>> pram = rate2amount(pr, out_units="pc")  # Get rain amount in parsecs. Why not.
    >>> pram.values
    array([7.00008327e-18, 1.63335276e-17, 1.63335276e-17])
    """
    return _rate_and_amount_converter(rate, dim=dim, to="amount", out_units=out_units)


def amount2rate(
    amount: xr.DataArray, dim: str = "time", out_units: str = None
) -> xr.DataArray:
    """Convert an amount variable to a rate by dividing by the sampling period length.

    If the sampling period length cannot be inferred, the amount values
    are divided by the duration between their time coordinate and the next one. The last period
    is estimated with the duration of the one just before.

    This is the inverse operation of :py:func:`rate2amount`.

    Parameters
    ----------
    amount : xr.DataArray
        "amount" variable. Ex: Precipitation amount in "mm".
    dim : str
        The time dimension.
    out_units : str, optional
        Output units to convert to.

    Returns
    -------
    xr.DataArray
    """
    return _rate_and_amount_converter(amount, dim=dim, out_units=out_units, to="rate")


@datacheck
def check_units(val: str | int | float | None, dim: str | None) -> None:
    """Check units for appropriate convention compliance."""
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

    # This is needed if dim is units in the CF-syntax,
    try:
        dim = str2pint(dim)
        expected = dim.dimensionality
    except pint.UndefinedUnitError:
        # Raised when it is not understood, we assume it was a dimensionality
        expected = units.get_dimensionality(dim.replace("dimensionless", ""))

    if isinstance(val, str):
        val_units = str2pint(val)
    else:  # a DataArray
        val_units = units2pint(val)
    val_dim = val_units.dimensionality

    if val_dim == expected:
        return

    # Check if there is a transformation available
    start = pint.util.to_units_container(val_dim)
    end = pint.util.to_units_container(expected)
    graph = units._active_ctx.graph  # noqa
    if pint.util.find_shortest_path(graph, start, end):
        return

    raise ValidationError(
        f"Data units {val_units} are not compatible with requested {dim}."
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

    Returns
    -------
    Callable

    Examples
    --------
    In the following function definition:

    .. code-block:: python

        @declare_units(tas=["temperature"])
        def func(tas):
            ...

    The decorator will check that `tas` has units of temperature (C, K, F).
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
                    if "units" not in outd.attrs:
                        raise ValueError(
                            "No units were assigned in one of the indice's outputs."
                        )
                    outd.attrs["units"] = ensure_cf_units(outd.attrs["units"])
            else:
                if "units" not in out.attrs:
                    raise ValueError("No units were assigned to the indice's output.")
                out.attrs["units"] = ensure_cf_units(out.attrs["units"])

            return out

        setattr(wrapper, "in_units", units_by_name)
        return wrapper

    return dec


def ensure_delta(unit: str = None):
    """Return delta units for temperature.

    For dimensions where delta exist in pint (Temperature), it replaces the temperature unit by delta_degC or
    delta_degF based on the input unit.
    For other dimensionality, it just gives back the input units.

    Parameters
    ----------
    unit : str
        unit to transform in delta (or not)
    """
    u = units2pint(unit)
    d = 1 * u
    #
    delta_unit = pint2cfunits(d - d)
    # replace kelvin/rankine by delta_degC/F
    if "kelvin" in u._units:
        delta_unit = pint2cfunits(u / units2pint("K") * units2pint("delta_degC"))
    if "degree_Rankine" in u._units:
        delta_unit = pint2cfunits(u / units2pint("°R") * units2pint("delta_degF"))
    return delta_unit
