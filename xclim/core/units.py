"""
Units Handling Submodule
========================

`Pint` is used to define the :py:data:`xclim.core.units.units` `UnitRegistry`.
This module defines most unit handling methods.
"""
from __future__ import annotations

import functools
import logging
import re
import warnings
from importlib.resources import open_text
from inspect import _empty, signature  # noqa
from typing import Any, Callable

import pint
import xarray as xr
from boltons.funcutils import wraps
from yaml import safe_load

from .calendar import date_range, get_calendar, parse_offset
from .options import datacheck
from .utils import Quantified, ValidationError

logging.getLogger("pint").setLevel(logging.ERROR)

__all__ = [
    "amount2lwethickness",
    "amount2rate",
    "check_units",
    "convert_units_to",
    "declare_units",
    "flux2rate",
    "infer_context",
    "infer_sampling_units",
    "lwethickness2amount",
    "pint2cfunits",
    "pint_multiply",
    "rate2amount",
    "rate2flux",
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
units.define("[radiation] = [power] / [area]")

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


CF_CONVERSIONS = safe_load(open_text("xclim.data", "variables.yml"))["conversions"]
_CONVERSIONS = {}


def _register_conversion(conversion, direction):
    """Register a conversion function to be automatically picked up in `convert_units_to`.

    The function must correspond to a name in `CF_CONVERSIONS`, so to a section in
    `xclim/data/variables.yml::conversions`.
    """
    if conversion not in CF_CONVERSIONS:
        raise NotImplementedError(
            "Automatic conversion functions must have a corresponding section in xclim/data/variables.yml"
        )

    def _func_register(func):
        _CONVERSIONS[(conversion, direction)] = func
        return func

    return _func_register


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


def pint2cfunits(value: units.Quantity | units.Unit) -> str:
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
    if isinstance(value, (pint.Quantity, units.Quantity)):
        value = value.units  # noqa reason: units.Quantity really have .units property

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


def pint_multiply(
    da: xr.DataArray, q: Any, out_units: str | None = None
) -> xr.DataArray:
    """Multiply xarray.DataArray by pint.Quantity.

    Parameters
    ----------
    da : xr.DataArray
        Input array.
    q : pint.Quantity
        Multiplicative factor.
    out_units : str, optional
        Units the output array should be converted into.

    Returns
    -------
    xr.DataArray
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
    source: Quantified,
    target: Quantified | units.Unit,
    context: str | None = None,
) -> Quantified:
    """Convert a mathematical expression into a value with the same units as a DataArray.

    If the dimensionalities of source and target units differ, automatic CF conversions
    will be applied when possible. See :py:func:`xclim.core.units.cf_conversion`.

    Parameters
    ----------
    source : str or xr.DataArray or units.Quantity
        The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str or xr.DataArray or units.Quantity or units.Unit
        Target array of values to which units must conform.
    context : str, optional
        The unit definition context. Default: None.
        If "infer", it will be inferred with :py:func:`xclim.core.units.infer_context` using
        the standard name from the `source` or, if none is found, from the `target`.
        This means that the 'hydro' context could be activated if any one of the standard names allows it.

    Returns
    -------
    str or xr.DataArray or units.Quantity
        The source value converted to target's units.
        The outputted type is always similar to `source` initial type.
        Attributes are preserved unless an automatic CF conversion is performed,
        in which case only the new `standard_name` appears in the result.

    See Also
    --------
    cf_conversion
    amount2rate
    rate2amount
    amount2lwethickness
    lwethickness2amount
    """
    context = context or "none"

    # Target units
    if isinstance(target, units.Unit):
        target_unit = target
    elif isinstance(target, (str, xr.DataArray)):
        target_unit = units2pint(target)
    else:
        raise NotImplementedError(
            "target must be either a pint Unit or a xarray DataArray."
        )

    if context == "infer":
        ctxs = []
        if isinstance(source, xr.DataArray):
            ctxs.append(infer_context(source.attrs.get("standard_name")))
        if isinstance(target, xr.DataArray):
            ctxs.append(infer_context(target.attrs.get("standard_name")))
        # If any one of the target or source is compatible with the "hydro" context, use it.
        if "hydro" in ctxs:
            context = "hydro"
        else:
            context = "none"

    if isinstance(source, str):
        q = str2pint(source)
        # Return magnitude of converted quantity. This is going to fail if units are not compatible.
        return q.to(target_unit, context).m

    if isinstance(source, units.Quantity):
        return source.to(target_unit, context).m

    if isinstance(source, xr.DataArray):
        source_unit = units2pint(source)
        target_cf_unit = pint2cfunits(target_unit)

        # Automatic pre-conversions based on the dimensionalities and CF standard names
        standard_name = source.attrs.get("standard_name")
        if (
            standard_name is not None
            and source_unit.dimensionality != target_unit.dimensionality
        ):
            dim_order_diff = source_unit.dimensionality / target_unit.dimensionality
            for convname, convconf in CF_CONVERSIONS.items():
                for direction, sign in [("to", 1), ("from", -1)]:
                    # If the dimensionality diff is compatible with this conversion
                    compatible = all(
                        [
                            dimdiff == (sign * dim_order_diff.get(f"[{dim}]"))
                            for dim, dimdiff in convconf["dimensionality"].items()
                        ]
                    )
                    # Does the input cf standard name have an equivalent after conversion
                    valid = cf_conversion(standard_name, convname, direction)
                    if compatible and valid:
                        # The new cf standard name is inserted by the converter
                        try:
                            source = _CONVERSIONS[(convname, direction)](source)
                        except Exception:
                            # FIXME: This is a broad exception. Bad practice.
                            # Failing automatic conversion
                            # It will anyway fail further down with a correct error message.
                            pass
                        else:
                            source_unit = units2pint(source)

        if source_unit == target_unit:
            # The units are the same, but the symbol may not be.
            source.attrs["units"] = target_cf_unit
            return source

        with units.context(context or "none"):
            out = source.copy(data=units.convert(source.data, source_unit, target_unit))
            out.attrs["units"] = target_cf_unit
            return out

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(source, (float, int)):
        if context == "hydro":
            source_unit = units.mm / units.day
        else:
            source_unit = units.degC
        warnings.warn(
            "Future versions of xclim will require explicit unit specifications.",
            FutureWarning,
            stacklevel=3,
        )
        return units.Quantity(source, units=source_unit).to(target_unit).m

    raise NotImplementedError(f"Source of type `{type(source)}` is not supported.")


def cf_conversion(standard_name: str, conversion: str, direction: str) -> str | None:
    """Get the standard name of the specific conversion for the given standard name.

    Parameters
    ----------
    standard_name : str
        Standard name of the input.
    conversion : {'amount2rate', 'amount2lwethickness'}
        Type of conversion. Available conversions are the keys of the `conversions` entry in `xclim/data/variables.yml`.
        See :py:data:`xclim.core.units.CF_CONVERSIONS`. They also correspond to functions in this module.
    direction : {'to', 'from'}
        The direction of the requested conversion. "to" means the conversion as given by the `conversion` name,
        while "from" means the reverse operation. For example `conversion="amount2rate"` and `direction="from"`
        will search for a conversion from a rate or flux to an amount or thickness for the given standard name.

    Returns
    -------
    str or None
        If a string, this means the conversion is possible and the result should have this standard name.
        If None, the conversion is not possible within the CF standards.
    """
    i = ["to", "from"].index(direction)
    for names in CF_CONVERSIONS[conversion]["valid_names"]:
        if names[i] == standard_name:
            return names[int(not i)]
    return None


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
Resampling frequency units for :py:func:`xclim.core.units.infer_sampling_units`.

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
    da: xr.DataArray,
    dim: str = "time",
    to: str = "amount",
    sampling_rate_from_coord: bool = False,
    out_units: str = None,
) -> xr.DataArray:
    """Internal converter for :py:func:`xclim.core.units.rate2amount` and :py:func:`xclim.core.units.amount2rate`."""
    m = 1
    u = None  # Default to assume a non-uniform axis
    label = "lower"
    time = da[dim]

    try:
        freq = xr.infer_freq(da[dim])
    except ValueError as err:
        if sampling_rate_from_coord:
            freq = None
        else:
            raise ValueError(
                "The variables' sampling frequency could not be inferred, "
                "which is needed for conversions between rates and amounts. "
                f"If the derivative of the variables' {dim} coordinate "
                "can be used as the sampling rate, pass `sampling_rate_from_coord=True`."
            ) from err
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
            raise ValueError("Argument `to` must be one of 'amount' or 'rate'.")

        out.attrs["units"] = pint2cfunits(tu)
    else:
        q = units.Quantity(m, u)
        if to == "amount":
            out = pint_multiply(da, q)
        elif to == "rate":
            out = pint_multiply(da, 1 / q)
        else:
            raise ValueError("Argument `to` must be one of 'amount' or 'rate'.")

    old_name = da.attrs.get("standard_name")
    if old_name and (
        new_name := cf_conversion(
            old_name, "amount2rate", "to" if to == "rate" else "from"
        )
    ):
        out.attrs["standard_name"] = new_name

    if out_units:
        out = convert_units_to(out, out_units)

    return out


@_register_conversion("amount2rate", "from")
def rate2amount(
    rate: xr.DataArray,
    dim: str = "time",
    sampling_rate_from_coord: bool = False,
    out_units: str = None,
) -> xr.DataArray:
    """Convert a rate variable to an amount by multiplying by the sampling period length.

    If the sampling period length cannot be inferred, the rate values
    are multiplied by the duration between their time coordinate and the next one. The last period
    is estimated with the duration of the one just before.

    This is the inverse operation of :py:func:`xclim.core.units.amount2rate`.

    Parameters
    ----------
    rate : xr.DataArray
        "Rate" variable, with units of "amount" per time. Ex: Precipitation in "mm / d".
    dim : str
        The time dimension.
    sampling_rate_from_coord : boolean
        For data with irregular time coordinates. If True, the diff of the time coordinate will be used as the sampling rate,
        meaning each data point will be assumed to apply for the interval ending at the next point. See notes.
        Defaults to False, which raises an error if the time coordinate is irregular.
    out_units : str, optional
        Output units to convert to.

    Raises
    ------
    ValueError
        If the time coordinate is irregular and `sampling_rate_from_coord` is False (default).

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
    starting on the values timestamp to the next timestamp. This option is activated with `sampling_rate_from_coord=True`.

    >>> time = time[[0, 9, 30]]  # The time axis is Jan 1st, Jan 10th, Jan 31st
    >>> pr = xr.DataArray(
    ...     [1] * 3, dims=("time",), coords={"time": time}, attrs={"units": "mm/h"}
    ... )
    >>> pram = rate2amount(pr, sampling_rate_from_coord=True)
    >>> pram.values
    array([216., 504., 504.])

    Finally, we can force output units:

    >>> pram = rate2amount(pr, out_units="pc")  # Get rain amount in parsecs. Why not.
    >>> pram.values
    array([7.00008327e-18, 1.63335276e-17, 1.63335276e-17])

    See Also
    --------
    amount2rate
    """
    return _rate_and_amount_converter(
        rate,
        dim=dim,
        to="amount",
        sampling_rate_from_coord=sampling_rate_from_coord,
        out_units=out_units,
    )


@_register_conversion("amount2rate", "to")
def amount2rate(
    amount: xr.DataArray,
    dim: str = "time",
    sampling_rate_from_coord: bool = False,
    out_units: str = None,
) -> xr.DataArray:
    """Convert an amount variable to a rate by dividing by the sampling period length.

    If the sampling period length cannot be inferred, the amount values
    are divided by the duration between their time coordinate and the next one. The last period
    is estimated with the duration of the one just before.

    This is the inverse operation of :py:func:`xclim.core.units.rate2amount`.

    Parameters
    ----------
    amount : xr.DataArray
        "amount" variable. Ex: Precipitation amount in "mm".
    dim : str
        The time dimension.
    sampling_rate_from_coord : boolean
        For data with irregular time coordinates.
        If True, the diff of the time coordinate will be used as the sampling rate,
        meaning each data point will be assumed to span the interval ending at the next point.
        See notes of :py:func:`xclim.core.units.rate2amount`.
        Defaults to False, which raises an error if the time coordinate is irregular.
    out_units : str, optional
        Output units to convert to.

    Raises
    ------
    ValueError
        If the time coordinate is irregular and `sampling_rate_from_coord` is False (default).

    Returns
    -------
    xr.DataArray

    See Also
    --------
    rate2amount
    """
    return _rate_and_amount_converter(
        amount,
        dim=dim,
        to="rate",
        sampling_rate_from_coord=sampling_rate_from_coord,
        out_units=out_units,
    )


@_register_conversion("amount2lwethickness", "to")
def amount2lwethickness(
    amount: xr.DataArray, out_units: str = None
) -> xr.DataArray | Quantified:
    """Convert a liquid water amount (mass over area) to its equivalent area-averaged thickness (length).

    This will simply divide the amount by the density of liquid water, 1000 kg/m³.
    This is equivalent to using the "hydro" context of :py:data:`xclim.core.units.units`.

    Parameters
    ----------
    amount : xr.DataArray
        A DataArray storing a liquid water amount quantity.
    out_units : str
        Specific output units if needed.

    Returns
    -------
    xr.DataArray or Quantified
        The standard_name of `amount` is modified if a conversion is found
        (see :py:func:`xclim.core.units.cf_conversion`), it is removed otherwise.
        Other attributes are left untouched.

    See Also
    --------
    lwethickness2amount
    """
    water_density = str2pint("1000 kg m-3")
    out = pint_multiply(amount, 1 / water_density)
    old_name = amount.attrs.get("standard_name", None)
    if old_name and (new_name := cf_conversion(old_name, "amount2lwethickness", "to")):
        out.attrs["standard_name"] = new_name
    if out_units:
        out = convert_units_to(out, out_units)
    return out


@_register_conversion("amount2lwethickness", "from")
def lwethickness2amount(
    thickness: xr.DataArray, out_units: str = None
) -> xr.DataArray | Quantified:
    """Convert a liquid water thickness (length) to its equivalent amount (mass over area).

    This will simply multiply the thickness by the density of liquid water, 1000 kg/m³.
    This is equivalent to using the "hydro" context of :py:data:`xclim.core.units.units`.

    Parameters
    ----------
    thickness : xr.DataArray
        A DataArray storing a liquid water thickness quantity.
    out_units : str
        Specific output units if needed.

    Returns
    -------
    xr.DataArray or Quantified
        The standard_name of `amount` is modified if a conversion is found (see :py:func:`xclim.core.units.cf_conversion`),
        it is removed otherwise. Other attributes are left untouched.

    See Also
    --------
    amount2lwethickness
    """
    water_density = str2pint("1000 kg m-3")
    out = pint_multiply(thickness, water_density)
    old_name = thickness.attrs.get("standard_name")
    if old_name and (
        new_name := cf_conversion(old_name, "amount2lwethickness", "from")
    ):
        out.attrs["standard_name"] = new_name
    if out_units:
        out = convert_units_to(out, out_units)
    return out


def _flux_and_rate_converter(
    da: xr.DataArray,
    density: Quantified | str,
    to: str = "rate",
    out_units: str = None,
) -> xr.DataArray:
    """Internal converter for :py:func:`xclim.core.units.flux2rate` and :py:func:`xclim.core.units.rate2flux`."""
    if to == "rate":
        # output: rate = da / density, with da = flux
        density_exp = -1
    elif to == "flux":
        # output: flux = da * density, with da = rate
        density_exp = 1
    else:
        raise ValueError("Argument `to` must be one of 'rate' or 'flux'.")

    in_u = units2pint(da)
    density_u = (
        str2pint(density).units if isinstance(density, str) else units2pint(density)
    )
    if out_units:
        out_u = str2pint(out_units).units

        if (in_u * density_u**density_exp).dimensionality != out_u.dimensionality:
            op = {1: "*", -1: "/"}[density_exp]
            raise ValueError(
                f"Dimensions incompatible for {to} = da {op} density:\n"
                f"da: {in_u.dimensionality}\n"
                f"density: {density_u.dimensionality}\n"
                f"out_units ({to}): {out_u.dimensionality}\n"
            )
    else:
        out_u = in_u * density_u**density_exp

    density = convert_units_to(density, (out_u / in_u) ** density_exp)
    out = (da * density**density_exp).assign_attrs(da.attrs)
    out.attrs["units"] = pint2cfunits(out_u)
    if "standard_name" in out.attrs.keys():
        out.attrs.pop("standard_name")
    return out


def rate2flux(
    rate: xr.DataArray,
    density: Quantified,
    out_units: str = None,
) -> xr.DataArray:
    """Convert a rate variable to a flux by multiplying with a density.

    This is the inverse operation of :py:func:`xclim.core.units.flux2rate`.

    Parameters
    ----------
    rate : xr.DataArray
        "Rate" variable. Ex: Snowfall rate in "mm / d".
    density : Quantified
        Density used to convert from a rate to a flux. Ex: Snowfall density "312 kg m-3".
        Density can also be an array with the same shape as `rate`.
    out_units : str, optional
        Output units to convert to.

    Returns
    -------
    flux: xr.DataArray

    Examples
    --------
    The following converts an array of snowfall rate in mm/s to snowfall flux in kg m-2 s-1,
    assuming a density of 100 kg m-3:

    >>> time = xr.cftime_range("2001-01-01", freq="D", periods=365)
    >>> prsnd = xr.DataArray(
    ...     [1] * 365, dims=("time",), coords={"time": time}, attrs={"units": "mm/s"}
    ... )
    >>> prsn = rate2flux(prsnd, density="100 kg m-3", out_units="kg m-2 s-1")
    >>> prsn.units
    'kg m-2 s-1'
    >>> float(prsn[0])
    0.1

    See Also
    --------
    flux2rate
    """
    return _flux_and_rate_converter(
        rate,
        density=density,
        to="flux",
        out_units=out_units,
    )


def flux2rate(
    flux: xr.DataArray,
    density: Quantified,
    out_units: str = None,
) -> xr.DataArray:
    """Convert a flux variable to a rate by dividing with a density.

    This is the inverse operation of :py:func:`xclim.core.units.rate2flux`.

    Parameters
    ----------
    flux : xr.DataArray
        "flux" variable. Ex: Snowfall flux in "kg m-2 s-1".
    density : Quantified
        Density used to convert from a flux to a rate. Ex: Snowfall density "312 kg m-3".
        Density can also be an array with the same shape as `flux`.
    out_units : str, optional
        Output units to convert to.

    Returns
    -------
    rate: xr.DataArray

    Examples
    --------
    The following converts an array of snowfall flux in kg m-2 s-1 to snowfall flux in mm/s,
    assuming a density of 100 kg m-3:

    >>> time = xr.cftime_range("2001-01-01", freq="D", periods=365)
    >>> prsn = xr.DataArray(
    ...     [0.1] * 365,
    ...     dims=("time",),
    ...     coords={"time": time},
    ...     attrs={"units": "kg m-2 s-1"},
    ... )
    >>> prsnd = flux2rate(prsn, density="100 kg m-3", out_units="mm/s")
    >>> prsnd.units
    'mm s-1'
    >>> float(prsnd[0])
    1.0

    See Also
    --------
    rate2flux
    """
    return _flux_and_rate_converter(
        flux,
        density=density,
        to="rate",
        out_units=out_units,
    )


@datacheck
def check_units(val: str | int | float | None, dim: str | None) -> None:
    """Check units for appropriate convention compliance."""
    if dim is None or val is None:
        return

    context = infer_context(dimension=dim)

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
    with units.context(context):
        start = pint.util.to_units_container(val_dim)
        end = pint.util.to_units_container(expected)
        graph = units._active_ctx.graph  # noqa
        if pint.util.find_shortest_path(graph, start, end):
            return

    raise ValidationError(
        f"Data units {val_units} are not compatible with requested {dim}."
    )


def declare_units(
    **units_by_name: str,
) -> Callable:
    """Create a decorator to check units of function arguments.

    The decorator checks that input and output values have units that are compatible with expected dimensions.
    It also stores the input units as a 'in_units' attribute.

    Parameters
    ----------
    units_by_name : dict[str, str]
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

        # Check that all Quantified parameters have their dimension declared.
        for name, val in sig.parameters.items():
            if (
                (val.annotation == "Quantified")
                and (val.default is not _empty)
                and (name not in units_by_name)
            ):
                raise ValueError(
                    f"Argument {name} of function {func.__name__} has no declared dimension."
                )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments, so we can check units
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
    delta_degF based on the input unit. For other dimensionality, it just gives back the input units.

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


def infer_context(standard_name=None, dimension=None):
    """Return units context based on either the variable's standard name or the pint dimension.

    Valid standard names for the hydro context are those including the terms "rainfall",
    "lwe" (liquid water equivalent) and "precipitation". The latter is technically incorrect,
    as any phase of precipitation could be referenced. Standard names for evapotranspiration,
    evaporation and canopy water amounts are also associated with the hydro context.

    Parameters
    ----------
    standard_name: str
      CF-Convention standard name.
    dimension: str
      Pint dimension, e.g. '[time]'.

    Returns
    -------
    str
      "hydro" if variable is a liquid water flux, otherwise "none"
    """
    csn = (
        (
            standard_name
            in [
                "water_potential_evapotranspiration_flux",
                "canopy_water_amount",
                "water_evaporation_amount",
            ]
            or "rainfall" in standard_name
            or "lwe" in standard_name
            or "precipitation" in standard_name
        )
        if standard_name is not None
        else False
    )
    cdim = (dimension == "[precipitation]") if dimension is not None else False

    return "hydro" if csn or cdim else "none"
