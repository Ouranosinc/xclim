"""
Units Handling Submodule
========================

`xclim`'s `pint`-based unit registry is an extension of the registry defined in `cf-xarray`.
This module defines most unit handling methods.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from importlib.resources import files
from inspect import _empty, signature  # noqa
from typing import Any, Callable, Literal, cast

import cf_xarray.units
import numpy as np
import pint
import xarray as xr
from boltons.funcutils import wraps
from yaml import safe_load

from .calendar import date_range, get_calendar, parse_offset
from .options import datacheck
from .utils import InputKind, Quantified, ValidationError, infer_kind_from_parameter

logging.getLogger("pint").setLevel(logging.ERROR)

__all__ = [
    "amount2lwethickness",
    "amount2rate",
    "check_units",
    "convert_units_to",
    "declare_relative_units",
    "declare_units",
    "ensure_cf_units",
    "ensure_delta",
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
units = deepcopy(cf_xarray.units.units)
# Changing the default string format for units/quantities. cf is implemented by cf-xarray
# g is the most versatile float format.
units.default_format = "gcf"
# Switch this flag back to False. Not sure what that implies, but it breaks some tests.
units.force_ndarray_like = False  # noqa: F841
# Another alias not included by cf_xarray
units.define("@alias percent = pct")

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


with (files("xclim.data") / "variables.yml").open() as f:
    CF_CONVERSIONS = safe_load(f)["conversions"]
_CONVERSIONS = {}


# FIXME: This needs to be properly annotated for mypy compliance.
# See: https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
def _register_conversion(conversion, direction):
    """Register a conversion function to be automatically picked up in `convert_units_to`.

    The function must correspond to a name in `CF_CONVERSIONS`, so to a section in
    `xclim/data/variables.yml::conversions`.
    """
    if conversion not in CF_CONVERSIONS:
        raise NotImplementedError(
            "Automatic conversion functions must have a corresponding section in xclim/data/variables.yml"
        )

    def _func_register(func: Callable) -> Callable:
        _CONVERSIONS[(conversion, direction)] = func
        return func

    return _func_register


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
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value.units)
    else:
        raise NotImplementedError(f"Value of type `{type(value)}` not supported.")

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
        value = value.units

    # Issue originally introduced in https://github.com/hgrecco/pint/issues/1486
    # Should be resolved in pint v0.24. See: https://github.com/hgrecco/pint/issues/1913
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return f"{value:cf}".replace("dimensionless", "")


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
    out: xr.DataArray = da * f.magnitude
    out = out.assign_attrs(units=pint2cfunits(f.units))
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


# FIXME: The typing here is difficult to determine, as Generics cannot be used to track the type of the output.
def convert_units_to(  # noqa: C901
    source: Quantified,
    target: Quantified | units.Unit,
    context: Literal["infer", "hydro", "none"] | None = None,
) -> xr.DataArray | float:
    """Convert a mathematical expression into a value with the same units as a DataArray.

    If the dimensionalities of source and target units differ, automatic CF conversions
    will be applied when possible. See :py:func:`xclim.core.units.cf_conversion`.

    Parameters
    ----------
    source : str or xr.DataArray or units.Quantity
        The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str or xr.DataArray or units.Quantity or units.Unit
        Target array of values to which units must conform.
    context : {"infer", "hydro", "none"}, optional
        The unit definition context. Default: None.
        If "infer", it will be inferred with :py:func:`xclim.core.units.infer_context` using
        the standard name from the `source` or, if none is found, from the `target`.
        This means that the "hydro" context could be activated if any one of the standard names allows it.

    Returns
    -------
    xr.DataArray or float
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

    m: float
    if isinstance(source, str):
        q = str2pint(source)
        # Return magnitude of converted quantity. This is going to fail if units are not compatible.
        m = q.to(target_unit, context).m
        return m
    if isinstance(source, units.Quantity):
        m = source.to(target_unit, context).m
        return m

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

        out: xr.DataArray
        if source_unit == target_unit:
            # The units are the same, but the symbol may not be.
            out = source.assign_attrs(units=target_cf_unit)
            return out

        with units.context(context or "none"):
            out = source.copy(data=units.convert(source.data, source_unit, target_unit))
            out = out.assign_attrs(units=target_cf_unit)
            return out

    # TODO remove backwards compatibility of int/float thresholds after v1.0 release
    if isinstance(source, (float, int)):
        raise TypeError("Please specify units explicitly.")

    raise NotImplementedError(f"Source of type `{type(source)}` is not supported.")


def cf_conversion(
    standard_name: str, conversion: str, direction: Literal["to", "from"]
) -> str | None:
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
            cf_name: str = names[int(not i)]
            return cf_name
    return None


FREQ_UNITS = {
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
        out = multi, FREQ_UNITS.get(base, base)
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
    op : {'min', 'max', 'mean', 'std', 'var', 'doymin', 'doymax',  'count', 'integral', 'sum'}
        The type of aggregation operation performed. "integral" is mathematically equivalent to "sum",
        but the units are multiplied by the timestep of the data (requires an inferrable frequency).
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
    ... )
    >>> dt = (tas - 16).assign_attrs(units="delta_degC")
    >>> degdays = dt.clip(0).sum("time")  # Integral of temperature above a threshold
    >>> degdays = to_agg_units(degdays, dt, op="integral")
    >>> degdays.units
    'week delta_degC'

    Which we can always convert to the more common "K days":

    >>> degdays = convert_units_to(degdays, "K days")
    >>> degdays.units
    'K d'
    """
    if op in ["amin", "min", "amax", "max", "mean", "std", "sum"]:
        out.attrs["units"] = orig.attrs["units"]

    elif op in ["var"]:
        out.attrs["units"] = pint2cfunits(str2pint(orig.units) ** 2)

    elif op in ["doymin", "doymax"]:
        out.attrs.update(
            units="", is_dayofyear=np.int32(1), calendar=get_calendar(orig)
        )

    elif op in ["count", "integral"]:
        m, freq_u_raw = infer_sampling_units(orig[dim])
        orig_u = str2pint(orig.units)
        freq_u = str2pint(freq_u_raw)
        out = out * m

        if op == "count":
            out.attrs["units"] = freq_u_raw
        elif op == "integral":
            if "[time]" in orig_u.dimensionality:
                # We need to simplify units after multiplication
                out_units = (orig_u * freq_u).to_reduced_units()
                out = out * out_units.magnitude
                out.attrs["units"] = pint2cfunits(out_units)
            else:
                out.attrs["units"] = pint2cfunits(orig_u * freq_u)
    else:
        raise ValueError(
            f"Unknown aggregation op {op}. "
            "Known ops are [min, max, mean, std, var, doymin, doymax, count, integral, sum]."
        )

    return out


def _rate_and_amount_converter(
    da: xr.DataArray,
    dim: str = "time",
    to: str = "amount",
    sampling_rate_from_coord: bool = False,
    out_units: str | None = None,
) -> xr.DataArray:
    """Internal converter for :py:func:`xclim.core.units.rate2amount` and :py:func:`xclim.core.units.amount2rate`."""
    m = 1
    u = None  # Default to assume a non-uniform axis
    label: Literal["lower", "upper"] = "lower"  # Default to "lower" label for diff
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
        if base in ["M", "Q", "A", "Y"]:
            start = time.indexes[dim][0]
            if not start_anchor:
                # Anchor is on the end of the period, subtract 1 period.
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

    out: xr.DataArray
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
        out = out.assign_attrs(standard_name=new_name)

    if out_units:
        out = cast(xr.DataArray, convert_units_to(out, out_units))

    return out


@_register_conversion("amount2rate", "from")
def rate2amount(
    rate: xr.DataArray,
    dim: str = "time",
    sampling_rate_from_coord: bool = False,
    out_units: str | None = None,
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
        Specific output units, if needed.

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
    out_units: str | None = None,
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
        Specific output units, if needed.

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
    amount: xr.DataArray, out_units: str | None = None
) -> xr.DataArray | Quantified:
    """Convert a liquid water amount (mass over area) to its equivalent area-averaged thickness (length).

    This will simply divide the amount by the density of liquid water, 1000 kg/m³.
    This is equivalent to using the "hydro" context of :py:data:`xclim.core.units.units`.

    Parameters
    ----------
    amount : xr.DataArray
        A DataArray storing a liquid water amount quantity.
    out_units : str, optional
        Specific output units, if needed.

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
        out = cast(xr.DataArray, convert_units_to(out, out_units))
    return out


@_register_conversion("amount2lwethickness", "from")
def lwethickness2amount(
    thickness: xr.DataArray, out_units: str | None = None
) -> xr.DataArray | Quantified:
    """Convert a liquid water thickness (length) to its equivalent amount (mass over area).

    This will simply multiply the thickness by the density of liquid water, 1000 kg/m³.
    This is equivalent to using the "hydro" context of :py:data:`xclim.core.units.units`.

    Parameters
    ----------
    thickness : xr.DataArray
        A DataArray storing a liquid water thickness quantity.
    out_units : str, optional
        Specific output units, if needed.

    Returns
    -------
    xr.DataArray or Quantified
        The standard_name of `amount` is modified if a conversion is found
        (see :py:func:`xclim.core.units.cf_conversion`), it is removed otherwise. Other attributes are left untouched.

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
        out = cast(xr.DataArray, convert_units_to(out, out_units))
    return out


def _flux_and_rate_converter(
    da: xr.DataArray,
    density: Quantified,
    to: str = "rate",
    out_units: str | None = None,
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
    if isinstance(density, str):
        density_u = str2pint(density).units
    else:
        density_u = units2pint(density)

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

    density_conv = convert_units_to(density, (out_u / in_u) ** density_exp)
    out: xr.DataArray = (da * density_conv**density_exp).assign_attrs(da.attrs)
    out = out.assign_attrs(units=pint2cfunits(out_u))
    if "standard_name" in out.attrs.keys():
        out.attrs.pop("standard_name")
    return out


def rate2flux(
    rate: xr.DataArray,
    density: Quantified,
    out_units: str | None = None,
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
        Specific output units, if needed.

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
    out_units: str | None = None,
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
        Specific output units, if needed.

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
def check_units(
    val: str | xr.DataArray | None, dim: str | xr.DataArray | None = None
) -> None:
    """Check that units are compatible with dimensions, otherwise raise a `ValidationError`.

    Parameters
    ----------
    val : str or xr.DataArray, optional
        Value to check.
    dim : str or xr.DataArray, optional
        Expected dimension, e.g. [temperature]. If a quantity or DataArray is given, the dimensionality is extracted.
    """
    if dim is None or val is None:
        return

    if isinstance(dim, xr.DataArray):
        _dim = str(dim.dims[0])
    else:
        _dim = dim

    # In case val is a DataArray, we try to get a standard_name
    if hasattr(val, "attrs"):
        standard_name = val.attrs.get("standard_name", None)
    else:
        standard_name = None
    context = infer_context(standard_name=standard_name, dimension=_dim)

    # Issue originally introduced in https://github.com/hgrecco/pint/issues/1486
    # Should be resolved in pint v0.24. See: https://github.com/hgrecco/pint/issues/1913
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if str(val).startswith("UNSET "):
            warnings.warn(
                "This index calculation will soon require user-specified thresholds.",
                FutureWarning,
                stacklevel=4,
            )
            val = str(val).replace("UNSET ", "")

    if isinstance(val, (int, float)):
        raise TypeError("Please set units explicitly using a string.")

    try:
        dim_units: pint.Unit | pint.Quantity
        if isinstance(dim, str):
            dim_units = str2pint(dim)
        else:
            dim_units = units2pint(dim)
        expected = dim_units.dimensionality
    except pint.UndefinedUnitError:
        # Raised when it is not understood, we assume it was a dimensionality
        expected = units.get_dimensionality(dim.replace("dimensionless", ""))

    val_units: pint.Unit | pint.Quantity
    if isinstance(val, str):
        val_units = str2pint(val)
    else:
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

    # Issue originally introduced in https://github.com/hgrecco/pint/issues/1486
    # Should be resolved in pint v0.24. See: https://github.com/hgrecco/pint/issues/1913
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        raise ValidationError(
            f"Data units {val_units} are not compatible with requested {dim}."
        )


def _check_output_has_units(out: xr.DataArray | tuple[xr.DataArray]) -> None:
    """Perform very basic sanity check on the output.

    Indices are responsible for unit management. If this fails, it's a developer's error.
    """
    if not isinstance(out, tuple):
        out = (out,)

    for outd in out:
        if "units" not in outd.attrs:
            raise ValueError("No units were assigned in one of the indice's outputs.")
        outd.attrs["units"] = ensure_cf_units(outd.attrs["units"])


# FIXME: This needs to be properly annotated for mypy compliance.
# See: https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
def declare_relative_units(**units_by_name) -> Callable:
    r"""Function decorator checking the units of arguments.

    The decorator checks that input values have units that are compatible with each other.
    It also stores the input units as a 'relative_units' attribute.

    Parameters
    ----------
    \*\*kwargs
        Mapping from the input parameter names to dimensions relative to other parameters.
        The dimensions can be a single parameter name as `<other_var>` or more complex expressions,
        like: `<other_var> * [time]`.

    Returns
    -------
    Callable

    Examples
    --------
    In the following function definition:

    .. code-block:: python

        @declare_relative_units(thresh="<da>", thresh2="<da> / [time]")
        def func(da, thresh, thresh2): ...

    The decorator will check that `thresh` has units compatible with those of da
    and that `thresh2` has units compatible with the time derivative of da.

    Usually, the function would be decorated further by :py:func:`declare_units` to create
    a unit-aware index:

    .. code-block:: python

        temperature_func = declare_units(da="[temperature]")(func)

    This call will replace the "<da>" by "[temperature]" everywhere needed.

    See Also
    --------
    declare_units
    """

    def dec(func):
        sig = signature(func)

        # Check if units are valid
        for name, dim in units_by_name.items():
            for ref, refparam in sig.parameters.items():
                if f"<{ref}>" in dim:
                    if infer_kind_from_parameter(refparam) not in [
                        InputKind.QUANTIFIED,
                        InputKind.OPTIONAL_VARIABLE,
                        InputKind.VARIABLE,
                    ]:
                        raise ValueError(
                            f"Dimensions of {name} are declared relative to {ref}, "
                            f"but that argument doesn't have a type that supports units. Got {refparam.annotation}."
                        )
                    # Put something simple to check validity
                    dim = dim.replace(f"<{ref}>", "(m)")
            if "<" in dim:
                raise ValueError(
                    f"Unit declaration of {name} relative to variables absent from the function's signature."
                )
            try:
                str2pint(dim)
            except pint.UndefinedUnitError:
                # Raised when it is not understood, we assume it was a dimensionality
                try:
                    units.get_dimensionality(dim.replace("dimensionless", ""))
                except Exception as e:
                    raise ValueError(
                        f"Relative units for {name} are invalid. Got {dim}. (See stacktrace for more information)."
                    ) from e

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed values to their proper arguments, so we can check units
            bound_args = sig.bind(*args, **kwargs)
            for name, dim in units_by_name.items():
                context = None
                for ref, refvar in bound_args.arguments.items():
                    if f"<{ref}>" in dim:
                        # Issue originally introduced in https://github.com/hgrecco/pint/issues/1486
                        # Should be resolved in pint v0.24. See: https://github.com/hgrecco/pint/issues/1913
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=DeprecationWarning)
                            dim = dim.replace(f"<{ref}>", f"({units2pint(refvar)})")

                        # check_units will guess the hydro context if "precipitation" appears in dim,
                        # but here we pass a real unit. It will also check the standard name of the arg,
                        # but we give it another chance by checking the ref arg.
                        context = context or infer_context(
                            standard_name=getattr(refvar, "attrs", {}).get(
                                "standard_name"
                            )
                        )
                with units.context(context):
                    check_units(bound_args.arguments.get(name), dim)

            out = func(*args, **kwargs)

            _check_output_has_units(out)

            return out

        setattr(wrapper, "relative_units", units_by_name)
        return wrapper

    return dec


# FIXME: This needs to be properly annotated for mypy compliance.
# See: https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
def declare_units(**units_by_name) -> Callable:
    r"""Create a decorator to check units of function arguments.

    The decorator checks that input and output values have units that are compatible with expected dimensions.
    It also stores the input units as a 'in_units' attribute.

    Parameters
    ----------
    \*\*units_by_name
        Mapping from the input parameter names to their units or dimensionality ("[...]").
        If this decorates a function previously decorated with :py:func:`declare_relative_units`,
        the relative unit declarations are made absolute with the information passed here.

    Returns
    -------
    Callable

    Examples
    --------
    In the following function definition:

    .. code-block:: python

        @declare_units(tas="[temperature]")
        def func(tas): ...

    The decorator will check that `tas` has units of temperature (C, K, F).

    See Also
    --------
    declare_relative_units
    """

    def dec(func):
        # The `_in_units` attr denotes a previously partially-declared function, update with that info.
        if hasattr(func, "relative_units"):
            # Make relative declarations absolute if possible
            for arg, dim in func.relative_units.items():
                if arg in units_by_name:
                    continue

                for ref, refdim in units_by_name.items():
                    if f"<{ref}>" in dim:
                        dim = dim.replace(f"<{ref}>", f"({refdim})")
                if "<" in dim:
                    raise ValueError(
                        f"Units for {arg} are declared relative to arguments absent from this decorator ({dim})."
                        "Pass units for the missing arguments."
                    )
                units_by_name[arg] = dim

        # Check that all Quantified parameters have their dimension declared.
        sig = signature(func)
        for name, param in sig.parameters.items():
            if infer_kind_from_parameter(param) == InputKind.QUANTIFIED and (
                name not in units_by_name
            ):
                raise ValueError(f"Argument {name} has no declared dimensions.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments, so we can check units
            bound_args = sig.bind(*args, **kwargs)
            for name, dim in units_by_name.items():
                check_units(bound_args.arguments.get(name), dim)

            out = func(*args, **kwargs)

            _check_output_has_units(out)

            return out

        setattr(wrapper, "in_units", units_by_name)
        return wrapper

    return dec


def ensure_delta(unit: str) -> str:
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


def infer_context(
    standard_name: str | None = None, dimension: str | None = None
) -> str:
    """Return units context based on either the variable's standard name or the pint dimension.

    Valid standard names for the hydro context are those including the terms "rainfall",
    "lwe" (liquid water equivalent) and "precipitation". The latter is technically incorrect,
    as any phase of precipitation could be referenced. Standard names for evapotranspiration,
    evaporation and canopy water amounts are also associated with the hydro context.

    Parameters
    ----------
    standard_name : str, optional
        CF-Convention standard name.
    dimension : str, optional
        Pint dimension, e.g. '[time]'.

    Returns
    -------
    str
        "hydro" if variable is a liquid water flux, otherwise "none".
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
    c_dim = ("[precipitation]" in dimension) if dimension is not None else False

    return "hydro" if csn or c_dim else "none"
