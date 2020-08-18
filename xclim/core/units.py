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
from typing import Any, Optional, Union

import pint.converters
import pint.unit
import xarray as xr
from boltons.funcutils import wraps
from packaging import version

from .options import datacheck
from .utils import ValidationError

__all__ = [
    "convert_units_to",
    "declare_units",
    "pint_multiply",
    "pint2cfunits",
    "units",
    "units2pint",
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
    if unit == "1":
        unit = ""

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
        return (source * fu).to(tu).m

    raise NotImplementedError(f"Source of type `{type(source)}` is not supported.")


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
        (1 * units2pint(val)).to(tu, "hydro")
    except (pint.UndefinedUnitError, pint.DimensionalityError):
        raise ValidationError(
            f"Value's dimension `{val_dim}` does not match expected units `{expected}`."
        )


def declare_units(out_units, check_output=True, **units_by_name):
    """Create a decorator to check units of function arguments.

    The decorator checks that input and output values have units that are compatible with expected dimensions.

    Parameters
    ----------
    out_units : str or Sequence[str]
      The units of the output(s). If the indice outputs multiple DataArray, a sequence of string of the same length must be given.
      Pass "" for unitless quantities.
    check_output : bool
      Set to False to skip the output units check.
    units_by_name : Mapping[str, str]
      Mapping from the input parameter names to their units or dimensionality ("[...]").

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

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments so we can check units
            bound_args = sig.bind(*args, **kwargs)
            for name, val in bound_args.arguments.items():
                check_units(val, bound_units.arguments.get(name, None))

            out = func(*args, **kwargs)

            def _check_out_units(out, out_units):
                if "units" in out.attrs:
                    # Check that output units dimensions match expectations, e.g. [temperature]
                    if "[" in out_units:
                        check_units(out, out_units)
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

            if check_output:
                if isinstance(out, tuple):
                    out = tuple(
                        _check_out_units(o, o_u) for o, o_u in zip(out, out_units)
                    )
                else:
                    out = _check_out_units(out, out_units)

            return out

        return wrapper

    return dec
