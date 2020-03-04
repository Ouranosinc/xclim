# -*- coding: utf-8 -*-
"""
xclim units core submodule

Functions to help the handling of units in xclim with xarray.DataArrays.
"""
import functools
import re
import warnings
from inspect import signature
from typing import Any
from typing import Optional
from typing import Union

import pint.converters
import pint.unit
import xarray as xr

from ._registry import units


__all__ = [
    "convert_units_to",
    "declare_units",
    "pint_multiply",
    "pint2cfunits",
    "units2pint",
]


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


def declare_units(out_units, check_output=True, **units_by_name):
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
