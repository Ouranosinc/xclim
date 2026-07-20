"""Type annotations and constants used throughout xclim."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Literal, NewType, TypeVar

import xarray as xr
from pint import Quantity
from yaml import safe_load

__all__ = ["VARIABLES", "Condition", "DateStr", "DayOfYearStr", "Freq", "Quantified", "Reducer", "TimeRange"]

#: Type annotation for strings representing full dates (YYYY[-MM[-DD[THH[:MM]]]]), may include time.
DateStr = NewType("DateStr", str)

#: Type annotation for a range between to full dates (YYYY[-MM[-DD]])
TimeRange = tuple[DateStr, DateStr]

#: Type annotation for strings representing dates without a year (MM-DD).
DayOfYearStr = NewType("DayOfYearStr", str)

#: Type annotation for frequency strings
Freq = Literal[
    "D",
    "ME",
    "MS",
    "QE",
    "QE-APR",
    "QE-AUG",
    "QE-DEC",
    "QE-FEB",
    "QE-JAN",
    "QE-JUL",
    "QE-JUN",
    "QE-MAR",
    "QE-MAY",
    "QE-NOV",
    "QE-OCT",
    "QE-SEP",
    "QS",
    "QS-APR",
    "QS-AUG",
    "QS-DEC",
    "QS-FEB",
    "QS-JAN",
    "QS-JUL",
    "QS-JUN",
    "QS-MAR",
    "QS-MAY",
    "QS-NOV",
    "QS-OCT",
    "QS-SEP",
    "YE",
    "YE-APR",
    "YE-AUG",
    "YE-DEC",
    "YE-FEB",
    "YE-JAN",
    "YE-JUL",
    "YE-JUN",
    "YE-MAR",
    "YE-MAY",
    "YE-NOV",
    "YE-OCT",
    "YE-SEP",
    "YS",
    "YS-APR",
    "YS-AUG",
    "YS-DEC",
    "YS-FEB",
    "YS-JAN",
    "YS-JUL",
    "YS-JUN",
    "YS-MAR",
    "YS-MAY",
    "YS-NOV",
    "YS-OCT",
    "YS-SEP",
    "h",
    "min",
    "ms",
    "s",
    "us",
]

#: Type annotation for thresholds and other not-exactly-a-variable quantities
Quantified = TypeVar("Quantified", xr.DataArray, str, Quantity)

#: Type annotation of the condition/comparison operators
Condition = Literal[">", "gt", "<", "lt", ">=", "ge", "<=", "le"]

#: Type annotation for reducing/resampling function names, or a function that reduces the "time" dimension.
Reducer = Literal["min", "max", "mean", "std", "var", "count", "sum", "integral", "doymin", "doymax"]
# FIXME : I want to do Literal[...] | Callable, but pylint won't allow it

with as_file(files("xclim.data")) as data_dir:
    with (data_dir / "variables.yml").open() as f:
        VARIABLES = safe_load(f)["variables"]
        """Official variables definitions.

A mapping from variable name to a dict with the following keys:

- canonical_units [required] : The conventional units used by this variable.
- cell_methods [optional] : The conventional `cell_methods` CF attribute
- description [optional] : A description of the variable, to populate dynamically generated docstrings.
- dimensions [optional] : The dimensionality of the variable, an abstract version of the units.
  See `xclim.units.units._dimensions.keys()` for available terms. This is especially useful for making xclim aware of
  "[precipitation]" variables.
- standard_name [optional] : If it exists, the CF standard name.
- data_flags [optional] : Data flags methods (:py:mod:`xclim.core.dataflags`) applicable to this variable.
  The method names are keys and values are dicts of keyword arguments to pass
  (an empty dict if there's nothing to configure).
"""
