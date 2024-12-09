"""Type annotations and constants used throughout xclim."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import NewType, TypeVar

import xarray as xr
from pint import Quantity
from yaml import safe_load

__all__ = [
    "VARIABLES",
    "DateStr",
    "DayOfYearStr",
    "Quantified",
]

#: Type annotation for strings representing full dates (YYYY-MM-DD), may include time.
DateStr = NewType("DateStr", str)

#: Type annotation for strings representing dates without a year (MM-DD).
DayOfYearStr = NewType("DayOfYearStr", str)

#: Type annotation for thresholds and other not-exactly-a-variable quantities
Quantified = TypeVar("Quantified", xr.DataArray, str, Quantity)


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
