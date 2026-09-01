"""Type annotations and constants used throughout xclim."""

from __future__ import annotations

from enum import IntEnum
from importlib.resources import as_file, files
from inspect import _empty
from typing import Literal, NewType, TypeVar

import xarray as xr
from pint import Quantity
from yaml import safe_load

__all__ = [
    "KIND_ANNOTATION",
    "VARIABLES",
    "Condition",
    "DataType",
    "DateStr",
    "DayOfYearStr",
    "Freq",
    "InputKind",
    "Quantified",
    "Reducer",
    "TimeRange",
    "infer_kind_from_parameter",
    "is_percentile_dataarray",
]

# Type hint for xarray DataArray and Dataset
DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)

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


class InputKind(IntEnum):
    """
    Constants for input parameter kinds.

    For use by external parsers to determine what kind of data the indicator expects.
    On the creation of an indicator, the appropriate constant is stored in
    :py:attr:`xclim.core.indicator.Indicator.parameters`. The integer value is what gets stored in the output
    of :py:meth:`xclim.core.indicator.Indicator.json`.

    For developers: For each constant, the docstring specifies the annotation a parameter of a compute function
    should use in order to be picked up by the indicator constructor. Notice that we are using the annotation format
    as described in `PEP 604 <https://peps.python.org/pep-0604/>`_, i.e. with '|' indicating a union and without import
    objects from `typing`.
    """

    VARIABLE = 0
    """A data variable (DataArray or variable name).

       Annotation : ``xr.DataArray``. May not include anything else, may not be optional.
    """
    OPTIONAL_VARIABLE = 1
    """An optional data variable (DataArray or variable name).

       Annotation : ``xr.DataArray | None``. The default should be None.
    """
    QUANTIFIED = 2
    """A quantity with units, either as a string (scalar), a pint.Quantity (scalar) or a DataArray (with units set).

       Annotation : ``xclim.core.utils.Quantified`` and an entry in the :py:func:`xclim.core.units.declare_units`
       decorator. "Quantified" translates to ``str | xr.DataArray | pint.util.Quantity``.
    """
    FREQ_STR = 3
    """A string representing an "offset alias", as defined by pandas.

       See the Pandas documentation on :ref:`timeseries.offset_aliases` for a list of valid aliases.

       Annotation : ``str`` + ``freq`` as the parameter name.
    """
    NUMBER = 4
    """A number.

       Annotation : ``int``, ``float`` and unions thereof, potentially optional.
    """
    STRING = 5
    """A simple string.

       Annotation : ``str`` or ``str | None``. In most cases, this kind of parameter makes sense
       with choices indicated in the docstring's version of the annotation with curly braces.
       See :ref:`notebooks/extendxclim:Defining new index-like compute functions`.
    """
    DAY_OF_YEAR = 6
    """A date, but without a year, in the MM-DD format.

       Annotation : :py:obj:`xclim.core.utils.DayOfYearStr` (may be optional).
    """
    DATE = 7
    """A date in the YYYY-MM-DD format, may include a time.

       Annotation : :py:obj:`xclim.core.utils.DateStr` (may be optional).
    """
    NUMBER_SEQUENCE = 8
    """A sequence of numbers

       Annotation : ``Sequence[int]``, ``Sequence[float]`` and unions thereof, may include single ``int`` and ``float``,
       may be optional.
    """
    BOOL = 9
    """A boolean flag.

       Annotation : ``bool``, may be optional.
    """
    DICT = 10
    """A dictionary.

       Annotation : ``dict`` or ``dict | None``, may be optional.
    """
    MASK = 11
    """A mask or flag or scalar. Any value without units that might be passed as a non-temporal DataArray.
       Can be a DataArray, a single bool or a single float.

        Annotation : ``xr.DataArray | bool`` or ``xr.DataArray | float``, may be optional.
    """
    KWARGS = 50
    """A mapping from argument name to value.

       Developers : maps the ``**kwargs``. Please use as little as possible.
    """
    DATASET = 70
    """An xarray dataset.

       Developers : as compute functions only accept DataArrays, this should only be added by the indicator.
    """
    OTHER_PARAMETER = 99
    """An object that fits None of the previous kinds.

       Developers : This is the fallback kind, it will raise an error in xclim's unit tests if used.
    """


def infer_kind_from_parameter(param) -> InputKind:
    """
    Return the appropriate InputKind constant from an ``inspect.Parameter`` object.

    Parameters
    ----------
    param : Parameter
        An inspect.Parameter instance.

    Returns
    -------
    InputKind
        The appropriate InputKind constant.

    Notes
    -----
    The correspondence between parameters and kinds is documented in :py:class:`xclim.core.utils.InputKind`.
    """
    if param.annotation is not _empty:
        annot = set(param.annotation.replace("xarray.", "").replace("xr.", "").split(" | "))
    else:
        annot = {"no_annotation"}

    if annot == {"DataArray"} and param.default is not None:
        return InputKind.VARIABLE

    annot = annot - {"None"}

    if annot in ({"DataArray", "bool"}, {"DataArray", "float"}, {"DataArray", "int"}):
        return InputKind.MASK

    # Not a mask and not a required variable
    if "DataArray" in annot:
        return InputKind.OPTIONAL_VARIABLE

    if param.name == "freq":
        return InputKind.FREQ_STR

    if param.kind == param.VAR_KEYWORD:
        return InputKind.KWARGS

    if annot == {"Quantified"}:
        return InputKind.QUANTIFIED

    if "DayOfYearStr" in annot:
        return InputKind.DAY_OF_YEAR

    if annot.issubset({"int", "float"}):
        return InputKind.NUMBER

    if annot.issubset({"int", "float", "Sequence[int]", "Sequence[float]"}):
        return InputKind.NUMBER_SEQUENCE

    if (
        annot.issuperset({"str"})
        or annot.issuperset({"Reducer"})
        or annot.issuperset({"Condition"})
        or any(a.startswith("Literal['") for a in annot)
    ):
        return InputKind.STRING

    if annot == {"DateStr"}:
        return InputKind.DATE

    if annot == {"bool"}:
        return InputKind.BOOL

    if annot == {"dict"}:
        return InputKind.DICT

    if annot == {"Dataset"}:
        return InputKind.DATASET

    return InputKind.OTHER_PARAMETER


KIND_ANNOTATION = {
    InputKind.VARIABLE: "str or DataArray",
    InputKind.OPTIONAL_VARIABLE: "str or DataArray, optional",
    InputKind.QUANTIFIED: "quantity (string or DataArray, with units)",
    InputKind.MASK: "DataArray or scalar",
    InputKind.FREQ_STR: "offset alias (string)",
    InputKind.NUMBER: "number",
    InputKind.NUMBER_SEQUENCE: "number or sequence of numbers",
    InputKind.STRING: "str",
    InputKind.DAY_OF_YEAR: "date (string, MM-DD)",
    InputKind.DATE: "date (string, YYYY-MM-DD)",
    InputKind.BOOL: "boolean",
    InputKind.DICT: "dict",
    InputKind.DATASET: "Dataset, optional",
    InputKind.KWARGS: "",
    InputKind.OTHER_PARAMETER: "Any",
}
"""
Mapping from InputKind to human-readable annotations, to use in the Parameters section of a numpydoc style docstring
(and not for function signatures).
"""


def is_percentile_dataarray(source: xr.DataArray) -> bool:
    """
    Evaluate whether a DataArray is a Percentile.

    A percentile DataArray must have 'climatology_bounds' attributes and either a
    quantile or percentiles coordinate, the window is not mandatory.

    Parameters
    ----------
    source : xr.DataArray
        The DataArray to evaluate.

    Returns
    -------
    bool
        True if the DataArray is a percentile.
    """
    return (
        isinstance(source, xr.DataArray)
        and source.attrs.get("climatology_bounds", None) is not None
        and ("quantile" in source.coords or "percentiles" in source.coords)
    )
