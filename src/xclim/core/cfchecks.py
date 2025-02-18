"""
CF-Convention Checking
======================

Utilities designed to verify the compliance of metadata with the CF-Convention.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence

import xarray as xr

from xclim.core._exceptions import ValidationError
from xclim.core._types import VARIABLES
from xclim.core.options import cfcheck


@cfcheck
def check_valid(var: xr.DataArray, key: str, expected: str | Sequence[str]):
    r"""
    Check that a variable's attribute has one of the expected values and raise a ValidationError if otherwise.

    Parameters
    ----------
    var : xr.DataArray
        The variable to check.
    key : str
        The attribute to check.
    expected : str or sequence of str
        The expected value(s).

    Raises
    ------
    ValidationError
        If the attribute is not present or does not match the expected value(s).
    """
    att = getattr(var, key, None)
    if att is None:
        raise ValidationError(f"Variable does not have a `{key}` attribute.")
    if isinstance(expected, str):
        expected = [expected]
    for exp in expected:
        if fnmatch.fnmatch(att, exp):
            break
    else:
        raise ValidationError(
            f"Variable has a non-conforming {key}: Got `{att}`, expected `{expected}`",
        )


def cfcheck_from_name(varname: str, vardata: xr.DataArray, attrs: list[str] | None = None):
    """
    Perform cfchecks on a DataArray using specifications from xclim's default variables.

    Parameters
    ----------
    varname : str
        The name of the variable to check.
    vardata : xr.DataArray
        The variable to check.
    attrs : list of str, optional
        The attributes to check. Default is ["cell_methods", "standard_name"].

    Raises
    ------
    ValidationError
        If the variable does not meet the expected CF-Convention.
    """
    if attrs is None:
        attrs = ["cell_methods", "standard_name"]

    data = VARIABLES[varname]
    if "cell_methods" in data and "cell_methods" in attrs:
        _check_cell_methods(getattr(vardata, "cell_methods", None), data["cell_methods"])
    if "standard_name" in data and "standard_name" in attrs:
        check_valid(vardata, "standard_name", data["standard_name"])


@cfcheck
def _check_cell_methods(data_cell_methods: str, expected_method: str) -> None:
    if data_cell_methods is None:
        raise ValidationError("Variable does not have a `cell_methods` attribute.")
    EXTRACT_CELL_METHOD_REGEX = r"(\s*\S+\s*:(\s+[\w()-]+)+)(?!\S*:)"
    for m in re.compile(EXTRACT_CELL_METHOD_REGEX).findall(data_cell_methods):
        if expected_method in m[0]:
            return None
    raise ValidationError(
        f"Variable has a non-conforming cell_methods: "
        f"Got `{data_cell_methods}`, which do not include the expected "
        f"`{expected_method}`."
    )
