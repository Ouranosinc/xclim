"""
CF-Convention Checking
======================

Utilities designed to verify the compliance of metadata with the CF-Convention.
"""
from __future__ import annotations

import fnmatch
import re
from typing import Sequence

from .options import cfcheck
from .utils import VARIABLES, ValidationError


@cfcheck
def check_valid(var, key: str, expected: str | Sequence[str]):
    r"""Check that a variable's attribute has one of the expected values. Raise a ValidationError otherwise."""
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


def cfcheck_from_name(varname, vardata, attrs: list[str] = None):
    """Perform cfchecks on a DataArray using specifications from xclim's default variables."""
    if attrs is None:
        attrs = ["cell_methods", "standard_name"]

    data = VARIABLES[varname]
    if "cell_methods" in data and "cell_methods" in attrs:
        _check_cell_methods(
            getattr(vardata, "cell_methods", None), data["cell_methods"]
        )
    if "standard_name" in data and "standard_name" in attrs:
        check_valid(vardata, "standard_name", data["standard_name"])


@cfcheck
def _check_cell_methods(data_cell_methods: str, expected_method: str) -> None:
    if data_cell_methods is None:
        raise ValidationError("Variable does not have a `cell_methods` attribute.")
    EXTRACT_CELL_METHOD_REGEX = r"(\s*\S+\s*:(\s+[\w()-]+)+)(?!\S*:)"
    for m in re.compile(EXTRACT_CELL_METHOD_REGEX).findall(data_cell_methods):
        # FIXME: Can this be replaced by "in"?
        if m[0].__contains__(expected_method):
            return None
    raise ValidationError(
        f"Variable has a non-conforming cell_methods: "
        f"Got `{data_cell_methods}`, which do not include the expected "
        f"`{expected_method}`"
    )
