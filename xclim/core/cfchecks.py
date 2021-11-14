# noqa: D205,D400
"""
CF-Convention checking
======================

Utilities designed to verify the compliance of metadata with the CF-Convention.
"""
import fnmatch
import re
from typing import Sequence, Union

from .options import cfcheck
from .utils import VARIABLES, ValidationError

# TODO: Implement pandas infer_freq in xarray with CFTimeIndex. >> PR pydata/xarray#4033


@cfcheck
def check_valid(var, key: str, expected: Union[str, Sequence[str]]):
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


def cfcheck_from_name(varname, vardata):
    """Perform cfchecks on a DataArray using specifications from xclim's default variables."""
    data = VARIABLES[varname]
    if "cell_methods" in data:
        _check_cell_methods(
            getattr(vardata, "cell_methods", None), data["cell_methods"]
        )
    if "standard_name" in data:
        check_valid(vardata, "standard_name", data["standard_name"])


@cfcheck
def _check_cell_methods(data_cell_methods: str, expected_method: str) -> None:
    if data_cell_methods is None:
        raise ValidationError("Variable does not have a `cell_methods` attribute.")
    EXTRACT_CELL_METHOD_REGEX = r"(\s*\S+\s*:(\s+[\w()-]+)+)(?!\S*:)"
    for m in re.compile(EXTRACT_CELL_METHOD_REGEX).findall(data_cell_methods):
        if m[0].__contains__(expected_method):
            return None
    raise ValidationError(
        f"Variable has a non-conforming cell_methods: "
        f"Got `{data_cell_methods}`, which do not include the expected "
        f"`{expected_method}`"
    )
