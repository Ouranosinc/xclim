"""
CF-Convention checks
====================

Utilities designed to verify the compliance of metadata with the CF-Convention.
"""
import fnmatch

from .options import cfcheck
from .utils import ValidationError

# TODO: Implement pandas infer_freq in xarray with CFTimeIndex. >> PR pydata/xarray#4033


@cfcheck
def check_valid(var, key, expected):
    r"""Check that a variable's attribute has the expected value. Warn user otherwise."""

    att = getattr(var, key, None)
    if att is None:
        raise ValidationError(f"Variable does not have a `{key}` attribute.")
    if not fnmatch.fnmatch(att, expected):
        raise ValidationError(
            f"Variable has a non-conforming {key}. Got `{att}`, expected `{expected}`",
        )
