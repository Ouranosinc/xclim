# noqa: D205,D400
"""
CF-Convention checking
======================

Utilities designed to verify the compliance of metadata with the CF-Convention.
"""
import fnmatch
from typing import Sequence, Union

from .options import cfcheck
from .utils import ValidationError

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
            f"Variable has a non-conforming {key}. Got `{att}`, expected `{expected}`",
        )


def check_valid_temperature(var, units):
    r"""Check that variable is air temperature."""
    check_valid(var, "standard_name", "air_temperature")
    check_valid(var, "units", units)


def check_valid_discharge(var):
    r"""Check that the variable is a discharge."""
    check_valid(var, "standard_name", "water_volume_transport_in_river_channel")
    check_valid(var, "units", "m3 s-1")


def check_valid_min_temperature(var, units="K"):
    r"""Check that a variable is a valid daily minimum temperature."""
    check_valid_temperature(var, units)
    check_valid(var, "cell_methods", "time: minimum within days")


def check_valid_mean_temperature(var, units="K"):
    r"""Check that a variable is a valid daily mean temperature."""
    check_valid_temperature(var, units)
    check_valid(var, "cell_methods", "time: mean within days")


def check_valid_max_temperature(var, units="K"):
    r"""Check that a variable is a valid daily maximum temperature."""
    check_valid_temperature(var, units)
    check_valid(var, "cell_methods", "time: maximum within days")
