from inspect import signature

import xarray as xr

from ..indices.run_length import suspicious_run
from .calendar import clim_mean_doy, within_bnds_doy
from .units import convert_units_to, declare_units
from .utils import VARIABLES, InputKind, MissingVariableError, infer_kind_from_parameter

_REGISTRY = dict()


def _register_methods(func):
    _REGISTRY[func.__name__] = func
    return func


@_register_methods
@declare_units(tasmax="[temperature]", tas="[temperature]", check_output=False)
def tasmax_below_tas(tasmax: xr.DataArray, tas: xr.DataArray):
    return (tasmax < tas).any()


@_register_methods
@declare_units(tasmax="[temperature]", tasmin="[temperature]", check_output=False)
def tasmax_below_tasmin(tasmax: xr.DataArray, tasmin: xr.DataArray):
    return (tasmax < tasmin).any()


@_register_methods
@declare_units(tas="[temperature]", tasmax="[temperature]", check_output=False)
def tas_exceeds_tasmax(tas: xr.DataArray, tasmax: xr.DataArray):
    return (tas > tasmax).any()


@_register_methods
@declare_units(tas="[temperature]", tasmin="[temperature]", check_output=False)
def tas_below_tasmin(tas: xr.DataArray, tasmin: xr.DataArray):
    return (tas < tasmin).any()


@_register_methods
@declare_units(tasmin="[temperature]", tasmax="[temperature]", check_output=False)
def tasmin_exceeds_tasmax(tasmin: xr.DataArray, tasmax: xr.DataArray):
    return (tasmin > tasmax).any()


@_register_methods
@declare_units(tasmin="[temperature]", tas="[temperature]", check_output=False)
def tasmin_exceeds_tas(tasmin: xr.DataArray, tas: xr.DataArray):
    return (tasmin > tas).any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_low(da: xr.DataArray, thresh: str = "-90 degC"):
    thresh = convert_units_to(thresh, da)
    return (da < thresh).any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_high(da: xr.DataArray, thresh: str = "60 degC"):
    thresh = convert_units_to(thresh, da)
    return (da > thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def negative_precipitation_values(pr: xr.DataArray):
    return (pr < 0).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def very_large_precipitation_events(pr: xr.DataArray, thresh="300 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return (pr > thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_1mm_repetitions(pr: xr.DataArray, window=10, op="==", thresh="1 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return suspicious_run(pr, window=window, op=op, thresh=thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_5mm_repetitions(pr: xr.DataArray, window=5, op="==", thresh="5 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return suspicious_run(pr, window=window, op=op, thresh=thresh).any()


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@_register_methods
def outside_5_standard_deviations_of_climatology(
    da: xr.DataArray, window: int = 5, n: int = 5
):
    """Check if any value is outside `n` standard deviations from the day of year mean."""
    mu, sig = clim_mean_doy(da, window=window)
    return ~within_bnds_doy(da, mu + n * sig, mu - n * sig).all()


@_register_methods
def values_repeating_for_5_or_more_days(da: xr.DataArray, window: int = 5):
    return suspicious_run(da, window=window).any()


def missing_vars(func, ds: xr.Dataset):
    sig = signature(func)
    sig = sig.parameters
    extras = dict()
    for i, (arg, value) in enumerate(sig.items()):
        if i == 0:
            continue
        kind = infer_kind_from_parameter(value)
        if kind == InputKind.VARIABLE:
            if arg in ds:
                extras[arg] = ds[arg]
            else:
                raise MissingVariableError()
    return extras


def data_flags(da: xr.DataArray, ds: xr.Dataset):
    var = str(da.name)
    flagfunc = VARIABLES.get(var)["data_flags"]

    flags = dict()
    for name, kwargs in flagfunc.items():
        func = _REGISTRY[name]

        try:
            extras = missing_vars(func, ds)
        except MissingVariableError:
            flags[name] = None
        else:
            flags[name] = func(da, **extras, **(kwargs or dict()))

    dsflags = xr.Dataset(data_vars=flags)
    return dsflags
