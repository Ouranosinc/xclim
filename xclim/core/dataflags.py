from inspect import signature

import xarray as xr

from ..indices.run_length import suspicious_run
from .calendar import clim_mean_doy, within_bnds_doy
from .units import convert_units_to, declare_units
from .utils import VARIABLES, MissingVariableError


@declare_units(tasmax="[temperature]", tas="[temperature]")
def tasmax_below_tas(tasmax: xr.Dataset, tas: xr.Dataset):
    return (tasmax < tas).any()


@declare_units(tasmax="[temperature]", tasmin="[temperature]")
def tasmax_below_tasmin(tasmax: xr.Dataset, tasmin: xr.Dataset):
    return (tasmax < tasmin).any()


@declare_units(tas="[temperature]", tasmax="[temperature]")
def tas_exceeds_tasmax(tas: xr.Dataset, tasmax: xr.Dataset):
    return (tas > tasmax).any()


@declare_units(tas="[temperature]", tasmin="[temperature]")
def tas_below_tasmin(tas: xr.Dataset, tasmin: xr.Dataset):
    return (tas < tasmin).any()


@declare_units(tasmin="[temperature]", tasmax="[temperature]")
def tasmin_exceeds_tasmax(tasmin: xr.Dataset, tasmax: xr.Dataset):
    return (tasmin > tasmax).any()


@declare_units(tasmin="[temperature]", tas="[temperature]")
def tasmin_exceeds_tas(tasmin: xr.Dataset, tas: xr.Dataset):
    return (tasmin > tas).any()


@declare_units(da="[temperature]")
def temperature_extremely_low(da: xr.DataArray, thresh: str = "-90 degC"):
    thresh = convert_units_to(thresh, da)
    return (da < thresh).any()


@declare_units(da="[temperature]")
def temperature_extremely_high(da: xr.DataArray, thresh: str = "60 degC"):
    thresh = convert_units_to(thresh, da)
    return (da > thresh).any()


@declare_units(pr="[precipitation]")
def negative_precipitation_values(pr: xr.Dataset):
    return (pr < 0).any()


@declare_units(pr="[precipitation]")
def very_large_precipitation_events(pr: xr.Dataset, thresh="300 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return (pr > thresh).any()


@declare_units(pr="[precipitation]")
def many_1mm_repetitions(pr: xr.DataArray, window=10, op="==", thresh="1 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return not suspicious_run(pr, window=window, op=op, thresh=thresh)


@declare_units(pr="[precipitation]")
def many_5mm_repetitions(pr: xr.DataArray, window=5, op="==", thresh="5 mm d-1"):
    thresh = convert_units_to(thresh, pr)
    return not suspicious_run(pr, window=window, op=op, thresh=thresh)


def outside_5_standard_deviations_of_climatology(
    da: xr.DataArray, window: int = 5, n: int = 5
):
    """Check if any value is outside `n` standard deviations from the day of year mean."""
    mu, sig = clim_mean_doy(da, window=window)
    return not within_bnds_doy(da, mu + n * sig, mu - n * sig).all()


def values_repeating_for_5_or_more_days(da: xr.DataArray, window: int = 5):
    return not suspicious_run(da, window=window)


def missing_vars(func, var: str, ds: xr.Dataset):
    sig = signature(func)
    sig = sig.pop(var)
    extras = dict()
    for arg in sig:
        if arg in ds:
            extras[arg] = ds[arg]
        else:
            raise MissingVariableError()


def data_flags(da: xr.DataArray, ds: xr.Dataset):
    var = str(da.name)
    flagfunc = VARIABLES.get(var)["data_flags"]

    flags = dict()
    for func in flagfunc:
        kwargs = VARIABLES["data_flags"][func.name]
        try:
            extras = missing_vars(func, var, ds)
        except MissingVariableError:
            flags[func.name] = None
        else:
            flags[func.name] = func(da, **extras, **kwargs)

    dsflags = xr.Dataset(data_vars=flags)
    return dsflags


# # TODO: Migrated from Data Quality Assurance Checks
# def flag(arr, conditions):
#     for msg, func in conditions.items():
#         if func(arr):
#             UserWarning(msg)
#
#
# # TODO: Migrated from Data Quality Assurance Checks
# def icclim_precipitation_flags():
#     """Return a dictionary of conditions that would flag a suspicious precipitation time series."""
#     conditions = {
#         "Contains negative values": lambda x: (x < 0).any(),
#         "Values too large (> 300mm)": lambda x: (x > 300).any(),
#         "Many 1mm repetitions": lambda x: suspicious_run(x, window=10, thresh=1.0),
#         "Many 5mm repetitions": lambda x: suspicious_run(x, window=5, thresh=5.0)
#         # TODO: Create a check for dry days in precipitation runs
#         # . . . dry periods receive flag = 1 (suspect), if the amount of dry days lies
#         # outside a 14·bivariate standard deviation ?
#         # 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation
#     }
#
#     return conditions
#
#
# # TODO: Migrated from Data Quality Assurance Checks
# def icclim_tasmean_flags():
#     """Return a dictionary of conditions that would flag a suspicious tas time series."""
#     conditions = {
#         "Extremely low (< -90℃)": lambda x: (x < -90).any(),
#         "Extremely high (> 60℃)": lambda x: (x > 60).any(),
#         "Exceeds maximum temperature": lambda x, tasmax: (x > tasmax).any(),
#         "Below minimum temperature": lambda x, tasmin: (x < tasmin).any(),
#         "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
#         "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
#     }
#
#     return conditions
#
#
# # TODO: Migrated from Data Quality Assurance Checks
# def icclim_tasmax_flags():
#     """Return a dictionary of conditions that would flag a suspicious tasmax time series."""
#     conditions = {
#         "Extremely low (< -90℃)": lambda x: (x < -90).any(),
#         "Extremely high (> 60℃)": lambda x: (x > 60).any(),
#         "Below mean temperature": lambda x, tas: (x < tas).any(),
#         "Below minimum temperature": lambda x, tasmin: (x < tasmin).any(),
#         "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
#         "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
#     }
#
#     return conditions
#
#
# # TODO: Migrated from Data Quality Assurance Checks
# def icclim_tasmin_flags():
#     """Return a dictionary of conditions that would flag a suspicious tasmin time series."""
#     conditions = {
#         "Extremely low (< -90℃)": lambda x: (x < -90).any(),
#         "Extremely high (> 60℃)": lambda x: (x > 60).any(),
#         "Exceeds maximum temperature": lambda x, tasmax: (x > tasmax).any(),
#         "Exceeds mean temperature": lambda x, tas: (x > tas).any(),
#         "Identical values for 5 or more days": lambda x: suspicious_run(x, window=5),
#         "Outside 5 standard deviations of mean": lambda x: outside_climatology(x, n=5),
#     }
#
#     return conditions
