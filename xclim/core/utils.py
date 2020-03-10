# -*- coding: utf-8 -*-
"""
Miscellaneous indices utilities
===============================

Helper functions for the indices computation, things that do not belong in neither
`xclim.indices.calendar`, `xclim.indices.fwi`, `xclim.indices.generic` or `xclim.indices.run_length`.
"""
from collections import defaultdict
from functools import partial
from inspect import _empty
from inspect import signature
from types import FunctionType

import numpy as np
import xarray as xr
from boltons.funcutils import FunctionBuilder
from boltons.funcutils import NO_DEFAULT

from xclim.core.units import convert_units_to


def wrapped_partial(func: FunctionType, suggested: dict = None, **fixed):
    """Wrap a function, updating its signature but keeping its docstring.

    Parameters
    ----------
    func : FunctionType
        The function to be wrapped
    suggested : dict
        Keyword arguments that should have new default values
        but still appear in the signature.
    fixed : dict
        Keyword arguments that should be fixed by the wrapped
        and removed from the signature.

    Examples
    --------

    >>> from inspect import signature
    >>> def func(a, b=1, c=1):
            print(a, b, c)
    >>> newf = wrapped_partial(func, b=2)
    >>> signature(newf)
    (a, *, c=1)
    >>> newf(1)
    1, 2, 1
    >>> newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    >>> signature(newf)
    (a, *, c=2)
    >>> newf(1)
    1, 2, 2
    """
    # Adapted from the code of boltons.funcutils.wraps
    suggested = suggested or {}

    sig = signature(func)

    partial_func = partial(func, **suggested, **fixed)

    fb = FunctionBuilder.from_func(func)
    # To be sure the signature is correct,
    # remove everyting and put back only what we want
    for arg in sig.parameters.keys():
        fb.remove_arg(arg)

    kwonly = False  # To preserve order, once a kwonly arg or a fixed arg is found, everything after is kwonly.
    for arg, param in sig.parameters.items():
        if arg in fixed:  # Don't put argument back
            kwonly = True
            continue
        if arg in suggested:
            default = suggested[arg]  # Change default
            kwonly = True  # partial moves keyword args to keyword only.
        else:
            default = param.default
            if param.kind > 1:  # 0 and 1 are positional args.
                kwonly = True
        fb.add_arg(arg, default if default is not _empty else NO_DEFAULT, kwonly=kwonly)

    fb.body = f"return _call({fb.get_invocation_str()})"

    execdict = dict(_call=partial_func, _func=func)
    fully_wrapped = fb.get_func(execdict, with_dict=True)
    # fully_wrapped.__wrapped__ = func  # If this line is uncommented "help" and common IDE will show func's signature, not the updated one.

    return fully_wrapped


# TODO Reconsider the utility of this
def walk_map(d: dict, func: FunctionType):
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
      Input dictionary, possibly nested.
    func : FunctionType
      Function to apply to dictionary values.

    Returns
    -------
    dict
      Dictionary whose values are the output of the given function.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, (dict, defaultdict)):
            out[k] = walk_map(v, func)
        else:
            out[k] = func(v)
    return out


def uas_vas_2_sfcwind(uas: xr.DataArray = None, vas: xr.DataArray = None):
    """Converts eastward and northward wind components to wind speed and direction.

    Parameters
    ----------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    Returns
    -------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Notes
    -----
    Northerly winds with a velocity less than 0.5 m/s are given a wind direction of 0°,
    while stronger winds are set to 360°.
    """
    # TODO: Add an attribute check to switch between sfcwind and wind

    # Converts the wind speed to m s-1
    uas = convert_units_to(uas, "m/s")
    vas = convert_units_to(vas, "m/s")

    # Wind speed is the hypotenuse of "uas" and "vas"
    wind = np.hypot(uas, vas)

    # Add attributes to wind. This is done by copying uas' attributes and overwriting a few of them
    wind.attrs = uas.attrs
    wind.name = "sfcWind"
    wind.attrs["standard_name"] = "wind_speed"
    wind.attrs["long_name"] = "Near-Surface Wind Speed"
    wind.attrs["units"] = "m s-1"

    # Calculate the angle
    # TODO: This creates decimal numbers such as 89.99992. Do we want to round?
    windfromdir_math = np.degrees(np.arctan2(vas, uas))

    # Convert the angle from the mathematical standard to the meteorological standard
    windfromdir = (270 - windfromdir_math) % 360.0

    # According to the meteorological standard, calm winds must have a direction of 0°
    # while northerly winds have a direction of 360°
    # On the Beaufort scale, calm winds are defined as < 0.5 m/s
    windfromdir = xr.where((windfromdir.round() == 0) & (wind >= 0.5), 360, windfromdir)
    windfromdir = xr.where(wind < 0.5, 0, windfromdir)

    # Add attributes to winddir. This is done by copying uas' attributes and overwriting a few of them
    windfromdir.attrs = uas.attrs
    windfromdir.name = "sfcWindfromdir"
    windfromdir.attrs["standard_name"] = "wind_from_direction"
    windfromdir.attrs["long_name"] = "Near-Surface Wind from Direction"
    windfromdir.attrs["units"] = "degree"

    return wind, windfromdir


def sfcwind_2_uas_vas(wind: xr.DataArray = None, windfromdir: xr.DataArray = None):
    """Converts wind speed and direction to eastward and northward wind components.

    Parameters
    ----------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Returns
    -------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    """
    # TODO: Add an attribute check to switch between sfcwind and wind

    # Converts the wind speed to m s-1
    wind = convert_units_to(wind, "m/s")

    # Converts the wind direction from the meteorological standard to the mathematical standard
    windfromdir_math = (-windfromdir + 270) % 360.0

    # TODO: This commented part should allow us to resample subdaily wind, but needs to be cleaned up and put elsewhere
    # if resample is not None:
    #     wind = wind.resample(time=resample).mean(dim='time', keep_attrs=True)
    #
    #     # nb_per_day is the number of values each day. This should be calculated
    #     windfromdir_math_per_day = windfromdir_math.reshape((len(wind.time), nb_per_day))
    #     # Averages the subdaily angles around a circle, i.e. mean([0, 360]) = 0, not 180
    #     windfromdir_math = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
    #                                       for angles in windfromdir_math_per_day])

    uas = wind * np.cos(np.radians(windfromdir_math))
    vas = wind * np.sin(np.radians(windfromdir_math))

    # Add attributes to uas and vas. This is done by copying wind' attributes and overwriting a few of them
    uas.attrs = wind.attrs
    uas.name = "uas"
    uas.attrs["standard_name"] = "eastward_wind"
    uas.attrs["long_name"] = "Near-Surface Eastward Wind"
    wind.attrs["units"] = "m s-1"

    vas.attrs = wind.attrs
    vas.name = "vas"
    vas.attrs["standard_name"] = "northward_wind"
    vas.attrs["long_name"] = "Near-Surface Northward Wind"
    wind.attrs["units"] = "m s-1"

    return uas, vas
