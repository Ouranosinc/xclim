from boltons.funcutils import wraps

from .cfchecks import check_valid
from .datachecks import check_daily
from .missing import *


def check_valid_temperature(var, units):
    r"""Check that variable is air temperature."""

    check_valid(var, "standard_name", "air_temperature")
    check_valid(var, "units", units)
    check_daily(var)


def check_valid_discharge(var):
    r"""Check that the variable is a discharge."""
    #
    check_valid(var, "standard_name", "water_volume_transport_in_river_channel")
    check_valid(var, "units", "m3 s-1")


def valid_daily_min_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmin, *args, **kwds):
        check_valid_temperature(tasmin, units)
        check_valid(tasmin, "cell_methods", "time: minimum within days")
        return comp(tasmin, **kwds)

    return func


def valid_daily_mean_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tas, *args, **kwds):
        check_valid_temperature(tas, units)
        check_valid(tas, "cell_methods", "time: mean within days")
        return comp(tas, *args, **kwds)

    return func


def valid_daily_max_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmax, *args, **kwds):
        check_valid_temperature(tasmax, units)
        check_valid(tasmax, "cell_methods", "time: maximum within days")
        return comp(tasmax, *args, **kwds)

    return func


def valid_daily_max_min_temperature(comp, units="K"):
    r"""Decorator to check that a computation runs on valid min and max temperature datasets."""

    @wraps(comp)
    def func(tasmax, tasmin, **kwds):
        valid_daily_max_temperature(tasmax, units)
        valid_daily_min_temperature(tasmin, units)

        return comp(tasmax, tasmin, **kwds)

    return func


def valid_daily_mean_discharge(comp):
    r"""Decorator to check that a computation runs on valid discharge data."""

    @wraps(comp)
    def func(q, **kwds):
        check_valid_discharge(q)
        return comp(q, **kwds)

    return func
