from functools import wraps
from warnings import warn

"""
Dev notes
---------

`functools.wraps` is used to copy the docstring and the function's original name from the source
function to the decorated function. This allows sphinx to correctly find and document
functions.
"""


def check_valid(var, key, expected):
    """Check that a variable's attribute has the expected value. Warn user otherwise."""
    att = getattr(var, key, None)
    if att is None:
        warn('Variable does not have a `{}` attribute.'.format(key))
    elif att != expected:
        warn('Variable has a non-conforming {}. Got `{}`, expected `{}`'.format(key, att, expected))


def check_monotonic(var):
    """Assert that the series is continuous (no jumps in time index)."""
    if not var.time.to_pandas().is_monotonic:
        raise ValueError("time index is not monotonically increasing.")


def check_valid_temperature(var, units):
    """Check that variable is a temperature."""
    check_valid(var, 'standard_name', 'air_temperature')
    check_valid(var, 'units', units)


def check_valid_discharge(var):
    """Check that the variable is a discharge."""
    check_valid(var, 'standard_name', 'discharge')


def valid_daily_min_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmin, *args, **kwds):
        check_valid_temperature(tasmin, units)
        check_valid(tasmin, 'cell_methods', 'time: minimum within days')
        return comp(tasmin, **kwds)

    return func


def valid_daily_mean_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tas, *args, **kwds):
        check_valid_temperature(tas, units)
        check_valid(tas, 'cell_methods', 'time: mean within days')
        return comp(tas, *args, **kwds)

    return func


def valid_daily_max_temperature(comp, units='K'):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    @wraps(comp)
    def func(tasmax, *args, **kwds):
        check_valid_temperature(tasmax, units)
        check_valid(tasmax, 'cell_methods', 'time: maximum within days')
        return comp(tasmax, *args, **kwds)

    return func


def valid_daily_max_min_temperature(comp, units='K'):
    """Decorator to check that a computation runs on valid min and max temperature datasets."""

    @wraps(comp)
    def func(tasmax, tasmin, **kwds):
        valid_daily_max_temperature(tasmax, units)
        valid_daily_min_temperature(tasmin, units)

        return comp(tasmin, tasmax, **kwds)

    return func


def valid_daily_mean_discharge(comp):
    """Decorator to check that a computation runs on valid discharge data."""

    @wraps(comp)
    def func(q, **kwds):
        check_valid_discharge(q)
        return comp(q, **kwds)

    return func


def valid_missing_data_threshold(comp, threshold=0):
    """Check that the relative number of missing data points does not exceed a threshold."""
    # TODO
    pass


def check_is_dataarray(comp):
    """Decorator to check that a computation has an instance of xarray.DataArray
     as first argument."""

    @wraps(comp)
    def func(data_array, *args, **kwds):
        import xarray as xr
        assert (isinstance(data_array, xr.DataArray))
        return comp(data_array, *args, **kwds)

    return func
