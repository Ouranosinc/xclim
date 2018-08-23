from warnings import warn


def check_valid(var, key, expected):
    """Check that a variable's attribute has the expected value. Warn user otherwise."""
    att = getattr(var, key)
    if att != expected:
        warn('Variable has a non-conforming {}. Got `{}`, expected `{}`'.format(key, att, expected))


# TODO: check that the series are continuous (no jumps in time index).
def check_valid_temperature(var):
    """Check that variable is a temperature."""
    check_valid(var, 'standard_name', 'air_temperature')
    # check_valid(var, 'units', 'K')


def check_valid_discharge(var):
    """Check that the variable is a discharge."""
    check_valid(var, 'standard_name', 'discharge')


def valid_daily_min_temperature(comp):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    def func(tasmin, *args, **kwds):
        check_valid_temperature(tasmin)
        check_valid(tasmin, 'cell_methods', 'time: minimum within days')
        return comp(tasmin, **kwds)

    return func


def valid_daily_mean_temperature(comp):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    def func(tas, *args, **kwds):
        check_valid_temperature(tas)
        check_valid(tas, 'cell_methods', 'time: mean within days')
        return comp(tas, *args, **kwds)

    return func


def valid_daily_max_temperature(comp):
    """Decorator to check that a computation runs on a valid temperature dataset."""

    def func(tasmax, **kwds):
        check_valid_temperature(tasmax)
        check_valid(tasmax, 'cell_methods', 'time: maximum within days')
        return comp(tasmax, **kwds)

    return func


def valid_daily_max_min_temperature(comp):
    """Decorator to check that a computation runs on valid min and max temperature datasets."""

    def func(tasmax, tasmin, **kwds):
        valid_daily_max_temperature(tasmax)
        valid_daily_min_temperature(tasmin)

        return comp(tasmin, tasmax, **kwds)

    return func


def valid_daily_mean_discharge(comp):
    """Decorator to check that a computation runs on valid discharge data."""

    def func(q, **kwds):
        check_valid_discharge(q)
        return comp(q, **kwds)

    return func


def valid_missing_data_threshold(comp, threshold=0):
    """Check that the relative number of missing data points does not exceed a threshold."""
    # TODO
    pass
