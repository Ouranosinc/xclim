"""
Scaling method
==============

Array `x` is scaled by an additive or multiplicative factor so that the mean of x matches the mean of y. These
factors can be computed independently per season, month or day of the year.

"""
from .utils import add_cyclic_bounds
from .utils import apply_correction
from .utils import broadcast
from .utils import get_correction
from .utils import group_apply
from .utils import parse_group


def train(x, y, group="time.month", kind="+", window=1):
    """Return correction factors such that the mean of `x` matches the mean of `y`.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually a reference time series drawn from observations.
    kind : {"+", "*"}
      The type of correction, either additive (+) or multiplicative (*). Multiplicative correction factors are
      typically used with lower bounded variables, such as precipitation, while additive factors are used for
      unbounded variables, such as temperature.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping dimension and property. If only the dimension is given (e.g. 'time'), the correction is computed over
      the entire series.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      used with group `time.dayofyear` to increase the number of samples.

    Returns
    -------
    xr.DataArray
      The correction factors indexed by group properties.

    References
    ----------


    """
    sx = group_apply("mean", x, group, window)
    sy = group_apply("mean", y, group, window)

    return get_correction(sx, sy, kind)


def predict(x, cf, interp=False):
    """Apply correction to data.
    """
    dim, prop = parse_group(cf.group)

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        cf = add_cyclic_bounds(cf, prop)

    factor = broadcast(cf, x, interp)

    out = apply_correction(x, factor, cf.kind)
    out.attrs["bias_corrected"] = True
    return out
