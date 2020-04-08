"""
Scaling method
--------------

This bias-correction method scales variables by an additive or multiplicative factor so that the mean of x matches
the mean of y.

"""
import xarray as xr

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
    xr.Dataset with variable:
        - qf : The correction factors indexed by group properties and quantiles.

        The type of correction used is stored in the "kind" attribute and grouping informations are in the
        "group" and "group_window" attributes.

    References
    ----------


    """
    sx = group_apply("mean", x, group, window)
    sy = group_apply("mean", y, group, window)

    return xr.Dataset(
        data_vars={"qf": sx + sy},
        attrs={"group": group, "group_window": window, "kind": kind},
    )


def predict(x, cf, interp: str = "nearest"):
    """
    Return a bias-corrected timeseries using the scaling method.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Time series to be bias-corrected, usually a model output.
    qm : xr.DataArray
      Correction factors indexed by group properties, as given by the `scaling.train` function.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method used to find the correction factors from the quantile map. See utils.broadcast.

    Returns
    -------
    xr.DataArray
      The bias-corrected time series.
    """
    dim, prop = parse_group(cf.group)

    # Add cyclical values to the scaling factors for interpolation
    if interp != "nearest" and prop is not None:
        cf["qf"] = add_cyclic_bounds(cf.qf, prop, cyclic_coords=False)

    factor = broadcast(cf.qf, x, group=cf.group, interp=interp)

    out = apply_correction(x, factor, cf.kind)
    out.attrs["bias_corrected"] = True
    return out
