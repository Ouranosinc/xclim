"""
Local Intensity Scaling (LOCI)
==============================


References
----------
Schmidli, J., Frei, C., & Vidale, P. L. (2006). Downscaling from GCM precipitation: A benchmark for dynamical and
statistical downscaling methods. International Journal of Climatology, 26(5), 679–689. DOI:10.1002/joc.1287
"""
import numpy as np
import xarray as xr

from .utils import add_cyclic_bounds
from .utils import broadcast
from .utils import ecdf
from .utils import get_correction
from .utils import group_apply
from .utils import map_cdf
from .utils import MULTIPLICATIVE
from .utils import parse_group


def train(
    x, y, group="time.dayofyear", window=1, thresh=None,
):
    r"""
    Return multiplicative correction factors such that the mean of `x` matches the mean of `y` for values above a
    threshold.

    The threshold on the training target `y` is first mapped to `x` by finding the quantile in `x` having the same
    exceedance probability as thresh in `y`. The correction factor is then given by

    .. math::

       s = \frac{\left \langle y: y \geq t_y \right\rangle - t_y}{\left \langle x : x \geq t_x \right\rangle - t_x}

    In the case of precipitations, the correction factor is the ratio of wet-days intensity.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually a reference time series drawn from observations.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping dimension and property. If only the dimension is given (e.g. 'time'), the correction is computed over
      the entire series.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      used with group `time.dayofyear` to increase the number of samples.
    thresh : float
      Threshold under which values are assumed null for `y`. For precipitations, this is the wet-day threshold.

    Returns
    -------
    xr.Dataset with variables:
        - cf : The correction factors indexed by group properties
        - x_thresh : The threshold over `x` indexed by group properties
        - y_thresh : The threshold over `y`.

        The grouping informations are in the "group" and "group_window" attributes.

    """

    x_thresh = map_cdf(x, y, thresh, group).isel(x=0)  # Selecting the first threshold.

    # Compute scaling factor on wet-day intensity
    xth = broadcast(x_thresh, x)
    wx = xr.where(x >= xth, x, np.nan)
    wy = xr.where(y >= thresh, y, np.nan)

    mx = group_apply("mean", wx, group, window, skipna=True)
    my = group_apply("mean", wy, group, window, skipna=True)

    # Correction factor
    cf = get_correction(mx - x_thresh, my - thresh, MULTIPLICATIVE)

    return xr.Dataset(
        data_vars={"cf": cf, "x_thresh": x_thresh, "y_thresh": thresh},
        attrs={"group": group, "group_window": window, "kind": MULTIPLICATIVE},
    )


def predict(x, c, interp="linear"):
    r"""
    Return a multiplicative bias-corrected time series for values above a threshold.

    Given a correction factor `s`, return the series

    .. math::

      p(t) = \max\left(t_y + s \cdot (x(t) - t_x), 0\right)

    Parameters
    ----------
    x : xr.DataArray
      Time series to be bias-corrected, usually a model output.
    c : xr.Dataset
      Dataset returned by `loci.train`.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method used to find the correction factors from the quantile map. See utils.broadcast.

    Returns
    -------
    xr.DataArray
      The bias-corrected time series.

    """
    dim, prop = parse_group(c.group)
    cf = c.cf

    if interp != "nearest" and prop is not None:
        cf = add_cyclic_bounds(cf, prop, cyclic_coords=False)

    xth = broadcast(c.x_thresh, x)
    factor = broadcast(cf, x, group=c.group, interp=interp)

    with xr.set_options(keep_attrs=True):
        out = (factor * (x - xth) + c.y_thresh).clip(min=0)

    out.attrs["bias_corrected"] = True
    return out
