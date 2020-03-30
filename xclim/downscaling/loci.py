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
    """

    """

    x_thresh = map_cdf(x, y, thresh, group).isel(x=0)  # Selecting the first threshold.

    # Compute scaling factor on wet-day intensity
    xth = broadcast(x_thresh, x)
    wx = xr.where(x > xth, x, np.nan)
    wy = xr.where(y > thresh, y, np.nan)

    mx = group_apply("mean", wx, group, window, skipna=True)
    my = group_apply("mean", wy, group, window, skipna=True)

    # Correction factor
    cf = get_correction(mx - x_thresh, my - thresh, MULTIPLICATIVE)

    return xr.Dataset(
        data_vars={"cf": cf, "x_thresh": x_thresh, "y_thresh": thresh},
        attrs={"group": group, "group_window": window, "kind": MULTIPLICATIVE},
    )


def predict(x, c, interp="linear"):
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
