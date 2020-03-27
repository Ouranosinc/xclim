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
from .utils import get_correction
from .utils import group_apply
from .utils import map_threshold
from .utils import MULTIPLICATIVE
from .utils import parse_group


def train(
    x, y, group="time.dayofyear", window=1, thresh=None,
):
    """

    """
    dim, prop = parse_group(group)

    # Determine model (x) wet-day threshold.
    x_thresh = group_apply(
        lambda x, dim: map_threshold(x.x, x.y, thresh),
        xr.Dataset({"x": x, "y": y}),
        group=group,
    )

    # Compute scaling factor on wet-day intensity
    xth = broadcast(x_thresh, x)
    wx = xr.where(x > xth, x, np.nan)
    wy = xr.where(y > thresh, y, np.nan)

    mx = group_apply("mean", wx, group, window, skipna=True)
    my = group_apply("mean", wy, group, window, skipna=True)

    return get_correction(mx - x_thresh, my - thresh, MULTIPLICATIVE), x_thresh


def predict(x, cf, x_thresh, interp=False):
    dim, prop = parse_group(cf.group)

    if interp:
        cf = add_cyclic_bounds(cf, prop)

    xth = broadcast(x_thresh, x)
    factor = broadcast(cf, x - xth, interp)

    with xr.set_options(keep_attrs=True):
        out = (x * factor + xth).clip(min=0)
    out.attrs["bias_corrected"] = True
    return out
