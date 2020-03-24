"""
Empirical quantile mapping
==========================


Notes
-----
In principle, whether the `correction` factor is multiplicative or additive does not change much within the
interpolation space, it's only in extrapolation (I assume) that it does potentially change something.

References
----------

"""
import numpy as np
import xarray as xr

from .utils import add_cyclic_bounds
from .utils import apply_correction
from .utils import equally_spaced_nodes
from .utils import get_correction
from .utils import get_index
from .utils import group_apply
from .utils import parse_group
from .utils import reindex


def train(
    x, y, group="time.dayofyear", kind="+", nq=40, window=1, extrapolation="constant"
):
    """Compute quantile bias-adjustment factors.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually an observed at-site time-series.
    nq : int
      Number of quantiles.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.

    Returns
    -------
    xr.Dataset
      Quantiles for the source and target series.
    """
    q = equally_spaced_nodes(nq, eps=1e-6)
    xq = group_apply("quantile", x, group, window, q=q)
    yq = group_apply("quantile", y, group, window, q=q)

    qqm = get_correction(xq, yq, kind)

    # Reindex the correction factor so they can be interpolated from quantiles instead of CDF.
    xqm = reindex(qqm, xq, extrapolation)
    return xqm


def predict(x, qm, interp=False):
    dim, prop = parse_group(qm.group)

    # Add cyclical values to the scaling factors for interpolation
    if interp and prop is not None:
        qm = add_cyclic_bounds(qm, prop)

    sel = {"x": x}
    if prop:
        sel.update({prop: get_index(x, dim, prop, interp)})

    # Extract the correct quantile for each time step.
    if interp:  # Interpolate both the time group and the quantile.
        factor = qm.interp(sel)
    else:  # Find quantile for nearest time group and quantile.
        factor = qm.sel(sel, method="nearest")

    # Apply the correction factors
    out = apply_correction(x, factor, qm.kind)
    out["bias_corrected"] = True
    # Remove time grouping and quantile coordinates
    if prop:
        return out.drop_vars(prop)
    return out
