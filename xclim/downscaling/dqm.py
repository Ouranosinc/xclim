"""Detrended Quantile Matching (Cannon et al. 2015), code inspired from Santander's downscaleR"""
import numpy as np
import xarray as xr

from .temp import polyfit
from .temp import polyval
from .utils import add_cyclic
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import broadcast
from .utils import get_correction
from .utils import get_index
from .utils import group_apply
from .utils import interp_quantiles
from .utils import invert
from .utils import jitter_under_thresh
from .utils import MULTIPLICATIVE
from .utils import nodes
from .utils import parse_group
from .utils import reindex


def train(
    x,
    y,
    group="time.month",
    kind=ADDITIVE,
    window=1,
    mult_thresh=None,
    nq=40,
    extrapolation="constant",
):
    """
    Return the quantile mapping factors and the change in mean using the detrended quantile mapping method.

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    # Nodes
    q = nodes(nq, eps=1e-6)

    # Add random noise to small values
    if kind == MULTIPLICATIVE and mult_thresh is not None:
        # Replace every thing under mult_thresh by a non-zero random number under mult_thresh
        x = jitter_under_thresh(x, mult_thresh)
        y = jitter_under_thresh(y, mult_thresh)

    # Compute mean per period
    mu_x = group_apply("mean", x, group, window)
    # mu_y = group_apply("mean", y, group, window)

    # Compute quantile per period
    xq = group_apply("quantile", x, group, window=window, q=q)
    yq = group_apply("quantile", y, group, window=window, q=q)

    # Remove mean from quantiles
    nxq = apply_correction(xq, invert(mu_x, kind), kind)
    # nyq = apply_correction(yq, invert(mu_y, kind), kind)

    # Compute quantile correction from scaled x quantiles
    qm = get_correction(nxq, yq, kind)  # qy / qx or qy - qx

    # Reindex the quantile correction factors according to scaled x values instead of CDF.
    xqm = reindex(qm, nxq, extrapolation)

    # Compute mean correction
    # mf = get_correction(mu_y, mu_x, kind)  # mx / my or mx - my

    return xqm, mu_x


def predict(
    x,
    qm,
    mu_r,
    window=1,
    mult_thresh=None,
    detrend=True,  # Should this be non-optional ?
    interp=False,
):
    """
    # TODO

    """
    dim, prop = parse_group(qm.group)
    kind = qm.kind

    # Compute mean correction
    mu_x = group_apply("mean", x, qm.group, window)
    mf = get_correction(mu_x, mu_r, kind)

    # Add random noise to small values
    if qm.kind == MULTIPLICATIVE and mult_thresh is not None:
        x = jitter_under_thresh(x, mult_thresh)

    # Add cyclical values to the scaling factors for interpolation
    if interp and prop is not None:
        qm = add_cyclic(qm, prop)
        mf = add_cyclic(mf, prop)

    # Apply mean correction factor nx = x / <x> * <h>
    nx = apply_correction(x, broadcast(mf, x, interp), kind)

    # Detrend series
    if detrend:
        coeffs = polyfit(nx, deg=1, dim="time")
        x_trend = polyval(x.time, coeffs)

        # Normalize with trend instead
        nx = apply_correction(nx, invert(x_trend, qm.kind), qm.kind)

    # Quantile mapping

    sel = {"x": nx}
    out = apply_correction(nx, broadcast(qm, nx, interp, sel), qm.kind)

    if detrend:
        return apply_correction(out, x_trend, qm.kind)

    return out
