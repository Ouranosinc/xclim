"""Detrended Quantile Matching (Cannon et al. 2015), code inspired from Santander's downscaleR"""
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

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

    # Compute quantile per period
    xq = group_apply("quantile", x, group, window=window, q=q)
    yq = group_apply("quantile", y, group, window=window, q=q)

    # Note that the order of these two operations is critical.
    # We're computing the correction factor based on x' = x - <x>.
    xqp = apply_correction(xq, invert(mu_x, kind), kind)

    # Compute quantile correction factors
    qm = get_correction(xqp, yq, kind)  # qy / qx or qy - qx

    # Reindex the quantile correction factors with x'
    xqm = reindex(qm, xqp, extrapolation)

    return xqm


def predict(
    x,
    qm,
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

    # Add random noise to small values
    if qm.kind == MULTIPLICATIVE and mult_thresh is not None:
        x = jitter_under_thresh(x, mult_thresh)

    # Add cyclical values to the scaling factors for interpolation
    if interp and prop is not None:
        qm = add_cyclic(qm, prop)
        mu_x = add_cyclic(mu_x, prop)

    # Apply mean correction factor nx = x / <x>
    mfx = broadcast(mu_x, x, interp)
    nx = apply_correction(x, invert(mfx, kind), kind)

    # Detrend series
    if detrend:
        np.testing.assert_allclose(nx.mean(dim="time"), 0, atol=1e-6)

        ax = nx.resample(time="Y").mean()
        fit_ds = ax.polyfit(deg=1, dim="time")
        x_trend = xr.polyval(coord=nx.time, coeffs=fit_ds.polyfit_coefficients)
        x_trend -= x_trend.mean(dim="time")

        # Detrended
        nxt = apply_correction(nx, invert(x_trend, kind), kind)

        np.testing.assert_allclose(nxt.mean(dim="time"), 0, atol=1e-6)

    else:
        nxt = nx

    # Quantile mapping
    sel = {"x": nxt}
    qf = broadcast(qm, nxt, interp, sel)
    corrected = apply_correction(nxt, qf, qm.kind)

    # Reapply trend
    if detrend:
        out = apply_correction(corrected, x_trend, kind)
    else:
        out = corrected

    return out
