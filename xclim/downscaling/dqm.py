"""Detrended Quantile Matching (Cannon et al. 2015), code inspired from Santander's downscaleR"""
import numpy as np
import xarray as xr

from .utils import add_cyclic_bounds
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import broadcast
from .utils import equally_spaced_nodes
from .utils import get_correction
from .utils import group_apply
from .utils import invert
from .utils import jitter_under_thresh
from .utils import MULTIPLICATIVE
from .utils import parse_group
from .utils import reindex


def train(
    x,
    y,
    kind=ADDITIVE,
    group="time.month",
    window=1,
    mult_thresh=None,
    nq=40,
    extrapolation="constant",
):
    """
    Return the quantile mapping factors using the detrended quantile mapping method.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually an observed at-site time-series.
    kind : {ADDITIVE, MULTIPLICATIVE, '+', '*'}
      Whether the correction are computed and applied in a multiplicative (ex. precip) or additive (ex: temp) manner.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.
    mult_thresh : float
      If kind is MULTIPLICATIVE, all values under this threshold are replaced by a non-zero random number smaller then the threshold.
      This is done to remove values that are exactly 0.
    nq : int
      Number of quantiles.
    extrapolation : {'constant', 'nan'}
      What type of extrapolation should be done when predicting on values outside the range of 'x'. See `utils.extrapolate_qm`.

    Returns
    -------
    xr.DataArray
      The correction factors along the group and value coordinates.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    # nq nodes + limit nodes at 1E-6 and 1 - 1E-6
    q = equally_spaced_nodes(nq, eps=1e-6)

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
    mult_thresh=None,
    detrend=True,  # Should this be non-optional ?
    interp=False,
):
    """
    Return a bias-corrected timeseries using the detrended quantile mapping method.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Uncorrected data, usually a model output for a time period subsequent to the training period.
    qm : xr.DataArray
      The array of correction factors along the value and group dimensions, as given by the train() method.
    mult_thresh : float
      If qm.kind is MULTIPLICATIVE, all values under this threshold are replaced by a non-zero random number smaller then the threshold.
      This is done to remove values that are exactly 0.
    detrend : bool
      Whether to detrend the input uncorrected data.
    interp : bool
      Whether to linearly interpolate the correction factors (True) or to find the closest factor (False).

    Returns
    -------
    xr.DataArray
      The bias-corrected timeseries.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    dim, prop = parse_group(qm.group)
    window = qm.group_window
    kind = qm.kind

    # Compute mean correction
    mu_x = group_apply("mean", x, qm.group, window)

    # Add random noise to small values
    if kind == MULTIPLICATIVE and mult_thresh is not None:
        x = jitter_under_thresh(x, mult_thresh)

    # Add cyclical values to the scaling factors for interpolation
    if interp and prop is not None:
        qm = add_cyclic_bounds(qm, prop)
        mu_x = add_cyclic_bounds(mu_x, prop)

    # Apply mean correction factor nx = x / <x>
    mfx = broadcast(mu_x, x, interp)
    nx = apply_correction(x, invert(mfx, kind), kind)

    # Detrend series
    if detrend:
        null = 0 if kind == ADDITIVE else 1
        np.testing.assert_allclose(nx.mean(dim="time"), null, atol=1e-6)

        ax = nx.resample(time="Y").mean()
        fit_ds = ax.polyfit(deg=1, dim="time")
        x_trend = xr.polyval(coord=nx.time, coeffs=fit_ds.polyfit_coefficients)
        x_trend = apply_correction(
            x_trend, invert(x_trend.mean(dim="time"), kind), kind
        )

        # Detrended
        nxt = apply_correction(nx, invert(x_trend, kind), kind)

        # np.testing.assert_allclose(nxt.mean(dim="time"), null, atol=1e-5)

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

    out.attrs["bias_corrected"] = True

    return out
