"""
Detrended quantile mapping
==========================

Quantiles from detrended `x` are mapped onto quantiles from `y`.
"""
import xarray as xr

from .base import Grouper
from .base import PolyDetrend
from .utils import add_cyclic_bounds
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import broadcast
from .utils import equally_spaced_nodes
from .utils import extrapolate_qm
from .utils import get_correction
from .utils import interp_on_quantiles
from .utils import invert
from .utils import jitter_under_thresh
from .utils import MULTIPLICATIVE
from .utils import parse_group


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
    mult_thresh : float, None
      In the multiplicative case, all values under this threshold are replaced by a non-zero random number smaller
      then the threshold. This is done to remove values that are exactly or close to 0 and create numerical
      instabilities.
    nq : int
      Number of equally spaced quantile nodes. Limit nodes are added at both ends for extrapolation.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation method used when predicting on values outside the range of 'x'. See
      `utils.extrapolate_qm`.

    Returns
    -------
    xr.Dataset with variables:
        - qf : The correction factors indexed by group properties and quantiles.
        - xq : The quantile values residuals (x/<x> or x-<x>).

        The type of correction used is stored in the "kind" attribute and grouping informations are in the
        "group" and "group_window" attributes.

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

    gr = Grouper(group, window)

    # Compute mean per period
    mu_x = gr.apply("mean", x)

    # Compute quantile per period
    xq = gr.apply("quantile", x, q=q).rename(quantile="quantiles")
    yq = gr.apply("quantile", y, q=q).rename(quantile="quantiles")

    # Compute quantile correction factors
    qf = get_correction(xq, yq, kind)

    # Add bounds for extrapolation
    qf, xq = extrapolate_qm(qf, xq, method=extrapolation)

    qm = xr.Dataset(
        data_vars={"mu_x": mu_x, "qf": qf, "xq": xq},
        attrs={"group": group, "group_window": window, "kind": kind},
    )
    return qm


# TODO: Add `deg` parameter and associated tests.
def predict(
    x: xr.DataArray, qm: xr.Dataset, mult_thresh: float = None, interp: str = "nearest",
):
    """
    Return a bias-corrected timeseries using the detrended quantile mapping method.

    Parameters
    ----------
    x : xr.DataArray
      Time series to be bias-corrected, usually a model output.
    qm : xr.Dataset
      Correction factors indexed by group properties and residuals of `x` over the training period, as given by the
      `dqm.train` function.
    mult_thresh : float, None
      In the multiplicative case, all values under this threshold are replaced by a non-zero random number smaller
      then the threshold. This is done to remove values that are exactly or close to 0 and create numerical
      instabilities.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method used to find the correction factors from the quantile map. See utils.interp_on_quantiles.

    Returns
    -------
    xr.DataArray
      The bias-corrected time series.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    dim, prop = parse_group(qm.group)
    kind = qm.kind

    gr = Grouper(qm.group, qm.group_window)

    # Add random noise to small values
    if kind == MULTIPLICATIVE and mult_thresh is not None:
        x = jitter_under_thresh(x, mult_thresh)

    # Compute mean correction - applied to trend
    xm = x.mean()

    # Detrend series while preserving mean
    pfit = PolyDetrend(degree=1, kind=kind).fit(x)
    xt = apply_correction(pfit.detrend(x), xm, kind)

    # Mean by period
    mu_x = gr.apply("mean", xt)

    # Add cyclical values to the scaling factors for interpolation
    if interp != "nearest" and prop is not None:
        qm["mu_x"] = add_cyclic_bounds(qm.mu_x, prop, cyclic_coords=False)
        qm["xq"] = add_cyclic_bounds(qm.xq, prop, cyclic_coords=False)
        qm["qf"] = add_cyclic_bounds(qm.qf, prop, cyclic_coords=False)
        mu_x = add_cyclic_bounds(mu_x, prop, cyclic_coords=False)

    # Adjust mean so it matches the mean of the training x.
    mf = get_correction(qm["mu_x"], mu_x, kind)
    mfx = broadcast(mf, xt, interp=interp)
    nxt = apply_correction(xt, invert(mfx, kind), kind)

    # Testing :
    # null = 0 if kind == ADDITIVE else 1
    # np.testing.assert_allclose(nx.mean(dim="time"), null, atol=1e-6)

    # Quantile mapping
    nxt = gr.add_index(nxt)
    qf = interp_on_quantiles(nxt, qm.xq, qm.qf, group=qm.attrs["group"], method=interp,)
    corrected = apply_correction(nxt, qf, kind)

    # Reapply mean
    out = apply_correction(corrected, mfx, kind)

    # Reapply trend
    out = apply_correction(pfit.retrend(out), invert(xm, kind), kind)

    out.attrs["bias_corrected"] = True

    return out
