"""
Quantile delta mapping
======================

Applies model-projected changes in quantiles.
The logic is very similar to Empirical quantile mapping, but it is applied as a delta instead of as bias-correction
factor, and these factors are indexed by CDF instead of values.

"""
import xarray as xr

from .utils import add_cyclic_bounds
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import broadcast
from .utils import equally_spaced_nodes
from .utils import extrapolate_qm
from .utils import get_correction
from .utils import group_apply
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
    Return the quantile mapping factors using the quantile delta mapping method.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output over a reference period.
    y : xr.DataArray
      Training target, usually a model output over a future period.
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
        - xq : The quantile values.

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

    # Compute quantile per period
    xq = group_apply("quantile", x, group, window=window, q=q)
    yq = group_apply("quantile", y, group, window=window, q=q)

    # Compute quantile correction factors
    qf = get_correction(xq, yq, kind)  # qy / qx or qy - qx

    qm = xr.Dataset(
        data_vars={"xq": xq, "qf": qf},
        attrs={"group": group, "group_window": window, "kind": kind},
    )
    qm = qm.rename(quantile="quantiles")

    # Add bounds for extrapolation
    qm["qf"], qm["xq"] = extrapolate_qm(qm.qf, qm.xq, method=extrapolation)
    return qm


def predict(
    x: xr.DataArray, qm: xr.Dataset, mult_thresh: float = None, interp: str = "nearest",
):
    """
    Apply quantile delta to series.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Time series delta is applied to.
    qm : xr.Dataset
      Delta factors indexed by group properties and quantiles, as given by the `qdm.train` function.
    mult_thresh : float, None
      In the multiplicative case, all values under this threshold are replaced by a non-zero random number smaller
      then the threshold. This is done to remove values that are exactly or close to 0 and create numerical
      instabilities.
    interp : {'nearest', 'linear'}
      The interpolation method used to find the correction factors from the quantile map. See utils.broadcast.

    Returns
    -------
    xr.DataArray
      The time series with delta applied.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    dim, prop = parse_group(qm.group)
    window = qm.group_window
    kind = qm.kind

    # Add random noise to small values
    if kind == MULTIPLICATIVE and mult_thresh is not None:
        x = jitter_under_thresh(x, mult_thresh)

    # Add cyclical values to the scaling factors for interpolation
    if interp != "nearest" and prop is not None:
        qm["qf"] = add_cyclic_bounds(qm.qf, prop, cyclic_coords=False)

    # Compute quantile of x
    xq = group_apply(xr.DataArray.rank, x, qm.group, window=window, pct=True)

    # Quantile mapping
    sel = {"quantiles": xq}
    qf = broadcast(qm.qf, x, interp=interp, sel=sel)
    out = apply_correction(x, qf, qm.kind)

    out.attrs["bias_corrected"] = True
    return out
