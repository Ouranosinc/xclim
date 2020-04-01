"""
Empirical quantile mapping
==========================

Quantiles from `x` are mapped onto quantiles from `y`.

"""
import xarray as xr

from .utils import add_cyclic_bounds
from .utils import ADDITIVE
from .utils import adjust_freq
from .utils import apply_correction
from .utils import equally_spaced_nodes
from .utils import extrapolate_qm
from .utils import get_correction
from .utils import get_index
from .utils import group_apply
from .utils import interp_on_quantiles
from .utils import MULTIPLICATIVE
from .utils import parse_group


def train(
    x,
    y,
    kind=ADDITIVE,
    group="time.dayofyear",
    window=1,
    nq=40,
    thresh=None,
    extrapolation="constant",
):
    """
    Return the quantile mapping factors using the empirical quantile mapping method.

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
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is
      mostly
      used with group `time.dayofyear` to increase the number of samples.
    nq : int
      Number of equally spaced quantile nodes. Limit nodes are added at both ends for extrapolation.
    thresh : float, None
      Value below which data for `y` is considered null (only used for multiplicative correction).
    extrapolation : {'constant', 'nan'}
      The type of extrapolation method used when predicting on values outside the range of 'x'. See
      `utils.extrapolate_qm`.

    Returns
    -------
    xr.Dataset with variables:
        - qf : The correction factors indexed by group properties and quantiles.
        - xq : The quantile values

        The type of correction used is stored in the "kind" attribute and grouping informations are in the
        "group" and "group_window" attributes.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
   """
    q = equally_spaced_nodes(nq, eps=1e-6)

    if kind == MULTIPLICATIVE and thresh is not None:
        y = adjust_freq(y, x, thresh, group)

    xq = group_apply("quantile", x, group, window, q=q).rename(quantile="quantiles")
    yq = group_apply("quantile", y, group, window, q=q).rename(quantile="quantiles")

    qf = get_correction(xq, yq, kind)

    # Add bounds for extrapolation
    qf, xq = extrapolate_qm(qf, xq, method=extrapolation)

    qm = xr.Dataset(
        data_vars={"xq": xq, "qf": qf},
        attrs={"group": group, "group_window": window, "kind": kind},
    )
    return qm


def predict(x: xr.DataArray, qm: xr.Dataset, interp: str = "nearest"):
    """
    Return a bias-corrected timeseries using the detrended quantile mapping method.

    This method acts on a single point (timeseries) only.

    Parameters
    ----------
    x : xr.DataArray
      Time series to be bias-corrected, usually a model output.
    qm : xr.Dataset
      Correction factors indexed by group properties and residuals of `x` over the training period, as given by the
      `eqm.train` function.
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

    # Add cyclical values to the scaling factors for interpolation
    if interp != "nearest" and prop is not None:
        qm["qf"] = add_cyclic_bounds(qm.qf, prop)

    if prop is not None:
        x = x.assign_coords({prop: get_index(x, dim, prop, interp)})

    # Broadcast correction factors onto x
    qf = interp_on_quantiles(x, qm.xq, qm.qf, group=qm.group, method=interp)

    # Apply the correction factors
    out = apply_correction(x, qf, qm.kind)

    out.attrs["bias_corrected"] = True
    return out
