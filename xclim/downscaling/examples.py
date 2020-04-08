"""
===============
Examples of use
===============

Methods defined here try to be as close as possible to the published reference, providing examples of how to use the downscaling module.
"""
from .base import Grouper
from .base import parse_group
from .correction import QuantileDeltaMapping
from .correction import QuantileMapping
from .detrending import PolyDetrend
from .processing import normalize
from .utils import ADDITIVE


def dqm(
    obs,
    sim,
    fut,
    kind=ADDITIVE,
    group="time.month",
    interp="nearest",
    nquantiles=40,
    extrapolation="constant",
):
    """
    Method DQM (Detrended Quantile Mapping) as described in [Cannon2015]_.

    This method is provided as an example of how the xclim.downscaling library should be used.

    Parameters
    ----------
    obs : xr.DataArray
      Training target, usually a reference time series drawn from observations.
    sim : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    fut : : xr.DataArray
      Projected data, usually a model output in the future.
    kind : {"+", "*"}
      The type of correction, either additive (+) or multiplicative (*). Multiplicative correction factors are
      typically used with lower bounded variables, such as precipitation, while additive factors are used for
      unbounded variables, such as temperature.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping dimension and property. If only the dimension is given (e.g. 'time'), the correction is computed over
      the entire series.
    nquantiles : int
      Number of equally spaced quantile nodes. Limit nodes are added at both ends for extrapolation.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation method used when predicting on values outside the range of 'x'. See `utils.extrapolate_qm`.

    Returns
    -------
    xr.DataArray of the bias-corrected projected data

    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    detrend = PolyDetrend(degree=4, kind=kind)
    DQM = QuantileMapping(
        nquantiles=nquantiles,
        group=group,
        extrapolation=extrapolation,
        kind=kind,
        interp=interp,
    )

    sim = normalize(sim, group=group, kind=kind)
    fut = normalize(fut, group=group, kind=kind)

    DQM.train(obs, sim)

    fut_fit = detrend.fit(fut)
    fut_detrended = fut_fit.detrend(fut)

    fut_corr_detrended = DQM.predict(fut_detrended)

    fut_corr = fut_fit.retrend(fut_corr_detrended)
    return fut_corr, DQM.ds


def qdm(
    obs,
    sim,
    fut,
    kind=ADDITIVE,
    group="time.month",
    interp="nearest",
    nquantiles=40,
    extrapolation="constant",
):
    """
    Method QDM (Quantile Delta Mapping) as described in [Cannon2015]_.

    This method is provided as an example of how the xclim.downscaling library should be used.

    Parameters
    ----------
    obs : xr.DataArray
      Training target, usually a reference time series drawn from observations.
    sim : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    fut : : xr.DataArray
      Projected data, usually a model output in the future.
    kind : {"+", "*"}
      The type of correction, either additive (+) or multiplicative (*). Multiplicative correction factors are
      typically used with lower bounded variables, such as precipitation, while additive factors are used for
      unbounded variables, such as temperature.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping dimension and property. If only the dimension is given (e.g. 'time'), the correction is computed over
      the entire series.
    nquantiles : int
      Number of equally spaced quantile nodes. Limit nodes are added at both ends for extrapolation.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation method used when predicting on values outside the range of 'x'. See `utils.extrapolate_qm`.

    Returns
    -------
    xr.DataArray of the bias-corrected projected data

    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    QDM = QuantileDeltaMapping(
        nquantiles=nquantiles,
        group=group,
        extrapolation=extrapolation,
        kind=kind,
        interp=interp,
    )
    QDM.train(obs, sim)
    fut_corr = QDM.predict(fut)
    return fut_corr, QDM.ds


def eqm(
    obs,
    sim,
    fut,
    kind=ADDITIVE,
    group="time.month",
    interp="nearest",
    nquantiles=40,
    extrapolation="constant",
):
    """
    Method EQM (Empirical Quantile Mapping) as described in [Cannon2015]_.

    This method is provided as an example of how the xclim.downscaling library should be used.

    Parameters
    ----------
    obs : xr.DataArray
      Training target, usually a reference time series drawn from observations.
    sim : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    fut : : xr.DataArray
      Projected data, usually a model output in the future.
    kind : {"+", "*"}
      The type of correction, either additive (+) or multiplicative (*). Multiplicative correction factors are
      typically used with lower bounded variables, such as precipitation, while additive factors are used for
      unbounded variables, such as temperature.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping dimension and property. If only the dimension is given (e.g. 'time'), the correction is computed over
      the entire series.
    nquantiles : int
      Number of equally spaced quantile nodes. Limit nodes are added at both ends for extrapolation.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation method used when predicting on values outside the range of 'x'. See `utils.extrapolate_qm`.

    Returns
    -------
    xr.DataArray of the bias-corrected projected data

    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping:
    How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959.
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """
    EQM = QuantileMapping(
        nquantiles=nquantiles,
        group=group,
        extrapolation=extrapolation,
        kind=kind,
        interp=interp,
    )
    EQM.train(obs, sim)
    fut_corr = EQM.predict(fut)
    return fut_corr, EQM.ds
