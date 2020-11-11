# -*- encoding: utf8 -*-
# noqa: D205,D400
"""
Robustness metrics
==================
"""
from collections import namedtuple
from typing import Mapping, Sequence, Union

import numpy as np
import scipy.stats as spstats
import xarray as xr
from boltons.funcutils import wraps
from scipy.integrate import quad

import xclim.indices.stats as xcstats

metrics = {}
norm = xcstats.get_dist("norm")
Metric = namedtuple("Metric", ["func", "mapping", "dtype", "hist_dims"])


def ensemble_robustness(hist: xr.DataArray, sims: xr.DataArray, method: str, **params):
    """Compute the robustness of an ensemble for a given indicator and change.

    Parameters
    ----------
    hist : xr.DataArray
      The indicator for the reference period. Shape and source depends on the metric.
    sims : xr.DataArray
      The simulated projected values of the indicator (along `time`) for different models ('realization`).
    methodc : {'knutti_sedlacek', 'tebaldi_et_al'}
      The metric to compute.
    **params:
      Other metric-dependent parameters. See the method's docstring for more info.

    Returns
    -------
    metric : DataArray
      The robustness metric category as an integer (except for method `knutti_sedlacek` where a float is returned.).
    mapping : dict or None
      A mapping from values in `metric` to their signification. None for `knutti_sedlacek`.
    """

    try:
        metric = metrics[method]
    except KeyError:
        raise ValueError(
            f"Method {method} is not implemented. Available methods are : {','.join(metrics.keys())}."
        )

    out = xr.apply_ufunc(
        metric.func,
        hist,
        sims,
        input_core_dims=[metric.hist_dims, ("realization", "time")],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[metric.dtype],
        kwargs=params,
    )

    out.name = "robustness"
    out.attrs.update(
        long_name=f"Ensemble robustness according to method {method}",
        method=method,
    )
    return out, metric.mapping


def metric(
    mapping: Mapping[int, str] = None,
    na_value: Union[int, float] = 999,
    hist_dims: Sequence[str] = ["realization", "time"],
):
    """Register a metric function in the `metrics` mapping and add some preparation/checking code.

    Parameters
    ----------
    mapping : Mapping[int, str], optional
      A mapping from a robustness category to a short english description.
    na_value : Union[int, float]
      The value returned when any input value is NaN. It is added (if not present),
      to the mapping with the description "Invalid values - no metric."
    hist_dims : {['realization'], ['time'], ['realization', 'time']}
      The dimension(s) expected on hist, used in the shape check.

    All metric functions are invalid when any non-finite values are present in the inputs.
    """

    def _metric(func):
        @wraps(func)
        def _metric_overhead(hist, sims, *args, **kwargs):
            if np.any(np.isnan(hist)) or np.any(np.isnan(sims)):
                return na_value

            if (
                (hist_dims == ["realization"] and hist.shape[0] != sims.shape[0])
                or (hist_dims == ["time"] and hist.shape[0] != sims.shape[1])
                or (hist_dims == ["realization", "time"] and hist.shape != sims.shape)
            ):
                raise AttributeError("Shape mismatch")

            return func(hist, sims, *args, **kwargs)

        if mapping is not None:
            mapping.setdefault(na_value, "Invalid values - no metric.")
        metrics[func.__name__] = Metric(
            _metric_overhead, mapping, type(na_value), hist_dims
        )
        return _metric_overhead

    return _metric


@metric(na_value=np.nan, hist_dims=["time"])
def knutti_sedlacek(hist, sims):
    """Robustness metric from Knutti and Sedlacek (2013).

    The robustness metric is defined as R = 1 − A1 / A2 , where A1 is defined
    as the integral of the squared area between two cumulative density functions
    characterizing the individual model projections and the multi-model mean
    projection and A2 is the integral of the squared area between two cumulative
    density functions characterizing the multi-model mean projection and the historical
    climate. (Description taken from [knutti2013]_)

    A value of R equal to one implies perfect model agreement. Higher model spread or
    smaller signal decreases the value of R.

    Parameters
    ----------
    hist : ndarray
      1D Array, historical values along 'time'.
    sims : ndarray
      2D Array, simulated values along 'realization' and 'time'.

    Returns
    -------
    R, float
      The robustness metric.

    References
    ----------
    .. [knnuti2013] Knutti, R. and Sedláček, J. (2013) Robustness and uncertainties in the new CMIP5 climate model projections. Nat. Clim. Change. doi:10.1038/nclimate1716
    """

    def cum_cdf(x, ls):  # Cumulative CDF renormalized
        return sum([norm.cdf(x, loc=loc, scale=sc) for loc, sc in ls]) / len(ls)

    def diff_cum_cdf_area_sq(
        x, sims_ls, l2, s2
    ):  # Squared difference between cumcdf and cdf
        return (cum_cdf(x, sims_ls) - norm.cdf(x, loc=l2, scale=s2)) ** 2

    def diff_cdf_area_sq(x, l1, s1, l2, s2):  # Squared difference between 2 cdfs
        return (norm.cdf(x, loc=l1, scale=s1) - norm.cdf(x, loc=l2, scale=s2)) ** 2

    # Get gaussian parameters
    sims_ls = []
    for r in range(sims.shape[0]):  # For each model
        sims_ls.append(norm.fit(sims[r, ...]))
    ls, ss = norm.fit(sims.mean(axis=-1).flatten())  # For the multi-model mean
    lh, sh = norm.fit(hist.flatten())  # For the historical mean

    int_bnds = norm.ppf(0.0001, loc=ls, scale=ss), norm.ppf(0.9999, loc=ls, scale=ss)
    A1 = quad(diff_cum_cdf_area_sq, *int_bnds, args=(sims_ls, ls, ss))[0]
    A2 = quad(diff_cdf_area_sq, *int_bnds, args=(lh, sh, ls, ss))[0]

    return 1 - A1 / A2


@metric(
    {
        0: "No significant change",
        1: "No agreement on sign of change",
        2: "Agreement on sign of change",
    },
    hist_dims=["realization"],
)
def tebaldi_et_al(hist, sims, X=0.5, Y=0.8, p_change=0.05):
    """Robustness categories from Tebaldi et al. (2011).

    Compares the historical mean to the projected values for each model.
    Category 0 is if less than X% of the models show significant change.
    Category 1 is if models show significant change but less then Y% agree on its sign.
    Category 2 is if models show significant change and agree on its sign.

    Parameters
    ----------
    hists : ndarray
      1D Array, multi-model historical mean along 'realization'.
    sims : ndarray
      2D Array, simulated values along 'realization' and 'time'.
    X : float, 0 .. 1
      Threshold fraction of models agreeing on significant change.
    Y : float, 0 .. 1
      Threshold fraction of models agreeing on the sign of the change.
    p_change : float
      p-value threshold for rejecting the hypothesis of no significant
      change in the t-test.

    Returns
    -------
    int
      0 for no significant change, 1 for change but no agreement on sign of change
      and 2 for change and agreement on sign.

    References
    ----------
    Tebaldi C., Arblaster, J.M. and Knutti, R. (2011) Mapping model agreement on future climate projections. GRL. doi:10.1029/2011GL049863
    """
    # Test hypothesis of no significant change
    t, p = spstats.ttest_1samp(sims, hist, axis=1)

    # When p < p_change, the hypothesis of no significant change is rejected.
    # We need at least X % models showing significant change
    if (p < p_change).sum() / hist.size <= X:
        return 0  # No significant change

    # Test that models agree on the sign of the change
    change_sign = np.sign(sims.mean(axis=1) - hist).clip(0, 1)
    if (1 - Y) <= change_sign.sum() / hist.size <= Y:
        return 1  # Significant change but no agreement on sign of change
    return 2  # Significant change and agreement on sign of change
