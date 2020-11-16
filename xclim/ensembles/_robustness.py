# -*- encoding: utf8 -*-
# noqa: D205,D400
"""Ensemble Robustness metrics."""
from collections import namedtuple
from typing import Mapping, Sequence, Union

import numpy as np
import scipy.stats as spstats
import xarray as xr
from boltons.funcutils import wraps

metrics = {}
Metric = namedtuple("Metric", ["func", "mapping", "dtype", "hist_dims"])


def ensemble_robustness(hist: xr.DataArray, sims: xr.DataArray, method: str, **params):
    """Compute the robustness of an ensemble for a given indicator and its change.

    The method offered here are implementations of a subset of the "Methods to Quantify
    Model Agreement in Maps" described in box 12.1, chapter 12 of IPCC's AR5 WG1 report.

    Given historical values (as an ensemble or a single realization) and an ensemble
    of simulated future values, this function returns a map indicating how robust the
    changes represented in the ensembles are. They also usually involve  an assessment
    of the signifiance of the change when compared to internal variability.
    (See [AR5WG1C12]_).

    Parameters
    ----------
    hist : xr.DataArray
      The indicator for the reference period. Shape and source depends on the metric.
    sims : xr.DataArray
      The simulated projected values of the indicator (along `time`) for different models ('realization`).
    method : {'knutti_sedlacek', 'tebaldi_et_al'}
      The metric to compute.
    **params:
      Other metric-dependent parameters. See the method's docstring for more info.

    Returns
    -------
    metric : DataArray
      The robustness metric category as an integer (except for method `knutti_sedlacek` where a float is returned.).
    mapping : dict or None
      A mapping from values in `metric` to their signification. None for `knutti_sedlacek`.

    References
    ----------
    .. [AR5WG1C12] Collins, M., et al., 2013: Long-term Climate Change: Projections, Commitments and Irreversibility. In: Climate Change 2013: The Physical Science Basis. Contribution of Working Group I to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change [Stocker, T.F. et al. (eds.)]. Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA.
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
      Historical values along 'time' (nt).
    sims : ndarray
      Simulated values along 'realization' and 'time' (nr, nt).

    Returns
    -------
    R, float
      The robustness metric.

    References
    ----------
    .. [knutti2013] Knutti, R. and Sedláček, J. (2013) Robustness and uncertainties in the new CMIP5 climate model projections. Nat. Clim. Change. doi:10.1038/nclimate1716
    """

    def diff_cdf_sq_area_int(x1, x2):
        """Exact integral of the squared area between the non-parametric CDFs of 2 vectors."""
        y1 = (
            np.arange(x1.size) + 1
        ) / x1.size  # Non-parametric CDF on points x1 and x2
        y2 = (
            np.arange(x2.size) + 1
        ) / x2.size  # i.e. y1(x) is the proportion of x1 <= x

        x2_in_1 = np.searchsorted(x1, x2, side="right")  # Where to insert x2 in x1
        x1_in_2 = np.searchsorted(x2, x1, side="right")  # Where to insert x1 in x2

        x = np.insert(
            x1, x2_in_1, x2
        )  # Merge to get all "discontinuities" of the CDF difference
        y1_f = np.insert(
            y1, x2_in_1, np.r_[0, y1][x2_in_1]
        )  # y1 with repeated value (to the right) where x2 is inserted
        y2_f = np.insert(
            y2, x1_in_2, np.r_[0, y2][x1_in_2]
        )  # Same for y2. 0s are prepended where needed.

        # Discrete integral of the squared difference (distance) between the two CDFs.
        return np.sum(np.diff(x) * (y1_f - y2_f)[:-1] ** 2)

    # Get sorted vectors
    v_sims = np.sort(sims.flatten())  # "cumulative" models distribution
    v_means = np.sort(sims.mean(axis=-1))  # Multi-model mean
    v_hist = np.sort(hist)  # Historical values

    A1 = diff_cdf_sq_area_int(v_sims, v_means)
    A2 = diff_cdf_sq_area_int(v_hist, v_means)

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

    Compare the historical mean to the projected values for each model.
    Categories:
      - 0 : less than X% of the models show significant change.
      - 1 : models show significant change, but less then Y% agree on sign of change.
      - 2 : models show significant change and agree on sign of change.

    Parameters
    ----------
    hists : ndarray
      Multi-model historical mean along 'realization' (nr).
    sims : ndarray
      Simulated values along 'realization' and 'time' (nr, nt).
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
        Categories of agreement on the change and its sign.

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
