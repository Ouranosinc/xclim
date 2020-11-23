# -*- encoding: utf8 -*-
# noqa: D205,D400
"""Ensemble Robustness metrics."""
from collections import namedtuple
from typing import Mapping, Sequence, Union

import numpy as np
import scipy.stats as spstats
import xarray as xr
from boltons.funcutils import wraps

from xclim.core.formatting import update_history

metrics = {}
Metric = namedtuple("Metric", ["func", "mapping", "dtype", "hist_dims"])


def robustness_map(stats, method: str, **params):
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
    stats : xr.Dataset
      Dataset with the two robustness statistics as calculated with ensemble_robustness.
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
    pass


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


def ensemble_robustness(
    ref: xr.DataArray, fut: xr.DataArray, stat: str = "ttest", **kwargs
):
    """Robustness statistics qualifying how the members of an ensemble agree on the existence of change and on its sign.

    Parameters
    ----------
    ref : xr.DataArray
      Reference period values along 'time' and 'realization'  (nt, nr).
    fut : xr.DataArray
      Future period values along 'time' and 'realization' (nt, nr).
    stat : {'ttest'}
      Name of the statistical test used to determine if there was significant change.
    **kwargs
      Other arguments specific to the statistical test.

      For 'ttest':
        p_change : float (default : 0.05)
          p-value threshold for rejecting the hypothesis of no significant change.

    Returns
    -------
    xr.Dataset
        Dataset with the following variables:
            change_frac: The proportion of members that show significant change. (0..1)
            sign_frac : The proportion of member showing significant change that agree on its sign. (0..1)


    References
    ----------
    Tebaldi C., Arblaster, J.M. and Knutti, R. (2011) Mapping model agreement on future climate projections. GRL. doi:10.1029/2011GL049863
    """
    test_params = {"ttest": ["p_change"]}
    if stat == "ttest":
        p_change = kwargs.set_default("p_change", 0.05)

        # Test hypothesis of no significant change
        pvals = xr.apply_ufunc(
            lambda f, r: spstats.ttest_1samp(f, r)[1],
            fut,
            ref.mean("time"),
            input_core_dims=[["realization", "time"], ["realization"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals > p_change
        fut_chng = fut.where(changed)
        ref_chng = ref.where(changed)
    else:
        raise ValueError(
            f"Statistical test {stat} must be one of {', '.join(test_params.keys())}."
        )
    change_frac = changed.sum("realization") / fut.realization.size

    # Test that models agree on the sign of the change
    pos_frac = ((fut_chng.mean("time") - ref_chng.mean("time")) > 0).sum(
        "realization"
    ) / fut.realization.size
    sign_frac = xr.concat((pos_frac, 1 - pos_frac), "sign").max("sign")

    # Metadata
    kwargs_str = ", ".join(
        [f"{k}: {v}" for k, v in kwargs.items() if k in test_params[stat]]
    )
    test_str = (
        f"Significant change was tested with test {stat} with parameters {kwargs_str}."
    )
    sign_frac.attrs.update(
        description="Fraction of members showing significant change that agree on the sign of change. "
        + test_str,
        units="",
    )
    change_frac.attrs.update(
        description="Fraction of members showing significant change. " + test_str,
        units="",
    )
    return xr.Dataset(
        data_vars={"sign_frac": sign_frac, "change_frac": change_frac},
        attrs={
            "history": update_history(
                f"ensemble_robustness with test {stat}, {kwargs_str}", ref, fut
            )
        },
    )
