# -*- encoding: utf8 -*-
# noqa: D205,D400
"""
Robustness metrics
==================
"""
import numpy as np
import xarray as xr
from scipy.integrate import quad

import xclim.indices.stats as xcstats


def knutti_sedlacek(hist, sims):
    """Compute the robustness measure R as described by Knutti and Sedláček (2013).

    Parameters
    ----------
    hist : xr.DataArray
      Array with the historical values of the indicator along 'time'.
    sims : xr.DataArray
      Array with the simulated projected values of the indicators for different models ('ensembles') along 'time'.

    Returns
    -------
    DataArray
      The robustness metric R.
    """
    norm = xcstats.get_dist("norm")

    out = xr.apply_ufunc(
        _knutti_sedlacek,
        hist,
        sims,
        input_core_dims=[("realization",), ("realization", "time")],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"dist": norm},
    )
    return out


def _knutti_sedlacek(hist, sims, dist):
    """Pointwise computation function."""

    def diff_cdf_area(x, l1, s1, l2, s2, dist):
        return (dist.cdf(x, loc=l1, scale=s1) - dist.cdf(x, loc=l2, scale=s2)) ** 2

    l1, s1 = dist.fit(sims.flatten())
    l2, s2 = dist.fit(sims.mean(axis=-1).flatten())
    lh, sh = dist.fit(hist.flatten())

    int_bnds = dist.ppf(0.0001, loc=l1, scale=s1), dist.ppf(0.9999, loc=l1, scale=s1)
    A1 = quad(diff_cdf_area, *int_bnds, args=(l1, s1, l2, s2, dist))[0]
    A2 = quad(diff_cdf_area, *int_bnds, args=(lh, sh, l2, s2, dist))[0]

    return 1 - A1 / A2
