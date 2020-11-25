# -*- encoding: utf8 -*-
# noqa: D205,D400
"""
Ensemble Robustness metrics.
============================

Robustness metrics are used to estimate the confidence of the climate change signal
of an ensemble. This submodule is inspired by and tries to follow the guidelines of
the IPCC, more specifically the 12th chapter of the Working Group 1's contribution to
the AR5 [AR5WG1C12]_ (see box 12.1).

References
----------
.. [AR5WG1C12] https://www.ipcc.ch/site/assets/uploads/2018/02/WG1AR5_Chapter12_FINAL.pdf
"""
import numpy as np
import scipy.stats as spstats
import xarray as xr

from xclim.core.formatting import update_history


def change_significance(
    ref: xr.DataArray, fut: xr.DataArray, test: str = "ttest", **kwargs
) -> xr.Dataset:
    """Robustness statistics qualifying how the members of an ensemble agree on the existence of change and on its sign.

    Parameters
    ----------
    ref : xr.DataArray
      Reference period values along 'time' and 'realization'  (nt1, nr).
    fut : xr.DataArray
      Future period values along 'time' and 'realization' (nt2, nr). The size of the
      'time' axis does not need to match the one of `ref`. But their 'realization' axes
      must be identical.
    test : {'ttest', 'welch-ttest', None}
      Name of the statistical test used to determine if there was significant change. See notes.
    **kwargs
      Other arguments specific to the statistical test.

      For 'ttest' and 'welch-ttest':
        p_change : float (default : 0.05)
          p-value threshold for rejecting the hypothesis of no significant change.

    Returns
    -------
    change_frac: DataArray
      The fraction of members that show significant change [0, 1].
      Passing `test=None` yields change_frac = 1 everywhere,
      or NaN where any of `ref` or `fut` was NaN.
    sign_frac : DataArray
      The fraction of members showing significant change that agree on its sign [0, 1].

    Notes
    -----
    Available statistical tests are :

      'ttest' :
        Single sample T-test. Same test as used by [tebaldi2011]_. The future
        values are compared against the reference mean (over 'time'). Change is qualified
        as 'significant' when the test's p-value is below the user-provided `p_change`
        value.
      'welch-ttest' :
         Two-sided T-test, without assuming equal population variance. Same
        significance criterion as 'ttest'.
      None :
        Significant change is not tested and, thus, members showing no change are
        included in the `sign_frac` output.

    References
    ----------
    .. [tebaldi2011] Tebaldi C., Arblaster, J.M. and Knutti, R. (2011) Mapping model agreement on future climate projections. GRL. doi:10.1029/2011GL049863
    """
    test_params = {"ttest": ["p_change"], "welch-ttest": ["p_change"]}
    if test == "ttest":
        p_change = kwargs.setdefault("p_change", 0.05)

        # Test hypothesis of no significant change
        pvals = xr.apply_ufunc(
            lambda f, r: spstats.ttest_1samp(f, r, axis=-1)[1],
            fut,
            ref.mean("time"),
            input_core_dims=[["realization", "time"], ["realization"]],
            output_core_dims=[["realization"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change
        mask = pvals.isnull().any("realization")
    elif test == "welch-ttest":
        p_change = kwargs.setdefault("p_change", 0.05)

        # Test hypothesis of no significant change
        # equal_var=False -> Welch's T-test
        pvals = xr.apply_ufunc(
            lambda f, r: spstats.ttest_ind(f, r, axis=-1, equal_var=False)[1],
            fut,
            ref,
            input_core_dims=[["realization", "time"], ["realization", "time"]],
            output_core_dims=[["realization"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change
        mask = pvals.isnull().any("realization")
    elif test is not None:
        raise ValueError(
            f"Statistical test {test} must be one of {', '.join(test_params.keys())}."
        )

    if test is not None:
        fut_chng = fut.where(changed & ~mask)
        ref_chng = ref.where(changed & ~mask)
        change_frac = (changed.sum("realization") / fut.realization.size).where(~mask)
    else:
        fut_chng = fut
        ref_chng = ref
        change_frac = xr.ones_like(ref.isel(time=0, realization=0)).where(
            fut.notnull().all(["time", "realization"])
        )

    # Test that models agree on the sign of the change
    pos_frac = ((fut_chng.mean("time") - ref_chng.mean("time")) > 0).sum(
        "realization"
    ) / (change_frac * fut.realization.size)
    sign_frac = xr.concat((pos_frac, 1 - pos_frac), "sign").max("sign")

    # Metadata
    kwargs_str = ", ".join(
        [f"{k}: {v}" for k, v in kwargs.items() if k in test_params[test]]
    )
    test_str = (
        f"Significant change was tested with test {test} with parameters {kwargs_str}."
    )
    sign_frac.attrs.update(
        description="Fraction of members showing significant change that agree on the sign of change. "
        + test_str,
        units="",
        test=test,
        xclim_history=update_history(
            f"sign_frac from change_significance(ref=ref, fut=fut, test={test}, {kwargs_str})",
            ref=ref,
            fut=fut,
        ),
    )
    change_frac.attrs.update(
        description="Fraction of members showing significant change. " + test_str,
        units="",
        test=test,
        xclim_history=update_history(
            f"change_frac from change_significance(ref=ref, fut=fut, test={test}, {kwargs_str})",
            ref=ref,
            fut=fut,
        ),
    )
    return change_frac, sign_frac


def knutti_sedlacek(ref: xr.DataArray, fut: xr.DataArray) -> xr.Dataset:
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
    ref : DataArray
      Reference period values along 'time' (nt).
    fut : DataArray
      Future ensemble values along 'realization' and 'time' (nr, nt).

    Returns
    -------
    R, float
      The robustness metric.

    References
    ----------
    .. [knutti2013] Knutti, R. and Sedláček, J. (2013) Robustness and uncertainties in the new CMIP5 climate model projections. Nat. Clim. Change. doi:10.1038/nclimate1716
    """

    def _knutti_sedlacek(ref, fut):
        def diff_cdf_sq_area_int(x1, x2):
            """Exact integral of the squared area between the non-parametric CDFs of 2 vectors."""
            # Non-parametric CDF on points x1 and x2
            # i.e. y1(x) is the proportion of x1 <= x
            y1 = (np.arange(x1.size) + 1) / x1.size
            y2 = (np.arange(x2.size) + 1) / x2.size

            x2_in_1 = np.searchsorted(x1, x2, side="right")  # Where to insert x2 in x1
            x1_in_2 = np.searchsorted(x2, x1, side="right")  # Where to insert x1 in x2

            # Merge to get all "discontinuities" of the CDF difference
            # y1 with repeated value (to the right) where x2 is inserted
            # Same for y2. 0s are prepended where needed.
            x = np.insert(x1, x2_in_1, x2)
            y1_f = np.insert(y1, x2_in_1, np.r_[0, y1][x2_in_1])
            y2_f = np.insert(y2, x1_in_2, np.r_[0, y2][x1_in_2])

            # Discrete integral of the squared difference (distance) between the two CDFs.
            return np.sum(np.diff(x) * (y1_f - y2_f)[:-1] ** 2)

        # Get sorted vectors
        v_fut = np.sort(fut.flatten())  # "cumulative" models distribution
        v_favg = np.sort(fut.mean(axis=-1))  # Multi-model mean
        v_ref = np.sort(ref)  # Historical values

        A1 = diff_cdf_sq_area_int(v_fut, v_favg)
        A2 = diff_cdf_sq_area_int(v_ref, v_favg)

        return 1 - A1 / A2

    R = xr.apply_ufunc(
        _knutti_sedlacek,
        ref,
        fut,
        input_core_dims=[["time"], ["realization", "time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    R.attrs.update(
        name="R",
        long_name="Ensemble robustness metric",
        description="Ensemble robustness metric as defined by Knutti and Sedláček (2013).",
        reference="Knutti, R. and Sedláček, J. (2013) Robustness and uncertainties in the new CMIP5 climate model projections. Nat. Clim. Change.",
        units="",
        xclim_history=update_history("knutti_sedlacek(ref, fut)", ref=ref, fut=fut),
    )
    return R
