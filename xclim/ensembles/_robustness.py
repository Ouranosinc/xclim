"""
Ensemble Robustness Metrics
===========================

Robustness metrics are used to estimate the confidence of the climate change signal of an ensemble.
This submodule is inspired by and tries to follow the guidelines of the IPCC, more specifically
the 12th chapter of the Working Group 1's contribution to the AR5 :cite:p:`collins_long-term_2013` (see box 12.1).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy
import scipy.stats as spstats  # noqa
import xarray as xr
from pkg_resources import parse_version

from xclim.core.formatting import update_history


def change_significance(
    fut: xr.DataArray | xr.Dataset,
    ref: xr.DataArray | xr.Dataset = None,
    test: str | None = "ttest",
    weights: xr.DataArray = None,
    p_vals: bool = False,
    **kwargs,
) -> (
    tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]
    | tuple[
        xr.DataArray | xr.Dataset,
        xr.DataArray | xr.Dataset,
        xr.DataArray | xr.Dataset | None,
    ]
):
    r"""Robustness statistics qualifying how members of an ensemble agree on the existence of change and on its sign.

    Parameters
    ----------
    fut : xr.DataArray or xr.Dataset
        Future period values along 'realization' and 'time' (..., nr, nt1)
        or if `ref` is None, Delta values along `realization` (..., nr).
    ref : Union[xr.DataArray, xr.Dataset], optional
        Reference period values along realization' and 'time'  (..., nt2, nr).
        The size of the 'time' axis does not need to match the one of `fut`.
        But their 'realization' axes must be identical.
        If `None` (default), values of `fut` are assumed to be deltas instead of
        a distribution across the future period.
        `fut` and `ref` must be of the same type (Dataset or DataArray). If they are
        Dataset, they must have the same variables (name and coords).
    test : {'ttest', 'welch-ttest', 'mannwhitney-utest', 'brownforsythe-test', 'threshold', None}
        Name of the statistical test used to determine if there was significant change. See notes.
    weights : xr.DataArray
        Weights to apply along the 'realization' dimension. This array cannot contain missing values.
        Only tests "threshold" and "None" are currently supported with weighted arrays.
    p_vals : bool
        If True, return the estimated p-values.
    \*\*kwargs
        Other arguments specific to the statistical test.

        For 'ttest', 'welch-ttest', 'mannwhitney-utest' and 'brownforsythe-test':
            p_change : float (default : 0.05)
                p-value threshold for rejecting the hypothesis of no significant change.
        For 'threshold': (Only one of those must be given.)
            abs_thresh : float (no default)
                Threshold for the (absolute) change to be considered significative.
            rel_thresh : float (no default, in [0, 1])
                Threshold for the relative change (in reference to ref) to be significative.
                Only valid if `ref` is given.

    Returns
    -------
    change_frac :  xr.DataArray or xr.Dataset
        The fraction of members that show significant change [0, 1].
        Passing `test=None` yields change_frac = 1 everywhere. Same type as `fut`.
    pos_frac : xr.DataArray or xr.Dataset
        The fraction of members showing significant change that show a positive change ]0, 1].
        Null values are returned where no members show significant change.
    pvals [Optional] : xr.DataArray or xr.Dataset or None
        The p-values estimated by the significance tests. Only returned if `p_vals` is True. None
        if `test` is one of 'ttest', 'welch-ttest', 'mannwhitney-utest' or 'brownforsythe-test'.

        The table below shows the coefficient needed to retrieve the number of members
        that have the indicated characteristics, by multiplying it to the total
        number of members (`fut.realization.size`).

        +-----------------+------------------------------+------------------------+
        |                 | Significant change           | Non-significant change |
        +-----------------+------------------------------+------------------------+
        | Any direction   | change_frac                  | 1 - change_frac        |
        +-----------------+------------------------------+------------------------+
        | Positive change | pos_frac * change_frac       | N.A.                   |
        +-----------------+------------------------------+                        |
        | Negative change | (1 - pos_frac) * change_frac |                        |
        +-----------------+------------------------------+------------------------+

    Notes
    -----
    Available statistical tests are :

      'ttest' :
        Single sample T-test. Same test as used by :cite:t:`tebaldi_mapping_2011`.
        The future values are compared against the reference mean (over 'time').
        Change is qualified as 'significant' when the test's p-value is below the user-provided `p_change` value.
      'welch-ttest' :
        Two-sided T-test, without assuming equal population variance. Same significance criterion as 'ttest'.
      'mannwhitney-utest' :
        Two-sided Mann-Whiney U-test. Same significance criterion as 'ttest'.
      'brownforsythe-test' :
        Brown-Forsythe test assuming skewed, non-normal distributions. Same significance criterion as 'ttest'.
      'threshold' :
        Change is considered significative if the absolute delta exceeds a given threshold (absolute or relative).
      None :
        Significant change is not tested and, thus, members showing no change are
        included in the `sign_frac` output.

    References
    ----------
    :cite:cts:`tebaldi_mapping_2011`

    Example
    -------
    This example computes the mean temperature in an ensemble and compares two time
    periods, qualifying significant change through a single sample T-test.

    >>> from xclim import ensembles
    >>> ens = ensembles.create_ensemble(temperature_datasets)
    >>> tgmean = xclim.atmos.tg_mean(tas=ens.tas, freq="YS")
    >>> fut = tgmean.sel(time=slice("2020", "2050"))
    >>> ref = tgmean.sel(time=slice("1990", "2020"))
    >>> chng_f, pos_f = ensembles.change_significance(fut, ref, test="ttest")

    If the deltas were already computed beforehand, the 'threshold' test can still
    be used, here with a 2 K threshold.

    >>> delta = fut.mean("time") - ref.mean("time")
    >>> chng_f, pos_f = ensembles.change_significance(
    ...     delta, test="threshold", abs_thresh=2
    ... )
    """
    # Realization dimension name
    realization = "realization"

    # Assign dummy realization dimension if not present.
    if realization not in fut.dims:
        fut = fut.assign_coords({realization: "dummy"})
        fut = fut.expand_dims(realization)
    if ref is not None and realization not in ref.dims:
        ref = ref.assign_coords({realization: "dummy"})
        ref = ref.expand_dims(realization)

    # Get dummy weights to simplify code
    if weights is not None:
        w = weights
    else:
        w = xr.DataArray(
            [1] * fut[realization].size,
            dims=(realization,),
            coords={"realization": fut[realization]},
        )

    # Significance tests parameter names
    test_params = {
        "ttest": ["p_change"],
        "welch-ttest": ["p_change"],
        "mannwhitney-utest": ["p_change"],
        "brownforsythe-test": ["p_change"],
        "threshold": ["abs_thresh", "rel_thresh"],
    }

    # Get delta, either from fut or from fut - ref
    changed = None
    if ref is None:
        delta = fut
        n_valid_real = w.where(delta.notnull()).sum(realization)
        if test not in ["threshold", None]:
            raise ValueError(
                "When deltas are given (ref=None), 'test' must be one of ['threshold', None]"
            )
    else:
        delta = fut.mean("time") - ref.mean("time")
        n_valid_real = w.where(fut.notnull().all("time")).sum(realization)

    pvals = None
    if test == "ttest":
        if weights is not None:
            raise NotImplementedError(
                "'ttest' is not currently supported for weighted arrays."
            )
        p_change = kwargs.setdefault("p_change", 0.05)

        if parse_version(scipy.__version__) < parse_version("1.9.0"):
            warnings.warn(
                "`xclim` will be dropping support for `scipy<1.9.0` in a future release. "
                "Please consider updating your environment dependencies accordingly",
                FutureWarning,
                stacklevel=3,
            )

            def _ttest_func(f, r):
                if np.isnan(f).all() or np.isnan(r).all():
                    return np.NaN

                return spstats.ttest_1samp(f, r, axis=-1, nan_policy="omit")[1]

        else:

            def _ttest_func(f, r):
                # scipy>=1.9: popmean.axis[-1] must equal 1 for both fut and ref
                if np.isnan(f).all() or np.isnan(r).all():
                    return np.NaN

                return spstats.ttest_1samp(
                    f, r[..., np.newaxis], axis=-1, nan_policy="omit"
                )[1]

        # Test hypothesis of no significant change
        pvals = xr.apply_ufunc(
            _ttest_func,
            fut,
            ref.mean("time"),
            input_core_dims=[["time"], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change

    elif test == "welch-ttest":
        if weights is not None:
            raise NotImplementedError(
                "'welch-ttest' is not currently supported for weighted arrays."
            )
        p_change = kwargs.setdefault("p_change", 0.05)

        # Test hypothesis of no significant change
        # equal_var=False -> Welch's T-test
        def wtt_wrapper(f, r):  # This specific test can't manage an all-NaN slice
            if np.isnan(f).all() or np.isnan(r).all():
                return np.NaN
            return spstats.ttest_ind(f, r, axis=-1, equal_var=False, nan_policy="omit")[
                1
            ]

        pvals = xr.apply_ufunc(
            wtt_wrapper,
            fut,
            ref,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[]],
            exclude_dims={"time"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change
    elif test == "mannwhitney-utest":
        if weights is not None:
            raise NotImplementedError(
                "'mannwhitney-utest' is not currently supported for weighted arrays."
            )
        if parse_version(scipy.__version__) < parse_version("1.8.0"):
            raise ImportError(
                "The Mann-Whitney test requires `scipy>=1.8.0`. "
                "`xclim` will be dropping support for `scipy<1.9.0` in a future release. "
                "Please consider updating your environment dependencies accordingly"
            )

        p_change = kwargs.setdefault("p_change", 0.05)

        # Test hypothesis of no significant change
        # -> Mann-Whitney U-test

        def mwu_wrapper(f, r):  # This specific test can't manage an all-NaN slice
            if np.isnan(f).all() or np.isnan(r).all():
                return np.NaN
            return spstats.mannwhitneyu(f, r, axis=-1, nan_policy="omit")[1]

        pvals = xr.apply_ufunc(
            mwu_wrapper,
            fut,
            ref,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[]],
            exclude_dims={"time"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change
    elif test == "brownforsythe-test":
        if weights is not None:
            raise NotImplementedError(
                "'brownforsythe-test' is not currently supported for weighted arrays."
            )

        p_change = kwargs.setdefault("p_change", 0.05)
        # Test hypothesis of no significant change
        # -> Brown-Forsythe test
        pvals = xr.apply_ufunc(
            lambda f, r: spstats.levene(f, r, center="median")[1],
            fut,
            ref,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[]],
            exclude_dims={"time"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # When p < p_change, the hypothesis of no significant change is rejected.
        changed = pvals < p_change
    elif test == "threshold":
        if "abs_thresh" in kwargs and "rel_thresh" not in kwargs:
            changed = abs(delta) > kwargs["abs_thresh"]
        elif "rel_thresh" in kwargs and "abs_thresh" not in kwargs and ref is not None:
            changed = abs(delta / ref.mean("time")) > kwargs["rel_thresh"]
        else:
            raise ValueError("Invalid argument combination for test='threshold'.")

    elif test is not None:
        raise ValueError(
            f"Statistical test {test} must be one of {', '.join(test_params.keys())}."
        )

    # Compute `change_frac`: ratio of realizations with significant changes.
    if test is not None:
        delta_chng = delta.where(changed)
        change_frac = changed.weighted(w).sum(realization) / n_valid_real
    else:
        delta_chng = delta
        change_frac = xr.ones_like(delta.isel({realization: 0}))

    # Test that models agree on the sign of the change
    # This returns NaN (cause 0 / 0) where no model show significant change.
    pos_frac = (delta_chng > 0).weighted(w).sum(realization) / (
        change_frac * n_valid_real
    )

    # Metadata
    kwargs_str = ", ".join(
        [f"{k}: {v}" for k, v in kwargs.items() if k in test_params[test]]
    )
    test_str = (
        f"Significant change was tested with test {test} with parameters {kwargs_str}."
    )
    das = {"fut": fut} if ref is None else {"fut": fut, "ref": ref}

    if pvals is not None:
        pvals.attrs.update(
            description="P-values from change significance test. " + test_str,
            units="",
            test=str(test),
            history=update_history(
                f"pvals from change_significance(fut=fut, ref=ref, test={test}, {kwargs_str})",
                **das,
            ),
        )
    pos_frac.attrs.update(
        description="Fraction of members showing significant change that agree on a positive change. "
        + test_str,
        units="",
        test=str(test),
        history=update_history(
            f"pos_frac from change_significance(fut=fut, ref=ref, test={test}, {kwargs_str})",
            **das,
        ),
    )
    change_frac.attrs.update(
        description="Fraction of members showing significant change. " + test_str,
        units="",
        test=str(test),
        history=update_history(
            f"change_frac from change_significance(fut=fut, ref=ref, test={test}, {kwargs_str})",
            **das,
        ),
    )

    # Returns either two (2) or three (3) variables. This should be adjusted.
    if p_vals:
        return change_frac, pos_frac, pvals
    return change_frac, pos_frac


def robustness_coefficient(
    fut: xr.DataArray | xr.Dataset, ref: xr.DataArray | xr.Dataset
) -> xr.DataArray | xr.Dataset:
    """Robustness coefficient quantifying the robustness of a climate change signal in an ensemble.

    Taken from :cite:ts:`knutti_robustness_2013`.

    The robustness metric is defined as R = 1 − A1 / A2 , where A1 is defined as the integral of the squared area
    between two cumulative density functions characterizing the individual model projections and the multimodel mean
    projection and A2 is the integral of the squared area between two cumulative density functions characterizing
    the multimodel mean projection and the historical climate.
    Description taken from :cite:t:`knutti_robustness_2013`.

    A value of R equal to one implies perfect model agreement. Higher model spread or
    smaller signal decreases the value of R.

    Parameters
    ----------
    fut : Union[xr.DataArray, xr.Dataset]
        Future ensemble values along 'realization' and 'time' (nr, nt). Can be a dataset,
        in which case the coefficient is computed on each variable.
    ref : Union[xr.DataArray, xr.Dataset]
        Reference period values along 'time' (nt). Same type as `fut`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The robustness coefficient, ]-inf, 1], float. Same type as `fut` or `ref`.

    References
    ----------
    :cite:cts:`knutti_robustness_2013`
    """

    def _knutti_sedlacek(reference, future):
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
        v_fut = np.sort(future.flatten())  # "cumulative" models distribution
        v_favg = np.sort(future.mean(axis=-1))  # Multimodel mean
        v_ref = np.sort(reference)  # Historical values

        A1 = diff_cdf_sq_area_int(v_fut, v_favg)  # noqa
        A2 = diff_cdf_sq_area_int(v_ref, v_favg)  # noqa

        return 1 - A1 / A2

    R = xr.apply_ufunc(  # noqa
        _knutti_sedlacek,
        ref,
        fut,
        input_core_dims=[["time"], ["realization", "time"]],
        exclude_dims={"time"},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    R.attrs.update(
        name="R",
        long_name="Ensemble robustness coefficient",
        description="Ensemble robustness coefficient as defined by Knutti and Sedláček (2013).",
        reference="Knutti, R. and Sedláček, J. (2013) Robustness and uncertainties in the new CMIP5 climate model projections. Nat. Clim. Change.",
        units="",
        history=update_history("knutti_sedlacek(fut, ref)", ref=ref, fut=fut),
    )
    return R
