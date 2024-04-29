"""
Ensemble Robustness Metrics
===========================

Robustness metrics are used to estimate the confidence of the climate change signal of an ensemble.
This submodule is inspired by and tries to follow the guidelines of the IPCC,
more specifically :cite:p:`collins_long-term_2013` (AR5) and :cite:cts:`ipccatlas_ar6wg1` (AR6).
"""

from __future__ import annotations

from inspect import Parameter, signature

import numpy as np
import scipy.stats as spstats  # noqa
import xarray as xr

from xclim.core.formatting import gen_call_string, update_xclim_history
from xclim.indices.generic import compare, detrend

__all__ = [
    "robustness_categories",
    "robustness_coefficient",
    "robustness_fractions",
]


SIGNIFICANCE_TESTS = {}
"""Registry of change significance tests.

New tests must be decorated with :py:func:`significance_test` and fulfill the following requirements:

- Function name should begin by "_", registered test name is the function name without its first character and with _ replaced by -.
- Function must accept 2 positional arguments : fut and ref (see :py:func:`robustness_fractions` for definitions)
- Function may accept other keyword-only arguments.
- Function must return 2 values :
    + `changed` : 1D boolean array along `realization`. True for realization with significant change.
    + `pvals` : 1D float array along `realization`. P-values of the statistical test. Should be `None` for test where is doesn't apply.
"""


def significance_test(func):
    """Register a significance test for use in :py:func:`robustness_fractions`.

    See :py:data:`SIGNIFICANCE_TESTS`.
    """
    SIGNIFICANCE_TESTS[func.__name__[1:].replace("_", "-")] = func
    return func


# This function's docstring is modified to include the registered test names and docs.
# See end of this file.
@update_xclim_history
def robustness_fractions(  # noqa: C901
    fut: xr.DataArray,
    ref: xr.DataArray | None = None,
    test: str | None = None,
    weights: xr.DataArray | None = None,
    **kwargs,
) -> xr.Dataset:
    r"""Robustness statistics qualifying how members of an ensemble agree on the existence of change and on its sign.

    Parameters
    ----------
    fut : xr.DataArray
        Future period values along 'realization' and 'time' (..., nr, nt1)
        or if `ref` is None, Delta values along `realization` (..., nr).
    ref : xr.DataArray, optional
        Reference period values along realization' and 'time'  (..., nr, nt2).
        The size of the 'time' axis does not need to match the one of `fut`.
        But their 'realization' axes must be identical and the other coordinates should be the same.
        If `None` (default), values of `fut` are assumed to be deltas instead of
        a distribution across the future period.
    test : {tests_list}, optional
        Name of the statistical test used to determine if there was significant change. See notes.
    weights : xr.DataArray
        Weights to apply along the 'realization' dimension. This array cannot contain missing values.
    \*\*kwargs
        Other arguments specific to the statistical test. See notes.

    Returns
    -------
    xr.Dataset
        Same coordinates as `fut` and  `ref`, but no `time` and no `realization`.

        Variables:

        changed :
                The weighted fraction of valid members showing significant change.
                Passing `test=None` yields change_frac = 1 everywhere. Same type as `fut`.
        positive :
                The weighted fraction of valid members showing strictly positive change, no matter if it is significant or not.
        changed_positive :
                The weighted fraction of valid members showing significant and positive change.
        negative :
                The weighted fraction of valid members showing strictly negative change, no matter if it is significant or not.
        changed_negative :
                The weighted fraction of valid members showing significant and negative change.
        agree :
                The weighted fraction of valid members agreeing on the sign of change. It is the maximum between positive, negative and the rest.
        valid :
                The weighted fraction of valid members. A member is valid is there are no NaNs along the time axes of `fut` and  `ref`.
        pvals :
                The p-values estimated by the significance tests. Only returned if the test uses `pvals`. Has the  `realization` dimension.

    Notes
    -----
    The table below shows the coefficient needed to retrieve the number of members
    that have the indicated characteristics, by multiplying it by the total
    number of members (`fut.realization.size`) and by `valid_frac`, assuming uniform weights.
    For compactness, we rename the outputs cf, pf, cpf, nf and cnf.

    +-----------------+--------------------+------------------------+------------+
    |                 | Significant change | Non-significant change | Any change |
    +-----------------+--------------------+------------------------+------------+
    | Any direction   | cf                 | 1 - cf                 | 1          |
    +-----------------+--------------------+------------------------+------------+
    | Positive change | cpf                | pf - cpf               | pf         |
    +-----------------+--------------------+------------------------+------------+
    | Negative change | cnf                | nf - cnf               | nf         |
    +-----------------+--------------------+------------------------+------------+

    And members showing absolutely no change are ``1 - nf - pf``.

    Available statistical tests are :

    {tests_doc}
        threshold :
                Change is considered significant when it exceeds an absolute or relative threshold.
                Accepts one argument, either "abs_thresh" or "rel_thresh".
        None :
                Significant change is not tested. Members showing any positive change are
                included in the `pos_frac` output.

    References
    ----------
    :cite:cts:`tebaldi_mapping_2011`
    :cite:cts:`ipccatlas_ar6wg1`

    Example
    -------
    This example computes the mean temperature in an ensemble and compares two time
    periods, qualifying significant change through a single sample T-test.

    >>> from xclim import ensembles
    >>> ens = ensembles.create_ensemble(temperature_datasets)
    >>> tgmean = xclim.atmos.tg_mean(tas=ens.tas, freq="YS")
    >>> fut = tgmean.sel(time=slice("2020", "2050"))
    >>> ref = tgmean.sel(time=slice("1990", "2020"))
    >>> fractions = ensembles.robustness_fractions(fut, ref, test="ttest")
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

    if ref is None:
        delta = fut
        valid = delta.notnull()
        if test not in [None, "threshold"]:
            raise ValueError(
                "When deltas are given (ref=None), 'test' must be None or 'threshold'."
            )
    else:
        delta = fut.mean("time") - ref.mean("time")
        valid = fut.notnull().all("time") & ref.notnull().all("time")

    if test is None:
        test_params = {}
        changed = xr.ones_like(delta).astype(bool)
        pvals = None
    elif test == "threshold":
        abs_thresh = kwargs.get("abs_thresh")
        rel_thresh = kwargs.get("rel_thresh")
        if abs_thresh is not None and rel_thresh is None:
            changed = abs(delta) > abs_thresh
            test_params = {"abs_thresh": abs_thresh}
        elif rel_thresh is not None and abs_thresh is None:
            changed = abs(delta / ref.mean("time")) > rel_thresh
            test_params = {"rel_thresh": rel_thresh}
        else:
            raise ValueError(
                "One and only one of abs_thresh or rel_thresh must be given if test='threshold'."
            )
        pvals = None
    elif test in SIGNIFICANCE_TESTS:
        test_func = SIGNIFICANCE_TESTS[test]
        test_params = {
            n: kwargs.get(n, p.default)
            for n, p in signature(test_func).parameters.items()
            if p.kind == Parameter.KEYWORD_ONLY
        }

        changed, pvals = test_func(fut, ref, **test_params)
    else:
        raise ValueError(
            f"Statistical test {test} must be one of {', '.join(SIGNIFICANCE_TESTS.keys())}."
        )

    valid_frac = valid.weighted(w).sum(realization) / fut[realization].size
    n_valid = valid.weighted(w).sum(realization)
    change_frac = changed.where(valid).weighted(w).sum(realization) / n_valid
    pos_frac = (delta > 0).where(valid).weighted(w).sum(realization) / n_valid
    neg_frac = (delta < 0).where(valid).weighted(w).sum(realization) / n_valid
    change_pos_frac = ((delta > 0) & changed).where(valid).weighted(w).sum(
        realization
    ) / n_valid
    change_neg_frac = ((delta < 0) & changed).where(valid).weighted(w).sum(
        realization
    ) / n_valid
    agree_frac = xr.concat((pos_frac, neg_frac, 1 - pos_frac - neg_frac), "sign").max(
        "sign"
    )

    # Metadata
    kwargs_str = gen_call_string("", **test_params)[1:-1]
    test_str = (
        f"Significant change was tested with test {test} and parameters {kwargs_str}."
    )

    out = xr.Dataset(
        {
            "changed": change_frac.assign_attrs(
                description="Fraction of members showing significant change. "
                + test_str,
                units="",
                test=str(test),
            ),
            "positive": pos_frac.assign_attrs(
                description="Fraction of valid members showing strictly positive change.",
                units="",
            ),
            "changed_positive": change_pos_frac.assign_attrs(
                description="Fraction of valid members showing significant and positive change. "
                + test_str,
                units="",
                test=str(test),
            ),
            "negative": neg_frac.assign_attrs(
                description="Fraction of valid members showing strictly negative change.",
                units="",
            ),
            "changed_negative": change_neg_frac.assign_attrs(
                description="Fraction of valid members showing significant and negative change. "
                + test_str,
                units="",
                test=str(test),
            ),
            "valid": valid_frac.assign_attrs(
                description="Fraction of valid members (No missing values along time).",
                units="",
            ),
            "agree": agree_frac.assign_attrs(
                description=(
                    "Fraction of valid members agreeing on the sign of change. "
                    "Maximum between the positive, negative and no change fractions."
                ),
                units="",
            ),
        },
        attrs={"description": "Significant change and sign of change fractions."},
    )

    if pvals is not None:
        pvals.attrs.update(
            description="P-values from change significance test. " + test_str,
            units="",
        )
        out = out.assign(pvals=pvals)

    # Keep attrs on non-modified coordinates
    for ncrd, crd in fut.coords.items():
        if ncrd in out.coords:
            out[ncrd].attrs.update(crd.attrs)

    return out


def robustness_categories(
    changed_or_fractions: xr.Dataset | xr.DataArray,
    agree: xr.DataArray | None = None,
    *,
    categories: list[str] | None = None,
    ops: list[tuple[str, str]] | None = None,
    thresholds: list[tuple[float, float]] | None = None,
) -> xr.DataArray:
    """Create a categorical robustness map for mapping hatching patterns.

    Each robustness category is defined by a double threshold, one on the fraction of members showing significant
    change (`change_frac`) and one on the fraction of member agreeing on the sign of change (`agree_frac`).
    When the two thresholds are fulfilled, the point is assigned to the given category.
    The default values for the comparisons are the ones suggested by the IPCC for its "Advanced approach" described
    in the Cross-Chapter Box 1 of the Atlas of the AR6 WGI report (:cite:t:`ipccatlas_ar6wg1`).

    Parameters
    ----------
    changed_or_fractions : xr.Dataset or xr.DataArray
        Either the fraction of members showing significant change as an array or
        directly the output of :py:func:`robustness_fractions`.
    agree : xr.DataArray, optional
        The fraction of members agreeing on the sign of change. Only needed if the first argument is
        the `changed` array.
    categories : list of str, optional
        The label of each robustness categories. They are stored in the semicolon separated flag_descriptions
        attribute as well as in a compressed form in the flag_meanings attribute.
        If a point is mapped to two categories, priority is given to the first one in this list.
    ops : list of tuples of str, optional
        For each category, the comparison operators for `change_frac` and `agree_frac`.
        None or an empty string means the variable is not needed for this category.
    thresholds : list of tuples of float, optional
        For each category, the threshold to be used with the corresponding operator. All should be between 0 and 1.

    Returns
    -------
    xr.DataArray
        Categorical (int) array following the flag variables CF conventions.
        99 is used as a fill value for points that do not fall in any category.
    """
    if categories is None:
        categories = [
            "Robust signal",
            "No change or no signal",
            "Conflicting signal",
        ]

    if ops is None:
        ops = [(">=", ">="), ("<", None), (">=", "<")]

    if thresholds is None:
        thresholds = [(0.66, 0.8), (0.66, None), (0.66, 0.8)]

    src = changed_or_fractions.copy()  # Ensure no inplace changing of coords...
    if isinstance(src, xr.Dataset):
        # Output of robustness fractions
        changed = src.changed
        agree = src.agree
    else:
        changed = src

    # Initial map is all 99, same shape as change_frac
    robustness = (changed.copy() * 0).astype(int) + 99
    # We go in reverse gear so that the first categories have precedence in the case of multiple matches.
    for i, ((chg_op, agr_op), (chg_thresh, agr_thresh)) in reversed(
        list(enumerate(zip(ops, thresholds), 1))
    ):
        if not agr_op:
            cond = compare(changed, chg_op, chg_thresh)
        elif not chg_op:
            cond = compare(agree, agr_op, agr_thresh)
        else:
            cond = compare(changed, chg_op, chg_thresh) & compare(
                agree, agr_op, agr_thresh
            )
        robustness = xr.where(~cond, robustness, i, keep_attrs=True)

    robustness = robustness.assign_attrs(
        flag_values=list(range(1, len(categories) + 1)),
        _FillValue=99,
        flag_descriptions=categories,
        flag_meanings=" ".join(
            map(lambda cat: cat.casefold().replace(" ", "_"), categories)
        ),
    )
    return robustness


@update_xclim_history
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
    )
    return R


@significance_test
def _ttest(fut, ref, *, p_change=0.05):
    """Single sample T-test. Same test as used by :cite:t:`tebaldi_mapping_2011`.

    The future values are compared against the reference mean (over 'time').
    Accepts argument p_change (float, default : 0.05) the p-value threshold for rejecting the hypothesis of no significant change.
    """

    def _ttest_func(f, r):
        # scipy>=1.9: popmean.axis[-1] must equal 1 for both fut and ref
        if np.isnan(f).all() or np.isnan(r).all():
            return np.NaN

        return spstats.ttest_1samp(f, r[..., np.newaxis], axis=-1, nan_policy="omit")[1]

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
    return changed, pvals


@significance_test
def _welch_ttest(fut, ref, *, p_change=0.05):
    """Two-sided T-test, without assuming equal population variance.

    Same significance criterion and argument as 'ttest'.
    """

    # Test hypothesis of no significant change
    # equal_var=False -> Welch's T-test
    def wtt_wrapper(f, r):  # This specific test can't manage an all-NaN slice
        if np.isnan(f).all() or np.isnan(r).all():
            return np.NaN
        return spstats.ttest_ind(f, r, axis=-1, equal_var=False, nan_policy="omit")[1]

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
    return changed, pvals


@significance_test
def _mannwhitney_utest(ref, fut, *, p_change=0.05):
    """Two-sided Mann-Whiney U-test. Same significance criterion and argument as 'ttest'."""

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
    return changed, pvals


@significance_test
def _brownforsythe_test(fut, ref, *, p_change=0.05):
    """Brown-Forsythe test assuming skewed, non-normal distributions.

    Same significance criterion and argument as 'ttest'.
    """
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
    return changed, pvals


@significance_test
def _ipcc_ar6_c(fut, ref, *, ref_pi=None):
    r"""The advanced approach used in the IPCC Atlas chapter (:cite:t:`ipccatlas_ar6wg1`).

    Change is considered significant if the delta exceeds a threshold related to the internal variability.
    If pre-industrial data is given in argument `ref_pi`, the threshold is defined as
    :math:`\sqrt{2}*1.645*\sigma_{20yr}`, where :math:`\sigma_{20yr}` is the standard deviation of 20-year
    means computed from non-overlapping periods after detrending with a quadratic fit.
    Otherwise, when such pre-industrial control data is not available, the threshold is defined in relation to
    the historical data (`ref`) as :math:`\sqrt{\frac{2}{20}}*1.645*\sigma_{1yr}, where :math:`\sigma_{1yr}`
    is the inter-annual standard deviation measured after linearly detrending the data.
    See notebook :ref:`notebooks/ensembles:Ensembles` for more details.
    """
    # Ensure annual
    refy = ref.resample(time="YS").mean()
    if ref_pi is None:
        ref_detrended = detrend(refy, dim="time", deg=1)
        gamma = np.sqrt(2 / 20) * 1.645 * ref_detrended.std("time")
    else:
        ref_detrended = detrend(refy, dim="time", deg=2)
        gamma = (
            np.sqrt(2) * 1.645 * ref_detrended.resample(time="20YS").mean().std("time")
        )

    delta = fut.mean("time") - ref.mean("time")
    changed = abs(delta) > gamma
    return changed, None


# Add doc of each significance test to `robustness_fractions` output's doc.
def _gen_test_entry(namefunc):
    name, func = namefunc
    doc = func.__doc__.replace("\n    ", "\n\t\t").rstrip()
    return f"\t{name}:\n\t\t{doc}"


robustness_fractions.__doc__ = robustness_fractions.__doc__.format(
    tests_list="{" + ", ".join(list(SIGNIFICANCE_TESTS.keys()) + ["threshold"]) + "}",
    tests_doc="\n".join(map(_gen_test_entry, SIGNIFICANCE_TESTS.items())),
)
