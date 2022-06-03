# noqa: D205,D400
"""
Properties Submodule
====================
SDBA diagnostic tests are made up of statistical properties and measures. Properties are calculated on both simulation
and reference datasets. They collapse the time dimension to one value.

This framework for the diagnostic tests was inspired by the [VALUE]_ project.
Statistical Properties is the xclim term for 'indices' in the VALUE project.

.. [VALUE] http://www.value-cost.eu/
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr
from boltons.funcutils import wraps
from scipy import stats
from statsmodels.tsa import stattools

import xclim as xc
from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to
from xclim.core.utils import uses_dask
from xclim.indices import run_length as rl
from xclim.indices.generic import select_resample_op
from xclim.indices.stats import fit, parametric_quantile

from .base import Grouper, map_groups, parse_group

STATISTICAL_PROPERTIES: dict[str, Callable] = {}
""" Dictionary of all the statistical properties available."""


def register_statistical_properties(
    aspect: str, seasonal: bool, annual: bool
) -> Callable:
    """Register statistical properties in the STATISTICAL_PROPERTIES dictionary with its aspect and time resolutions."""

    def _register_statistical_properties(func):
        func.aspect = aspect
        func.seasonal = seasonal
        func.annual = annual
        allowed = []
        if annual:
            allowed.append("group")
        if seasonal:
            allowed.extend(["season", "month"])

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = parse_group(func, kwargs=kwargs, allow_only=allowed)
            out = func(*args, **kwargs)
            if "group" in out.dims:
                out = out.squeeze("group", drop=True)
            return out

        STATISTICAL_PROPERTIES[func.__name__] = wrapper
        return wrapper

    return _register_statistical_properties


@update_xclim_history
@register_statistical_properties(aspect="marginal", seasonal=True, annual=True)
def mean(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """Mean.

    Mean over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the temporal average is performed separately for each month.

    Returns
    -------
    xr.DataArray,
      Mean of the variable.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> mean(da=pr, group="time.season")
    """
    attrs = da.attrs
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.mean(dim=group.dim)
    out.attrs.update(attrs)
    out.attrs["long_name"] = "Mean"
    out.name = "mean"
    return out


@update_xclim_history
@register_statistical_properties(aspect="marginal", seasonal=True, annual=True)
def var(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """Variance.

    Variance of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the variance is performed separately for each month.

    Returns
    -------
    xr.DataArray
      Variance of the variable.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> var(da=pr, group="time.season")
    """
    attrs = da.attrs
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.var(dim=group.dim)
    out.attrs.update(attrs)
    out.attrs["long_name"] = "Variance"
    u = xc.core.units.units2pint(attrs["units"])
    u2 = u**2
    out.attrs["units"] = xc.core.units.pint2cfunits(u2)
    out.name = "variance"
    return out


@update_xclim_history
@register_statistical_properties(aspect="marginal", seasonal=True, annual=True)
def skewness(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """Skewness.

    Skewness of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the skewness is performed separately for each month.

    Returns
    -------
    xr.DataArray
      Skewness of the variable.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> skewness(da=pr, group="time.season")

    See Also
    --------
    scipy.stats.skew
    """
    attrs = da.attrs
    if group.prop != "group":
        da = da.groupby(group.name)
    out = xr.apply_ufunc(
        stats.skew,
        da,
        input_core_dims=[[group.dim]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs.update(attrs)
    out.attrs["long_name"] = "Skewness"
    out.attrs["units"] = ""
    out.name = "skewness"
    return out


@update_xclim_history
@register_statistical_properties(aspect="marginal", seasonal=True, annual=True)
def quantile(
    da: xr.DataArray, *, q: float = 0.98, group: str | Grouper = "time"
) -> xr.DataArray:
    """Quantile.

    Returns the quantile q of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    q: float
      Quantile to be calculated. Should be between 0 and 1.
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the quantile is computed separately for each month.

    Returns
    -------
    xr.DataArray
      Quantile {q} of the variable.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> quantile(da=pr, q=0.9, group="time.season")
    """
    attrs = da.attrs
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.quantile(q, dim=group.dim, keep_attrs=True).drop_vars("quantile")
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Quantile {q}"
    out.name = "quantile"
    return out


@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=True, annual=True)
def spell_length_distribution(
    da: xr.DataArray,
    *,
    method: str = "amount",
    op: str = ">=",
    thresh: str | float = "1 mm d-1",
    stat: str = "mean",
    group: str | Grouper = "time",
) -> xr.DataArray:
    """Spell length distribution.

    Statistic of spell length distribution when the variable respects a condition (defined by an operation, a method and
     a threshold).

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    method: {'amount', 'quantile'}
      Method to choose the threshold.
      'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
      'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op: {">", "<", ">=", "<="}
      Operation to verify the condition for a spell.
      The condition for a spell is variable {op} threshold.
    thresh: str or float
      Threshold on which to evaluate the condition to have a spell.
      Str with units if the method is "amount".
      Float of the quantile if the method is "quantile".
    stat: {'mean','max','min'}
      Statistics to apply to the resampled input at the {group} (e.g. 1-31 Jan 1980)
      and then over all years (e.g. Jan 1980-2010)
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the spell lengths are coputed separately for each month.

    Returns
    -------
    xr.DataArray
      {stat} of spell length distribution when the variable is {op} the {method} {thresh}.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> spell_length_distribution(da=pr, op="<", thresh="1mm d-1", group="time.season")
    """
    attrs = da.attrs
    ops = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _spell_stats(ds, *, dim, method, thresh, op, freq, stat):
        # PB: This prevents an import error in the distributed dask scheduler, but I don't know why.
        import xarray.core.resample_cftime  # noqa

        da = ds.data
        mask = ~(da.isel({dim: 0}).isnull()).drop_vars(
            dim
        )  # mask of the ocean with NaNs
        if method == "quantile":
            thresh = da.quantile(thresh, dim=dim).drop_vars("quantile")

        cond = op(da, thresh)
        out = cond.resample(time=freq).map(rl.rle_statistics, dim=dim, reducer=stat)
        out = getattr(out, stat)(dim=dim)
        out = out.where(mask)
        return out.rename("out").to_dataset()

    # threshold is an amount that will be converted to the right units
    if method == "amount":
        thresh = convert_units_to(thresh, da)
    elif method != "quantile":
        raise ValueError(
            f"{method} is not a valid method. Choose 'amount' or 'quantile'."
        )

    out = _spell_stats(
        da.rename("data").to_dataset(),
        group=group,
        method=method,
        thresh=thresh,
        op=ops[op],
        freq=group.freq,
        stat=stat,
    ).out
    out.attrs.update(attrs)
    out.attrs[
        "long_name"
    ] = f"{stat} of spell length when input variable {op} {method} {thresh}"
    out.name = "spell_length_distribution"
    out.attrs["units"] = "day"
    return out


@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=True, annual=False)
def acf(
    da: xr.DataArray, *, lag: int = 1, group: str | Grouper = "time.season"
) -> xr.DataArray:
    """Autocorrelation function.

    Autocorrelation with a lag over a time resolution and averaged over all years.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    lag: int
      Lag.
    group : {'time.season', 'time.month'}
      Grouping of the output.
      E.g. If 'time.month', the autocorrelation is calculated over each month separately for all years.
      Then, the autocorrelation for all Jan/Feb/... is averaged over all years, giving 12 outputs for each grid point.

    Returns
    -------
    xr.DataArray
      lag-{lag} autocorrelation of the variable over a {group.prop} and averaged over all years.

    See Also
    --------
    statsmodels.tsa.stattools.acf

    References
    ----------
    Alavoine M., and Grenier P. (under review) The distinct problems of physical inconsistency and of multivariate bias potentially involved in the statistical adjustment of climate simulations. International Journal of Climatology, submitted on September 19th 2021. (Preprint: https://doi.org/10.31223/X5C34C)

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> acf(da=pr, lag=3, group="time.season")
    """
    attrs = da.attrs

    def acf_last(x, nlags):
        # noqa: D403
        """statsmodels acf calculates acf for lag 0 to nlags, this return only the last one."""
        # As we resample + group, timeseries are quite short and fft=False seems more performant
        out_last = stattools.acf(x, nlags=nlags, fft=False)
        return out_last[-1]

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _acf(ds, *, dim, lag, freq):
        out = xr.apply_ufunc(
            acf_last,
            ds.dat.resample({dim: freq}),
            input_core_dims=[[dim]],
            vectorize=True,
            kwargs={"nlags": lag},
        )
        out = out.mean("__resample_dim__")
        return out.rename("out").to_dataset()

    out = _acf(da.rename("dat").to_dataset(), group=group, lag=lag, freq=group.freq).out
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"lag-{lag} autocorrelation"
    out.attrs["units"] = ""
    out.name = "acf"
    return out


# group was kept even though "time" is the only acceptable arg to keep the signature similar to other properties
@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=False, annual=True)
def annual_cycle_amplitude(
    da: xr.DataArray,
    *,
    amplitude_type: str = "absolute",
    group: str | Grouper = "time",
) -> xr.DataArray:
    r"""Annual cycle amplitude.

    The amplitudes of the annual cycle are calculated for each year, then averaged over the all years.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    amplitude_type: {'absolute','relative'}
      Type of amplitude.
      'absolute' is the peak-to-peak amplitude. (max - min).
      'relative' is a relative percentage. 100 * (max - min) / mean (Recommended for precipitation).

    Returns
    -------
    xr.DataArray
      {amplitude_type} amplitude of the annual cycle.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> annual_cycle_amplitude(da=pr, amplitude_type="relative")
    """
    attrs = da.attrs
    da = da.resample({group.dim: group.freq})
    # amplitude
    amp = da.max(dim=group.dim) - da.min(dim=group.dim)
    amp.attrs.update(attrs)
    amp.attrs["units"] = xc.core.units.ensure_delta(attrs["units"])
    if amplitude_type == "relative":
        amp = amp * 100 / da.mean(dim=group.dim, keep_attrs=True)
        amp.attrs["units"] = "%"
    amp = amp.mean(dim=group.dim, keep_attrs=True)
    amp.attrs["long_name"] = f"{amplitude_type} amplitude of the annual cycle"
    amp.name = "annual_cycle_amplitude"
    return amp


# group was kept even though "time" is the only acceptable arg to keep the signature similar to other properties
@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=False, annual=True)
def annual_cycle_phase(
    da: xr.DataArray, *, group: str | Grouper = "time"
) -> xr.DataArray:
    """Annual cycle phase.

    The phases of the annual cycle are calculated for each year, then averaged over the all years.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    group : {"time", 'time.season', 'time.month'}
      Grouping of the output. Default: "time".

    Returns
    -------
    xr.DataArray
      Phase of the annual cycle. The position (day-of-year) of the maximal value.

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> annual_cycle_phase(da=pr)
    """
    attrs = da.attrs
    mask = ~(da.isel({group.dim: 0}).isnull()).drop_vars(
        group.dim
    )  # mask of the ocean with NaNs
    da = da.resample({group.dim: group.freq})

    # +1  at the end to go from index to doy
    phase = (
        xr.apply_ufunc(
            np.argmax,
            da,
            input_core_dims=[[group.dim]],
            vectorize=True,
            dask="parallelized",
        )
        + 1
    )
    phase = phase.mean(dim="__resample_dim__")
    # put nan where there was nan in the input, if not phase = 0 + 1
    phase = phase.where(mask, np.nan)
    phase.attrs.update(attrs)
    phase.attrs["long_name"] = "Phase of the annual cycle"
    phase.attrs.update(units="", is_dayofyear=1)
    phase.name = "annual_cycle_phase"
    return phase


@update_xclim_history
@register_statistical_properties(aspect="multivariate", seasonal=True, annual=True)
def corr_btw_var(
    da1: xr.DataArray,
    da2: xr.DataArray,
    *,
    corr_type: str = "Spearman",
    group: str | Grouper = "time",
    output: str = "correlation",
) -> xr.DataArray:
    r"""Correlation between two variables.

    Spearman or Pearson correlation coefficient between two variables at the time resolution.

    Parameters
    ----------
    da1 : xr.DataArray
      First variable on which to calculate the diagnostic.
    da2 : xr.DataArray
      Second variable on which to calculate the diagnostic.
    corr_type: {'Pearson','Spearman'}
      Type of correlation to calculate.
    output: {'correlation', 'pvalue'}
      Wheter to return the correlation coefficient or the p-value.
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output.
      Eg. For 'time.month', the correlation would be calculated on each month separately,
      but with all the years together.

    Returns
    -------
    xr.DataArray
      {corr_type} correlation coefficient

    Examples
    --------
    >>> pr = open_dataset(path_to_pr_file).pr
    >>> tasmax = open_dataset("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc").tasmax
    >>> corr_btw_var(da1=pr, da2=tasmax, group="time.season")
    """
    attrs1 = da1.attrs

    if corr_type.lower() not in {"pearson", "spearman"}:
        raise ValueError(
            f"{corr_type} is not a valid type. Choose 'Pearson' or 'Spearman'."
        )

    index = {"correlation": 0, "pvalue": 1}[output]

    def _first_output_1d(a, b, index, corr_type):
        """Only keep the correlation (first output) from the scipy function."""
        if corr_type == "Pearson":
            # for points in the water with NaNs
            if np.isnan(a).any():
                return np.nan
            return stats.pearsonr(a, b)[index]
        return stats.spearmanr(a, b, nan_policy="propagate")[index]

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _first_output(ds, *, dim, index, corr_type):
        out = xr.apply_ufunc(
            _first_output_1d,
            ds.a,
            ds.b,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            kwargs={"index": index, "corr_type": corr_type},
        )
        return out.rename("out").to_dataset()

    out = _first_output(
        xr.Dataset({"a": da1, "b": da2}), group=group, index=index, corr_type=corr_type
    ).out
    out.attrs.update(attrs1)
    out.attrs["long_name"] = f"{corr_type} correlation coefficient"
    out.attrs["units"] = ""
    out.name = "corr_btw_varr"
    return out


@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=True, annual=True)
def relative_frequency(
    da: xr.DataArray,
    *,
    op: str = ">=",
    thresh: str = "1mm d-1",
    group: str | Grouper = "time",
) -> xr.DataArray:
    """Relative Frequency.

    Relative Frequency of days with variable  respecting a condition (defined by an operation and a threshold) at the
    time resolution. The relative freqency is the number of days that satisfy the condition divided by the total number
    of days.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    op: {">", "<", ">=", "<="}
      Operation to verify the condition.
      The condition is variable {op} threshold.
    thresh: str
      Threshold on which to evaluate the condition.
    group : {'time', 'time.season', 'time.month'}
      Grouping on the output.
      Eg. For 'time.month', the relative frequency would be calculated on each month,
      with all years included.

    Returns
    -------
    xr.DataArray
      Relative frequency of the variable.

    Examples
    --------
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax
    >>> relative_frequency(da=tasmax, op="<", thresh="0 degC", group="time.season")
    """
    attrs = da.attrs
    mask = ~(da.isel({group.dim: 0}).isnull()).drop_vars(
        group.dim
    )  # mask of the ocean with NaNs
    ops = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}
    t = convert_units_to(thresh, da)
    length = da.sizes[group.dim]
    cond = ops[op](da, t)
    if group.prop != "group":  # change the time resolution if necessary
        cond = cond.groupby(group.name)
        # length of the groupBy groups
        length = np.array([len(v) for k, v in cond.groups.items()])
        for i in range(da.ndim - 1):  # add empty dimension(s) to match input
            length = np.expand_dims(length, axis=-1)
    # count days with the condition and divide by total nb of days
    out = cond.sum(dim=group.dim, skipna=False) / length
    out = out.where(mask, np.nan)
    out.attrs.update(attrs)
    out.attrs[
        "long_name"
    ] = f"Relative frequency of days with input variable {op} {thresh}"
    out.attrs["units"] = ""
    out.name = "relative frequency"
    return out


@update_xclim_history
@register_statistical_properties(aspect="temporal", seasonal=True, annual=True)
def trend(
    da: xr.DataArray,
    *,
    group: str | Grouper = "time",
    output: str = "slope",
) -> xr.DataArray:
    """Linear Trend.

    The data is averaged over each time resolution and the interannual trend is returned.

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    output: {'slope', 'pvalue'}
      Attributes of the linear regression to return.
      'slope' is the slope of the regression line.
      'pvalue' is  for a hypothesis test whose null hypothesis is that the slope is zero,
      using Wald Test with t-distribution of the test statistic.
    group : {'time', 'time.season', 'time.month'}
      Grouping on the output.

    Returns
    -------
    xr.DataArray
      Trend of the variable.

    See Also
    --------
    scipy.stats.linregress

    numpy.polyfit

    Examples
    --------
    >>> tas = open_dataset(path_to_tas_file).tas
    >>> trend(da=tas, group="time.season")
    """
    attrs = da.attrs
    da = da.resample({group.dim: group.freq})  # separate all the {group}
    da_mean = da.mean(dim=group.dim)  # avg over all {group}
    if uses_dask(da_mean):
        da_mean = da_mean.chunk({group.dim: -1})
    if group.prop != "group":
        da_mean = da_mean.groupby(group.name)  # group all month/season together

    def modified_lr(
        x,
    ):  # modify linregress to fit into apply_ufunc and only return slope
        return getattr(stats.linregress(list(range(len(x))), x), output)

    out = xr.apply_ufunc(
        modified_lr,
        da_mean,
        input_core_dims=[[group.dim]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"{output} of the interannual linear trend"
    out.attrs["units"] = f"{attrs['units']}/year"
    out.name = "trend"
    return out


@update_xclim_history
@register_statistical_properties(aspect="marginal", seasonal=True, annual=True)
def return_value(
    da: xr.DataArray,
    *,
    period: int = 20,
    op: str = "max",
    method: str = "ML",
    group: str | Grouper = "time",
) -> xr.DataArray:
    r"""Return value.

    Return the value corresponding to a return period.
    On average, the return value will be exceeded (or not exceed for op='min') every return period (eg. 20 years).
    The return value is computed by first extracting the variable annual maxima/minima,
    fitting a statistical distribution to the maxima/minima,
    then estimating the percentile associated with the return period (eg. 95th percentile (1/20) for 20 years)

    Parameters
    ----------
    da : xr.DataArray
      Variable on which to calculate the diagnostic.
    period: int
      Return period. Number of years over which to check if the value is exceeded (or not for op='min').
    op: {'max','min'}
      Whether we are looking for a probability of exceedance ('max', right side of the distribution)
      or a probability of non-exceedance (min, left side of the distribution).
    method : {"ML", "PWM"}
      Fitting method, either maximum likelihood (ML) or probability weighted moments (PWM), also called L-Moments.
      The PWM method is usually more robust to outliers. However, it requires the lmoments3 libraryto be installed
      from the `develop` branch.
      ``pip install git+https://github.com/OpenHydrology/lmoments3.git@develop#egg=lmoments3``
    group : {'time', 'time.season', 'time.month'}
      Grouping of the output. A distribution of the extremums is done for each group.

    Returns
    -------
    xr.DataArray
      {period}-{group} {op} return level of the variable.

    Examples
    --------
    >>> tas = open_dataset(path_to_tas_file).tas
    >>> return_value(da=tas, group="time.season")
    """

    @map_groups(out=[Grouper.PROP], main_only=True)
    def frequency_analysis_method(ds, *, dim, method):
        sub = select_resample_op(ds.x, op=op)
        params = fit(sub, dist="genextreme", method=method)
        out = parametric_quantile(params, q=1 - 1.0 / period)
        return out.isel(quantile=0, drop=True).rename("out").to_dataset()

    out = frequency_analysis_method(
        da.rename("x").to_dataset(), method=method, group=group
    ).out
    out.attrs.update(da.attrs)
    out.attrs["long_name"] = f"{period}-{group.prop_name} {op} return level"
    out.name = "return_value"
    return out
