"""
Properties submodule
=================
 Statistical Properties is the xclim term for 'indices' in the VALUE project.
 SDBA diagnostics are made up of properties and measures.

 This framework for the diagnostics was inspired by the VALUE project (www.value-cost.eu/).
"""
import xarray
import xclim as xc
from typing import Callable, Dict, Tuple
from xclim.core.units import convert_units_to
from xclim.indices import run_length as rl
import numpy as np
from statsmodels.tsa import stattools
from warnings import warn
import scipy
import pandas as pd
from xclim.indices.stats import frequency_analysis
from xclim.indices.stats import frequency_analysis, fit, fa, parametric_quantile
from xclim.indices.generic import select_resample_op

res2freq = {'year': 'YS', 'season': 'QS-DEC', 'month': 'MS'}

STATISTICAL_PROPERTIES: Dict[str, Callable] = dict()


def register_statistical_properties(aspect: str, seasonal: bool, annual: bool) -> Callable:
    """Register missing properties."""
    def _register_statistical_properties(func):
        func.aspect = aspect
        func.seasonal = seasonal
        func.annual = annual
        STATISTICAL_PROPERTIES[func.__name__] = func
        return func
    return _register_statistical_properties


@register_statistical_properties(aspect='marginal', seasonal=True, annual=True)
def mean(da: xarray.DataArray, time_res: str = 'year') -> xarray.DataArray:
    """Mean.

    Mean over all years at the time resolution.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : {'year', 'season', 'month'}
      Time resolution.
      Eg. If 'month', the average is performed on 12 arrays for each grid point.
      Each array contains the days in a given month for all available years.

    Returns
    -------
    xarray.DataArray,
      Mean of the variable.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> mean(da=pr, time_res='season')
    """
    attrs = da.attrs
    if time_res != 'year':
        da = da.groupby(f'time.{time_res}')
    out = da.mean(dim='time')
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Mean {attrs['standard_name']}"
    return out


@register_statistical_properties(aspect='marginal', seasonal=True, annual=True)
def var(da: xarray.DataArray, time_res: str = 'year') -> xarray.DataArray:
    """Variance.

    Variance of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : {'year', 'season', 'month'}
      Time resolution.
      Eg. If 'month', the variance is calculated on 12 arrays for each grid point.
      Each array contains the days in a given month for all available years.

    Returns
    -------
    xarray.DataArray,
      Variance of the variable.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> var(da=pr, time_res='season')
    """
    attrs = da.attrs
    if time_res != 'year':
        da = da.groupby(f'time.{time_res}')
    out = da.var(dim='time')
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Variance of {attrs['standard_name']}"
    u = xc.core.units.units2pint(attrs['units'])
    u2 = u ** 2
    out.attrs['units'] = xc.core.units.pint2cfunits(u2)
    return out


@register_statistical_properties(aspect='marginal', seasonal=True, annual=True)
def skewness(da: xarray.DataArray, time_res: str = 'year') -> xarray.DataArray:
    """Skewness.

    Skewness of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : {'year', 'season', 'month'}
      Time resolution.
      Eg. If 'month', the skewness is calculated on 12 arrays for each grid point.
      Each array contains the days in a given month for all available years.

    Returns
    -------
    xarray.DataArray,
      Skewness of the variable.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> skewness(da=pr, time_res='season')

    Notes
    --------
     See scipy.stats.skew
    """
    attrs = da.attrs
    if time_res != 'year':
        da = da.groupby(f'time.{time_res}')
    out = xarray.apply_ufunc(scipy.stats.skew, da,
                             input_core_dims=[["time"]], vectorize=True, dask='parallelized')
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Skewness of {attrs['standard_name']}"
    out.attrs['units'] = ''
    return out


@register_statistical_properties(aspect='marginal', seasonal=True, annual=True)
def quantile(da: xarray.DataArray, q: float = 0.98, time_res: str = 'year') -> xarray.DataArray:
    """Quantile.

    Returns the quantile {q} of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    q: float
      Quantile to be calculated. Should be between 0 and 1.
    time_res : {'year', 'season', 'month'}
      Time resolution.
      Eg. If 'month', the quantile is calculated on 12 arrays for each grid point.
      Each array contains the days in a given month for all available years.

    Returns
    -------
    xarray.DataArray,
      Quantile {q} of the variable.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> quantile(da=pr, q=0.9, time_res='season')
    """
    attrs = da.attrs
    if time_res != 'year':
        da = da.groupby(f'time.{time_res}')
    out = da.quantile(q, dim="time", keep_attrs=True)
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Quantile {q} of {attrs['standard_name']}"
    return out


@register_statistical_properties(aspect='temporal', seasonal=True, annual=True)
def spell_length_distribution(
    da: xarray.DataArray,
    method: str = 'amount',
    op: str = '>=',
    thresh='1mm d-1',
    stat: str = 'mean',
    time_res: str = 'year'
) -> xarray.DataArray:
    f"""Spell length distribution.

    {stat} of spell length distribution when the variable is {op} the {method} {thresh}.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    method: {'amount', 'quantile'}:
      Method to choose the threshold
      'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as da.
      'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op: {">", "<", ">=", "<="}
      Operation to verify the condition for a spell.
      The condition for a spell is variable {op} threshold.
    thresh: str, float
      Threshold on which to evaluate the condition to have a spell.
      Str with units if the method is "amount".
      Float of the quantile if the method is "quantile".
    stat: {'mean','max','min','median'}
      Statistics to apply to the spell length distribution.
    time_res : str
      Time resolution.
      Eg. If 'month', the {stat} is calculated on 12 distributions for each grid point.
      Each distribution contains the spell lenghts in a given month for all available years.
    Returns
    -------
    xarray.DataArray,
      {stat} of spell length distribution when the variable is {op} the {method} {thresh}.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> spell_length_by_amount(da=pr, op='<',thresh ='1mm d-1', stat= 'p10', time_res='season')
    """
    attrs = da.attrs
    mask = ~(da.isel(time=0).isnull())  # mask of the ocean with NaNs
    ops = {">": np.greater,
           "<": np.less,
           ">=": np.greater_equal,
           "<=": np.less_equal}

    # threshold is an amount that will be converted to the right units
    if method == 'amount':
        t = convert_units_to(thresh, da)
    # threshold is calculated from quantile of distribution at the time_res
    elif method == 'quantile':
        if time_res != 'year':
            da = da.groupby(f'time.{time_res}')
        t = da.quantile(thresh, dim='time')

    cond = ops[op](da, t)
    if time_res != 'year':
        cond = cond.groupby(f'time.{time_res}')
        # DataArrayGroupBy need to call map
        out = cond.map(rl.rle_statistics, dim="time", reducer=stat)
    else:
        out = rl.rle_statistics(cond, dim="time", reducer=stat)
    out = out.where(mask, np.nan)  # put NaNs back over the ocean
    out.attrs.update(attrs)
    out.attrs["long_name"] = f" {stat} of spell length when {attrs['standard_name']} {op} {method} {thresh} "
    return out


@register_statistical_properties(aspect='temporal', seasonal=True, annual=False)
def acf(da: xarray.DataArray, lag: int = 1, time_res: str = 'season') -> xarray.DataArray:
    f"""Autocorrelation function.

    Autocorrelation with lag-{lag} over a {time_res} and averaged over all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    lag: int
      lag.
    time_res : {'season', 'month'}
      Time resolution. 'year' is not an option for this property.
      Eg. If 'month', the autocorrelation is calculated over each month separately for all years.
      Then, the autocorrelation for all Jan/Feb/... is averaged over all years, giving 12 outputs for each grid point.

    Returns
    -------
    xarray.DataArray,
      lag-{lag} autocorrelation of the variable over a {time_res} and averaged over all years.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> acf(da=pr, lag=3, time_res='season')
    Notes
    --------
    See statsmodels.tsa.stattools.acf

    References
    --------
    Alavoine, M., & Grenier, P. (2021). The distinct problems of physical inconsistency and of multivariate bias
    potentially involved in the statistical adjustment of climate simulations. California Digital Library (CDL).
    https://doi.org/10.31223/x5c34c
    """
    if time_res == 'year':
        warn("'year' is not a valid time resolution for this statistical property.")
        return None

    attrs = da.attrs
    da = da.resample(time=res2freq[time_res])

    def acf_last(x, nlags):
        """ statsmodels acf calculates acf for lag 0 to nlags, this return only the last one. """
        out_last = stattools.acf(x, nlags=nlags)
        return out_last[-1]
    out = xarray.apply_ufunc(acf_last, da, input_core_dims=[["time"]], vectorize=True, kwargs={'nlags': lag},
                             dask='parallelized')
    # average over the years
    out = out.groupby(f'__resample_dim__.{time_res}').mean(dim='__resample_dim__')

    out.attrs.update(attrs)
    out.attrs["long_name"] = f"lag-{lag} autocorrelation of {attrs['standard_name']}"
    out.attrs["units"] = ''
    return out


# time_res was kept even though 'year' it the only acceptable arg to keep the signature similar to other properties
@register_statistical_properties(aspect='temporal', seasonal=False, annual=True)
def annual_cycle_amplitude(
    da: xarray.DataArray,
    amplitude_type: str = 'absolute',
    time_res: str = 'year'
) -> xarray.DataArray:
    f"""Annual Cycle amplitude.

    The amplitudes of the annual cycle are calculated for each year, than averaged over the all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.

    amplitude_type: {'absolute','relative'}
        Type of amplitude.
        'absolute' is the peak-to-peak amplitude. (max - min)
        'relative' is a relative percentage. 100 * (max - min) / mean. Recommanded for precipitation.

    Returns
    -------
    out: xarray.DataArray,
      {amplitude_type} amplitude of the annual cycle.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> annual_cycle_amplitude(da=pr, amplitude_type='relative')
    """
    if time_res != 'year':
        warn("'year' is the only valid time resolution for this statistical property.")
        return None
    attrs = da.attrs
    da = da.resample(time='YS')
    # amplitude
    amp = da.max(dim='time') - da.min(dim='time')
    amp.attrs.update(attrs)
    if xc.core.units.units2pint(attrs['units']).dimensionality == xc.core.units.units2pint('degC').dimensionality:
        amp.attrs['units'] = 'delta_degree_Celsius'
    if amplitude_type == 'relative':
        amp = amp * 100 / da.mean(dim='time', keep_attrs=True)
        amp.attrs['units'] = '%'
    amp = amp.mean(dim='time', keep_attrs=True)
    amp.attrs["long_name"] = f"{amplitude_type} amplitude of the annual cycle of {attrs['standard_name']}"
    return amp


# time_res was kept even though 'year' it the only acceptable arg to keep the signature similar to other properties
@register_statistical_properties(aspect='temporal', seasonal=False, annual=True)
def annual_cycle_phase(da: xarray.DataArray, time_res: str = 'year') -> xarray.DataArray:
    f"""Annual Cycle phase.

    The phases of the annual cycle are calculated for each year, than averaged over the all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.

    Returns
    -------
    phase: xarray.DataArray,
      Phase of the annual cycle. The position (day-of-year) of the maximal value.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> annual_cycle_phase(da=pr, amplitude_type='relative')
    """
    if time_res != 'year':
        warn("'year' is the only valid time resolution for this statistical property.")
        return None

    attrs = da.attrs
    mask = ~(da.isel(time=0).isnull())  # mask of the ocean with NaNs
    da = da.resample(time='YS')

    # +1  at the end to go from index to doy
    phase = xarray.apply_ufunc(np.argmax, da, input_core_dims=[["time"]], vectorize=True, dask='parallelized')+1
    phase = phase.mean(dim='__resample_dim__')
    # put nan where there was nan in the input, if not phase = 0 + 1
    phase = phase.where(mask, np.nan)
    phase.attrs.update(attrs)
    phase.attrs["long_name"] = f"Phase of the annual cycle of {attrs['standard_name']}"
    phase.attrs.update(units="", is_dayofyear=1)
    return phase


@register_statistical_properties(aspect='multivariate', seasonal=True, annual=True)
def corr_btw_var(
    da1: xarray.DataArray,
    da2: xarray.DataArray,
    corr_type: str = 'Spearman',
    time_res: str = 'year',
    output: str = 'correlation',
) -> xarray.DataArray:
    f"""Correlation between two variables.

    {corr_type} correlation coefficient between two variables at the time resolution.

    Parameters
    ----------
    da1 : xarray.DataArray
      First variable on which to calculate the diagnostic.
    da2 : xarray.DataArray
      Second variable on which to calculate the diagnostic.
    corr_type: {'Pearson','Spearman'}
        Type of correlation to calculate.
    output: {'correlation', 'pvalue'}
      Wheter to return the correlation coefficient or the p-value.
    time_res : {'year', 'season', 'month'}
      Time resolution.
      Eg. For 'month', the correlation would be calculated on 12 arrays
      (each array contains all days in that month for all years).

    Returns
    -------
    xarray.DataArray,
      {corr_type} correlation coefficient

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> corr_btw_var(da=pr, q=0.9, time_res='season')
    """
    attrs1 = da1.attrs
    attrs2 = da2.attrs
    if time_res != 'year':
        da1 = da1.groupby(f'time.{time_res}')
        da2 = da2.groupby(f'time.{time_res}')

    def first_output(a, b):
        """Only keep the correlation (first output) from the scipy function"""
        index = {'correlation':0, 'pvalue':1}
        if corr_type == 'Pearson':
            # for points in the water with NaNs
            if np.isnan(a[0]):
                return np.nan
            else:
                return scipy.stats.pearsonr(a, b)[index[output]]
        elif corr_type == 'Spearman':
            return scipy.stats.spearmanr(a, b, nan_policy='propagate')[index[output]]
    out = xarray.apply_ufunc(first_output, da1, da2,
                             input_core_dims=[["time"], ["time"]], vectorize=True, dask='parallelized')
    out.attrs.update(attrs1)
    out.attrs["long_name"] = f"{corr_type} correlation coefficient between" \
                              f" {attrs1['standard_name']} and {attrs2['standard_name']}"
    out.attrs["units"] = ""
    return out


@register_statistical_properties(aspect='temporal', seasonal=True, annual=True)
def relative_frequency(
    da: xarray.DataArray,
    op: str = '>=',
    thresh='1mm d-1',
    time_res: str = 'year'
) -> xarray.DataArray:
    f"""Relative Frequency.

    Relative Frequency of days with variable {op} {thresh} at the time resolution.
    Number of days that satisfy the condition / total number of days.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    op: {">", "<", ">=", "<="}
      Operation to verify the condition.
      The condition is variable {op} threshold.
    thresh: str
      Threshold on which to evaluate the condition.
    time_res : {'year', 'season', 'month'}
      Time resolution
      Eg. For 'month', the relative frequency would be calculated on 12 arrays
      (each array contains all days in that month for all years).

    Returns
    -------
    xarray.DataArray,
      Relative frequency of the variable.

    Examples
    --------
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> relative_frequency(da=tas, op= '<', thresh= '0 degC', time_res='season')
    """
    attrs = da.attrs
    mask = ~(da.isel(time=0).isnull())  # mask of the ocean with NaNs
    ops = {">": np.greater,
           "<": np.less,
           ">=": np.greater_equal,
           "<=": np.less_equal}
    t = convert_units_to(thresh, da)
    length = da.sizes['time']
    cond = ops[op](da, t)#, where=np.isfinite(da), out=da.values)  # check the condition
    if time_res != 'year':  # change the time resolution if necessary
        cond = cond.groupby(f'time.{time_res}')
        length = np.array([len(v) for k, v in cond.groups.items()])  # length of the groupBy groups
        for i in range(da.ndim - 1):  # add empty dimension(s) to match input
            length = np.expand_dims(length, axis=-1)
    out = cond.sum(dim='time', skipna=False) / length  # count days with the condition and divide by total nb of days
    out = out.where(mask, np.nan)
    out.attrs.update(attrs)
    out.attrs["long_name"] = f"Relative frequency of days with {attrs['standard_name']} {op} {thresh}"
    out.attrs["units"] = ''
    return out


@register_statistical_properties(aspect='temporal', seasonal=True, annual=True)
def trend(
    da: xarray.DataArray,
    time_res: str = 'year',
    output: str = 'slope',
) -> xarray.DataArray:
    f"""Linear Trend.

    The data is averaged over each {time_res} and the interannual trend is returned.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.

    output: {'slope', 'pvalue'}
      Attributes of the linear regression to return.
      'slope' is the slope of the regression line.
      'pvalue' is  for a hypothesis test whose null hypothesis is that the slope is zero,
       using Wald Test with t-distribution of the test statistic.

    time_res : {'year', 'season', 'month'}
      Time resolution on which to do the initial averaging.

    Returns
    -------
    xarray.DataArray,
      Trend of the variable.

    Examples
    --------
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> trend(da=tas, time_res='season')
    Notes
    --------
    See scipy.stats.linregress and np.polyfit
    """
    attrs = da.attrs
    da = da.resample(time=res2freq[time_res])  # separate all the {time_res}
    da_mean = da.mean(dim='time')  # avg over all {time_res}
    da_mean = da_mean.chunk({'time': -1})
    if time_res != 'year':
        da_mean = da_mean.groupby(f'time.{time_res}')  # group all month/season together

    def modified_lr(x):  # modify linregress to fit into apply_ufunc and only return slope
        return getattr(scipy.stats.linregress(list(range(len(x))), x), output)

    out = xarray.apply_ufunc(modified_lr, da_mean,
                             input_core_dims=[["time"]], vectorize=True, dask='parallelized')
    out.attrs.update(attrs)
    out.attrs["long_name"] = f" {output} of the interannual linear trend of {attrs['standard_name']}"
    out.attrs["units"] = f"{attrs['units']}/year"
    return out


@register_statistical_properties(aspect='marginal', seasonal=True, annual=True)
def return_value(da: xarray.DataArray, period: int = 20, op: str = 'max', method: str ='PWM', time_res: str = 'year')\
    -> xarray.DataArray:
    f"""Return value.

    Return the value corresponding to a return period.
    On average, the return value will be exceeded (or not exceed for op='min') every {period} years.
    The return value is computed by first extracting the variable annual maxima/minima,
    fitting a statistical distribution to the maxima/minima,
    then estimating the percentile associated with the return period (eg. 95th percentile (1/20) for 20 years)

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.

    period: int
      Return period. Number of years over which to check if the value is exceeded (or not for op='min').

    op: {'max','min'}
      Whether we are looking for a probability of exceedance ('max', right side of the distribution)
       or a probability of non-exceedance (min, left side of the distribution).

    method : {"ML", "PWM"}
      Fitting method, either maximum likelihood (ML) or probability weighted moments (PWM), also called L-Moments.
      The PWM method is usually more robust to outliers. However, it requires the lmoments3 libraryto be installed
       from the `develop` branch. `pip install git+https://github.com/OpenHydrology/lmoments3.git@develop#egg=lmoments3`

    time_res : {'year', 'season', 'month'}
      Time resolution on which to create a distribution of the extremums.

    Returns
    -------
    xarray.DataArray,
      {period}-{time_res} {op} return level of the variable.

    Examples
    --------
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> return_value(da=tas, time_res='season')
    """
    attrs = da.attrs

    def frequency_analysis_method(x, method, **indexer):
        sub = select_resample_op(x, op=op, **indexer)
        params = fit(sub, dist="genextreme", method=method)
        out = parametric_quantile(params, q=1 - 1.0 / period)
        return out

    if time_res == 'year':
        out = frequency_analysis_method(da, method)
    else:
        # get coords of final output
        coords = da.groupby(f'time.{time_res}').mean(dim='time').coords
        # create empty dataArray in the shape of final output
        out = xarray.DataArray(coords=coords)
        # iterate through all the seasons/months to get the return value
        for ind in coords[time_res].values:
                #out.loc[{time_res: ind}] = frequency_analysis(da, t=period, dist="genextreme", mode=op,
                                                              #**{time_res: ind}).isel(return_period=0)
            out.loc[{time_res: ind}] = frequency_analysis_method(da, method, **{time_res: ind}).isel(quantile=0)

    out.attrs.update(attrs)
    out.attrs["long_name"] = f" {period}-{time_res} {op} return level of {attrs['standard_name']}"
    return out





