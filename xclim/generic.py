# -*- coding: utf-8 -*-
# Note: stats.dist.shapes: comma separated names of shape parameters
# The other parameters, common to all distribution, are loc and scale.

import dask
import numpy as np
import xarray as xr


def select_time(da, **indexer):
    """Select entries according to a time period.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xr.DataArray
      Selected input values.
    """
    if not indexer:
        selected = da
    else:
        key, val = indexer.popitem()
        time_att = getattr(da.time.dt, key)
        selected = da.sel(time=time_att.isin(val)).dropna(dim='time')

    return selected


def select_resample_op(da, op, freq="YS", **indexer):
    """Apply operation over each period that is part of the index selection.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'argmax', 'argmin'} or func
      Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xarray.DataArray
      The maximum value for each period.
    """
    da = select_time(da, **indexer)
    r = da.resample(time=freq, keep_attrs=True)
    if isinstance(op, str):
        return getattr(r, op)(dim='time', keep_attrs=True)

    return r.apply(op)


def doymax(da):
    """Return the day of year of the maximum value."""
    i = da.argmax(dim='time')
    out = da.time.dt.dayofyear[i]
    out.attrs['units'] = ''
    return out


def doymin(da):
    """Return the day of year of the minimum value."""
    i = da.argmax(dim='time')
    out = da.time.dt.dayofyear[i]
    out.attrs['units'] = ''
    return out


def fit(arr, dist='norm'):
    """Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    arr : xarray.DataArray
      Time series to be fitted along the time dimension.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).

    Returns
    -------
    xarray.DataArray
      An array of distribution parameters fitted using the method of Maximum Likelihood.
    """
    # Get the distribution
    dc = get_dist(dist)

    # Fit the parameters (lazy computation)
    data = dask.array.apply_along_axis(dc.fit, arr.get_axis_num('time'), arr)

    # Count the number of values used for the fit.
    # n = arr.count(dim='time')

    # Create a view to a DataArray with the desired dimensions to copy them over to the parameter array.
    mean = arr.mean(dim='time', keep_attrs=True)

    # Create coordinate for the distribution parameters
    coords = dict(mean.coords.items())
    coords['dparams'] = ([] if dc.shapes is None else dc.shapes.split(',')) + ['loc', 'scale']

    # TODO: add time and time_bnds coordinates (Low will work on this)
    # time.attrs['climatology'] = 'climatology_bounds'
    # coords['time'] =
    # coords['climatology_bounds'] =

    out = xr.DataArray(data=data, coords=coords, dims=(u'dparams',) + mean.dims)
    out.attrs = arr.attrs
    out.attrs['original_name'] = getattr(arr, 'standard_name', '')
    out.attrs['standard_name'] = '{0} distribution parameters'.format(dist)
    out.attrs['long_name'] = '{0} distribution parameters for {1}'.format(dist, getattr(arr, 'standard_name', ''))
    out.attrs['estimator'] = 'Maximum likelihood'
    out.attrs['cell_methods'] = (out.attrs.get('cell_methods', '') + ' time: fit').strip()
    out.attrs['units'] = ''
    msg = '\nData fitted with {0} statistical distribution using a Maximum Likelihood Estimator'
    out.attrs['history'] = out.attrs.get('history', '') + msg.format(dist)

    return out


def fa(arr, t, dist='norm', mode='high'):
    """Return the value corresponding to the given return period.

    Parameters
    ----------
    arr : xarray.DataArray
      Maximized/minimized input data with a `time` dimension.
    t : int or sequence
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).
    mode : {'min', 'max}
      Whether we are looking for a probability of exceedance (max) or a probability of non-exceedance (min).

    Returns
    -------
    xarray.DataArray
      An array of values with a 1/t probability of exceedance (if mode=='max').
    """
    t = np.atleast_1d(t)

    # Get the distribution
    dc = get_dist(dist)

    # Fit the parameters of the distribution
    p = fit(arr, dist)

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    if mode in ['max', 'high']:
        def func(x):
            return dc.isf(1. / t, *x)
    elif mode in ['min', 'low']:
        def func(x):
            return dc.ppf(1. / t, *x)
    else:
        raise ValueError("mode `{}` should be either 'max' or 'min'".format(mode))

    data = dask.array.apply_along_axis(func, p.get_axis_num('dparams'), p)

    # Create coordinate for the return periods
    coords = dict(p.coords.items())
    coords.pop('dparams')
    coords['return_period'] = t

    # Create dimensions
    dims = list(p.dims)
    dims.remove('dparams')
    dims.insert(0, u'return_period')

    # TODO: add time and time_bnds coordinates (Low will work on this)
    # time.attrs['climatology'] = 'climatology_bounds'
    # coords['time'] =
    # coords['climatology_bounds'] =

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = p.attrs
    out.attrs['standard_name'] = '{0} quantiles'.format(dist)
    out.attrs['long_name'] = '{0} return period values for {1}'.format(dist, getattr(arr, 'standard_name', ''))
    out.attrs['cell_methods'] = (out.attrs.get('cell_methods', '') + ' dparams: ppf').strip()
    out.attrs['units'] = arr.attrs.get('units', '')
    out.attrs['mode'] = mode
    out.attrs['history'] = out.attrs.get('history', '') + "Compute values corresponding to return periods."

    return out


def frequency_analysis(da, mode, t, dist, window=1, freq=None, **indexer):
    """Return the value corresponding to a return period.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    t : int or sequence
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).
    mode : {'min', 'max'}
      Whether we are looking for a probability of exceedance (high) or a probability of non-exceedance (low).
    window : int
      Averaging window length (days).
    freq : str
      Resampling frequency. If None, the frequency is assumed to be 'YS' unless the indexer is season='DJF',
      in which case `freq` would be set to `YS-DEC`.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xarray.DataArray
      An array of values with a 1/t probability of exceedance or non-exceedance when mode is high or low respectively.

    """
    # Apply rolling average
    if window > 1:
        da = da.rolling(time=window, center=False).mean()

    # Assign default resampling frequency if not provided
    freq = freq or default_freq(**indexer)

    # Extract the time series of min or max over the period
    sel = select_resample_op(da, op=mode, freq=freq, **indexer).dropna(dim='time')

    # Frequency analysis
    return fa(sel, t, dist, mode)


def default_freq(**indexer):
    """Return the default frequency."""
    freq = 'AS-JAN'
    if indexer:
        if 'DJF' in indexer.values():
            freq = 'AS-DEC'
        if 'month' in indexer and sorted(indexer.values()) != indexer.values():
            raise (NotImplementedError)

    return freq


def get_dist(dist):
    """Return a distribution object from scipy.stats.
    """
    from scipy import stats

    dc = getattr(stats, dist, None)
    if dc is None:
        e = "Statistical distribution `{}` is not in scipy.stats.".format(dist)
        raise ValueError(e)
    return dc
