# -*- coding: utf-8 -*-
"""
Statistical distribution fit module
"""
# Note: stats.dist.shapes: comma separated names of shape parameters
# The other parameters, common to all distribution, are loc and scale.

import dask
import xarray as xr


def get_dist(dist):
    """Return a distribution object from scipy.stats.
    """
    from scipy import stats

    dc = getattr(stats, dist, None)
    if dc is None:
        e = "Statistical distribution `{}` is not in scipy.stats.".format(dist)
        raise ValueError(e)
    return dc


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
    out.attrs['history'] = out.attrs.get('history', '') + \
                           'Data fitted with {0} statistical distribution using a Maximum Likelihood ' \
                           'Estimator'.format(dist)

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
    mode : {'low', 'high'}
      Whether we are looking for a probability of exceedance (high) or a probability of non-exceedance (low).

    Returns
    -------
    xarray.DataArray
      An array of values with a 1/T probability of exceedance (if mode=='high').
    """

    # Get the distribution
    dc = get_dist(dist)

    p = fit(arr, dist)

    if mode == 'high':
        func = lambda x: dc.isf(1./t, *x)
    elif mode == 'low':
        func = lambda x: dc.ppf(1./t, *x)
    else:
        raise ValueError("mode `{}` should be either 'high' or 'low'".format(mode))

    data = dask.array.apply_along_axis(func, p.get_axis_num('dparams'), p)
    # TODO: Update attributes
    return data
