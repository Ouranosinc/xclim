# -*- coding: utf-8 -*-
"""
Statistical distribution fit module
"""

import dask
# import logging
import xarray as xr
from scipy import stats

# log = logging.getLogger(__name__)


def fit(arr, dist='norm'):
    """Fit an array to a distribution along the time dimension."""

    # Note: stats.dist.shapes: comma separated names of shape parameters
    # The other parameters, common to all distribution, are loc and scale.

    # Get the distribution object
    dc = getattr(stats, dist, None)
    if dc is None:
        e = "Statistical distribution {} is not in scipy.stats.".format(dist)
        raise ValueError(e)

    # Fit the parameters (lazy computation)
    data = dask.array.apply_along_axis(dc.fit, arr.get_axis_num('time'), arr)

    # Create a DataArray with the desired dimensions to copy them over to the parameter array.
    mean = arr.mean(dim='time', keep_attrs=True)
    coords = dict(mean.coords.items())
    coords['dparams'] = ([] if dc.shapes is None else dc.shapes.split(',')) + ['loc', 'scale']

    # TODO: add time and time_bnds coordinates
    # time.attrs['climatology'] = 'climatology_bounds'
    # coords['time'] =
    # coords['climatology_bounds'] =

    out = xr.DataArray(data=data, coords=coords, dims=(u'dparams',) + mean.dims)
    out.attrs = arr.attrs
    out.attrs['original_name'] = arr.standard_name
    out.attrs['standard_name'] = '{0} distribution parameters'.format(dist)
    out.attrs['long_name'] = '{0} distribution parameters for {1}'.format(dist, arr.standard_name)
    out.attrs['estimator'] = 'Maximum likelihood'
    out.attrs['cell_methods'] += ' time: fit'
    out.attrs['units'] = ''
    out.attrs['history'] += 'Data fitted with {0} statistical distribution using a Maximum Likelihood Estimator'

    return out


# FIXME: Write a name for this test
def test():
    fn = '~/src/flyingpigeon/flyingpigeon/tests/testdata/cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc'
    D = xr.open_dataset(fn, chunks={'lat': 1}, decode_cf=True)
    p = fit(D.tas)
    stats.norm.cdf(.99, *p.values)
