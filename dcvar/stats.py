# -*- coding: utf-8 -*-
"""Statistical distribution fit"""

import numpy as np
import xarray as xr
from scipy import stats

def fit(arr, dist='norm'):
    """Fit an array to a distribution along the time dimension."""
    dc = getattr(stats, dist, None)
    if dc is None:
        raise ValueError("Statistical distribution {} is not in scipy.stats.".format(dist))

    # Fit the parameters
    params = np.apply_along_axis(dc.fit, arr.get_axis_num('time'), arr.data)


#    out = xr.apply_ufunc(lambda x: np.apply_along_axis(dc.fit, -1, x,),
#                         arr,
#                         input_core_dims=[['time']],
#                         output_core_dims=(('params')),
#                         keep_attrs=True,
#                         dask='allowed',
#                         )
    return xr.DataArray(params, )


    # Create a DataArray with the parameters
#    return params
