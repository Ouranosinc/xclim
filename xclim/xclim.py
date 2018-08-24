# -*- coding: utf-8 -*-

"""Main module."""
import dask
import numpy as np
# import pandas as pd
import xarray as xr


from .checks import *
from .run_length import windowed_run_count

K2C = 273.15




@valid_daily_max_temperature
def HWI(tasmax, thresh=25, window=5, freq='YS'):
    """Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature.
    thresh : float
      Threshold temperature to designate a heatwave [℃].
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.

    Returns
    -------
    DataArray
      Heat wave index.
    """
    # TODO: Deal better with boundary effects.
    # TODO: Deal with attributes

    over = tasmax > K2C + thresh
    group = over.resample(time=freq)
    func = lambda x: xr.apply_ufunc(windowed_run_count,
                          x,
                          input_core_dims=[['time'],],
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.int,],
                          keep_attrs=True,
                          kwargs={'window': window})

    return group.apply(func)
