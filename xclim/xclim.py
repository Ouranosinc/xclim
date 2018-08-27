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

@valid_daily_mean_temperature
def CSI(tas, thresh=-10, window=5, freq='AS-JUL'):
    """Cold spell index.
    """
    over = tas < K2C + thresh
    group = over.resample(time=freq)
    func = lambda x: xr.apply_ufunc(windowed_run_count,
                                    x,
                                    input_core_dims=[['time'], ],
                                    vectorize=True,
                                    dask='parallelized',
                                    output_dtypes=[np.int, ],
                                    keep_attrs=True,
                                    kwargs={'window': window})

    return group.apply(func)

@valid_daily_max_min_temperature
def daily_freezethaw_cycles(tasmax, tasmin, freq='YS'):
    """Number of days with a freeze-thaw cycle.

    The number of days where Tmax > 0℃ and Tmin < 0℃.
    """
    ft = (tasmin < K2C) * (tasmax > K2C) * 1
    return ft.resample(time=freq).sum(dim='time')

@valid_daily_max_temperature
def hotdays(tasmax, thresh=30, freq):
    """Number of very hot days.

    The number of days exceeding a threshold. """
    hd = (tasmax > K2C + thresh)*1
    return hd.resample(time=freq).sum(dim='time')

@valid_daily_mean_temperature
def CoolingDD(tas, thresh=18):
    """Cooling degree days above threshold."""
    cdd = (tas > K2C + thresh) * 1
    return cdd.resample(time=freq).sum(dim='time')




