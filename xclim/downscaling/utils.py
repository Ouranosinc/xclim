"""Utilities for the downscaling module"""
import numpy as np
import xarray as xr

MULTIPLICATIVE = "*"
ADDITIVE = "+"


class ParametrizableClass(object):
    """Helper base class that sets as attribute every kwarg it receives in __init__.

    Parameters are all public attributes. Subclasses should use private attributes (starting with _).
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def parameters(self):
        """Return all parameters as a dictionary"""
        return {
            key: val for key, val in self.__dict__.items() if not key.startswith("_")
        }


def interp_quantiles(xq, yq, x):
    return xr.apply_ufunc(
        np.interp,
        x,
        xq,
        yq,
        input_core_dims=[["time"], ["quantile"], ["quantile"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float],
    )
