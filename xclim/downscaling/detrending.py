"""Detrending classes"""
from warnings import warn

import xarray as xr

from .base import ParametrizableClass


class NoDetrend(ParametrizableClass):
    """Base class for detrending objects

    Defines three methods:

    fit(da)     : Compute trend from da and return a new _fitted_ Detrend object.
    detrend(da) : Return detrended array.
    retrend(da) : Puts trend back on da.

    * Subclasses should implement _fit(), _detrend() and _retrend(), not the methods themselves.
    Only _fit() should store data. _detrend() and _retrend() are meant to be used on any dataarray with the trend computed in fit.
    """

    __fitted = False

    def fit(self, da: xr.DataArray):
        new = self.__class__(**self.parameters)
        new._fit(da)
        new.__fitted = True
        return new

    def detrend(self, da: xr.DataArray):
        if not self.__fitted:
            raise ValueError("You must call fit() before detrending.")
        return self._detrend(da)

    def retrend(self, da: xr.DataArray):
        if not self.__fitted:
            raise ValueError("You must call fit() before retrending")
        return self._retrend(da)

    def _fit(self, da):
        pass

    def _detrend(self, da):
        return da

    def _retrend(self, da):
        return da


class MeanDetrend(NoDetrend):
    def _fit(self, da):
        self._mean = da.mean(dim="time")

    def _detrend(self, da):
        return da - self._mean

    def _retrend(self, da):
        return da + self._mean


class PolyDetrend(NoDetrend):
    def __init__(self, degree=4):
        super().__init__(self, degree=degree)

    def _fit(self, da):
        self._fitds = da.polyfit(dim="time", deg=self.degree, full=True)

    def _detrend(self, da):
        trend = xr.polyval(coord=da["time"], coeffs=self._fitds.polyfit_coefficients)
        return da - trend

    def _retrend(self, da):
        trend = xr.polyval(coord=da["time"], coeffs=self._fitds.polyfit_coefficients)
        return da - trend
