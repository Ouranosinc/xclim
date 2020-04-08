"""Detrending objects"""
import xarray as xr

from .base import ParametrizableClass
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import invert
from .utils import loffsets


class BaseDetrend(ParametrizableClass):
    """Base class for detrending objects

    Defines three methods:

    fit(da)     : Compute trend from da and return a new _fitted_ Detrend object.
    detrend(da) : Return detrended array.
    retrend(da) : Puts trend back on da.

    * Subclasses should implement _fit(), _detrend() and _retrend(), not the methods themselves.
    Only _fit() should store data. _detrend() and _retrend() are meant to be used on any dataarray with the trend computed in fit.
    """

    __fitted = False

    def fit(self, da: xr.DataArray, dim="time"):
        new = self.__class__(**self.parameters)
        new._fit(da, dim=dim)
        new._fitted_dim = dim
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
        raise NotImplementedError

    def _detrend(self, da):
        raise NotImplementedError

    def _retrend(self, da):
        raise NotImplementedError


class NoDetrend(BaseDetrend):
    def _fit(self, da, dim=None):
        pass

    def _detrend(self, da):
        return da

    def _retrend(self, da):
        return da


class MeanDetrend(BaseDetrend):
    def _fit(self, da, dim="time"):
        self._mean = da.mean(dim=dim)

    def _detrend(self, da):
        return da - self._mean

    def _retrend(self, da):
        return da + self._mean


class PolyDetrend(BaseDetrend):
    """
    Detrend time series using a polynomial.

    Notes
    -----
    If freq is used to resample at a lower frequency, make sure the series includes full periods.
    """

    def __init__(self, degree=4, freq=None, kind=ADDITIVE):
        super().__init__(degree=degree, freq=freq, kind=kind)

    def _fit(self, da, dim="time"):
        if self.freq is not None:
            da = da.resample(
                time=self.freq, label="left", loffset=loffsets[self.freq]
            ).mean()
        self._fitds = da.polyfit(dim=dim, deg=self.degree, full=True)

    def _detrend(self, da):
        # Estimate trend over da
        trend = xr.polyval(
            coord=da[self._fitted_dim], coeffs=self._fitds.polyfit_coefficients
        )

        # Remove trend from series
        return apply_correction(da, invert(trend, self.kind), self.kind)

    def _retrend(self, da):
        # Estimate trend over da
        trend = xr.polyval(
            coord=da[self._fitted_dim], coeffs=self._fitds.polyfit_coefficients
        )

        # Add trend to series
        return apply_correction(da, trend, self.kind)
