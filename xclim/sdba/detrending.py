"""Detrending objects"""
import xarray as xr

from .base import Parametrizable
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import invert
from .utils import loffsets


class BaseDetrend(Parametrizable):
    """Base class for detrending objects

    Defines three methods:

    fit(da)     : Compute trend from da and return a new _fitted_ Detrend object.
    detrend(da) : Return detrended array.
    retrend(da) : Puts trend back on da.

    * Subclasses should implement _fit(), _detrend() and _retrend(), not the methods themselves.
    Only _fit() should store data. _detrend() and _retrend() are meant to be used on any dataarray with the trend computed in fit.
    """

    def __init__(self, **kwargs):
        self.__fitted = False
        super().__init__(**kwargs)

    def fit(self, da: xr.DataArray, dim="time"):
        """Extract the trend of a DataArray along a specific dimension.

        Returns a new object storing the fit data that can be used for detrending and retrending.
        """
        new = self.copy()
        new._fit(da, dim=dim)
        new._fitted_dim = dim
        new.__fitted = True
        return new

    def detrend(self, da: xr.DataArray):
        """Removes the previously fitted trend from a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before detrending.")
        return self._detrend(da)

    def retrend(self, da: xr.DataArray):
        """Puts back the previsouly fitted trend on a DataArray."""
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
    """Convenience class for polymorphism. Does nothing."""

    def _fit(self, da, dim=None):
        pass

    def _detrend(self, da):
        return da

    def _retrend(self, da):
        return da


class MeanDetrend(BaseDetrend):
    """Simple detrending removing only the mean from the data, quite similar to normalizing in additive mode."""

    def _fit(self, da, dim="time"):
        self._mean = da.mean(dim=dim)

    def _detrend(self, da):
        return da - self._mean

    def _retrend(self, da):
        return da + self._mean


class PolyDetrend(BaseDetrend):
    """
    Detrend time series using a polynomial regression.

    Parameters
    ----------
    degree : int
      The order of the polynomial to fit.
    freq : Optional[str]
      If given, resamples the data to this frequency before computing the trend.
    kind : {'+', '*'}
      The way the trend is removed and put back, either additively or multiplicatively.

    Notes
    -----
    If freq is used to resample at a lower frequency, make sure the series includes full periods.
    """

    def __init__(self, degree=4, freq=None, kind=ADDITIVE, preserve_mean=False):
        super().__init__(
            degree=degree, freq=freq, kind=kind, preserve_mean=preserve_mean
        )

    def _fit(self, da, dim="time"):
        if self.freq is not None:
            da = da.resample(
                time=self.freq, label="left", loffset=loffsets[self.freq]
            ).mean()
        self._fitds = da.polyfit(dim=dim, deg=self.degree, full=True)

    def _get_trend(self, da):
        # Estimate trend over da
        trend = xr.polyval(
            coord=da[self._fitted_dim], coeffs=self._fitds.polyfit_coefficients
        )

        if self.preserve_mean:
            trend = apply_correction(
                trend, invert(trend.mean(dim=self._fitted_dim), self.kind), self.kind
            )

        return trend

    def _detrend(self, da):
        # Remove trend from series
        return apply_correction(da, invert(self._get_trend(da), self.kind), self.kind)

    def _retrend(self, da):
        # Add trend to series
        return apply_correction(da, self._get_trend(da), self.kind)
