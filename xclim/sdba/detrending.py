"""Detrending objects"""
import xarray as xr

from .base import Parametrizable
from .base import parse_group
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

    @parse_group
    def __init__(self, *, group="time", kind="+", **kwargs):
        self.__fitted = False
        super().__init__(group=group, kind=kind, **kwargs)

    def fit(self, da: xr.DataArray):
        """Extract the trend of a DataArray along a specific dimension.

        Returns a new object storing the fit data that can be used for detrending and retrending.
        """
        new = self.copy()
        new._set_fitds(new.group.apply(new._fit, da, main_only=True))
        new.__fitted = True
        return new

    def get_trend(self, da: xr.DataArray):
        return self.group.apply(
            self._get_trend,
            {self.group.dim: da[self.group.dim], **self.fit_ds.data_vars},
            main_only=True,
        )

    def detrend(self, da: xr.DataArray):
        """Removes the previously fitted trend from a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before detrending.")
        trend = self.get_trend(da)
        return self._detrend(da, trend)

    def retrend(self, da: xr.DataArray):
        """Puts back the previsouly fitted trend on a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before retrending")
        trend = self.get_trend(da)
        return self._retrend(da, trend)

    def _set_fitds(self, ds):
        self.fit_ds = ds

    def _detrend(self, da, trend):
        # Remove trend from series
        return apply_correction(da, invert(trend, self.kind), self.kind)

    def _retrend(self, da, trend):
        # Add trend to series
        return apply_correction(da, trend, self.kind)

    def _get_trend(self, grpd, dim="time"):
        raise NotImplementedError

    def _fit(self, da):
        raise NotImplementedError


class NoDetrend(BaseDetrend):
    """Convenience class for polymorphism. Does nothing."""

    def _fit(self, da, dim=None):
        return da.isel({dim: 0})

    def _detrend(self, da, trend):
        return da

    def _retrend(self, da, trend):
        return da


class MeanDetrend(BaseDetrend):
    """Simple detrending removing only the mean from the data, quite similar to normalizing in additive mode."""

    def _fit(self, da, dim="time"):
        mean = da.mean(dim=dim)
        mean.name = "mean"
        return mean

    def _get_trend(self, grpd, dim="time"):
        return grpd.mean


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

    def __init__(self, degree=4, preserve_mean=False, **kwargs):
        super().__init__(degree=degree, preserve_mean=preserve_mean, **kwargs)

    def _fit(self, da, dim="time"):
        # if self.freq is not None:
        #     da = da.resample(
        #         time=self.freq, label="left", loffset=loffsets[self.freq]
        #     ).mean()
        return da.polyfit(dim=dim, deg=self.degree)

    def _get_trend(self, grpd, dim="time"):
        # Estimate trend over da
        trend = xr.polyval(coord=grpd[dim], coeffs=grpd.polyfit_coefficients)

        if self.preserve_mean:
            trend = apply_correction(
                trend, invert(trend.mean(dim=dim), self.kind), self.kind
            )

        return trend
