"""Detrending objects."""
from typing import Union

import xarray as xr

from .base import Grouper, Parametrizable, parse_group
from .loess import loess_smoothing
from .utils import ADDITIVE, apply_correction, invert


class BaseDetrend(Parametrizable):
    """Base class for detrending objects.

    Defines three methods:

    fit(da)      : Compute trend from da and return a new _fitted_ Detrend object.
    get_trend(da): Return the fitted trend along da's coordinate.
    detrend(da)  : Return detrended array.
    retrend(da)  : Puts trend back on da.

    * Subclasses should implement _fit() and _get_trend(). Both will be called in a `group.apply()`.
    `_fit()` is called with the dataarray and str `dim` that indicates the fitting dimension,
        it should return a dataset that will be set as `.fitds`.
    `_get_trend()` is called with .fitds broadcasted on the main dim of the input DataArray.
    """

    @parse_group
    def __init__(
        self, *, group: Union[Grouper, str] = "time", kind: str = "+", **kwargs
    ):
        """Initialize Detrending object.

        Parameters
        ----------
        group : Union[str, Grouper]
            The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
            The fit is performed along the group's main dim.
        kind : {'*', '+'}
            The way the trend is removed or added, either additive or multiplicative.
        """
        self.__fitted = False
        super().__init__(group=group, kind=kind, **kwargs)

    def fit(self, da: xr.DataArray):
        """Extract the trend of a DataArray along a specific dimension.

        Returns a new object storing the fit data that can be used for detrending and retrending.
        """
        new = self.copy()
        new._set_ds(new.group.apply(new._fit, da, main_only=True))
        new.__fitted = True
        return new

    def get_trend(self, da: xr.DataArray):
        """Get the trend computed from the fit, along the self.group.dim as found on da.

        If da is a DataArray (and has a "dtype" attribute), the trend is casted to have the same dtype.
        """
        out = self.group.apply(
            self._get_trend,
            {self.group.dim: da[self.group.dim], **self.ds.data_vars},
            main_only=True,
        )
        if hasattr(da, "dtype"):
            out = out.astype(da.dtype)
        return out

    def detrend(self, da: xr.DataArray):
        """Remove the previously fitted trend from a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before detrending.")
        trend = self.get_trend(da)
        return self._detrend(da, trend)

    def retrend(self, da: xr.DataArray):
        """Replace the previously fitted trend on a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before retrending")
        trend = self.get_trend(da)
        return self._retrend(da, trend)

    def _set_ds(self, ds):
        self.ds = ds
        self.ds.attrs["fit_params"] = str(self)

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
    """Simple detrending removing only the mean from the data, quite similar to normalizing."""

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
    group : Union[str, Grouper]
        The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        The fit is performed along the group's main dim.
    kind : {'*', '+'}
        The way the trend is removed or added, either additive or multiplicative.
    degree : int
        The order of the polynomial to fit.
    preserve_mean : bool
        Whether to preserve the mean when de/re-trending. If True, the trend has its mean
        removed before it is used.
    """

    def __init__(self, group="time", kind=ADDITIVE, degree=4, preserve_mean=False):
        super().__init__(
            group=group, kind=kind, degree=degree, preserve_mean=preserve_mean
        )

    def _fit(self, da, dim="time"):
        return da.polyfit(dim=dim, deg=self.degree)

    def _get_trend(self, grpd, dim="time"):
        # Estimate trend over da
        trend = xr.polyval(coord=grpd[dim], coeffs=grpd.polyfit_coefficients)

        if self.preserve_mean:
            trend = apply_correction(
                trend, invert(trend.mean(dim=dim), self.kind), self.kind
            )

        return trend


class LoessDetrend(BaseDetrend):
    """
    Detrend time series using a LOESS regression.

    The fit is a piecewise linear regression. For each point, the contribution of all
    neighbors is weighted by a bell-shaped curve (gaussian) with parameters sigma (std).
    The x-coordinate of the dataarray is scaled to [0,1] before the regression is computed.

    Parameters
    ----------
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      The fit is performed along the group's main dim.
    kind : {'*', '+'}
      The way the trend is removed or added, either additive or multiplicative.
    d: [0, 1]
      Order of the local regression. Only 0 and 1 currently implemented.
    f : float
      Parameter controling the span of the weights, between 0 and 1.
    niter : int
      Number of robustness iterations to execute.
    weights : ["tricube", "gaussian"]
      Shape of the weighting function:
      "tricube" : a smooth top-hat like curve, f gives the span of non-zero values.
      "gaussian" : a gaussian curve, f gives the span for 95% of the values.

    Notes
    -----
    LOESS smoothing is computationally expensive. As it relies on a loop on gridpoints, it
    can be useful to use smaller than usual chunks.
    Moreover, it suffers from heavy boundary effects. As a rule of thumb, the outermost N * f/2 points
    should be considered dubious. (N is the number of points along each group)
    """

    def __init__(
        self, group="time", kind=ADDITIVE, f=0.2, niter=1, d=0, weights="tricube"
    ):
        super().__init__(group=group, kind=kind, f=f, niter=niter, d=0, weights=weights)

    def _fit(self, da, dim="time"):
        trend = loess_smoothing(
            da,
            dim=self.group.dim,
            f=self.f,
            niter=self.niter,
            d=self.d,
            weights=self.weights,
        )
        trend.name = "trend"
        return trend.to_dataset()

    def get_trend(self, da: xr.DataArray):
        """Get the trend computed from the fit, along the self.group.dim as found on da.

        If da is a DataArray (and has a "dtype" attribute), the trend is casted to have the same dtype.
        """
        # Check if we need to interpolate
        if da[self.group.dim].equals(self.ds[self.group.dim]):
            out = self.ds.trend
        else:
            out = self.ds.trend.interp(coords={self.group.dim: da[self.group.dim]})

        if hasattr(da, "dtype"):
            out = out.astype(da.dtype)
        return out
