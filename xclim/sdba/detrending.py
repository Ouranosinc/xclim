"""
Detrending Objects Utilities
============================
"""
from __future__ import annotations

from typing import Union

import xarray as xr

from xclim.core.units import convert_units_to

from .base import Grouper, ParametrizableWithDataset, map_groups, parse_group
from .loess import loess_smoothing
from .utils import ADDITIVE, apply_correction, invert


class BaseDetrend(ParametrizableWithDataset):
    """Base class for detrending objects.

    Defines three methods:

    fit(da)      : Compute trend from da and return a new _fitted_ Detrend object.
    detrend(da)  : Return detrended array.
    retrend(da)  : Puts trend back on da.

    A fitted `Detrend` object is unique to the trend coordinate of the object used in `fit`, (usually 'time').
    The computed trend is stored in ``Detrend.ds.trend``.

    Subclasses should implement ``_get_trend_group()`` or ``_get_trend()``.
    The first will be called in a ``group.apply(..., main_only=True)``, and should return a single DataArray.
    The second allows the use of functions wrapped in :py:func:`map_groups` and should also return a single DataArray.

    The subclasses may reimplement ``_detrend`` and ``_retrend``.
    """

    @parse_group
    def __init__(self, *, group: Grouper | str = "time", kind: str = "+", **kwargs):
        """Initialize Detrending object.

        Parameters
        ----------
        group : Union[str, Grouper]
            The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
            The fit is performed along the group's main dim.
        kind : {'*', '+'}
            The way the trend is removed or added, either additive or multiplicative.
        """
        super().__init__(group=group, kind=kind, **kwargs)

    @property
    def fitted(self):
        """Return whether instance is fitted."""
        return hasattr(self, "ds")

    def fit(self, da: xr.DataArray):
        """Extract the trend of a DataArray along a specific dimension.

        Returns a new object that can be used for detrending and retrending.
        Fitted objects are unique to the fitted coordinate used.
        """
        new = self.__class__(**self.parameters)
        new.set_dataset(new._get_trend(da).rename("trend").to_dataset())
        new.ds.trend.attrs["units"] = da.attrs.get("units", "")
        return new

    def _get_trend(self, da: xr.DataArray):
        """Compute the trend along the self.group.dim as found on da.

        If da is a DataArray (and has a `dtype` attribute), the trend is cast to have the same dtype.

        Notes
        -----
        This method applies `_get_trend_group` with `self.group`.
        """
        out = self.group.apply(
            self._get_trend_group,
            da,
        )
        if hasattr(da, "dtype"):
            out = out.astype(da.dtype)
        return out.rename("trend")

    def detrend(self, da: xr.DataArray):
        """Remove the previously fitted trend from a DataArray."""
        if not self.fitted:
            raise ValueError("You must call fit() before detrending.")
        trend = self.ds.trend
        if "units" in da.attrs:
            trend = convert_units_to(self.ds.trend, da, context="infer")
        return self._detrend(da, trend)

    def retrend(self, da: xr.DataArray):
        """Put the previously fitted trend back on a DataArray."""
        if not self.fitted:
            raise ValueError("You must call fit() before retrending")
        trend = self.ds.trend
        if "units" in da.attrs:
            trend = convert_units_to(self.ds.trend, da, context="infer")
        return self._retrend(da, trend)

    def _detrend(self, da, trend):
        # Remove trend from series
        return apply_correction(da, invert(trend, self.kind), self.kind)

    def _retrend(self, da, trend):
        # Add trend to series
        return apply_correction(da, trend, self.kind)

    def _get_trend_group(self, grpd, *, dim):
        raise NotImplementedError

    def __repr__(self):
        """Format instance representation."""
        rep = super().__repr__()
        if not self.fitted:
            return f"<{rep} | unfitted>"
        return rep


class NoDetrend(BaseDetrend):
    """Convenience class for polymorphism. Does nothing."""

    def _get_trend_group(self, da, *, dim):
        return da.isel({d: 0 for d in dim})

    def _detrend(self, da, trend):
        return da

    def _retrend(self, da, trend):
        return da


class MeanDetrend(BaseDetrend):
    """Simple detrending removing only the mean from the data, quite similar to normalizing."""

    def _get_trend(self, da):
        return _meandetrend_get_trend(da, **self).trend


@map_groups(trend=[Grouper.DIM])
def _meandetrend_get_trend(da, *, dim, kind):
    trend = da.mean(dim).broadcast_like(da)
    return trend.rename("trend").to_dataset()


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

    def _get_trend(self, da):
        # Estimate trend over da
        trend = _polydetrend_get_trend(da, **self)
        return trend.trend


@map_groups(trend=[Grouper.DIM])
def _polydetrend_get_trend(da, *, dim, degree, preserve_mean, kind):
    """Polydetrend, atomic func on 1 group."""
    if len(dim) > 1:
        da = da.mean(dim[1:])
    dim = dim[0]
    pfc = da.polyfit(dim=dim, deg=degree)
    trend = xr.polyval(coord=da[dim], coeffs=pfc.polyfit_coefficients)

    if preserve_mean:
        trend = apply_correction(trend, invert(trend.mean(dim=dim), kind), kind)
    out = trend.rename("trend").to_dataset()
    return out


class LoessDetrend(BaseDetrend):
    """
    Detrend time series using a LOESS regression.

    The fit is a piecewise linear regression. For each point, the contribution of all
    neighbors is weighted by a bell-shaped curve (gaussian) with parameters sigma (std).
    The x-coordinate of the DataArray is scaled to [0,1] before the regression is computed.

    Parameters
    ----------
    group : str or Grouper
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      The fit is performed along the group's main dim.
    kind : {'*', '+'}
      The way the trend is removed or added, either additive or multiplicative.
    d: [0, 1]
      Order of the local regression. Only 0 and 1 currently implemented.
    f : float
      Parameter controlling the span of the weights, between 0 and 1.
    niter : int
      Number of robustness iterations to execute.
    weights : ["tricube", "gaussian"]
      Shape of the weighting function:
      "tricube" : a smooth top-hat like curve, f gives the span of non-zero values.
      "gaussian" : a gaussian curve, f gives the span for 95% of the values.
    skipna : bool
      If True (default), missing values are not included in the loess trend computation
      and thus are not propagated. The output will have the same missing values as the input.

    Notes
    -----
    LOESS smoothing is computationally expensive. As it relies on a loop on gridpoints, it can be useful to use
    smaller than usual chunks. Moreover, it suffers from heavy boundary effects. As a rule of thumb, the outermost
    N * f/2 points should be considered dubious. (N is the number of points along each group)
    """

    def __init__(
        self,
        group="time",
        kind=ADDITIVE,
        f=0.2,
        niter=1,
        d=0,
        weights="tricube",
        equal_spacing=None,
        skipna=True,
    ):
        super().__init__(
            group=group,
            kind=kind,
            f=f,
            niter=niter,
            d=0,
            weights=weights,
            equal_spacing=equal_spacing,
            skipna=skipna,
        )

    def _get_trend(self, da):
        # Estimate trend over da
        trend = _loessdetrend_get_trend(da, **self)
        return trend.trend


@map_groups(trend=[Grouper.DIM])
def _loessdetrend_get_trend(
    da, *, dim, f, niter, d, weights, equal_spacing, skipna, kind
):
    if len(dim) > 1:
        da = da.mean(dim[1:])
    trend = loess_smoothing(
        da,
        dim=dim[0],
        f=f,
        niter=niter,
        d=d,
        weights=weights,
        equal_spacing=equal_spacing,
        skipna=skipna,
    )
    return trend.rename("trend").to_dataset()


class RollingMeanDetrend(BaseDetrend):
    """
    Detrend time series using a rolling mean.

    Parameters
    ----------
    group : str or Grouper
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      The fit is performed along the group's main dim.
    kind : {'*', '+'}
      The way the trend is removed or added, either additive or multiplicative.
    win : int
      The size of the rolling window. Units are the steps of the grouped data, which
      means this detrending is best use with either `group='time'` or
      `group='time.dayofyear'`. Other grouping will have large jumps included within the
      windows and :py`:class:`LoessDetrend` might offer a better solution.
    weights : sequence of floats, optional
      Sequence of length `win`. Defaults to None, which means a flat window.
    min_periods: int, optional
      Minimum number of observations in window required to have a value, otherwise the
      result is NaN. See :py:meth:`xarray.DataArray.rolling`.
      Defaults to None, which sets it equal to `win`. Setting both `weights` and this
      is not implemented yet.

    Notes
    -----
    As for the :py:class:`LoessDetrend` detrending, important boundary effects are to be expected.
    """

    def __init__(
        self, group="time", kind=ADDITIVE, win=30, weights=None, min_periods=None
    ):
        if weights is not None:
            weights = xr.DataArray(weights, dims=("window",))
            weights = weights / weights.sum()
            if min_periods is not None:
                raise NotImplementedError(
                    "Setting both `min_periods` and `weights` is not implemented yet."
                )
        super().__init__(
            group=group, kind=kind, win=win, weights=weights, min_periods=min_periods
        )

    def _get_trend(self, da):
        # Estimate trend over da
        trend = _rollingmean_get_trend(da, **self)
        return trend.trend


@map_groups(trend=[Grouper.DIM])
def _rollingmean_get_trend(da, *, dim, kind, win, weights, min_periods):
    if len(dim) > 1:
        da = da.mean(dim[1:])
    roll = da.rolling(center=True, min_periods=min_periods, **{dim[0]: win})
    if weights is not None:
        trend = roll.construct("window").dot(weights)
    else:
        trend = roll.mean()
    return trend.rename("trend").to_dataset()
