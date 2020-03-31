"""Detrending classes"""
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.groupby import DataArrayGroupBy

from .utils import ADDITIVE
from .utils import apply_correction
from .utils import invert
from .utils import MULTIPLICATIVE


# ## Base classes for the downscaling module

# Time offsets to find the middle of the period
loffsets = {"MS": "14d", "M": "15d", "YS": "181d", "Y": "182d", "QS": "45d", "Q": "46d"}


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
    """
    Detrend time series using a polynomial.

    Notes
    -----
    If freq is used to resample at a lower frequency, make sure the series includes full periods.
    """

    def __init__(self, degree=4, freq=None, kind=ADDITIVE):
        super().__init__(degree=degree, freq=freq, kind=kind)

    def _fit(self, da):
        if self.freq is not None:
            da = da.resample(
                time=self.freq, label="left", loffset=loffsets[self.freq]
            ).mean()
        self._fitds = da.polyfit(dim="time", deg=self.degree, full=True)

    def _detrend(self, da):
        # Estimate trend over da
        trend = xr.polyval(coord=da["time"], coeffs=self._fitds.polyfit_coefficients)

        # Remove trend from series
        return apply_correction(da, invert(trend, self.kind), self.kind)

    def _retrend(self, da):
        # Estimate trend over da
        trend = xr.polyval(coord=da["time"], coeffs=self._fitds.polyfit_coefficients)

        # Add trend to series
        return apply_correction(da, trend, self.kind)


# ## Grouping objects


class BaseGrouping(ParametrizableClass):
    def group(self, da: xr.DataArray):
        return self._group(da)

    def add_group_axis(self, da: xr.DataArray):
        return self._add_group_axis(da)

    def _group(self, da):
        raise NotImplementedError

    def _add_group_axis(self, da):
        raise NotImplementedError


class EmptyGrouping(BaseGrouping):
    def _group(self, da):
        return da.expand_dims(group=xr.DataArray([1], dims=("group",), name="group"))

    def _add_group_axis(self, da):
        return da.assign_coords(
            group=xr.DataArray([1.0] * da["time"].size, dims=("time",), name="group")
        )


class MonthGrouping(BaseGrouping):
    def _group(self, da):
        group = da.time.dt.month
        group.name = "group"
        group.attrs.update(group_name="month")
        return da.groupby(group)

    def _add_group_axis(self, da):
        group = da.time.dt.month - 0.5 + da.time.dt.day / da.time.dt.daysinmonth
        group.name = "group"
        group.attrs.update(group_name="month")
        return da.assign_coords(group=group)


class DOYGrouping(BaseGrouping):
    def __init__(self, window=None):
        super().__init__(self, window=window)

    def _group(self, da):
        group = da.time.dt.dayofyear
        group.name = "group"
        group.attrs.update(group_name="dayofyear")
        if self.window is not None:
            da = da.rolling(time=self.window, center=True).construct(
                window_dim="window"
            )
            group = xr.concat([group] * self.window, da.window)
            da.rename(time="old_time").stack(time=("old_time", "window"))
            group.rename(time="old_time").stack(time=("old_time", "window"))
        return da.groupby(group)

    def _add_group_axis(self, da):
        group = da.time.dt.dayofyear
        group.name = "group"
        group.attrs.update(group_name="dayofyear")
        return da.assign_coords(group=group)


# ## Mapping objects


class BaseMapping(ParametrizableClass):

    __fitted = False

    def fit(
        self,
        obs: Union[DataArray, DataArrayGroupBy],
        sim: Union[DataArray, DataArrayGroupBy],
    ):
        if self.__fitted:
            warn("fit() was already called, overwriting old results.")
        self._fit(obs, sim)
        self.__fitted = True

    def predict(self, fut: xr.DataArray):
        if not self.__fitted:
            raise ValueError("fit() must be called before predicting.")
        return self._predict(fut)

    def _fit(self):
        raise NotImplementedError

    def _predict(self, fut):
        raise NotImplementedError


class DeltaMapping(BaseMapping):
    def _fit(self, obs, sim):
        self._delta = obs.mean("time") - sim.mean("time")

    def _predict(self, fut):
        return fut + self._delta


class ScaleMapping(BaseMapping):
    def _fit(self, obs, sim):
        self._scale = obs.mean("time") / sim.mean("time")

    def _predict(self, fut):
        return fut * self._scale


class QuantileMapping(BaseMapping):
    def __init__(self, nquantiles=20, kind=ADDITIVE, interp=False):
        super().__init__(nquantiles=nquantiles, kind=kind, interp=interp)

    def _fit(self, obs, sim):
        self._dq = (1 / self.nquantiles) / 2
        self._quantiles = np.append(
            np.insert(np.linspace(self._dq, 1 - self._dq, self.nquantiles), 0, 0.0001),
            0.9999,
        )

        obsq = obs.quantile(self._quantiles, dim="time")
        simq = sim.quantile(self._quantiles, dim="time")

        if self.kind == MULTIPLICATIVE:
            self._qmfit = simq / obsq
        elif self.kind == ADDITIVE:
            self._qmfit = simq - obsq

    def _predict(self, fut):
        if self.interp:
            raise NotImplementedError

        futq = fut.rank(dim="time", pct=True)

        if self.interp:
            factor = self._qmfit.interp(quantile=futq, group=futq.group)
        else:
            factor = self._qmfit.sel(quantile=futq, group=futq.group, method="nearest")

        if self.kind == MULTIPLICATIVE:
            out = fut * factor
        elif self.kind == ADDITIVE:
            out = fut + factor

        return out.drop("quantile")


# ## Pipeline draft


def basicpipeline(
    obs: DataArray,
    sim: DataArray,
    fut: DataArray,
    detrender=NoDetrend(),
    grouper=EmptyGrouping(),
    mapper=DeltaMapping(),
):
    obs_trend = detrender.fit(obs)
    sim_trend = detrender.fit(sim)
    fut_trend = detrender.fit(fut)

    obs = obs_trend.detrend(obs)
    sim = sim_trend.detrend(sim)
    fut = fut_trend.detrend(fut)

    mapper.fit(grouper.group(obs), grouper.group(sim))
    fut_corr = mapper.predict(grouper.add_group_axis(fut))

    fut_corr = fut_trend.retrend(fut_corr)

    return fut_corr.drop_vars("group")
