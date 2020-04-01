"""Detrending classes"""
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.groupby import DataArrayGroupBy

from .utils import add_cyclic_bounds
from .utils import ADDITIVE
from .utils import adjust_freq
from .utils import apply_correction
from .utils import broadcast
from .utils import equally_spaced_nodes
from .utils import extrapolate_qm
from .utils import get_correction
from .utils import get_index
from .utils import group_apply
from .utils import interp_on_quantiles
from .utils import invert
from .utils import MULTIPLICATIVE
from .utils import parse_group


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

    def parameters_to_json(self):
        return {
            key: val
            if isinstance(val, (str, float, int, bool, type(None)))
            else str(val)
            for key, val in self.parameters.items()
        }

    def __str__(self):
        params_str = ", ".join(
            [f"{key}: {val}" for key, val in self.parameters_to_json().items()]
        )
        return f"<{self.__class__.__name__}: {params_str}>"


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

    __trained = False

    def train(
        self,
        obs: Union[DataArray, DataArrayGroupBy],
        sim: Union[DataArray, DataArrayGroupBy],
    ):
        if self.__trained:
            warn("train() was already called, overwriting old results.")
        self._train(obs, sim)
        self.__trained = True

    def predict(self, fut: xr.DataArray):
        if not self.__trained:
            raise ValueError("train() must be called before predicting.")
        return self._predict(fut)

    def _train(self):
        raise NotImplementedError

    def _predict(self, fut):
        raise NotImplementedError

    def _qm_dataset(self, qf, xq):
        qm = xr.Dataset(data_vars={"qf": qf, "xq": xq})
        qm.qf.attrs.update(
            standard_name="Correction factors",
            long_name="Quantile mapping correction factors",
        )
        qm.xq.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        qm.attrs["QM_parameters"] = self.parameters_to_json()
        return qm


class QuantileMapping(BaseMapping):
    def __init__(
        self,
        nquantiles=20,
        kind=ADDITIVE,
        interp="nearest",
        extrapolation="constant",
        detrender=NoDetrend(),
        group="time",
        normalize=False,
        rank_from_fut=False,
    ):
        super().__init__(
            nquantiles=nquantiles,
            kind=kind,
            interp=interp,
            extrapolation=extrapolation,
            detrender=detrender,
            group=group,
            normalize=normalize,
            rank_from_fut=rank_from_fut,
        )

    def _train(self, sim, obs):
        quantiles = equally_spaced_nodes(self.nquantiles, eps=1e-6)
        obsq = group_apply("quantile", obs, self.group, 1, q=quantiles).rename(
            quantile="quantiles"
        )
        simq = group_apply("quantile", sim, self.group, 1, q=quantiles).rename(
            quantile="quantiles"
        )

        if self.normalize:
            mu_sim = group_apply("mean", sim, self.group, 1)
            simq = apply_correction(simq, invert(mu_sim, self.kind), self.kind)

        qf = get_correction(simq, obsq, self.kind)

        qf, simq = extrapolate_qm(qf, simq, method=self.extrapolation)
        self.qm = self._qm_dataset(qf, simq)
        return self.qm

    def _predict(self, fut):
        if self.normalize:
            mu_fut = group_apply("mean", fut, self.group, 1)
            fut = apply_correction(
                fut,
                broadcast(
                    invert(mu_fut, self.kind), fut, group=self.group, interp=self.interp
                ),
                self.kind,
            )

        fut_fit = self.detrender.fit(fut)
        fut_det = fut_fit.detrend(fut)

        dim, prop = parse_group(self.group)

        if self.rank_from_fut:
            xq = group_apply(xr.DataArray.rank, fut_det, self.group, window=1, pct=True)
            sel = {"quantiles": xq}
            qf = broadcast(self.qm.qf, fut_det, interp=self.interp, sel=sel)
        else:
            if prop is not None:
                fut_det = fut_det.assign_coords(
                    {prop: get_index(fut_det, dim, prop, self.interp)}
                )
            qf = interp_on_quantiles(
                fut_det, self.qm.xq, self.qm.qf, group=self.group, method=self.interp
            )

        corrected = apply_correction(fut_det, qf, self.kind)

        out = fut_fit.retrend(corrected)
        out.attrs["bias_corrected"] = True
        return out


# ## Pipeline draft


def basicpipeline(
    obs: DataArray,
    sim: DataArray,
    fut: DataArray,
    detrender=NoDetrend(),
    grouper=EmptyGrouping(),
    mapper=QuantileMapping(),
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
