"""Detrending classes"""
from types import FunctionType
from typing import Optional
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


class MeanDetrend(BaseDetrend):
    def _fit(self, da):
        self._mean = da.mean(dim="time")

    def _detrend(self, da):
        return da - self._mean

    def _retrend(self, da):
        return da + self._mean


# TODO: Add an option to preserve mean in detrend / retrend operations.
class PolyDetrend(BaseDetrend):
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


class Grouper(ParametrizableClass):
    """Applies `groupby` method to one or more data arrays.

    Parameters
    ----------
    """

    def __init__(self, group: str, window: int = 1, interp: Union[bool, str] = False):
        if "." in group:
            dim, prop = group.split(".")
        else:
            dim, prop = group, None

        if isinstance(interp, str):
            interp = interp != "nearest"
        if window > 1:
            dims = ("window", dim)
        else:
            dims = dim
        super().__init__(
            dim=dim, dims=dims, prop=prop, name=group, window=window, interp=interp
        )

    def group(self, *das: xr.DataArray):
        if len(das) > 1:
            da = xr.Dataset(data_vars={da.name: da for da in das})
        else:
            da = das[0]

        if self.window > 1:
            da = da.rolling(center=True, **{self.dim: self.window}).construct(
                window_dim="window"
            )
        if self.prop:
            return da.groupby(self.name)
        else:  # TODO: Fix. time is hard-coded here, not so great.
            return da.groupby(da.time.dt.time.real == da.time.dt.time)

    def add_index(self, da: xr.DataArray):
        if self.prop is None:
            return da

        ind = da.indexes[self.dim]
        i = getattr(ind, self.prop)

        if self.interp:
            if self.dim == "time":
                if self.prop == "month":
                    i = ind.month - 0.5 + ind.day / ind.daysinmonth
                elif self.prop == "dayofyear":
                    i = ind.dayofyear
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        xi = xr.DataArray(
            i,
            dims=self.dim,
            coords={self.dim: da.coords[self.dim]},
            name=self.dim + " group index",
        )

        # Expand dimensions of index to match the dimensions of da
        # We want vectorized indexing with no broadcasting
        return da.assign_coords(
            {
                self.prop: xi.expand_dims(
                    **{k: v for (k, v) in da.coords.items() if k != self.dim}
                )
            }
        )

    def apply(self, func: Union[FunctionType, str], *das: xr.DataArray, **kwargs):
        grpd = self.group(*das)
        if isinstance(func, str):
            out = getattr(grpd, func)(dim=self.dims, **kwargs)
        else:
            if isinstance(grpd, xr.core.groupby.GroupBy):
                out = grpd.map(func, dim=self.dims, **kwargs)
            else:
                out = func(grpd, dim=self.dims, **kwargs)

        # Case where the function wants to return more than one variables
        # and that some have grouped dims and other have the same dimensions as the input.
        # In that specific case, groupby broadcasts everything back to the input's dim, copying the grouped data.
        if isinstance(out, xr.Dataset):
            for name, da in out.data_vars.items():
                if "_group_apply_reshape" in da.attrs:
                    if da.attrs["_group_apply_reshape"] and self.prop is not None:
                        out[name] = da.groupby(self.name).first(
                            skipna=False, keep_attrs=True
                        )
                    del out[name].attrs["_group_apply_reshape"]

        # Save input parameters as attributes of output DataArray.
        out.attrs["group"] = self.name
        out.attrs["group_window"] = self.window

        # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
        if self.dim in out.dims:
            out = out.sortby(self.dim)

        # TODO: Fix. Again, time dimension is hard-coded.
        # Because we are grouping on da.time.dt.time.real == da.time.dt.time, `apply` creates a time attribute that
        # conflicts with operations later on.
        if self.prop is None and len(out.time) == 1:
            out = out.squeeze("time", drop=True)

        return out


# ## Mapping objects


class BaseMapping(ParametrizableClass):

    __trained = False

    def __init__(self, group="time", **kwargs):
        if not isinstance(group, Grouper):
            group = Grouper(
                group,
                interp=kwargs.get("interp", False),
                window=kwargs.get("window", 1),
            )
        super().__init__(group=group, **kwargs)

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


class QuantileMapping(BaseMapping):
    def __init__(
        self,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        interp: str = "nearest",
        mode: Optional[str] = None,
        extrapolation: str = "constant",
        detrender: BaseDetrend = BaseDetrend(),
        group: Union[str, Grouper] = "time",
        normalize: bool = False,
        rank_from_fut: bool = False,
    ):
        if mode == "qdm":
            rank_from_fut = True
        elif mode == "dqm":
            normalize = True
            if detrender.__class__ == BaseDetrend:
                detrender = PolyDetrend()
        elif mode == "eqm":
            pass
        elif mode is not None:
            raise NotImplementedError
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
        obsq = self.group.apply("quantile", obs, q=quantiles).rename(
            quantile="quantiles"
        )
        simq = self.group.apply("quantile", sim, q=quantiles).rename(
            quantile="quantiles"
        )

        if self.normalize:
            mu_sim = self.group.apply("mean", sim)
            simq = apply_correction(simq, invert(mu_sim, self.kind), self.kind)

        qf = get_correction(simq, obsq, self.kind)

        qf, simq = extrapolate_qm(qf, simq, method=self.extrapolation)
        self.qm = self._qm_dataset(qf, simq)
        return self.qm

    def _predict(self, fut):
        if self.normalize:
            mu_fut = self.group.apply("mean", fut)
            fut = apply_correction(
                fut,
                broadcast(
                    invert(mu_fut, self.kind), fut, group=self.group, interp=self.interp
                ),
                self.kind,
            )

        fut_fit = self.detrender.fit(fut)
        fut_det = fut_fit.detrend(fut)

        if self.rank_from_fut:
            xq = self.group.apply(xr.DataArray.rank, fut_det, pct=True)
            sel = {"quantiles": xq}
            qf = broadcast(self.qm.qf, fut_det, interp=self.interp, sel=sel)
        else:
            fut_det = self.group.add_index(fut_det)
            qf = interp_on_quantiles(
                fut_det, self.qm.xq, self.qm.qf, group=self.group, method=self.interp
            )

        corrected = apply_correction(fut_det, qf, self.kind)

        out = fut_fit.retrend(corrected)
        out.attrs["bias_corrected"] = True
        return out

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
