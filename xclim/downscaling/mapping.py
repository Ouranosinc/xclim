"""Mapping classes"""
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.groupby import DataArrayGroupBy

from .base import ADDITIVE
from .base import MULTIPLICATIVE
from .base import ParametrizableClass


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
