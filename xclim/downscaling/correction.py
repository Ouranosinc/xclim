"""Mapping objects"""
from typing import Optional
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.groupby import DataArrayGroupBy

from .base import Grouper
from .base import ParametrizableClass
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import broadcast
from .utils import equally_spaced_nodes
from .utils import extrapolate_qm
from .utils import get_correction
from .utils import interp_on_quantiles
from .utils import map_cdf
from .utils import MULTIPLICATIVE


class BaseCorrection(ParametrizableClass):

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
        out = self._predict(fut)
        out.attrs["bias_corrected"] = True
        return out

    def make_dataset(self, **kwargs):
        self.ds = xr.Dataset(data_vars=kwargs)
        self.ds.attrs["corr_params"] = self.parameters_to_json()

    def _train(self):
        raise NotImplementedError

    def _predict(self, fut):
        raise NotImplementedError


class QuantileMapping(BaseCorrection):
    def __init__(
        self,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        interp: str = "nearest",
        mode: Optional[str] = None,
        extrapolation: str = "constant",
        group: Union[str, Grouper] = "time",
    ):
        super().__init__(
            nquantiles=nquantiles,
            kind=kind,
            interp=interp,
            extrapolation=extrapolation,
            group=group,
        )

    def _train(self, obs, sim):
        quantiles = equally_spaced_nodes(self.nquantiles, eps=1e-6)
        obs_q = self.group.apply("quantile", obs, q=quantiles).rename(
            quantile="quantiles"
        )
        sim_q = self.group.apply("quantile", sim, q=quantiles).rename(
            quantile="quantiles"
        )

        qf = get_correction(sim_q, obs_q, self.kind)

        qf.attrs.update(
            standard_name="Correction factors",
            long_name="Quantile mapping correction factors",
        )
        sim_q.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        self.make_dataset(qf=qf, sim_q=sim_q)

    def _predict(self, fut):
        qf, xq = extrapolate_qm(self.ds.qf, self.ds.xq, method=self.extrapolation)

        qf = interp_on_quantiles(fut, xq, qf, group=self.group, method=self.interp)

        return apply_correction(fut, qf, self.kind)


class QuantileDeltaMapping(QuantileMapping):
    def _predict(self, fut):
        qf, xq = extrapolate_qm(self.qm.qf, self.qm.xq, method=self.extrapolation)

        xq = self.group.apply(xr.DataArray.rank, fut, pct=True)
        sel = {"quantiles": xq}
        qf = broadcast(qf, fut, interp=self.interp, sel=sel)

        return apply_correction(fut, qf, self.kind)


class LOCI(BaseCorrection):
    def __init__(self, group="time", thresh=None, interp="linear"):
        super().__init__(group=group, thresh=thresh, interp=interp)

    def _train(self, obs, sim):
        s_thresh = map_cdf(sim, obs, self.thresh, self.group).isel(
            x=0
        )  # Selecting the first threshold.

        # Compute scaling factor on wet-day intensity
        sth = broadcast(s_thresh, sim)
        ws = xr.where(sim >= sth, sim, np.nan)
        wo = xr.where(obs >= self.thresh, obs, np.nan)

        ms = self.group.apply("mean", ws, skipna=True)
        mo = self.group.apply("mean", wo, skipna=True)

        # Correction factor
        cf = get_correction(ms - s_thresh, mo - self.thresh, MULTIPLICATIVE)
        cf.attrs.update(long_name="LOCI correction factors")
        s_thresh.attrs.update(long_name="Threshold over modeled data")
        self.make_dataset(sim_thresh=s_thresh, obs_thresh=self.thresh, cf=cf)

    def _predict(self, fut):
        sth = broadcast(self.ds.sim_thresh, fut)
        factor = broadcast(self.ds.cf, fut, group=self.group, interp=self.interp)
        with xr.set_options(keep_attrs=True):
            out = (factor * (fut - sth) + self.ds.obs_thresh).clip(min=0)
        return out


class Scaling(BaseCorrection):
    def __init__(self, group="time", kind=ADDITIVE, interp="nearest"):
        super().__init__(self, group=group, kind=kind, interp=interp)

    def _train(self, obs, sim):
        mean_sim = self.group.apply("mean", sim)
        mean_obs = self.group.apply("mean", obs)
        cf = get_correction(mean_sim, mean_obs, self.kind)
        cf.attrs.update(long_name="Scaling correction factors")
        self.make_dataset(cf=cf)

    def _predict(self, fut):
        factor = broadcast(self.ds.cf, fut, group=self.group, interp=self.interp)
        return apply_correction(fut, factor, self.kind)
