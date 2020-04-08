"""Mapping objects"""
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from .base import Grouper
from .base import ParametrizableClass
from .base import parse_group
from .detrending import PolyDetrend
from .processing import normalize
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
    """Base object for correction algorithms.

    Subclasses should implement the `_train` and `_predict` methods.
    """

    __trained = False

    def train(
        self, obs: DataArray, sim: DataArray,
    ):
        """Train the correction object. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        obs : DataArray
          Training target, usually a reference time series drawn from observations.
        sim : DataArray
          Training data, usually a model output whose biases are to be corrected.
        """
        if self.__trained:
            warn("train() was already called, overwriting old results.")
        self._train(obs, sim)
        self.__trained = True

    def predict(self, fut: DataArray):
        """Return bias-corrected data. Refer to the class documentationfor the algorithm details.

        Parameters
        ----------
        fut : DataArray
          Time series to be bias-corrected, usually a model output.
        """
        if not self.__trained:
            raise ValueError("train() must be called before predicting.")
        out = self._predict(fut)
        out.attrs["bias_corrected"] = True
        return out

    def _make_dataset(self, **kwargs):
        """Set the trained dataset from the passed variables.

        The trained dataset should at least have a `cf` variable storing the correction factors.
        Adds the correction parameters as the "corr_params" dictionary attribute.
        """
        self.ds = xr.Dataset(data_vars=kwargs)
        self.ds.attrs["corr_params"] = self.parameters_to_json()

    def _train(self):
        raise NotImplementedError

    def _predict(self, fut):
        raise NotImplementedError


class QuantileMapping(BaseCorrection):
    """Quantile Mapping bias-correction.

    Correction factors are computed between the quantiles of `obs` and `sim`.
    Values of `fut` are matched to the corresponding quantiles of `sim` and corrected accordingly.

    Algorithms here are based on [Cannon2015]_.

    Parameters
    ----------
    nquantiles : int
      The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
    kind : {'+', '*'}
      The correction kind, either additive or multiplicative.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the correction factors.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation to use. See :py:func:`xclim.downscaling.utils.extrapolate_qm` for details.
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.downscaling.base.Grouper` for details.
    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    @parse_group
    def __init__(
        self,
        *,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        interp: str = "nearest",
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

        cf = get_correction(sim_q, obs_q, self.kind)

        cf.attrs.update(
            standard_name="Correction factors",
            long_name="Quantile mapping correction factors",
        )
        sim_q.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        self._make_dataset(cf=cf, sim_q=sim_q)

    def _predict(self, fut):
        cf, sim_q = extrapolate_qm(self.ds.cf, self.ds.sim_q, method=self.extrapolation)
        cf = interp_on_quantiles(fut, sim_q, cf, group=self.group, method=self.interp)

        return apply_correction(fut, cf, self.kind)


class DetrendedQuantileMapping(QuantileMapping):
    """Detrended Quantile Mapping bias-correction.

    A scaling factor that would make the mean of `sim` match the mean of `obs` is computed.
    `obs` and `sim` are normalized by removing the group-wise mean.
    Correction factors are computed between the quantiles of the normalized `obs` and `sim`.
    `fut` is corrected by the scaling factor then detrended using a linear fit.
    Values of detrended `fut` are matched to the corresponding quantiles of normalized `sim` and corrected accordingly.
    The trend is put back on the result.

    Based on the DQM method of [Cannon2015]_.

    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    def _train(self, obs, sim):
        mu_obs = self.group.apply("mean", obs)
        mu_sim = self.group.apply("mean", sim)
        obs = normalize(obs, group=self.group, kind=self.kind)
        sim = normalize(sim, group=self.group, kind=self.kind)
        super()._train(obs, sim)

        self.ds["scaling"] = get_correction(mu_sim, mu_obs, kind=self.kind)
        self.ds.scaling.attrs.update(
            standard_name="Scaling factor",
            description="Scaling factor making the mean of sim match the one of sim.",
        )

    def _predict(self, fut):
        fut = apply_correction(
            fut,
            broadcast(self.ds.scaling, fut, group=self.group, interp=self.interp),
            self.kind,
        )
        fut_fit = PolyDetrend(degree=1, kind=self.kind).fit(fut)
        fut_detrended = fut_fit.detrend(fut)
        fut_corr_detrended = super()._predict(fut_detrended)
        fut_corr = fut_fit.retrend(fut_corr_detrended)
        return fut_corr


class QuantileDeltaMapping(QuantileMapping):
    """Quantile Delta Mapping bias-correction.

    Correction factors are computed between the quantiles of `obs` and `sim`.
    Quantiles of `fut` are matched to the corresponding quantiles of `sim` and corrected accordingly.

    The algorithm is based on the "QDM" method of [Cannon2015]_.

    References
    ----------
    [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    def _predict(self, fut):
        cf, _ = extrapolate_qm(self.ds.cf, self.ds.sim_q, method=self.extrapolation)

        fut_q = self.group.apply(xr.DataArray.rank, fut, main_only=True, pct=True)
        sel = {"quantiles": fut_q}
        cf = broadcast(cf, fut, group=self.group, interp=self.interp, sel=sel)

        return apply_correction(fut, cf, self.kind)


class LOCI(BaseCorrection):
    r"""Local Intensity Scaling (LOCI) bias-correction.

    This bias correction method is designed to correct daily precipitation time series by considering wet and dry days
    separately. Based on [Schmidli2006]_.

    Multiplicative correction factors are computed such that the mean of `sim` matches the mean of `obs` for values above a
    threshold.

    The threshold on the training target `obs` is first mapped to `sim` by finding the quantile in `sim` having the same
    exceedance probability as thresh in `obs`. The correction factor is then given by

    .. math::

       s = \frac{\left \langle obs: obs \geq t_{obs} \right\rangle - t_{obs}}{\left \langle sim : sim \geq t_{sim} \right\rangle - t_{sim}}

    In the case of precipitations, the correction factor is the ratio of wet-days intensity.

    For a correction factor `s`, the bias-correction of `fut` is:

    .. math::

      fut(t) = \max\left(t_{obs} + s \cdot (sim(t) - t_{sim}), 0\right)

    Parameters
    ----------
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.downscaling.base.Grouper` for details.
    thresh : float
      The threshold in `obs` above which the values are scaled.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the correction factors.

    References
    ----------
    [Schmidli2006] Schmidli, J., Frei, C., & Vidale, P. L. (2006). Downscaling from GCM precipitation: A benchmark for dynamical and statistical downscaling methods. International Journal of Climatology, 26(5), 679–689. DOI:10.1002/joc.1287
    """

    @parse_group
    def __init__(
        self,
        *,
        group: Union[str, Grouper] = "time",
        thresh: float = None,
        interp: str = "linear",
    ):
        super().__init__(group=group, thresh=thresh, interp=interp)

    def _train(self, obs, sim):
        s_thresh = map_cdf(sim, obs, self.thresh, group=self.group).isel(
            x=0
        )  # Selecting the first threshold.
        # Compute scaling factor on wet-day intensity
        sth = broadcast(s_thresh, sim, group=self.group)
        ws = xr.where(sim >= sth, sim, np.nan)
        wo = xr.where(obs >= self.thresh, obs, np.nan)

        ms = self.group.apply("mean", ws, skipna=True)
        mo = self.group.apply("mean", wo, skipna=True)

        # Correction factor
        cf = get_correction(ms - s_thresh, mo - self.thresh, MULTIPLICATIVE)
        cf.attrs.update(long_name="LOCI correction factors")
        s_thresh.attrs.update(long_name="Threshold over modeled data")
        self._make_dataset(sim_thresh=s_thresh, obs_thresh=self.thresh, cf=cf)

    def _predict(self, fut):
        sth = broadcast(self.ds.sim_thresh, fut, group=self.group, interp=self.interp)
        factor = broadcast(self.ds.cf, fut, group=self.group, interp=self.interp)
        with xr.set_options(keep_attrs=True):
            out = (factor * (fut - sth) + self.ds.obs_thresh).clip(min=0)
        return out


class Scaling(BaseCorrection):
    """Scaling bias-correction

    Simple bias-correction method scaling variables by an additive or multiplicative factor so that the mean of sim matches the mean of obs.

    Parameters
    ----------
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.downscaling.base.Grouper` for details.
    kind : {'+', '*'}
      The correction kind, either additive or multiplicative.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the correction factors.
    """

    @parse_group
    def __init__(self, *, group="time", kind=ADDITIVE, interp="nearest"):
        super().__init__(group=group, kind=kind, interp=interp)

    def _train(self, obs, sim):
        mean_sim = self.group.apply("mean", sim)
        mean_obs = self.group.apply("mean", obs)
        cf = get_correction(mean_sim, mean_obs, self.kind)
        cf.attrs.update(long_name="Scaling correction factors")
        self._make_dataset(cf=cf)

    def _predict(self, fut):
        factor = broadcast(self.ds.cf, fut, group=self.group, interp=self.interp)
        return apply_correction(fut, factor, self.kind)
