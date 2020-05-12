"""Adjustment objects"""
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from .base import Grouper
from .base import Parametrizable
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
from xclim.core.calendar import get_calendar
from xclim.core.formatting import update_history


class BaseAdjustment(Parametrizable):
    """Base object for adjustment algorithms.

    Subclasses should implement the `_train` and `_adjust` methods.
    """

    def __init__(self, **kwargs):
        self.__trained = False
        super().__init__(**kwargs)

    def train(
        self, ref: DataArray, hist: DataArray,
    ):
        """Train the adjustment object. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
          Training target, usually a reference time series drawn from observations.
        hist : DataArray
          Training data, usually a model output whose biases are to be adjusted.
        """
        if self.__trained:
            warn("train() was already called, overwriting old results.")
        if (
            hasattr(self, "group")
            and self.group.prop == "dayofyear"
            and get_calendar(ref) != get_calendar(hist)
        ):
            warn(
                (
                    "Input ref and hist are defined on different calendars, "
                    "this is not recommended when using 'dayofyear' grouping "
                    "and could give strange results. See `xclim.core.calendar` "
                    "for tools to convert your data to a common calendar."
                ),
                stacklevel=4,
            )
        self._train(ref, hist)
        self._hist_calendar = get_calendar(hist)
        self.__trained = True

    def adjust(self, sim: DataArray, **kwargs):
        """Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        sim : DataArray
          Time series to be bias-adjusted, usually a model output.
        """
        if not self.__trained:
            raise ValueError("train() must be called before adjusting.")
        if (
            hasattr(self, "group")
            and self.group.prop == "dayofyear"
            and get_calendar(sim) != self._hist_calendar
        ):
            warn(
                (
                    "This adjustment was trained on a simulation with the "
                    f"{self._hist_calendar} calendar but the sim input uses "
                    f"{get_calendar(sim)}. This is not recommended with dayofyear "
                    "grouping and could give strange results."
                ),
                stacklevel=4,
            )
        scen = self._adjust(sim, **kwargs)
        scen.attrs["history"] = update_history(
            f"Bias-adjusted with method {str(self)}", sim
        )
        return scen

    def _make_dataset(self, **kwargs):
        """Set the trained dataset from the passed variables.

        The trained dataset should at least have a `af` variable storing the adjustment factors.
        Adds the adjustment parameters as the "adj_params" dictionary attribute.
        """
        self.ds = xr.Dataset(data_vars=kwargs)
        self.ds.attrs["adj_params"] = str(self)

    def _train(self):
        raise NotImplementedError

    def _adjust(self, sim):
        raise NotImplementedError


class EmpiricalQuantileMapping(BaseAdjustment):
    """Empirical Quantile Mapping bias-adjustment.

    Adjustment factors are computed between the quantiles of `ref` and `sim`.
    Values of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

    .. math::

      F^{-1}_{ref} (F_{hist}(sim))

    where :math:`F` is the cumulative distribution function (CDF) and `mod` stands for model data.


    Parameters
    ----------
    nquantiles : int
      The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the adjustment factors.
    extrapolation : {'constant', 'nan'}
      The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details.
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.

    References
    ----------
    Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030
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

    def _train(self, ref, hist):
        quantiles = equally_spaced_nodes(self.nquantiles, eps=1e-6)
        ref_q = self.group.apply("quantile", ref, q=quantiles).rename(
            quantile="quantiles"
        )
        hist_q = self.group.apply("quantile", hist, q=quantiles).rename(
            quantile="quantiles"
        )

        af = get_correction(hist_q, ref_q, self.kind)

        af.attrs.update(
            standard_name="Adjustment factors",
            long_name="Quantile mapping adjustment factors",
        )
        hist_q.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        self._make_dataset(af=af, hist_q=hist_q)

    def _adjust(self, sim):
        af, hist_q = extrapolate_qm(
            self.ds.af, self.ds.hist_q, method=self.extrapolation
        )
        af = interp_on_quantiles(sim, hist_q, af, group=self.group, method=self.interp)

        return apply_correction(sim, af, self.kind)


class DetrendedQuantileMapping(EmpiricalQuantileMapping):
    r"""Detrended Quantile Mapping bias-adjustment.

    The algorithm follows these steps, 1-3 being the 'train' and 4-6, the 'adjust' steps.

    1. A scaling factor that would make the mean of `hist` match the mean of `ref` is computed.
    2. `ref` and `hist` are normalized by removing the group-wise mean.
    3. Adjustment factors are computed between the quantiles of the normalized `ref` and `hist`.
    4. `sim` is corrected by the scaling factor then detrended using a linear fit.
    5. Values of detrended `sim` are matched to the corresponding quantiles of normalized `hist` and corrected accordingly.
    6. The trend is put back on the result.

    .. math::

        F^{-1}_{ref}\left\{F_{hist}\left[\frac{\overline{hist}\cdot sim}{\overline{sim}}\right]\right\}\frac{\overline{sim}}{\overline{hist}}

    where :math:`F` is the cumulative distribution function (CDF) and :math:`\overline{xyz}` is the linear trend of the data.
    This equation is valid for multiplicative adjustment. Based on the DQM method of [Cannon2015]_.

    References
    ----------
    .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    def _train(self, ref, hist):
        mu_ref = self.group.apply("mean", ref)
        mu_hist = self.group.apply("mean", hist)
        ref = normalize(ref, group=self.group, kind=self.kind)
        hist = normalize(hist, group=self.group, kind=self.kind)
        super()._train(ref, hist)

        self.ds["scaling"] = get_correction(mu_hist, mu_ref, kind=self.kind)
        self.ds.scaling.attrs.update(
            standard_name="Scaling factor",
            description="Scaling factor making the mean of hist match the one of hist.",
        )

    def _adjust(self, sim, degree=0):
        sim = apply_correction(
            sim,
            broadcast(self.ds.scaling, sim, group=self.group, interp=self.interp),
            self.kind,
        )
        sim_fit = PolyDetrend(degree=degree, kind=self.kind).fit(sim)
        sim_detrended = sim_fit.detrend(sim)
        scen_detrended = super()._adjust(sim_detrended)
        scen = sim_fit.retrend(scen_detrended)
        return scen


class QuantileDeltaMapping(EmpiricalQuantileMapping):
    r"""Quantile Delta Mapping bias-adjustment.

    Adjustment factors are computed between the quantiles of `ref` and `hist`.
    Quantiles of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

    .. math::

        sim\frac{F^{-1}_{ref}\left[F_{sim}(sim)\right]}{F^{-1}_{hist}\left[F_{sim}(sim)\right]}

    where :math:`F` is the cumulative distribution function (CDF). This equation is valid for multiplicative adjustment.
    The algorithm is based on the "QDM" method of [Cannon2015]_.

    References
    ----------
    .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    def _adjust(self, sim):
        af, _ = extrapolate_qm(self.ds.af, self.ds.hist_q, method=self.extrapolation)

        sim_q = self.group.apply(xr.DataArray.rank, sim, main_only=True, pct=True)
        sel = {"quantiles": sim_q}
        af = broadcast(af, sim, group=self.group, interp=self.interp, sel=sel)

        return apply_correction(sim, af, self.kind)


class LOCI(BaseAdjustment):
    r"""Local Intensity Scaling (LOCI) bias-adjustment.

    This bias adjustment method is designed to correct daily precipitation time series by considering wet and dry days
    separately ([Schmidli2006]_).

    Multiplicative adjustment factors are computed such that the mean of `hist` matches the mean of `ref` for values above a
    threshold.

    The threshold on the training target `ref` is first mapped to `hist` by finding the quantile in `hist` having the same
    exceedance probability as thresh in `ref`. The adjustment factor is then given by

    .. math::

       s = \frac{\left \langle ref: ref \geq t_{ref} \right\rangle - t_{ref}}{\left \langle hist : hist \geq t_{hist} \right\rangle - t_{hist}}

    In the case of precipitations, the adjustment factor is the ratio of wet-days intensity.

    For an adjustment factor `s`, the bias-adjustment of `sim` is:

    .. math::

      sim(t) = \max\left(t_{ref} + s \cdot (hist(t) - t_{hist}), 0\right)

    Parameters
    ----------
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    thresh : float
      The threshold in `ref` above which the values are scaled.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the adjustment factors.

    References
    ----------
    .. [Schmidli2006] Schmidli, J., Frei, C., & Vidale, P. L. (2006). Downscaling from GCM precipitation: A benchmark for dynamical and statistical downscaling methods. International Journal of Climatology, 26(5), 679–689. DOI:10.1002/joc.1287
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

    def _train(self, ref, hist):
        s_thresh = map_cdf(hist, ref, self.thresh, group=self.group).isel(
            x=0
        )  # Selecting the first threshold.
        # Compute scaling factor on wet-day intensity
        sth = broadcast(s_thresh, hist, group=self.group)
        ws = xr.where(hist >= sth, hist, np.nan)
        wo = xr.where(ref >= self.thresh, ref, np.nan)

        ms = self.group.apply("mean", ws, skipna=True)
        mo = self.group.apply("mean", wo, skipna=True)

        # Adjustment factor
        af = get_correction(ms - s_thresh, mo - self.thresh, MULTIPLICATIVE)
        af.attrs.update(long_name="LOCI adjustment factors")
        s_thresh.attrs.update(long_name="Threshold over modeled data")
        self._make_dataset(hist_thresh=s_thresh, ref_thresh=self.thresh, af=af)

    def _adjust(self, sim):
        sth = broadcast(self.ds.hist_thresh, sim, group=self.group, interp=self.interp)
        factor = broadcast(self.ds.af, sim, group=self.group, interp=self.interp)
        with xr.set_options(keep_attrs=True):
            scen = (factor * (sim - sth) + self.ds.ref_thresh).clip(min=0)
        return scen


class Scaling(BaseAdjustment):
    """Scaling bias-adjustment

    Simple bias-adjustment method scaling variables by an additive or multiplicative factor so that the mean of `hist`
    matches the mean of `ref`.

    Parameters
    ----------
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the adjustment factors.
    """

    @parse_group
    def __init__(self, *, group="time", kind=ADDITIVE, interp="nearest"):
        super().__init__(group=group, kind=kind, interp=interp)

    def _train(self, ref, hist):
        mean_hist = self.group.apply("mean", hist)
        mean_ref = self.group.apply("mean", ref)
        af = get_correction(mean_hist, mean_ref, self.kind)
        af.attrs.update(long_name="Scaling adjustment factors")
        self._make_dataset(af=af)

    def _adjust(self, sim):
        factor = broadcast(self.ds.af, sim, group=self.group, interp=self.interp)
        return apply_correction(sim, factor, self.kind)
