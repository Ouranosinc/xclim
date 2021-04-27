"""Adjustment objects."""
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.calendar import get_calendar
from xclim.core.formatting import update_history
from xclim.core.units import convert_units_to
from xclim.indices import stats

from .base import Grouper, ParametrizableWithDataset, parse_group
from .detrending import PolyDetrend
from .processing import normalize
from .utils import (
    ADDITIVE,
    MULTIPLICATIVE,
    apply_correction,
    best_pc_orientation,
    broadcast,
    equally_spaced_nodes,
    extrapolate_qm,
    get_clusters,
    get_clusters_1d,
    get_correction,
    interp_on_quantiles,
    map_cdf,
    pc_matrix,
    rank,
)

__all__ = [
    "BaseAdjustment",
    "EmpiricalQuantileMapping",
    "DetrendedQuantileMapping",
    "LOCI",
    "PrincipalComponents",
    "QuantileDeltaMapping",
    "Scaling",
]


def _raise_on_multiple_chunk(da, main_dim):
    if da.chunks is not None and len(da.chunks[da.get_axis_num(main_dim)]) > 1:
        raise ValueError(
            f"Multiple chunks along the main adjustment dimension {main_dim} is not supported."
        )


class BaseAdjustment(ParametrizableWithDataset):
    """Base class for adjustment objects."""

    _attribute = "_xclim_adjustment"
    _repr_hide_params = ["hist_calendar"]

    @property
    def _trained(self):
        return hasattr(self, "ds")

    def train(
        self,
        ref: DataArray,
        hist: DataArray,
    ):
        """Train the adjustment object. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
          Training target, usually a reference time series drawn from observations.
        hist : DataArray
          Training data, usually a model output whose biases are to be adjusted.
        """
        if self._trained:
            warn("train() was already called, overwriting old results.")

        if hasattr(self, "group"):
            # Right now there is no other way of getting the main adjustment dimension
            _raise_on_multiple_chunk(ref, self.group.dim)
            _raise_on_multiple_chunk(hist, self.group.dim)

            if self.group.prop == "dayofyear" and get_calendar(ref) != get_calendar(
                hist
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

        self["hist_calendar"] = get_calendar(hist)
        self._train(ref, hist)

    def adjust(self, sim: DataArray, **kwargs):
        """Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        sim : DataArray
          Time series to be bias-adjusted, usually a model output.
        kwargs :
          Algorithm-specific keyword arguments, see class doc.
        """
        if not self._trained:
            raise ValueError("train() must be called before adjusting.")

        if hasattr(self, "group"):
            # Right now there is no other way of getting the main adjustment dimension
            _raise_on_multiple_chunk(sim, self.group.dim)

            if (
                self.group.prop == "dayofyear"
                and get_calendar(sim) != self.hist_calendar
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
        params = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        scen.attrs["xclim_history"] = update_history(
            f"Bias-adjusted with {str(self)}.adjust(sim, {params})", sim
        )
        return scen

    def set_dataset(self, ds: xr.Dataset):
        """Stores an xarray dataset in the `ds` attribute.

        Useful with custom object initialization or if some external processing was performed.
        """
        super().set_dataset(ds)
        self.ds.attrs["adj_params"] = str(self)

    def _train(self, ref: DataArray, hist: DataArray):
        raise NotImplementedError

    def _adjust(self, sim, **kwargs):
        raise NotImplementedError


class EmpiricalQuantileMapping(BaseAdjustment):
    """Conventional quantile mapping adjustment."""

    @parse_group
    def __init__(
        self,
        *,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        group: Union[str, Grouper] = "time",
    ):
        """Empirical Quantile Mapping bias-adjustment.

        Adjustment factors are computed between the quantiles of `ref` and `sim`.
        Values of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

        .. math::

          F^{-1}_{ref} (F_{hist}(sim))

        where :math:`F` is the cumulative distribution function (CDF) and `mod` stands for model data.

        Parameters
        ----------
        At instantiation:

        nquantiles : int
          The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
        kind : {'+', '*'}
          The adjustment kind, either additive or multiplicative.
        group : Union[str, Grouper]
          The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.

        In adjustment:

        interp : {'nearest', 'linear', 'cubic'}
          The interpolation method to use when interpolating the adjustment factors. Defaults to "nearset".
        extrapolation : {'constant', 'nan'}
          The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".

        References
        ----------
        Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030
        """
        super().__init__(
            nquantiles=nquantiles,
            kind=kind,
            group=group,
        )

    def _train(self, ref, hist):
        if np.isscalar(self.nquantiles):
            quantiles = equally_spaced_nodes(self.nquantiles, eps=1e-6)
        else:
            quantiles = self.nquantiles
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
        self.set_dataset(xr.Dataset(dict(af=af, hist_q=hist_q)))

    def _adjust(self, sim, interp="nearest", extrapolation="constant"):
        af, hist_q = extrapolate_qm(self.ds.af, self.ds.hist_q, method=extrapolation)
        af = interp_on_quantiles(sim, hist_q, af, group=self.group, method=interp)

        return apply_correction(sim, af, self.kind)


class DetrendedQuantileMapping(EmpiricalQuantileMapping):
    """Quantile mapping using normalized and detrended data."""

    @parse_group
    def __init__(
        self,
        *,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        group: Union[str, Grouper] = "time",
        norm_window: int = 1,
    ):
        r"""Detrended Quantile Mapping bias-adjustment.

        The algorithm follows these steps, 1-3 being the 'train' and 4-6, the 'adjust' steps.

        1. A scaling factor that would make the mean of `hist` match the mean of `ref` is computed.
        2. `ref` and `hist` are normalized by removing the "dayofyear" mean.
        3. Adjustment factors are computed between the quantiles of the normalized `ref` and `hist`.
        4. `sim` is corrected by the scaling factor, and either normalized by "dayofyear" and  detrended group-wise
           or directly detrended per "dayofyear", using a linear fit (modifiable).
        5. Values of detrended `sim` are matched to the corresponding quantiles of normalized `hist` and corrected accordingly.
        6. The trend is put back on the result.

        .. math::

            F^{-1}_{ref}\left\{F_{hist}\left[\frac{\overline{hist}\cdot sim}{\overline{sim}}\right]\right\}\frac{\overline{sim}}{\overline{hist}}

        where :math:`F` is the cumulative distribution function (CDF) and :math:`\overline{xyz}` is the linear trend of the data.
        This equation is valid for multiplicative adjustment. Based on the DQM method of [Cannon2015]_.

        Parameters
        ----------
        At instantiation:

        nquantiles : int
          The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
        kind : {'+', '*'}
          The adjustment kind, either additive or multiplicative.
        group : Union[str, Grouper]
          The grouping information used in the quantile mapping process. See :py:class:`xclim.sdba.base.Grouper` for details.
          the normalization step is always performed on each day of the year.
        norm_window : 1
          The window size used in the normalization grouping. Defaults to 1.

        In adjustment:

        interp : {'nearest', 'linear', 'cubic'}
          The interpolation method to use when interpolating the adjustment factors. Defaults to "nearest".
        detrend : int or BaseDetrend instance
          The method to use when detrending. If an int is passed, it is understood as a PolyDetrend (polynomial detrending) degree. Defaults to 1 (linear detrending)
        extrapolation : {'constant', 'nan'}
          The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".
        normalize_sim : bool
          If True, scaled sim is normalized by its "dayofyear" mean and then detrended using `group`.
          The norm is broadcasted and added back on scen using `interp='nearest'`, ignoring the passed `interp`.
          If False, scaled sim is detrended per "dayofyear".
          This is useful on large datasets using dask, in which case "dayofyear" is a very small division,
          because normalization is a more efficient operation than detrending for similarly sized groups.

        References
        ----------
        .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
        """
        super().__init__(
            nquantiles=nquantiles,
            kind=kind,
            group=group,
        )
        self["norm_group"] = Grouper("time.dayofyear", window=norm_window)

    def _train(self, ref, hist):
        refn = normalize(ref, group=self.norm_group, kind=self.kind)
        histn = normalize(hist, group=self.norm_group, kind=self.kind)
        super()._train(refn, histn)

        mu_ref = self.group.apply("mean", ref)
        mu_hist = self.group.apply("mean", hist)
        self.ds["scaling"] = get_correction(mu_hist, mu_ref, kind=self.kind)
        self.ds.scaling.attrs.update(
            standard_name="Scaling factor",
            description="Scaling factor making the mean of hist match the one of hist.",
        )

    def _adjust(
        self,
        sim,
        interp="nearest",
        extrapolation="constant",
        detrend=1,
        normalize_sim=False,
    ):

        # Apply preliminary scaling from obs to hist
        sim = apply_correction(
            sim,
            broadcast(self.ds.scaling, sim, group=self.group, interp=interp),
            self.kind,
        )

        if normalize_sim:
            sim_norm = self.norm_group.apply("mean", sim)
            sim = normalize(sim, group=self.norm_group, kind=self.kind, norm=sim_norm)

        # Find trend on sim
        if isinstance(detrend, int):
            detrend = PolyDetrend(
                degree=detrend,
                kind=self.kind,
                group=self.group if normalize_sim else self.norm_group,
            )

        sim_fit = detrend.fit(sim)
        sim_detrended = sim_fit.detrend(sim)

        # Adjust using `EmpiricalQuantileMapping.adjust`
        scen_detrended = super()._adjust(
            sim_detrended, extrapolation=extrapolation, interp=interp
        )
        # Retrend
        scen = sim_fit.retrend(scen_detrended)

        if normalize_sim:
            return apply_correction(
                scen,
                broadcast(sim_norm, scen, group=self.norm_group, interp="nearest"),
                self.kind,
            )
        return scen


class QuantileDeltaMapping(EmpiricalQuantileMapping):
    """Quantile mapping with sim's quantiles computed independently."""

    def __init__(self, **kwargs):
        r"""Quantile Delta Mapping bias-adjustment.

        Adjustment factors are computed between the quantiles of `ref` and `hist`.
        Quantiles of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

        .. math::

            sim\frac{F^{-1}_{ref}\left[F_{sim}(sim)\right]}{F^{-1}_{hist}\left[F_{sim}(sim)\right]}

        where :math:`F` is the cumulative distribution function (CDF). This equation is valid for multiplicative adjustment.
        The algorithm is based on the "QDM" method of [Cannon2015]_.

        Parameters
        ----------
        At instantiation:

        nquantiles : int
          The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
        kind : {'+', '*'}
          The adjustment kind, either additive or multiplicative.
        group : Union[str, Grouper]
          The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.

        In adjustment:

        interp : {'nearest', 'linear', 'cubic'}
          The interpolation method to use when interpolating the adjustment factors. Defaults to "nearest".
        extrapolation : {'constant', 'nan'}
          The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".

        References
        ----------
        Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
        """
        super().__init__(**kwargs)

    def _adjust(self, sim, interp="nearest", extrapolation="constant"):
        af, _ = extrapolate_qm(self.ds.af, self.ds.hist_q, method=extrapolation)

        sim_q = self.group.apply(rank, sim, main_only=True, pct=True)
        sel = {dim: sim_q[dim] for dim in set(af.dims).intersection(set(sim_q.dims))}
        sel["quantiles"] = sim_q
        af = broadcast(af, sim, group=self.group, interp=interp, sel=sel)

        return apply_correction(sim, af, self.kind)


class ExtremeValues(BaseAdjustment):
    """Second order adjustment for extreme values."""

    def __init__(
        self,
        cluster_thresh: str,
        *,
        q_thresh: float = 0.95,
    ):
        r"""Adjustement correction for extreme values.

        The tail of the distribution of adjusted data is corrected according to the
        parametric Generalized Pareto distribution of the reference data, [RRJF2021]_.
        The distributions are composed of the maximal values of clusters of "large" values.
        With "large" values being those above `cluster_thresh`. Only extreme values, whose
        quantile within the pool of large values are above `q_thresh`, are re-adjusted.
        See Notes.

        Parameters
        ----------
        At instantiation:

        cluster_thresh: Quantity (str with units)
          The threshold value for defining clusters.
        q_thresh : float
          The quantile of "extreme" values, [0, 1[.

        In training:

        ref_params :  xr.DataArray
          Distribution parameters to use in place of a fitted dist on `ref`.

        In adjustment:

        frac: float
          Fraction where the cutoff happens between the original scen and the corrected one.
          See Notes, ]0, 1].
        power: float
          Shape of the correction strength, see Notes.

        Notes
        -----
        Extreme values are extracted from `ref`, `hist` and `sim` by finding all "clusters",
        i.e. runs of consecutive values above `cluster_thresh`. The `q_thresh`th percentile
        of these values is taken on `ref` and `hist` and becomes `thresh`, the extreme value
        threshold. The maximal value of each cluster of `ref`, if it exceeds that new threshold,
        is taken and Generalized Pareto distribution is fitted to them.
        Similarly with `sim`. The cdf of the extreme values of `sim` is computed in reference
        to the distribution fitted on `sim` and then the corresponding values (quantile / ppf)
        in reference to the distribution fitted on `ref` are taken as the new bias-adjusted values.

        Once new extreme values are found, a mixture from the original scen and corrected scen
        is used in the result. For each original value :math:`S_i` and corrected value :math:`C_i`
        the final extreme value :math:`V_i` is:

        .. math::

            V_i = C_i * \tau + S_i * (1 - \tau)

        Where :math:`\tau` is a function of sim's extreme values :math:`F` and of arguments
        ``frac`` (:math:`f`) and ``power`` (:math:`p`):

        .. math::

            \tau = \left(\frac{1}{f}\frac{S - min(S)}{max(S) - min(S)}\right)^p

        Code based on the `biascorrect_extremes` function of the julia package [ClimateTools]_.

        References
        ----------
        .. [ClimateTools] https://juliaclimate.github.io/ClimateTools.jl/stable/
        .. [RRJF2021] Roy, P., Rondeau-Genesse, G., Jalbert, J., Fournier, É. 2021. Climate Scenarios of Extreme Precipitation Using a Combination of Parametric and Non-Parametric Bias Correction Methods. Submitted to Climate Services, April 2021.
        """
        super().__init__(q_thresh=q_thresh, cluster_thresh=cluster_thresh)

    def train(self, ref, hist, ref_params=None):
        """Train the second-order adjustment object. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
          Training target, usually a reference time series drawn from observations.
        hist : DataArray
          Training data, usually a model output whose biases are to be adjusted.
        ref_params: DataArray, optional
          Distribution parameters to use inplace of a Generalized Pareto fitted on `ref`.
          Must be similar to the output of `xclim.indices.stats.fit` called on `ref`.
          If the `scipy_dist` attribute is missing, `genpareto` is assumed.
          Only `genextreme` and `genpareto` are accepted as scipy_dist.
        """
        if self._trained:
            warn("train() was already called, overwriting old results.")

        cluster_thresh = convert_units_to(self.cluster_thresh, ref)
        hist = convert_units_to(hist, ref)

        # Extreme value threshold computed relative to "large values".
        # We use the mean between ref and hist here.
        thresh = (
            ref.where(ref >= cluster_thresh).quantile(self.q_thresh, dim="time")
            + hist.where(hist >= cluster_thresh).quantile(self.q_thresh, dim="time")
        ) / 2

        if ref_params is None:
            # All large value clusters
            ref_clusters = get_clusters(ref, thresh, cluster_thresh)
            # Parameters of a genpareto (or other) distribution, we force the location at thresh.
            fit_params = stats.fit(
                ref_clusters.maximum - thresh, "genpareto", dim="cluster", floc=0
            )
            # Param "loc" was fitted with 0, put thresh back
            fit_params = fit_params.where(
                fit_params.dparams != "loc", fit_params + thresh
            )
        else:
            dist = ref_params.attrs.get("scipy_dist", "genpareto")
            fit_params = ref_params.copy().transpose(..., "dparams")
            if dist == "genextreme":
                fit_params = xr.where(
                    fit_params.dparams == "loc",
                    fit_params.sel(dparams="scale")
                    + fit_params.sel(dparams="c") * (thresh - fit_params),
                    fit_params,
                )
            elif dist != "genpareto":
                raise ValueError(f"Unknown conversion from {dist} to genpareto.")

        ds = xr.Dataset(dict(fit_params=fit_params, thresh=thresh))
        ds.fit_params.attrs.update(
            long_name="Generalized Pareto distribution parameters of ref",
        )
        ds.thresh.attrs.update(
            long_name=f"{self.q_thresh * 100}th percentile extreme value threshold",
            description=f"Mean of the {self.q_thresh * 100}th percentile of large values (x > {self.cluster_thresh}) of ref and hist.",
        )
        self.set_dataset(ds)

    def adjust(
        self,
        scen: xr.DataArray,
        sim: xr.DataArray,
        frac: float = 0.25,
        power: float = 1.0,
    ):
        """Return second order bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        scen: DataArray
          Bias-adjusted time series.
        sim : DataArray
          Time series to be bias-adjusted, source of scen.
        kwargs :
          Algorithm-specific keyword arguments, see class doc.
        """
        if not self._trained:
            raise ValueError("train() must be called before adjusting.")

        def _adjust_extremes_1d(scen, sim, ref_params, thresh, *, dist, cluster_thresh):
            # Clusters of large values of sim
            _, _, sim_posmax, sim_maxs = get_clusters_1d(sim, thresh, cluster_thresh)

            new_scen = scen.copy()
            if sim_posmax.size == 0:
                # Happens if everything is under `cluster_thresh`
                return new_scen

            # Fit the dist, force location at thresh
            sim_fit = stats._fitfunc_1d(
                sim_maxs, dist=dist, nparams=len(ref_params), method="ML", floc=thresh
            )

            # Cumulative density function for extreme values in sim's distribution
            sim_cdf = dist.cdf(sim_maxs, *sim_fit)
            # Equivalent value of sim's CDF's but in ref's distribution.
            new_sim = dist.ppf(sim_cdf, *ref_params) + thresh

            # Get the transition weights based on frac and power values
            transition = (
                ((sim_maxs - sim_maxs.min()) / ((sim_maxs.max()) - sim_maxs.min()))
                / frac
            ) ** power
            np.clip(transition, None, 1, out=transition)

            # Apply smooth linear transition between scen and corrected scen
            new_scen_trans = (new_sim * transition) + (
                scen[sim_posmax] * (1.0 - transition)
            )

            # We change new_scen to the new data
            new_scen[sim_posmax] = new_scen_trans
            return new_scen

        new_scen = xr.apply_ufunc(
            _adjust_extremes_1d,
            scen,
            sim,
            self.ds.fit_params,
            self.ds.thresh,
            input_core_dims=[["time"], ["time"], ["dparams"], []],
            output_core_dims=[["time"]],
            vectorize=True,
            kwargs={
                "dist": stats.get_dist("genpareto"),
                "cluster_thresh": convert_units_to(self.cluster_thresh, sim),
            },
            dask="parallelized",
            output_dtypes=[scen.dtype],
        )

        params = f"frac={frac}, power={power}"
        new_scen.attrs["xclim_history"] = update_history(
            f"Second order bias-adjustment with {str(self)}.adjust(sim, {params})", sim
        )
        return new_scen


class LOCI(BaseAdjustment):
    """Local intensity scaling adjustment intended for daily precipitation."""

    @parse_group
    def __init__(self, *, group: Union[str, Grouper] = "time", thresh: float = None):
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
        At instantiation:

        group : Union[str, Grouper]
          The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        thresh : float
          The threshold in `ref` above which the values are scaled.

        In adjustment:

        interp : {'nearest', 'linear', 'cubic'}
          The interpolation method to use then interpolating the adjustment factors. Defaults to "linear".

        References
        ----------
        .. [Schmidli2006] Schmidli, J., Frei, C., & Vidale, P. L. (2006). Downscaling from GCM precipitation: A benchmark for dynamical and statistical downscaling methods. International Journal of Climatology, 26(5), 679–689. DOI:10.1002/joc.1287
        """
        super().__init__(group=group, thresh=thresh)

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
        self.set_dataset(
            xr.Dataset(dict(hist_thresh=s_thresh, ref_thresh=self.thresh, af=af))
        )

    def _adjust(self, sim, interp="linear"):
        sth = broadcast(self.ds.hist_thresh, sim, group=self.group, interp=interp)
        factor = broadcast(self.ds.af, sim, group=self.group, interp=interp)
        with xr.set_options(keep_attrs=True):
            scen = (factor * (sim - sth) + self.ds.ref_thresh).clip(min=0)
        return scen


class Scaling(BaseAdjustment):
    """Simple scaling adjustment."""

    @parse_group
    def __init__(self, *, group="time", kind=ADDITIVE):
        """Scaling bias-adjustment.

        Simple bias-adjustment method scaling variables by an additive or multiplicative factor so that the mean of `hist`
        matches the mean of `ref`.

        Parameters
        ----------
        At instantiation:

        group : Union[str, Grouper]
          The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        kind : {'+', '*'}
          The adjustment kind, either additive or multiplicative.

        In adjustment:

        interp : {'nearest', 'linear', 'cubic'}
          The interpolation method to use then interpolating the adjustment factors. Defaults to "nearest".
        """
        super().__init__(group=group, kind=kind)

    def _train(self, ref, hist):
        mean_hist = self.group.apply("mean", hist)
        mean_ref = self.group.apply("mean", ref)
        af = get_correction(mean_hist, mean_ref, self.kind)
        af.attrs.update(long_name="Scaling adjustment factors")
        self.set_dataset(xr.Dataset(dict(af=af)))

    def _adjust(self, sim, interp="nearest"):
        factor = broadcast(self.ds.af, sim, group=self.group, interp=interp)
        return apply_correction(sim, factor, self.kind)


class PrincipalComponents(BaseAdjustment):
    """Principal components inspired adjustment."""

    @parse_group
    def __init__(
        self,
        *,
        group: Union[str, Grouper] = "time",
        crd_dims: Optional[Sequence[str]] = None,
        pts_dims: Optional[Sequence[str]] = None,
    ):
        r"""Principal component adjustment.

        This bias-correction method maps model simulation values to the observation
        space through principal components ([hnilica2017]_). Values in the simulation
        space (multiple variables, or multiple sites) can be thought of as coordinates
        along axes, such as variable, temperature, etc. Principal components (PC) are a
        linear combinations of the original variables where the coefficients are the
        eigenvectors of the covariance matrix. Values can then be expressed as coordinates
        along the PC axes. The method makes the assumption that bias-corrected values have
        the same coordinates along the PC axes of the observations. By converting from the
        observation PC space to the original space, we get bias corrected values.
        See notes for a mathematical explanation.

        Note that *principal components* is meant here as the algebraic operation defining a coordinate system
        based on the eigenvectors, not statistical principal component analysis.

        Parameters
        ----------
        At instantiation:

        group : Union[str, Grouper]
          The grouping information. `pts_dims` can also be given through Grouper's
          `add_dims` argument. See Notes.
          See :py:class:`xclim.sdba.base.Grouper` for details.
          The adjustment will be performed on each group independently.
        crd_dims : Sequence of str, optional
          The data dimension(s) along which the multiple simulation space dimensions are taken.
          They are flattened into  "coordinate" dimension, see Notes.
          Default is `None` in which case all dimensions shared by `ref` and `hist`,
          except those in `pts_dims` are used.
          The training algorithm currently doesn't support any chunking
          along the coordinate and point dimensions.
        pts_dims : Sequence of str, optional
          The data dimensions to flatten into the "points" dimension, see Notes.
          They will be merged with those given through the `add_dims` property
          of `group`.

        Notes
        -----
        The input data is understood as a set of N points in a :math:`M`-dimensional space.

        - :math:`N` is taken along the data coordinates listed in `pts_dims` and the `group` (the main `dim` but also the `add_dims`).

        - :math:`M` is taken along the data coordinates listed in `crd_dims`, the default being all except those in `pts_dims`.

        For example, for a 3D matrix of data, say in (lat, lon, time), we could say that all spatial points
        are independent dimensions of the simulation space by passing  ``crd_dims=['lat', 'lon']``. For
        a (5, 5, 365) array, this results in a 25-dimensions space, i.e. :math:`M = 25` and :math:`N = 365`.

        Thus, the adjustment is equivalent to a linear transformation
        of these :math:`N` points in a :math:`M`-dimensional space.

        The principal components (PC) of `hist` and `ref` are used to defined new
        coordinate systems, centered on their respective means. The training step creates a
        matrix defining the transformation from `hist` to `ref`:

        .. math::

          scen = e_{R} + \mathrm{\mathbf{T}}(sim - e_{H})

        Where:

        .. math::

          \mathrm{\mathbf{T}} = \mathrm{\mathbf{R}}\mathrm{\mathbf{H}}^{-1}

        :math:`\mathrm{\mathbf{R}}` is the matrix transforming from the PC coordinates
        computed on `ref` to the data coordinates. Similarly, :math:`\mathrm{\mathbf{H}}`
        is transform from the `hist` PC to the data coordinates
        (:math:`\mathrm{\mathbf{H}}` is the inverse transformation). :math:`e_R` and
        :math:`e_H` are the centroids of the `ref` and `hist` distributions respectively.
        Upon running the  `adjust` step, one may decide to use :math:`e_S`, the centroid
        of the `sim` distribution, instead of :math:`e_H`.

        References
        ----------
        .. [hnilica2017] Hnilica, J., Hanel, M. and Puš, V. (2017), Multisite bias correction of precipitation data from regional climate models. Int. J. Climatol., 37: 2934-2946. https://doi.org/10.1002/joc.4890
        """
        pts_dims = set(pts_dims or []).intersection(set(group.add_dims) - {"window"})

        # Grouper used in the training part
        train_group = Grouper(group.name, window=group.window, add_dims=pts_dims)
        # Grouper used in the adjust part : no window, no add_dims
        adj_group = Grouper(group.name)

        super().__init__(
            train_group=train_group, adj_group=adj_group, crd_dims=crd_dims
        )

    def _train(self, ref, hist):
        all_dims = set(ref.dims).intersection(hist.dims)
        lbl_R = xr.core.utils.get_temp_dimname(all_dims, "crdR")
        lbl_M = xr.core.utils.get_temp_dimname(all_dims, "crdM")
        lbl_P = xr.core.utils.get_temp_dimname(all_dims, "points")

        # Dimensions that represent different points
        pts_dims = set(self.train_group.add_dims)
        pts_dims.update({self.train_group.dim})
        # Dimensions that represents different dimensions.
        crds_M = self.crd_dims or (set(ref.dims).union(hist.dims) - pts_dims)
        # Rename coords on ref, multiindex do not like conflicting coordinates names
        crds_R = [f"{name}_out" for name in crds_M]
        # Stack, so that we have a single "coordinate" axis
        ref = ref.rename(dict(zip(crds_M, crds_R))).stack({lbl_R: crds_R})
        hist = hist.stack({lbl_M: crds_M})

        # The real thing, acting on 2D numpy arrays
        def _compute_transform_matrix(reference, historical):
            """Return the transformation matrix converting simulation coordinates to observation coordinates."""
            # Get transformation matrix from PC coords to ref, dropping points with a NaN coord.
            ref_na = np.isnan(reference).any(axis=0)
            R = pc_matrix(reference[:, ~ref_na])
            # Get transformation matrix from PC coords to hist, dropping points with a NaN coord.
            hist_na = np.isnan(historical).any(axis=0)
            H = pc_matrix(historical[:, ~hist_na])
            # This step needs vectorize with dask, but vectorize doesn't work with dask, argh.
            # Invert to get transformation matrix from hist to PC coords.
            Hinv = np.linalg.inv(H)
            # Fancy trick to choose best orientation on each axes
            # (using eigenvectors, the output axes orientation is undefined)
            orient = best_pc_orientation(R, Hinv)
            # Get transformation matrix
            return (R * orient) @ Hinv

        # The group wrapper
        def _compute_transform_matrices(ds, dim):
            """Apply `_compute_transform_matrix` along dimensions other than time and the variables to map."""
            # The multiple PC-space dimensions are along "coordinate"
            # Matrix multiplication in xarray behaves as a dot product across
            # same-name dimensions, instead of reducing according to the dimension order,
            # as in numpy or normal maths. So crdX all refer to the same dimension,
            # but with names assuring correct matrix multiplication even if they are out of order.
            reference = ds.ref.stack({lbl_P: dim})
            historical = ds.hist.stack({lbl_P: dim})
            transformation = xr.apply_ufunc(
                _compute_transform_matrix,
                reference,
                historical,
                input_core_dims=[[lbl_R, lbl_P], [lbl_M, lbl_P]],
                output_core_dims=[[lbl_R, lbl_M]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            return transformation

        # Transformation matrix, from model coords to ref coords.
        trans = self.train_group.apply(
            _compute_transform_matrices, {"ref": ref, "hist": hist}
        )
        trans.attrs.update(long_name="Transformation from training to target spaces.")

        ref_mean = self.train_group.apply("mean", ref)  # Centroids of ref
        ref_mean.attrs.update(long_name="Centroid point of target.")

        hist_mean = self.train_group.apply("mean", hist)  # Centroids of hist
        hist_mean.attrs.update(long_name="Centroid point of training.")

        self.set_dataset(
            xr.Dataset(dict(trans=trans, ref_mean=ref_mean, hist_mean=hist_mean))
        )
        self.ds.attrs["_reference_coord"] = lbl_R
        self.ds.attrs["_model_coord"] = lbl_M

    def _adjust(self, sim):
        lbl_R = self.ds.attrs["_reference_coord"]
        lbl_M = self.ds.attrs["_model_coord"]
        crds_M = self.ds.indexes[lbl_M].names

        vmean = self.train_group.apply("mean", sim).stack({lbl_M: crds_M})

        sim = sim.stack({lbl_M: crds_M})

        def _compute_adjust(ds, dim):
            """Apply the mapping transformation."""
            scenario = ds.ref_mean + ds.trans.dot((ds.sim - ds.vmean), [lbl_M])
            return scenario

        scen = self.adj_group.apply(
            _compute_adjust,
            {
                "ref_mean": self.ds.ref_mean,
                "trans": self.ds.trans,
                "sim": sim,
                "vmean": vmean,
            },
        )
        scen[lbl_R] = sim.indexes[lbl_M]
        return scen.unstack(lbl_R)
