"""Adjustment objects."""
from typing import Any, Mapping, Optional, Sequence, Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.calendar import get_calendar
from xclim.core.formatting import update_history
from xclim.core.options import OPTIONS, SDBA_EXTRA_OUTPUT, set_options
from xclim.core.units import convert_units_to
from xclim.core.utils import uses_dask
from xclim.indices import stats

from ._adjustment import (
    dqm_adjust,
    dqm_train,
    eqm_train,
    loci_adjust,
    loci_train,
    npdf_transform,
    qdm_adjust,
    qm_adjust,
    scaling_adjust,
    scaling_train,
)
from .base import Grouper, ParametrizableWithDataset, parse_group
from .utils import (
    ADDITIVE,
    best_pc_orientation,
    equally_spaced_nodes,
    get_clusters,
    get_clusters_1d,
    pc_matrix,
    rand_rot_matrix,
)

__all__ = [
    "BaseAdjustment",
    "EmpiricalQuantileMapping",
    "DetrendedQuantileMapping",
    "LOCI",
    "PrincipalComponents",
    "QuantileDeltaMapping",
    "Scaling",
    "NpdfTransform",
]


class BaseAdjustment(ParametrizableWithDataset):
    """Base class for adjustment objects.

    Children classes should implement the `train` and / or the `adjust` method.

    This base class defined the basic input and output checks. It should only be used for a real adjustment
    if neither `TrainAdjust` or `Adjust` fit the algorithm.
    """

    _allow_diff_calendars = True
    _attribute = "_xclim_adjustment"

    def __init__(self, *args, _trained=False, **kwargs):
        if _trained:
            super().__init__(*args, **kwargs)
        else:
            raise ValueError(
                "As of xclim 0.29, Adjustment object should be initialized through their `train` or  `adjust` methods."
            )

    @classmethod
    def _check_inputs(cls, *inputs, group):
        """
        Raises an error if there are chunks along the main dimension.
        Also raises if cls._allow_diff_calendars is False and calendars differ.
        """
        for inda in inputs:
            if uses_dask(inda) and len(inda.chunks[inda.get_axis_num(group.dim)]) > 1:
                raise ValueError(
                    f"Multiple chunks along the main adjustment dimension {group.dim} is not supported."
                )

        # All calendars used by the inputs
        calendars = {get_calendar(inda, group.dim) for inda in inputs}
        if not cls._allow_diff_calendars and len(calendars) > 1:
            raise ValueError(
                "Inputs are defined on different calendars,"
                f" this is not supported for {cls.__name__} adjustment."
            )

        if group.prop == "dayofyear" and (
            "default" in calendars or "standard" in calendars
        ):
            warn(
                "Strange results could be returned when using dayofyear grouping "
                "on data defined in the proleptic_gregorian calendar "
            )

    @classmethod
    def _harmonize_units(cls, *inputs, target: Optional[str] = None):
        """Convert all inputs to the same units.

        If the target unit is not given, the units of the first input are used.

        Returns the converted inputs and the target units.
        """
        if target is None:
            target = inputs[0].units

        return (convert_units_to(inda, target) for inda in inputs), target

    @classmethod
    def _train(ref, hist, **kwargs):
        raise NotImplementedError()

    def _adjust(sim, *args, **kwargs):
        raise NotImplementedError()


class TrainAdjust(BaseAdjustment):
    """Base class for adjustment objects obeying the train-adjust scheme.

    Children classes should implement these methods:

    @classmethod
    _train(ref, hist, **kwargs)
      Receiving the training target and data, returning a training dataset and parameters to store in the object.

    _adjust(sim, **kwargs)
      Receiving the projected data and some arguments, returning the `scen` dataarray.

    """

    _allow_diff_calendars = True
    _attribute = "_xclim_adjustment"
    _repr_hide_params = ["hist_calendar", "train_units"]

    @classmethod
    def train(cls, ref: DataArray, hist: DataArray, **kwargs):
        """Train the adjustment object. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
          Training target, usually a reference time series drawn from observations.
        hist : DataArray
          Training data, usually a model output whose biases are to be adjusted.
        """
        kwargs = parse_group(cls._train, kwargs)

        (ref, hist), train_units = cls._harmonize_units(ref, hist)

        if "group" in kwargs:
            cls._check_inputs(ref, hist, group=kwargs["group"])

        hist = convert_units_to(hist, ref)

        ds, params = cls._train(ref, hist, **kwargs)
        obj = cls(
            _trained=True,
            hist_calendar=get_calendar(hist),
            train_units=train_units,
            **params,
        )
        obj.set_dataset(ds)
        return obj

    def adjust(self, sim: DataArray, *args, **kwargs):
        """Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        sim : DataArray
          Time series to be bias-adjusted, usually a model output.
        *args : xr.DataArray
          Other DataArrays needed for the adjustment (usually none).
        kwargs :
          Algorithm-specific keyword arguments, see class doc.
        """
        (sim, *args), _ = self._harmonize_units(sim, *args, target=self.train_units)

        if "group" in self:
            self._check_inputs(sim, *args, group=self.group)

        sim = convert_units_to(sim, self.train_units)
        out = self._adjust(sim, *args, **kwargs)

        if isinstance(out, xr.DataArray):
            out = out.rename("scen").to_dataset()

        scen = out.scen

        params = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        infostr = f"{str(self)}.adjust(sim, {params})"
        scen.attrs["history"] = update_history(f"Bias-adjusted with {infostr}", sim)
        scen.attrs["bias_adjustment"] = infostr
        scen.attrs["units"] = self.train_units

        if OPTIONS[SDBA_EXTRA_OUTPUT]:
            return out
        return scen

    def set_dataset(self, ds: xr.Dataset):
        """Stores an xarray dataset in the `ds` attribute.

        Useful with custom object initialization or if some external processing was performed.
        """
        super().set_dataset(ds)
        self.ds.attrs["adj_params"] = str(self)

    @classmethod
    def _train(cls, ref: DataArray, hist: DataArray, *kwargs):
        raise NotImplementedError

    def _adjust(self, sim, **kwargs):
        raise NotImplementedError


class Adjust(BaseAdjustment):
    """Adjustment with no intermediate trained object.

    Children classes should implement a `_adjust` classmethod taking as input the three DataArrays
    and returning the scen dataset/array.
    """

    @classmethod
    def adjust(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        sim: xr.DataArray,
        **kwargs,
    ):
        """Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
          Training target, usually a reference time series drawn from observations.
        hist : DataArray
          Training data, usually a model output whose biases are to be adjusted.
        sim : DataArray
          Time series to be bias-adjusted, usually a model output.
        kwargs :
          Algorithm-specific keyword arguments, see class doc.
        """
        kwargs = parse_group(cls._adjust, kwargs)
        if "group" in kwargs:
            cls._check_inputs(ref, hist, sim, group=kwargs["group"])

        (ref, hist, sim), _ = cls._harmonize_units(ref, hist, sim)

        out = cls._adjust(ref, hist, sim, **kwargs)

        if isinstance(out, xr.DataArray):
            out = out.rename("scen").to_dataset()

        scen = out.scen

        params = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        infostr = f"{cls.__name__}.adjust(ref, hist, sim, {params})"
        scen.attrs["history"] = update_history(f"Bias-adjusted with {infostr}", sim)
        scen.attrs["bias_adjustment"] = infostr
        scen.attrs["units"] = ref.units

        if OPTIONS[SDBA_EXTRA_OUTPUT]:
            return out
        return scen


class EmpiricalQuantileMapping(TrainAdjust):
    """Empirical Quantile Mapping bias-adjustment.

    Adjustment factors are computed between the quantiles of `ref` and `sim`.
    Values of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

    .. math::

      F^{-1}_{ref} (F_{hist}(sim))

    where :math:`F` is the cumulative distribution function (CDF) and `mod` stands for model data.

    Parameters
    ----------
    Train step:

    nquantiles : int or 1d array of floats
      The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
      An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles (+2 endpoints).
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative. Defaults to "+".
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      Default is "time", meaning an single adjustment group along dimension "time".

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use when interpolating the adjustment factors. Defaults to "nearset".
    extrapolation : {'constant', 'nan'}
      The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".

    References
    ----------
    Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030

    _allow_diff_calendars = False

    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        group: Union[str, Grouper] = "time",
    ):
        if np.isscalar(nquantiles):
            quantiles = equally_spaced_nodes(nquantiles, eps=1e-6).astype(ref.dtype)
        else:
            quantiles = nquantiles

        ds = eqm_train(
            xr.Dataset({"ref": ref, "hist": hist}),
            group=group,
            kind=kind,
            quantiles=quantiles,
        )

        ds.af.attrs.update(
            standard_name="Adjustment factors",
            long_name="Quantile mapping adjustment factors",
        )
        ds.hist_q.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        return ds, {"group": group, "kind": kind}

    def _adjust(self, sim, interp="nearest", extrapolation="constant"):
        return qm_adjust(
            xr.Dataset({"af": self.ds.af, "hist_q": self.ds.hist_q, "sim": sim}),
            group=self.group,
            interp=interp,
            extrapolation=extrapolation,
            kind=self.kind,
        ).scen


class DetrendedQuantileMapping(TrainAdjust):
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
    Train step:

    nquantiles : int or 1d array of floats
      The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
      An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles (+2 endpoints).
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative. Defaults to "+".
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      Default is "time", meaning an single adjustment group along dimension "time".

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use when interpolating the adjustment factors. Defaults to "nearest".
    detrend : int or BaseDetrend instance
      The method to use when detrending. If an int is passed, it is understood as a PolyDetrend (polynomial detrending) degree.
      Defaults to 1 (linear detrending)
    extrapolation : {'constant', 'nan'}
      The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".

    References
    ----------
    .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        nquantiles: int = 20,
        kind: str = ADDITIVE,
        group: Union[str, Grouper] = "time",
    ):
        if group.prop not in ["group", "dayofyear"]:
            warn(
                f"Using DQM with a grouping other than 'dayofyear' is not recommended (received {group.name})."
            )

        if np.isscalar(nquantiles):
            quantiles = equally_spaced_nodes(nquantiles, eps=1e-6).astype(ref.dtype)
        else:
            quantiles = nquantiles

        ds = dqm_train(
            xr.Dataset({"ref": ref, "hist": hist}),
            group=group,
            quantiles=quantiles,
            kind=kind,
        )

        ds.af.attrs.update(
            standard_name="Adjustment factors",
            long_name="Quantile mapping adjustment factors",
        )
        ds.hist_q.attrs.update(
            standard_name="Model quantiles",
            long_name="Quantiles of model on the reference period",
        )
        ds.scaling.attrs.update(
            standard_name="Scaling factor",
            description="Scaling factor making the mean of hist match the one of hist.",
        )
        return ds, {"group": group, "kind": kind}

    def _adjust(
        self,
        sim,
        interp="nearest",
        extrapolation="constant",
        detrend=1,
    ):

        scen = dqm_adjust(
            self.ds.assign(sim=sim),
            interp=interp,
            extrapolation=extrapolation,
            detrend=detrend,
            group=self.group,
            kind=self.kind,
        ).scen
        # Detrending needs units.
        scen.attrs["units"] = sim.units
        return scen


class QuantileDeltaMapping(EmpiricalQuantileMapping):
    r"""Quantile Delta Mapping bias-adjustment.

    Adjustment factors are computed between the quantiles of `ref` and `hist`.
    Quantiles of `sim` are matched to the corresponding quantiles of `hist` and corrected accordingly.

    .. math::

        sim\frac{F^{-1}_{ref}\left[F_{sim}(sim)\right]}{F^{-1}_{hist}\left[F_{sim}(sim)\right]}

    where :math:`F` is the cumulative distribution function (CDF). This equation is valid for multiplicative adjustment.
    The algorithm is based on the "QDM" method of [Cannon2015]_.

    Parameters
    ----------
    Train step:

    nquantiles : int or 1d array of floats
      The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
      An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles (+2 endpoints).
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative. Defaults to "+".
    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      Default is "time", meaning an single adjustment group along dimension "time".

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use when interpolating the adjustment factors. Defaults to "nearest".
    extrapolation : {'constant', 'nan'}
      The type of extrapolation to use. See :py:func:`xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".

    Extra diagnostics
    -----------------
    In adjustment:

    quantiles : The quantile of each value of `sim`. The adjustment factor is interpolated using this as the "quantile" axis on `ds.af`.

    References
    ----------
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    def _adjust(self, sim, interp="nearest", extrapolation="constant"):
        out = qdm_adjust(
            xr.Dataset({"sim": sim, "af": self.ds.af, "hist_q": self.ds.hist_q}),
            group=self.group,
            interp=interp,
            extrapolation=extrapolation,
            kind=self.kind,
        )
        if OPTIONS[SDBA_EXTRA_OUTPUT]:
            out.sim_q.attrs.update(long_name="Group-wise quantiles of `sim`.")
            return out
        return out.scen


class ExtremeValues(TrainAdjust):
    r"""Adjustment correction for extreme values.

    The tail of the distribution of adjusted data is corrected according to the
    parametric Generalized Pareto distribution of the reference data, [RRJF2021]_.
    The distributions are composed of the maximal values of clusters of "large" values.
    With "large" values being those above `cluster_thresh`. Only extreme values, whose
    quantile within the pool of large values are above `q_thresh`, are re-adjusted.
    See Notes.

    Parameters
    ----------
    Train step:

    cluster_thresh: Quantity (str with units)
      The threshold value for defining clusters.
    q_thresh : float
      The quantile of "extreme" values, [0, 1[. Defaults to 0.95.
    ref_params :  xr.DataArray, optional
      Distribution parameters to use instead of fitting a GenPareto distribution on `ref`.

    Adjust step:

    scen: DataArray
      This is a second-order adjustment, so the adjust method needs the first-order
      adjusted timeseries in addition to the raw "sim".
    frac: float
      Fraction where the cutoff happens between the original scen and the corrected one.
      See Notes, ]0, 1]. Defaults to 0.25.
    power: float
      Shape of the correction strength, see Notes. Defaults to 1.0.

    Extra diagnostics
    -----------------
    In training:

    nclusters : Number of extreme value clusters found for each gridpoint.

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

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        cluster_thresh: str,
        ref_params: xr.Dataset = None,
        q_thresh: float = 0.95,
    ):
        warn(
            "The ExtremeValues adjustment is a work in progress and not production-ready. The current version imitates the algorithm found in ClimateTools.jl, but results might not be accurate."
        )
        cluster_thresh = convert_units_to(cluster_thresh, ref)
        hist = convert_units_to(hist, ref)

        # Extreme value threshold computed relative to "large values".
        # We use the mean between ref and hist here.
        thresh = (
            ref.where(ref >= cluster_thresh).quantile(q_thresh, dim="time")
            + hist.where(hist >= cluster_thresh).quantile(q_thresh, dim="time")
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
            long_name=f"{q_thresh * 100}th percentile extreme value threshold",
            description=f"Mean of the {q_thresh * 100}th percentile of large values (x > {cluster_thresh}) of ref and hist.",
        )

        if OPTIONS[SDBA_EXTRA_OUTPUT] and ref_params is None:
            ds = ds.assign(nclusters=ref_clusters.nclusters)

        return ds, {"cluster_thresh": cluster_thresh}

    def _adjust(
        self,
        sim: xr.DataArray,
        scen: xr.DataArray,
        *,
        frac: float = 0.25,
        power: float = 1.0,
    ):
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
            # removed a +thresh that was a bug in the julia source.
            new_sim = dist.ppf(sim_cdf, *ref_params)

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

        return new_scen


class LOCI(TrainAdjust):
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
    Train step:

    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      Default is "time", meaning an single adjustment group along dimension "time".
    thresh : str
      The threshold in `ref` above which the values are scaled.

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the adjustment factors. Defaults to "linear".

    References
    ----------
    .. [Schmidli2006] Schmidli, J., Frei, C., & Vidale, P. L. (2006). Downscaling from GCM precipitation: A benchmark for dynamical and statistical downscaling methods. International Journal of Climatology, 26(5), 679–689. DOI:10.1002/joc.1287
    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        thresh: str,
        group: Union[str, Grouper] = "time",
    ):
        thresh = convert_units_to(thresh, ref)
        ds = loci_train(
            xr.Dataset({"ref": ref, "hist": hist}), group=group, thresh=thresh
        )
        ds.af.attrs.update(long_name="LOCI adjustment factors")
        ds.hist_thresh.attrs.update(long_name="Threshold over modeled data")
        return ds, {"group": group, "thresh": thresh}

    def _adjust(self, sim, interp="linear"):
        return loci_adjust(
            xr.Dataset(
                {"hist_thresh": self.ds.hist_thresh, "af": self.ds.af, "sim": sim}
            ),
            group=self.group,
            thresh=self.thresh,
            interp=interp,
        ).scen


class Scaling(TrainAdjust):
    """Scaling bias-adjustment.

    Simple bias-adjustment method scaling variables by an additive or multiplicative factor so that the mean of `hist`
    matches the mean of `ref`.

    Parameters
    ----------
    Train step:

    group : Union[str, Grouper]
      The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
      Default is "time", meaning an single adjustment group along dimension "time".
    kind : {'+', '*'}
      The adjustment kind, either additive or multiplicative. Defaults to "+".

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use then interpolating the adjustment factors. Defaults to "nearest".
    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        group: Union[str, Grouper] = "time",
        kind: str = ADDITIVE,
    ):
        ds = scaling_train(
            xr.Dataset({"ref": ref, "hist": hist}), group=group, kind=kind
        )
        ds.af.attrs.update(long_name="Scaling adjustment factors")
        return ds, {"group": group, "kind": kind}

    def _adjust(self, sim, interp="nearest"):
        return scaling_adjust(
            xr.Dataset({"sim": sim, "af": self.ds.af}),
            group=self.group,
            interp=interp,
            kind=self.kind,
        ).scen


class PrincipalComponents(TrainAdjust):
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
    group : Union[str, Grouper]
      The grouping information. `pts_dims` can also be given through Grouper's
      `add_dims` argument. See Notes.
      See :py:class:`xclim.sdba.base.Grouper` for details.
      The adjustment will be performed on each group independently.
      Default is "time", meaning an single adjustment group along dimension "time".
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

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        group: Union[str, Grouper] = "time",
        crd_dims: Optional[Sequence[str]] = None,
        pts_dims: Optional[Sequence[str]] = None,
    ):
        pts_dims = set(pts_dims or []).intersection(set(group.add_dims) - {"window"})

        # Grouper used in the training part
        train_group = Grouper(group.name, window=group.window, add_dims=pts_dims)
        # Grouper used in the adjust part : no window, no add_dims
        adj_group = Grouper(group.name)

        all_dims = set(ref.dims).intersection(hist.dims)
        lbl_R = xr.core.utils.get_temp_dimname(all_dims, "crdR")
        lbl_M = xr.core.utils.get_temp_dimname(all_dims, "crdM")
        lbl_P = xr.core.utils.get_temp_dimname(all_dims, "points")

        # Dimensions that represent different points
        pts_dims = set(train_group.add_dims)
        pts_dims.update({train_group.dim})
        # Dimensions that represents different dimensions.
        crds_M = crd_dims or (set(ref.dims).union(hist.dims) - pts_dims)
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
        trans = train_group.apply(
            _compute_transform_matrices, {"ref": ref, "hist": hist}
        )
        trans.attrs.update(long_name="Transformation from training to target spaces.")

        ref_mean = train_group.apply("mean", ref)  # Centroids of ref
        ref_mean.attrs.update(long_name="Centroid point of target.")

        hist_mean = train_group.apply("mean", hist)  # Centroids of hist
        hist_mean.attrs.update(long_name="Centroid point of training.")

        ds = xr.Dataset(dict(trans=trans, ref_mean=ref_mean, hist_mean=hist_mean))

        ds.attrs["_reference_coord"] = lbl_R
        ds.attrs["_model_coord"] = lbl_M
        return ds, {"train_group": train_group, "adj_group": adj_group}

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


class NpdfTransform(Adjust):
    r"""N-dimensional probability density function transform.

    This adjustment object combines both training and adjustent steps in the `adjust` class method.

    A multivariate bias-adjustment algorithm described by [Cannon18]_, as part of the MBCn algorithm,
    based on a color-correction algorithm described by [Pitie05]_.

    This algorithm in itself, when used with QuantileDeltaMapping, is NOT trend-preserving.
    The full MBCn algorithm includes a reordering step provided here by :py:func:`xclim.sdba.processing.reordering`.

    See notes for an explanation of the algorithm.

    Parameters
    ----------
    base: BaseAdjustment
      An univariate bias-adjustment class. This is untested for anything else than QuantileDeltaMapping.
    base_kws : dict, optional
      Arguments passed to the training of the univariate adjustment.
    n_escore : int
      The number of elements to send to the escore function. The default, 0, means all elements are included.
      Pass -1 to skip computing the escore completely.
      Small numbers result in less significative scores, but the execution time goes up quickly with large values.
    n_iter : int
      The number of iterations to perform. Defaults to 20.
    pts_dim : str
      The name of the "multivariate" dimension. Defaults to "variables", which is the
      normal case when using :py:func:`xclim.sdba.base.stack_variables`.
    adj_kws : dict, optional
      Dictionary of arguments to pass to the adjust method of the univariate adjustment.
    rot_matrices : xr.DataArray, optional
      The rotation matrices as a 3D array ('iterations', <pts_dim>, <anything>), with shape (n_iter, <N>, <N>).
      If left empty, random rotation matrices will be automatically generated.

    Notes
    -----
    The historical reference (:math:`T`, for "target"), simulated historical (:math:`H`) and simulated projected (:math:`S`)
    datasets are constructed by stacking the timeseries of N variables together. The algoriths goes into the
    following steps:

    1. Rotate the datasets in the N-dimensional variable space with :math:`\mathbf{R}`, a random rotation NxN matrix.

    ..math ::

        \tilde{\mathbf{T}} = \mathbf{T}\mathbf{R} \\
        \tilde{\mathbf{H}} = \mathbf{H}\mathbf{R} \\
        \tilde{\mathbf{S}} = \mathbf{S}\mathbf{R}

    2. An univariate bias-adjustment :math:`\mathcal{F}` is used on the rotated datasets.
    The adjustments are made in additive mode, for each variable :math:`i`.

    .. math::

        \hat{\mathbf{H}}_i, \hat{\mathbf{S}}_i = \mathcal{F}\left(\tilde{\mathbf{T}}_i, \tilde{\mathbf{H}}_i, \tilde{\mathbf{S}}_i\right)

    3. The bias-adjusted datasets are rotated back.

    .. math::

        \mathbf{H}' = \hat{\mathbf{H}}\mathbf{R} \\
        \mathbf{S}' = \hat{\mathbf{S}}\mathbf{R}


    These three steps are repeated a certain number of times, prescribed by argument ``n_iter``. At each
    iteration, a new random rotation matrix is generated.

    The original algorithm ([Pitie05]_), stops the iteration when some distance score converges. Following
    [Cannon18_] and the MBCn implementation in [CannonR]_, we instead fix the number of iterations.

    As done by [Cannon18]_, the distance score chosen is the "Energy distance" from [SkezelyRizzo]_
    (see :py:func:`xclim.sdba.processing.escore`).

    The random matrices are generated following a method laid out by [Mezzadri].

    This is only part of the full MBCn algorithm, see :ref:`The MBCn algorithm` for an example on how
    to replicate the full method with xclim. This includes a standardization of the simulated data beforehand,
    an initial univariate adjustment and the reordering of those adjusted series according to the
    rank structure of the output of this algorithm.

    References
    ----------
    .. [Cannon18] Cannon, A. J. (2018). Multivariate quantile mapping bias correction: An N-dimensional probability density function transform for climate model simulations of multiple variables. Climate Dynamics, 50(1), 31–49. https://doi.org/10.1007/s00382-017-3580-6
    .. [Mezzadri]Mezzadri, F. (2006). How to generate random matrices from the classical compact groups. arXiv preprint math-ph/0609050.
    .. [Pitie05] Pitie, F., Kokaram, A. C., & Dahyot, R. (2005). N-dimensional probability density function transfer and its application to color transfer. Tenth IEEE International Conference on Computer Vision (ICCV’05) Volume 1, 2, 1434-1439 Vol. 2. https://doi.org/10.1109/ICCV.2005.166
    .. [SkezelyRizzo] Szekely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
    """

    @classmethod
    def _adjust(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        sim: xr.DataArray,
        *,
        base: TrainAdjust = QuantileDeltaMapping,
        base_kws: Optional[Mapping[str, Any]] = None,
        n_escore: int = 0,
        n_iter: int = 20,
        pts_dim: str = "variables",
        adj_kws: Optional[Mapping[str, Any]] = None,
        rot_matrices: Optional[xr.DataArray] = None,
    ):
        base_kws or {}
        if "kind" in base_kws:
            warn(
                f'The adjustment kind cannot be controlled when using {cls.__name__}, it defaults to "+".'
            )
        base_kws.setdefault("kind", "+")

        # Assuming sim has the same coords as hist
        # We get the safest new name of the rotated dim.
        rot_dim = xr.core.utils.get_temp_dimname(
            set(ref.dims).union(hist.dims).union(sim.dims), pts_dim + "_prime"
        )

        # Get the rotation matrices
        rot_matrices = rot_matrices or rand_rot_matrix(
            ref[pts_dim], num=n_iter, new_dim=rot_dim
        ).rename(matrices="iterations")

        # Call a map_blocks on the iterative function
        # Sadly, this is a bit too complicated for map_blocks, we'll do it by hand.
        escores_tmpl = xr.broadcast(
            ref.isel({pts_dim: 0, "time": 0}),
            hist.isel({pts_dim: 0, "time": 0}),
        )[0].expand_dims(iterations=rot_matrices.iterations)

        template = xr.Dataset(
            data_vars={
                "scenh": xr.full_like(hist, np.NaN).rename(time="time_hist"),
                "scen": xr.full_like(sim, np.NaN),
                "escores": escores_tmpl,
            }
        )

        # Input data, rename time dim on sim since it can't be aligned with ref or hist.
        ds = xr.Dataset(
            data_vars={
                "ref": ref.rename(time="time_hist"),
                "hist": hist.rename(time="time_hist"),
                "sim": sim,
                "rot_matrices": rot_matrices,
            }
        )

        kwargs = {
            "base": base,
            "base_kws": base_kws,
            "n_escore": n_escore,
            "n_iter": n_iter,
            "pts_dim": pts_dim,
            "adj_kws": adj_kws or {},
        }

        with set_options(sdba_extra_output=False):
            out = ds.map_blocks(npdf_transform, template=template, kwargs=kwargs)

        out = out.assign(rotation_matrices=rot_matrices)
        out.scenh.attrs["units"] = hist.units
        return out
