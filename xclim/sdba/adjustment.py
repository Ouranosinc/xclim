"""
Adjustment Methods
==================
"""
from __future__ import annotations

from inspect import signature
from typing import Any
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.calendar import get_calendar
from xclim.core.formatting import gen_call_string, update_history
from xclim.core.options import OPTIONS, SDBA_EXTRA_OUTPUT, set_options
from xclim.core.units import convert_units_to
from xclim.core.utils import uses_dask
from xclim.indices import stats

from ._adjustment import (
    dqm_adjust,
    dqm_train,
    eqm_train,
    extremes_adjust,
    extremes_train,
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
    best_pc_orientation_full,
    best_pc_orientation_simple,
    equally_spaced_nodes,
    pc_matrix,
    rand_rot_matrix,
)

__all__ = [
    "BaseAdjustment",
    "DetrendedQuantileMapping",
    "EmpiricalQuantileMapping",
    "ExtremeValues",
    "LOCI",
    "NpdfTransform",
    "PrincipalComponents",
    "QuantileDeltaMapping",
    "Scaling",
]


class BaseAdjustment(ParametrizableWithDataset):
    """Base class for adjustment objects.

    Children classes should implement the `train` and / or the `adjust` method.

    This base class defined the basic input and output checks. It should only be used for a real adjustment
    if neither `TrainAdjust` nor `Adjust` fit the algorithm.
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
        """Raise an error if there are chunks along the main dimension.

        Also raises if :py:attr:`BaseAdjustment._allow_diff_calendars` is False and calendars differ.
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

        # Check multivariate dimensions
        mvcrds = []
        for inda in inputs:
            for crd in inda.coords.values():
                if crd.attrs.get("is_variables", False):
                    mvcrds.append(crd)
        if mvcrds and (
            not all(mvcrds[0].equals(mv) for mv in mvcrds[1:])
            or len(mvcrds) != len(inputs)
        ):
            raise ValueError(
                "Inputs have different multivariate coordinates "
                f"({set(mv.name for mv in mvcrds)})."
            )

        if group.prop == "dayofyear" and (
            "default" in calendars or "standard" in calendars
        ):
            warn(
                "Strange results could be returned when using dayofyear grouping "
                "on data defined in the proleptic_gregorian calendar "
            )

    @classmethod
    def _harmonize_units(cls, *inputs, target: str | None = None):
        """Convert all inputs to the same units.

        If the target unit is not given, the units of the first input are used.

        Returns the converted inputs and the target units.
        """
        if target is None:
            target = inputs[0].units

        return (
            convert_units_to(inda, target, context="infer") for inda in inputs
        ), target

    @classmethod
    def _train(cls, ref, hist, **kwargs):
        raise NotImplementedError()

    def _adjust(self, sim, *args, **kwargs):
        raise NotImplementedError()


class TrainAdjust(BaseAdjustment):
    """Base class for adjustment objects obeying the train-adjust scheme.

    Children classes should implement these methods:

    - ``_train(ref, hist, **kwargs)``, classmethod receiving the training target and data, returning a training dataset and parameters to store in the object.

    - ``_adjust(sim, **kwargs)``, receiving the projected data and some arguments, returning the `scen` DataArray.

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
        skip_checks = kwargs.pop("skip_input_checks", False)

        if not skip_checks:
            (ref, hist), train_units = cls._harmonize_units(ref, hist)

            if "group" in kwargs:
                cls._check_inputs(ref, hist, group=kwargs["group"])

            hist = convert_units_to(hist, ref)
        else:
            train_units = ""

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
        args : xr.DataArray
            Other DataArrays needed for the adjustment (usually none).
        kwargs
            Algorithm-specific keyword arguments, see class doc.
        """
        skip_checks = kwargs.pop("skip_input_checks", False)
        if not skip_checks:
            (sim, *args), _ = self._harmonize_units(sim, *args, target=self.train_units)

            if "group" in self:
                self._check_inputs(sim, *args, group=self.group)

            sim = convert_units_to(sim, self.train_units)
        out = self._adjust(sim, *args, **kwargs)

        if isinstance(out, xr.DataArray):
            out = out.rename("scen").to_dataset()

        scen = out.scen

        # Keep attrs
        scen.attrs.update(sim.attrs)
        for name, crd in sim.coords.items():
            if name in scen.coords:
                scen[name].attrs.update(crd.attrs)
        params = gen_call_string("", **kwargs)[1:-1]  # indexing to remove added ( )
        infostr = f"{str(self)}.adjust(sim, {params})"
        scen.attrs["history"] = update_history(f"Bias-adjusted with {infostr}", sim)
        scen.attrs["bias_adjustment"] = infostr
        scen.attrs["units"] = self.train_units

        if OPTIONS[SDBA_EXTRA_OUTPUT]:
            return out
        return scen

    def set_dataset(self, ds: xr.Dataset):
        """Store an xarray dataset in the `ds` attribute.

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
        r"""Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        ref : DataArray
            Training target, usually a reference time series drawn from observations.
        hist : DataArray
            Training data, usually a model output whose biases are to be adjusted.
        sim : DataArray
            Time series to be bias-adjusted, usually a model output.
        \*\*kwargs
            Algorithm-specific keyword arguments, see class doc.
        """
        kwargs = parse_group(cls._adjust, kwargs)
        skip_checks = kwargs.pop("skip_input_checks", False)

        if not skip_checks:
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

    Attributes
    ----------
    Train step

    nquantiles : int or 1d array of floats
        The number of quantiles to use. Two endpoints at 1e-6 and 1 - 1e-6 will be added.
        An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles.
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
    :cite:cts:`sdba-deque_frequency_2007`
    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        nquantiles: int | np.ndarray = 20,
        kind: str = ADDITIVE,
        group: str | Grouper = "time",
    ):
        if np.isscalar(nquantiles):
            quantiles = equally_spaced_nodes(nquantiles).astype(ref.dtype)
        else:
            quantiles = nquantiles.astype(ref.dtype)

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
    This equation is valid for multiplicative adjustment. Based on the DQM method of :cite:p:`sdba-cannon_bias_2015`.

    Parameters
    ----------
    Train step:

    nquantiles : int or 1d array of floats
        The number of quantiles to use. See :py:func:`~xclim.sdba.utils.equally_spaced_nodes`.
        An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles.
    kind : {'+', '*'}
        The adjustment kind, either additive or multiplicative. Defaults to "+".
    group : Union[str, Grouper]
        The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        Default is "time", meaning a single adjustment group along dimension "time".

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
    :cite:cts:`sdba-cannon_bias_2015`

    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        nquantiles: int | np.ndarray = 20,
        kind: str = ADDITIVE,
        group: str | Grouper = "time",
    ):
        if group.prop not in ["group", "dayofyear"]:
            warn(
                f"Using DQM with a grouping other than 'dayofyear' is not recommended (received {group.name})."
            )

        if np.isscalar(nquantiles):
            quantiles = equally_spaced_nodes(nquantiles).astype(ref.dtype)
        else:
            quantiles = nquantiles.astype(ref.dtype)

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
    The algorithm is based on the "QDM" method of :cite:p:`sdba-cannon_bias_2015`.

    Parameters
    ----------
    Train step:

    nquantiles : int or 1d array of floats
        The number of quantiles to use. See :py:func:`~xclim.sdba.utils.equally_spaced_nodes`.
        An array of quantiles [0, 1] can also be passed. Defaults to 20 quantiles.
    kind : {'+', '*'}
        The adjustment kind, either additive or multiplicative. Defaults to "+".
    group : Union[str, Grouper]
        The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        Default is "time", meaning a single adjustment group along dimension "time".

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
    :cite:cts:`sdba-cannon_bias_2015`
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

    The tail of the distribution of adjusted data is corrected according to the bias between the parametric Generalized
    Pareto distributions of the simulated and reference data :cite:p:`sdba-roy_extremeprecip_2023`. The distributions are composed of the
    maximal values of clusters of "large" values.  With "large" values being those above `cluster_thresh`. Only extreme
    values, whose quantile within the pool of large values are above `q_thresh`, are re-adjusted. See `Notes`.

    This adjustment method should be considered experimental and used with care.

    Parameters
    ----------
    Train step :

    cluster_thresh : Quantity (str with units)
        The threshold value for defining clusters.
    q_thresh : float
        The quantile of "extreme" values, [0, 1[. Defaults to 0.95.
    ref_params :  xr.DataArray, optional
        Distribution parameters to use instead of fitting a GenPareto distribution on `ref`.

    Adjust step:

    scen : DataArray
        This is a second-order adjustment, so the adjust method needs the first-order
        adjusted timeseries in addition to the raw "sim".
    interp : {'nearest', 'linear', 'cubic'}
        The interpolation method to use when interpolating the adjustment factors. Defaults to "linear".
    extrapolation : {'constant', 'nan'}
        The type of extrapolation to use. See :py:func:`~xclim.sdba.utils.extrapolate_qm` for details. Defaults to "constant".
    frac : float
        Fraction where the cutoff happens between the original scen and the corrected one.
        See Notes, ]0, 1]. Defaults to 0.25.
    power : float
        Shape of the correction strength, see Notes. Defaults to 1.0.

    Notes
    -----
    Extreme values are extracted from `ref`, `hist` and `sim` by finding all "clusters", i.e. runs of consecutive values
    above `cluster_thresh`. The `q_thresh`th percentile of these values is taken on `ref` and `hist` and becomes
    `thresh`, the extreme value threshold. The maximal value of each cluster, if it exceeds that new threshold, is taken
    and Generalized Pareto distributions are fitted to them, for both `ref` and `hist`. The probabilities associated
    with each of these extremes in `hist` is used to find the corresponding value according to `ref`'s distribution.
    Adjustment factors are computed as the bias between those new extremes and the original ones.

    In the adjust step, a Generalized Pareto distributions is fitted on the cluster-maximums of `sim` and it is used to
    associate a probability to each extreme, values over the `thresh` compute in the training, without the clustering.
    The adjustment factors are computed by interpolating the trained ones using these probabilities and the
    probabilities computed from `hist`.

    Finally, the adjusted values (:math:`C_i`) are mixed with the pre-adjusted ones (`scen`, :math:`D_i`) using the
    following transition function:

    .. math::

        V_i = C_i * \tau + D_i * (1 - \tau)

    Where :math:`\tau` is a function of sim's extreme values (unadjusted, :math:`S_i`)
    and of arguments ``frac`` (:math:`f`) and ``power`` (:math:`p`):

    .. math::

        \tau = \left(\frac{1}{f}\frac{S - min(S)}{max(S) - min(S)}\right)^p

    Code based on an internal Matlab source and partly ib the `biascorrect_extremes` function of the julia package
    "ClimateTools.jl" :cite:p:`sdba-roy_juliaclimateclimatetoolsjl_2021`.

    Because of limitations imposed by the lazy computing nature of the dask backend, it
    is not possible to know the number of cluster extremes in `ref` and `hist` at the
    moment the output data structure is created. This is why the code tries to estimate
    that number and usually overestimates it. In the training dataset, this translated
    into a `quantile` dimension that is too large and variables `af` and `px_hist` are
    assigned NaNs on extra elements. This has no incidence on the calculations
    themselves but requires more memory than is useful.

    References
    ----------
    :cite:cts:`sdba-roy_juliaclimateclimatetoolsjl_2021`
    :cite:cts:`sdba-roy_extremeprecip_2023`
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
        cluster_thresh = convert_units_to(cluster_thresh, ref, context="infer")

        # Approximation of how many "quantiles" values we will get:
        N = (1 - q_thresh) * ref.time.size

        # ref_params: cast nan to f32 not to interfere with map_blocks dtype parsing
        #   ref and hist are f32, we want to have f32 in the output.
        ds = extremes_train(
            xr.Dataset(
                {
                    "ref": ref,
                    "hist": hist,
                    "ref_params": ref_params or np.float32(np.NaN),
                }
            ),
            q_thresh=q_thresh,
            cluster_thresh=cluster_thresh,
            dist=stats.get_dist("genpareto"),
            quantiles=np.arange(int(N)),
            group="time",
        )

        ds.px_hist.attrs.update(
            long_name="Probability of extremes in hist",
            description="Parametric probabilities of extremes in the common domain of hist and ref.",
        )
        ds.af.attrs.update(
            long_name="Extremes adjustment factor",
            description="Multiplicative adjustment factor of extremes from hist to ref.",
        )
        ds.thresh.attrs.update(
            long_name=f"{q_thresh * 100}th percentile extreme value threshold",
            description=f"Mean of the {q_thresh * 100}th percentile of large values (x > {cluster_thresh}) of ref and hist.",
        )

        return ds.drop_vars(["quantiles"]), {"cluster_thresh": cluster_thresh}

    def _adjust(
        self,
        sim: xr.DataArray,
        scen: xr.DataArray,
        *,
        frac: float = 0.25,
        power: float = 1.0,
        interp: str = "linear",
        extrapolation: str = "constant",
    ):
        # Quantiles coord : cheat and assign 0 - 1, so we can use `extrapolate_qm`.
        ds = self.ds.assign(
            quantiles=(np.arange(self.ds.quantiles.size) + 1)
            / (self.ds.quantiles.size + 1)
        )

        scen = extremes_adjust(
            ds.assign(sim=sim, scen=scen),
            cluster_thresh=self.cluster_thresh,
            dist=stats.get_dist("genpareto"),
            frac=frac,
            power=power,
            interp=interp,
            extrapolation=extrapolation,
            group="time",
        )

        return scen


class LOCI(TrainAdjust):
    r"""Local Intensity Scaling (LOCI) bias-adjustment.

    This bias adjustment method is designed to correct daily precipitation time series by considering wet and dry days
    separately :cite:p:`sdba-schmidli_downscaling_2006`.

    Multiplicative adjustment factors are computed such that the mean of `hist` matches the mean of `ref` for values
    above a threshold.

    The threshold on the training target `ref` is first mapped to `hist` by finding the quantile in `hist` having the same
    exceedance probability as thresh in `ref`. The adjustment factor is then given by

    .. math::

       s = \frac{\left \langle ref: ref \geq t_{ref} \right\rangle - t_{ref}}{\left \langle hist : hist \geq t_{hist} \right\rangle - t_{hist}}

    In the case of precipitations, the adjustment factor is the ratio of wet-days intensity.

    For an adjustment factor `s`, the bias-adjustment of `sim` is:

    .. math::

      sim(t) = \max\left(t_{ref} + s \cdot (hist(t) - t_{hist}), 0\right)

    Attributes
    ----------
    Train step:

    group : Union[str, Grouper]
        The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        Default is "time", meaning a single adjustment group along dimension "time".
    thresh : str
        The threshold in `ref` above which the values are scaled.

    Adjust step:

    interp : {'nearest', 'linear', 'cubic'}
        The interpolation method to use then interpolating the adjustment factors. Defaults to "linear".

    References
    ----------
    :cite:cts:`sdba-schmidli_downscaling_2006`
    """

    _allow_diff_calendars = False

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        thresh: str,
        group: str | Grouper = "time",
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
        group: str | Grouper = "time",
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

    This bias-correction method maps model simulation values to the observation space through principal components
    :cite:p:`sdba-hnilica_multisite_2017`. Values in the simulation space (multiple variables, or multiple sites) can be
    thought of as coordinate along axes, such as variable, temperature, etc. Principal components (PC) are a
    linear combinations of the original variables where the coefficients are the eigenvectors of the covariance matrix.
    Values can then be expressed as coordinates along the PC axes. The method makes the assumption that bias-corrected
    values have the same coordinates along the PC axes of the observations. By converting from the observation PC space
    to the original space, we get bias corrected values. See `Notes` for a mathematical explanation.

    Warnings
    --------
    Be aware that *principal components* is meant here as the algebraic operation defining a coordinate system
    based on the eigenvectors, not statistical principal component analysis.

    Attributes
    ----------
    group : Union[str, Grouper]
        The main dimension and grouping information. See Notes.
        See :py:class:`xclim.sdba.base.Grouper` for details.
        The adjustment will be performed on each group independently.
        Default is "time", meaning a single adjustment group along dimension "time".
    best_orientation : {'simple', 'full'}
        Which method to use when searching for the best principal component orientation.
        See :py:func:`~xclim.sdba.utils.best_pc_orientation_simple` and
        :py:func:`~xclim.sdba.utils.best_pc_orientation_full`.
        "full" is more precise, but it is much slower.
    crd_dim : str
        The data dimension along which the multiple simulation space dimensions are taken.
        For a multivariate adjustment, this usually is "multivar", as returned by `sdba.stack_variables`.
        For a multisite adjustment, this should be the spatial dimension.
        The training algorithm currently doesn't support any chunking
        along either `crd_dim`. `group.dim` and `group.add_dims`.

    Notes
    -----
    The input data is understood as a set of N points in a :math:`M`-dimensional space.

    - :math:`M` is taken along `crd_dim`.

    - :math:`N` is taken along the dimensions given through `group` : (the main `dim` but also, if requested, the `add_dims` and `window`).

    The principal components (PC) of `hist` and `ref` are used to defined new coordinate systems, centered on their
    respective means. The training step creates a matrix defining the transformation from `hist` to `ref`:

    .. math::

      scen = e_{R} + \mathrm{\mathbf{T}}(sim - e_{H})

    Where:

    .. math::

      \mathrm{\mathbf{T}} = \mathrm{\mathbf{R}}\mathrm{\mathbf{H}}^{-1}

    :math:`\mathrm{\mathbf{R}}` is the matrix transforming from the PC coordinates computed on `ref` to the data
    coordinates. Similarly, :math:`\mathrm{\mathbf{H}}` is transform from the `hist` PC to the data coordinates
    (:math:`\mathrm{\mathbf{H}}` is the inverse transformation). :math:`e_R` and :math:`e_H` are the centroids of the
    `ref` and `hist` distributions respectively. Upon running the  `adjust` step, one may decide to use :math:`e_S`,
    the centroid of the `sim` distribution, instead of :math:`e_H`.

    References
    ----------
    :cite:cts:`sdba-hnilica_multisite_2017,sdba-alavoine_distinct_2022`

    """

    @classmethod
    def _train(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        *,
        crd_dim: str,
        best_orientation: str = "simple",
        group: str | Grouper = "time",
    ):
        all_dims = set(ref.dims + hist.dims)

        # Dimension name for the "points"
        lblP = xr.core.utils.get_temp_dimname(all_dims, "points")

        # Rename coord on ref, multiindex do not like conflicting coordinates names
        lblM = crd_dim
        lblR = xr.core.utils.get_temp_dimname(ref.dims, lblM + "_out")
        ref = ref.rename({lblM: lblR})

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
            # Fancy tricks to choose best orientation on each axes
            # (using eigenvectors, the output axes orientation is undefined)
            if best_orientation == "simple":
                orient = best_pc_orientation_simple(R, Hinv)
            elif best_orientation == "full":
                orient = best_pc_orientation_full(
                    R, Hinv, reference.mean(axis=1), historical.mean(axis=1), historical
                )
            # Get transformation matrix
            return (R * orient) @ Hinv

        # The group wrapper
        def _compute_transform_matrices(ds, dim):
            """Apply `_compute_transform_matrix` along dimensions other than time and the variables to map."""
            # The multiple PC-space dimensions are along lblR and lblM
            # Matrix multiplication in xarray behaves as a dot product across
            # same-name dimensions, instead of reducing according to the dimension order,
            # as in numpy or normal maths.
            if len(dim) > 1:
                reference = ds.ref.stack({lblP: dim})
                historical = ds.hist.stack({lblP: dim})
            else:
                reference = ds.ref.rename({dim[0]: lblP})
                historical = ds.hist.rename({dim[0]: lblP})
            transformation = xr.apply_ufunc(
                _compute_transform_matrix,
                reference,
                historical,
                input_core_dims=[[lblR, lblP], [lblM, lblP]],
                output_core_dims=[[lblR, lblM]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            return transformation

        # Transformation matrix, from model coords to ref coords.
        trans = group.apply(_compute_transform_matrices, {"ref": ref, "hist": hist})
        trans.attrs.update(long_name="Transformation from training to target spaces.")

        ref_mean = group.apply("mean", ref)  # Centroids of ref
        ref_mean.attrs.update(long_name="Centroid point of target.")

        hist_mean = group.apply("mean", hist)  # Centroids of hist
        hist_mean.attrs.update(long_name="Centroid point of training.")

        ds = xr.Dataset(dict(trans=trans, ref_mean=ref_mean, hist_mean=hist_mean))

        ds.attrs["_reference_coord"] = lblR
        ds.attrs["_model_coord"] = lblM
        return ds, {"group": group}

    def _adjust(self, sim):
        lblR = self.ds.attrs["_reference_coord"]
        lblM = self.ds.attrs["_model_coord"]

        vmean = self.group.apply("mean", sim)

        def _compute_adjust(ds, dim):
            """Apply the mapping transformation."""
            scenario = ds.ref_mean + ds.trans.dot((ds.sim - ds.vmean), [lblM])
            return scenario

        scen = (
            self.group.apply(
                _compute_adjust,
                {
                    "ref_mean": self.ds.ref_mean,
                    "trans": self.ds.trans,
                    "sim": sim,
                    "vmean": vmean,
                },
                main_only=True,
            )
            .rename({lblR: lblM})
            .rename("scen")
        )
        return scen


class NpdfTransform(Adjust):
    r"""N-dimensional probability density function transform.

    This adjustment object combines both training and adjust steps in the `adjust` class method.

    A multivariate bias-adjustment algorithm described by :cite:t:`sdba-cannon_multivariate_2018`, as part of the MBCn
    algorithm, based on a color-correction algorithm described by :cite:t:`sdba-pitie_n-dimensional_2005`.

    This algorithm in itself, when used with QuantileDeltaMapping, is NOT trend-preserving.
    The full MBCn algorithm includes a reordering step provided here by :py:func:`xclim.sdba.processing.reordering`.

    See notes for an explanation of the algorithm.

    Parameters
    ----------
    base : BaseAdjustment
        An univariate bias-adjustment class. This is untested for anything else than QuantileDeltaMapping.
    base_kws : dict, optional
        Arguments passed to the training of the univariate adjustment.
    n_escore : int
        The number of elements to send to the escore function. The default, 0, means all elements are included.
        Pass -1 to skip computing the escore completely.
        Small numbers result in less significant scores, but the execution time goes up quickly with large values.
    n_iter : int
        The number of iterations to perform. Defaults to 20.
    pts_dim : str
        The name of the "multivariate" dimension. Defaults to "multivar", which is the
        normal case when using :py:func:`xclim.sdba.base.stack_variables`.
    adj_kws : dict, optional
        Dictionary of arguments to pass to the adjust method of the univariate adjustment.
    rot_matrices : xr.DataArray, optional
        The rotation matrices as a 3D array ('iterations', <pts_dim>, <anything>), with shape (n_iter, <N>, <N>).
        If left empty, random rotation matrices will be automatically generated.

    Notes
    -----
    The historical reference (:math:`T`, for "target"), simulated historical (:math:`H`) and simulated projected (:math:`S`)
    datasets are constructed by stacking the timeseries of N variables together. The algorithm is broken into the
    following steps:

    1. Rotate the datasets in the N-dimensional variable space with :math:`\mathbf{R}`, a random rotation NxN matrix.

    .. math::

        \tilde{\mathbf{T}} = \mathbf{T}\mathbf{R} \
        \tilde{\mathbf{H}} = \mathbf{H}\mathbf{R} \
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

    The original algorithm :cite:p:`sdba-pitie_n-dimensional_2005`, stops the iteration when some distance score converges.
    Following cite:t:`sdba-cannon_multivariate_2018` and the MBCn implementation in :cite:t:`sdba-cannon_mbc_2020`, we
    instead fix the number of iterations.

    As done by cite:t:`sdba-cannon_multivariate_2018`, the distance score chosen is the "Energy distance" from
    :cite:t:`sdba-szekely_testing_2004`. (see: :py:func:`xclim.sdba.processing.escore`).

    The random matrices are generated following a method laid out by :cite:t:`sdba-mezzadri_how_2007`.

    This is only part of the full MBCn algorithm, see :ref:`notebooks/sdba:Statistical Downscaling and Bias-Adjustment`
    for an example on how to replicate the full method with xclim. This includes a standardization of the simulated data
    beforehand, an initial univariate adjustment and the reordering of those adjusted series according to the rank
    structure of the output of this algorithm.

    References
    ----------
    :cite:cts:`sdba-cannon_multivariate_2018,sdba-cannon_mbc_2020,sdba-pitie_n-dimensional_2005,sdba-mezzadri_how_2007,sdba-szekely_testing_2004`
    """

    @classmethod
    def _adjust(
        cls,
        ref: xr.DataArray,
        hist: xr.DataArray,
        sim: xr.DataArray,
        *,
        base: TrainAdjust = QuantileDeltaMapping,
        base_kws: dict[str, Any] | None = None,
        n_escore: int = 0,
        n_iter: int = 20,
        pts_dim: str = "multivar",
        adj_kws: dict[str, Any] | None = None,
        rot_matrices: xr.DataArray | None = None,
    ):
        if base_kws is None:
            base_kws = {}
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


try:
    import SBCK
except ImportError:
    pass
else:

    class _SBCKAdjust(Adjust):
        sbck = None  # The method

        @classmethod
        def _adjust(cls, ref, hist, sim, *, multi_dim=None, **kwargs):
            # Check inputs
            fit_needs_sim = "X1" in signature(cls.sbck.fit).parameters
            for k, v in signature(cls.sbck.__init__).parameters.items():
                if (
                    v.default == v.empty
                    and v.kind != v.VAR_KEYWORD
                    and k != "self"
                    and k not in kwargs
                ):
                    raise ValueError(
                        f"Argument {k} is not optional for SBCK method {cls.sbck.__name__}."
                    )

            ref = ref.rename(time="time_cal")
            hist = hist.rename(time="time_cal")
            sim = sim.rename(time="time_tgt")

            if multi_dim:
                input_core_dims = [
                    ("time_cal", multi_dim),
                    ("time_cal", multi_dim),
                    ("time_tgt", multi_dim),
                ]
            else:
                input_core_dims = [("time_cal",), ("time_cal",), ("time_tgt",)]

            return xr.apply_ufunc(
                cls._apply_sbck,
                ref,
                hist,
                sim,
                input_core_dims=input_core_dims,
                kwargs={"method": cls.sbck, "fit_needs_sim": fit_needs_sim, **kwargs},
                vectorize=True,
                keep_attrs=True,
                dask="parallelized",
                output_core_dims=[input_core_dims[-1]],
                output_dtypes=[sim.dtype],
            ).rename(time_tgt="time")

        @staticmethod
        def _apply_sbck(ref, hist, sim, method, fit_needs_sim, **kwargs):
            obj = method(**kwargs)
            if fit_needs_sim:
                obj.fit(ref, hist, sim)
            else:
                obj.fit(ref, hist)
            scen = obj.predict(sim)
            if sim.ndim == 1:
                return scen[:, 0]
            return scen

    def _parse_sbck_doc(cls):
        def _parse(s):
            s = s.replace("\t", "    ")
            n = min(len(line) - len(line.lstrip()) for line in s.split("\n") if line)
            lines = []
            for line in s.split("\n"):
                line = line[n:] if line else line
                if set(line).issubset({"=", " "}):
                    line = line.replace("=", "-")
                elif set(line).issubset({"-", " "}):
                    line = line.replace("-", "~")
                lines.append(line)
            return lines

        return "\n".join(
            [
                f"SBCK_{cls.__name__}",
                "=" * (5 + len(cls.__name__)),
                (
                    f"This Adjustment object was auto-generated from the {cls.__name__} "
                    " object of package SBCK. See :ref:`Experimental wrap of SBCK`."
                ),
                "",
                (
                    "The adjust method accepts ref, hist, sim and all arguments listed "
                    'below in "Parameters". It also accepts a `multi_dim` argument '
                    "specifying the dimension accross which to take the 'features' and "
                    "is valid for multivariate methods only. See :py:func:`xclim.sdba.stack_variables`."
                    "In the description below, `n_features` is the size of the `multi_dim` "
                    "dimension. There is no way of specifying parameters across other "
                    "dimensions for the moment."
                ),
                "",
                *_parse(cls.__doc__),
                *_parse(cls.__init__.__doc__),
                " Copyright(c) 2021 Yoann Robin.",
            ]
        )

    def _generate_SBCK_classes():
        classes = []
        for clsname in dir(SBCK):
            cls = getattr(SBCK, clsname)
            if (
                not clsname.startswith("_")
                and isinstance(cls, type)
                and hasattr(cls, "fit")
                and hasattr(cls, "predict")
            ):
                doc = _parse_sbck_doc(cls)
                classes.append(
                    type(
                        f"SBCK_{clsname}", (_SBCKAdjust,), {"sbck": cls, "__doc__": doc}
                    )
                )
        return classes
