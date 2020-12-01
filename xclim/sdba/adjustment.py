"""Adjustment objects."""
from typing import Union
from warnings import warn

import dask.array as dsk
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.calendar import get_calendar
from xclim.core.formatting import update_history

from .base import Grouper, Parametrizable, parse_group
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
    get_correction,
    interp_on_quantiles,
    map_cdf,
    pc_matrix,
    rank,
)

__all__ = [
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


class BaseAdjustment(Parametrizable):
    def __init__(self, **kwargs):
        """Base object for adjustment algorithms.

        Subclasses should implement the `_train` and `_adjust` methods.
        """
        self.__trained = False
        super().__init__(**kwargs)

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
        if self.__trained:
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

        self._train(ref, hist)
        self._hist_calendar = get_calendar(hist)
        self.__trained = True

    def adjust(self, sim: DataArray, **kwargs):
        """Return bias-adjusted data. Refer to the class documentation for the algorithm details.

        Parameters
        ----------
        sim : DataArray
          Time series to be bias-adjusted, usually a model output.
        kwargs :
          Algorithm-specific keyword arguments, see class doc.
        """
        if not self.__trained:
            raise ValueError("train() must be called before adjusting.")

        if hasattr(self, "group"):
            # Right now there is no other way of getting the main adjustment dimension
            _raise_on_multiple_chunk(sim, self.group.dim)

            if (
                self.group.prop == "dayofyear"
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
        params = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        scen.attrs["xclim_history"] = update_history(
            f"Bias-adjusted with {str(self)}.adjust(sim, {params})", sim
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

    def _adjust(self, sim, interp="nearest", extrapolation="constant"):
        af, hist_q = extrapolate_qm(self.ds.af, self.ds.hist_q, method=extrapolation)
        af = interp_on_quantiles(sim, hist_q, af, group=self.group, method=interp)

        return apply_correction(sim, af, self.kind)


class DetrendedQuantileMapping(EmpiricalQuantileMapping):
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


class LOCI(BaseAdjustment):
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
        self._make_dataset(hist_thresh=s_thresh, ref_thresh=self.thresh, af=af)

    def _adjust(self, sim, interp="linear"):
        sth = broadcast(self.ds.hist_thresh, sim, group=self.group, interp=interp)
        factor = broadcast(self.ds.af, sim, group=self.group, interp=interp)
        with xr.set_options(keep_attrs=True):
            scen = (factor * (sim - sth) + self.ds.ref_thresh).clip(min=0)
        return scen


class Scaling(BaseAdjustment):
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
        self._make_dataset(af=af)

    def _adjust(self, sim, interp="nearest"):
        factor = broadcast(self.ds.af, sim, group=self.group, interp=interp)
        return apply_correction(sim, factor, self.kind)


class PrincipalComponents(BaseAdjustment):
    @parse_group
    def __init__(self, *, group="time", dims=None):
        r"""Principal Component adjustment.

        Method remapping simulation values to the observation through principal component
        analysis ([hnilica2017]_)

        Parameters
        ----------
        At instantiation:

        dims : Sequence of str, optional
          The dimensions to flatten into the "coordinates" dimensions. Default is `None` in
          which case all dimensions except "time" are used. This information can also
          be given through the `add_dims` property of `group`.
          The training algorithm currently doesn't support any chunking along the
          coordinates and the time dimensions.
        group : Union[str, Grouper]
          The grouping information. Additional dims can also be given through the
          `dims` argument. The window option of `Grouper` can not be used with this
          adjustment method. See :py:class:`xclim.sdba.base.Grouper` for details.
          The adjustment will be performed on each group independently.


        In adjustment:

        norm_to : {'hist', 'sim'}
          Before the transformation, sim values are normalized by subtracting the mean of
          hist if norm_to == 'hist' (default) and its own mean if norm_to == 'sim'.

        Notes
        -----
        The input data is understood as a set of N points in a :math:`M`-dimensional space.
        Where :math:`N` is taken along the 'time' coordinates, but :math:`M` can be the
        concatenation of any number of pre-existing dimensions (the default being all
        except 'time'). Thus, the adjustment is equivalent to a linear transformation
        of these :math:`N` points in a the :math:`M`-dimensional space.

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
        if len(group.add_dims) == 0 and dims is not None:
            group.add_dims = dims
        elif "window" in group.add_dims:
            raise ValueError(
                f"Grouping with a window is not accepted for {self.__class__.__name__}. Received {group}."
            )
        elif dims is not None or len(group.add_dims) == 0:
            raise ValueError(
                f"Coordinate dimensions must be given through either `group` or `dims`, but not both. Received {group} and {dims}"
            )
        super().__init__(group=group)

    def _train(self, ref, hist):
        all_dims = set(ref.dims).union(hist.dims)
        crdR = xr.core.utils.get_temp_dimname(all_dims, "crdR")
        crdM = xr.core.utils.get_temp_dimname(all_dims, "crdM")

        ref_mean = self.group.apply("mean", ref)  # Centroids of ref
        hist_mean = self.group.apply("mean", hist)  # Centroids of hist

        # The real thing, acting on 2D numpy arrays
        def _compute_transform_matrix(ref, hist):
            R = pc_matrix(ref)  # Get transformation matrix from PC coords to ref
            H = pc_matrix(hist)  # Get transformation matrix from PC coords to hist
            # This step needs vectorize with dask, but vectorize doesn't work with dask, argh.
            # Invert to get transformation matrix from hist to PC coords.
            Hinv = np.linalg.inv(H)
            # Fancy trick to choose best orientation on each axes
            # (using eigenvectors, the output axes orientation is undefined)
            orient = best_pc_orientation(R, Hinv)
            # Get transformation matrix
            return (R * orient) @ Hinv

        # The group wrapper
        def _compute_transform_matrices(ds, dims):
            dims.pop("time")
            # The multiple PC-space dimensions are along "coordinate"
            # Matrix multiplication in xarray behaves as a dot product across
            # same-name dimensions, instead of reducing according to the dimension order,
            # as in numpy or normal maths. So crdX all refer to the same dimension,
            # but with names assuring correct matrix multiplication even if they are out of order.
            ref = ds.ref.stack({crdR: dims})
            hist = ds.hist.stack({crdM: dims})
            trans = xr.apply_ufunc(
                _compute_transform_matrix,
                ref,
                hist,
                input_core_dims=[[crdR, "time"], [crdM, "time"]],
                output_core_dims=[[crdR, crdM]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            return trans

        # Transformation matrix, from model coords to ref coords.
        trans = self.group.apply(
            _compute_transform_matrices, {"ref": ref, "hist": hist}
        )

        # Metadata
        ref_mean.attrs.update(long_name="Centroid point of target.")
        hist_mean.attrs.update(long_name="Centroid point of training.")
        trans.attrs.update(long_name="Transformation from training to target spaces.")
        # Datasets do not like conflicting multiindex level names.
        crdR_idx = trans.indexes[crdR]
        trans[crdR] = crdR_idx.rename(
            [f"{name}_out" for name in crdR_idx.names], crdR_idx.names
        )
        ref_mean[crdR] = trans[crdR]
        self._make_dataset(trans=trans, ref_mean=ref_mean, hist_mean=hist_mean)
        self.ds.attrs["_reference_coord"] = crdR
        self.ds.attrs["_model_coord"] = crdM

    def _adjust(self, sim, norm_to="hist"):
        crdR = self.ds.attrs["_reference_coord"]
        crdM = self.ds.attrs["_model_coord"]
        dims = self.ds.indexes[crdM].names

        sim = sim.stack({crdM: dims})

        if norm_to == "hist":
            vmean = self.ds.hist_mean
        elif norm_to == "sim":
            vmean = sim.mean("time")
        scen = self.ds.ref_mean + self.ds.trans.dot((sim - vmean), [crdM])

        scen[crdR] = sim.indexes[crdM]
        return scen.unstack(crdR)
