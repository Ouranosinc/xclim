# pylint: disable=no-value-for-parameter
"""
Adjustment Algorithms
=====================

This file defines the different steps, to be wrapped into the Adjustment objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr

from xclim import set_options
from xclim.core.units import convert_units_to, infer_context, units
from xclim.indices.stats import _fitfunc_1d  # noqa

from . import nbutils as nbu
from . import utils as u
from ._processing import _adapt_freq
from .base import Grouper, map_blocks, map_groups
from .detrending import PolyDetrend
from .processing import escore, jitter_under_thresh, reordering, standardize


def _adapt_freq_hist(ds: xr.Dataset, adapt_freq_thresh: str):
    """Adapt frequency of null values of `hist`    in order to match `ref`."""
    with units.context(infer_context(ds.ref.attrs.get("standard_name"))):
        thresh = convert_units_to(adapt_freq_thresh, ds.ref)
    dim = ["time"] + ["window"] * ("window" in ds.hist.dims)
    return _adapt_freq.func(
        xr.Dataset({"sim": ds.hist, "ref": ds.ref}), thresh=thresh, dim=dim
    ).sim_ad


@map_groups(
    af=[Grouper.PROP, "quantiles"],
    hist_q=[Grouper.PROP, "quantiles"],
    scaling=[Grouper.PROP],
)
def dqm_train(
    ds: xr.Dataset,
    *,
    dim: str,
    kind: str,
    quantiles: np.ndarray,
    adapt_freq_thresh: str | None = None,
    jitter_under_thresh_value: str | None = None,
) -> xr.Dataset:
    """
    Train step on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    dim : str
        The dimension along which to compute the quantiles.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.
    quantiles : array-like
        The quantiles to compute.
    adapt_freq_thresh : str, optional
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    jitter_under_thresh_value : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        Default is None, meaning that jitter under thresh is not performed.

    Returns
    -------
    xr.Dataset
        The dataset containing the adjustment factors, the quantiles over the training data, and the scaling factor.
    """
    ds["hist"] = (
        jitter_under_thresh(ds.hist, jitter_under_thresh_value)
        if jitter_under_thresh_value
        else ds.hist
    )
    ds["hist"] = (
        _adapt_freq_hist(ds, adapt_freq_thresh) if adapt_freq_thresh else ds.hist
    )

    refn = u.apply_correction(ds.ref, u.invert(ds.ref.mean(dim), kind), kind)
    histn = u.apply_correction(ds.hist, u.invert(ds.hist.mean(dim), kind), kind)

    ref_q = nbu.quantile(refn, quantiles, dim)
    hist_q = nbu.quantile(histn, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)
    mu_ref = ds.ref.mean(dim)
    mu_hist = ds.hist.mean(dim)
    scaling = u.get_correction(mu_hist, mu_ref, kind=kind)

    return xr.Dataset(data_vars={"af": af, "hist_q": hist_q, "scaling": scaling})


@map_groups(
    af=[Grouper.PROP, "quantiles"],
    hist_q=[Grouper.PROP, "quantiles"],
)
def eqm_train(
    ds: xr.Dataset,
    *,
    dim: str,
    kind: str,
    quantiles: np.ndarray,
    adapt_freq_thresh: str | None = None,
    jitter_under_thresh_value: str | None = None,
) -> xr.Dataset:
    """
    EQM: Train step on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    dim : str
        The dimension along which to compute the quantiles.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.
    quantiles : array-like
        The quantiles to compute.
    adapt_freq_thresh : str, optional
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    jitter_under_thresh_value : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        Default is None, meaning that jitter under thresh is not performed.

    Returns
    -------
    xr.Dataset
        The dataset containing the adjustment factors and the quantiles over the training data.
    """
    ds["hist"] = (
        jitter_under_thresh(ds.hist, jitter_under_thresh_value)
        if jitter_under_thresh_value
        else ds.hist
    )
    ds["hist"] = (
        _adapt_freq_hist(ds, adapt_freq_thresh) if adapt_freq_thresh else ds.hist
    )
    ref_q = nbu.quantile(ds.ref, quantiles, dim)
    hist_q = nbu.quantile(ds.hist, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)

    return xr.Dataset(data_vars={"af": af, "hist_q": hist_q})


def _npdft_train(ref, hist, rots, quantiles, method, extrap, n_escore, standardize):
    r"""
    Npdf transform to correct a source `hist` into target `ref`.

    Perform a rotation, bias correct `hist` into `ref` with QuantileDeltaMapping, and rotate back.
    Do this iteratively over all rotations `rots` and conserve adjustment factors `af_q` in each iteration.

    Notes
    -----
    This function expects numpy inputs. The input arrays `ref,hist` are expected to be 2-dimensional arrays with shape:
    `(len(nfeature), len(time))`, where `nfeature` is the dimension which is mixed by the multivariate bias adjustment
    (e.g. a `multivar` dimension), i.e. `pts_dims[0]` in :py:func:`mbcn_train`. `rots` are rotation matrices with shape
    `(len(iterations), len(nfeature), len(nfeature))`.
    """
    if standardize:
        ref = (ref - np.nanmean(ref, axis=-1, keepdims=True)) / (
            np.nanstd(ref, axis=-1, keepdims=True)
        )
        hist = (hist - np.nanmean(hist, axis=-1, keepdims=True)) / (
            np.nanstd(hist, axis=-1, keepdims=True)
        )
    af_q = np.zeros((len(rots), ref.shape[0], len(quantiles)))
    escores = np.zeros(len(rots)) * np.nan
    if n_escore > 0:
        ref_step, hist_step = (
            int(np.ceil(arr.shape[1] / n_escore)) for arr in [ref, hist]
        )
    for ii, _rot in enumerate(rots):
        rot = _rot if ii == 0 else _rot @ rots[ii - 1].T
        ref, hist = rot @ ref, rot @ hist
        # loop over variables
        for iv in range(ref.shape[0]):
            ref_q, hist_q = nbu._quantile(ref[iv], quantiles), nbu._quantile(
                hist[iv], quantiles
            )
            af_q[ii, iv] = ref_q - hist_q
            af = u._interp_on_quantiles_1D(
                u._rank_bn(hist[iv]),
                quantiles,
                af_q[ii, iv],
                method=method,
                extrap=extrap,
            )
            hist[iv] = hist[iv] + af
        if n_escore > 0:
            escores[ii] = nbu._escore(ref[:, ::ref_step], hist[:, ::hist_step])
    hist = rots[-1].T @ hist
    return af_q, escores


def mbcn_train(
    ds: xr.Dataset,
    rot_matrices: xr.DataArray,
    pts_dims: Sequence[str],
    quantiles: np.ndarray,
    gw_idxs: xr.DataArray,
    interp: str,
    extrapolation: str,
    n_escore: int,
) -> xr.Dataset:
    """
    Npdf transform training.

    Adjusting factors obtained for each rotation in the npdf transform and conserved to be applied in
    the adjusting step in :py:func:`mcbn_adjust`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    rot_matrices : xr.DataArray
        The rotation matrices as a 3D array ('iterations', <pts_dims[0]>, <pts_dims[1]>), with shape (n_iter, <N>, <N>).
    pts_dims : sequence of str
        The name of the "multivariate" dimension and its primed counterpart. Defaults to "multivar", which
        is the normal case when using :py:func:`xclim.sdba.base.stack_variables`, and "multivar_prime",
    quantiles : array-like
        The quantiles to compute.
    gw_idxs : xr.DataArray
        Indices of the times in each windowed time group
    interp : str
        The interpolation method to use.
    extrapolation : str
        The extrapolation method to use.
    n_escore : int
        Number of elements to include in the e_score test (0 for all, < 0 to skip)

    Returns
    -------
    xr.Dataset
        The dataset containing the adjustment factors and the quantiles over the training data
        (only the npdf transform of mbcn).
    """
    # unpack data
    ref = ds.ref
    hist = ds.hist
    gr_dim = gw_idxs.attrs["group_dim"]

    # npdf training core
    af_q_l = []
    escores_l = []

    # loop over time blocks
    for ib in range(gw_idxs[gr_dim].size):
        # indices in a given time block
        indices = gw_idxs[{gr_dim: ib}].fillna(-1).astype(int).values
        ind = indices[indices >= 0]

        # npdft training : multiple rotations on standardized datasets
        # keep track of adjustment factors in each rotation for later use
        af_q, escores = xr.apply_ufunc(
            _npdft_train,
            ref[{"time": ind}],
            hist[{"time": ind}],
            rot_matrices,
            quantiles,
            input_core_dims=[
                [pts_dims[0], "time"],
                [pts_dims[0], "time"],
                ["iterations", pts_dims[1], pts_dims[0]],
                ["quantiles"],
            ],
            output_core_dims=[
                ["iterations", pts_dims[1], "quantiles"],
                ["iterations"],
            ],
            dask="parallelized",
            output_dtypes=[hist.dtype, hist.dtype],
            kwargs={
                "method": interp,
                "extrap": extrapolation,
                "n_escore": n_escore,
                "standardize": True,
            },
            vectorize=True,
        )
        af_q_l.append(af_q.expand_dims({gr_dim: [ib]}))
        escores_l.append(escores.expand_dims({gr_dim: [ib]}))
    af_q = xr.concat(af_q_l, dim=gr_dim)
    escores = xr.concat(escores_l, dim=gr_dim)
    out = xr.Dataset({"af_q": af_q, "escores": escores}).assign_coords(
        {"quantiles": quantiles, gr_dim: gw_idxs[gr_dim].values}
    )
    return out


def _npdft_adjust(sim, af_q, rots, quantiles, method, extrap):
    """
    Npdf transform adjusting.

    Adjusting factors `af_q` obtained in the training step are applied on the simulated data `sim` at each iterated
    rotation, see :py:func:`_npdft_train`.

    This function expects numpy inputs. `sim` can be a 2-d array with shape: `(len(nfeature), len(time))`, or
    a 3-d array with shape: `(len(period), len(nfeature), len(time))`, allowing to adjust multiple climatological periods
    all at once. `nfeature` is the dimension which is mixed by the multivariate bias adjustment
    (e.g. a `multivar` dimension), i.e. `pts_dims[0]` in :py:func:`mbcn_train`. `rots` are rotation matrices with shape
    `(len(iterations), len(nfeature), len(nfeature))`.
    """
    # add dummy dim  if period_dim absent to uniformize the function below
    # This could be done at higher level, not sure where is best
    if dummy_dim_added := (len(sim.shape) == 2):
        sim = sim[:, np.newaxis, :]

    # adjust npdft
    for ii, _rot in enumerate(rots):
        rot = _rot if ii == 0 else _rot @ rots[ii - 1].T
        sim = np.einsum("ij,j...->i...", rot, sim)
        # loop over variables
        for iv in range(sim.shape[0]):
            af = u._interp_on_quantiles_1D_multi(
                u._rank_bn(sim[iv], axis=-1),
                quantiles,
                af_q[ii, iv],
                method=method,
                extrap=extrap,
            )
            sim[iv] = sim[iv] + af

    rot = rots[-1].T
    sim = np.einsum("ij,j...->i...", rot, sim)
    if dummy_dim_added:
        sim = sim[:, 0, :]

    return sim


def mbcn_adjust(
    ref: xr.DataArray,
    hist: xr.DataArray,
    sim: xr.DataArray,
    ds: xr.Dataset,
    pts_dims: Sequence[str],
    interp: str,
    extrapolation: str,
    base: Callable,
    base_kws_vars: dict,
    adj_kws: dict,
    period_dim: str | None,
) -> xr.Dataset:
    """
    Perform the adjustment portion MBCn multivariate bias correction technique.

    The function :py:func:`mbcn_train` pre-computes the adjustment factors for each rotation
    in the npdf portion of the MBCn algorithm. The rest of adjustment is performed here
    in `mbcn_adjust``.

    Parameters
    ----------
    ref : xr.DataArray
        training target
    hist : xr.DataArray
        training data
    sim : xr.DataArray
        data to adjust (stacked with multivariate dimension)
    ds : xr.Dataset
        Dataset variables:
            rot_matrices : Rotation matrices used in the training step.
            af_q : Adjustment factors obtained in the training step for the npdf transform
            g_idxs : Indices of the times in each time group
            gw_idxs: Indices of the times in each windowed time group
    pts_dims : [str, str]
        The name of the "multivariate" dimension and its primed counterpart. Defaults to "multivar", which
        is the normal case when using :py:func:`xclim.sdba.base.stack_variables`, and "multivar_prime"
    interp : str
        Interpolation method for the npdf transform (same as in the training step)
    extrapolation : str
        Extrapolation method for the npdf transform (same as in the training step)
    base : BaseAdjustment
        Bias-adjustment class used for the univariate bias correction.
    base_kws_vars : Dict
        Options for univariate training for the scenario that is reordered with the output of npdf transform.
        The arguments are those expected by TrainAdjust classes along with
        - kinds : Dict of correction kinds for each variable (e.g. {"pr":"*", "tasmax":"+"})
    adj_kws : Dict
        Options for univariate adjust for the scenario that is reordered with the output of npdf transform
    period_dim : str, optional
        Name of the period dimension used when stacking time periods of `sim`  using :py:func:`xclim.core.calendar.stack_periods`.
        If specified, the interpolation of the npdf transform is performed only once and applied on all periods simultaneously.
        This should be more performant, but also more memory intensive. Defaults to `None`: No optimization will be attempted.

    Returns
    -------
    xr.Dataset
        The adjusted data.
    """
    # unpacking training parameters
    rot_matrices = ds.rot_matrices
    af_q = ds.af_q
    quantiles = af_q.quantiles
    g_idxs = ds.g_idxs
    gw_idxs = ds.gw_idxs
    gr_dim = gw_idxs.attrs["group_dim"]
    win = gw_idxs.attrs["group"][1]

    # this way of handling was letting open the possibility to perform
    # interpolation for multiple periods in the simulation all at once
    # in principle, avoiding redundancy. Need to test this on small data
    # to confirm it works,  and on big data to check performance.
    dims = ["time"] if period_dim is None else [period_dim, "time"]

    # mbcn core
    scen_mbcn = xr.zeros_like(sim)
    for ib in range(gw_idxs[gr_dim].size):
        # indices in a given time block (with and without the window)
        indices_gw = gw_idxs[{gr_dim: ib}].fillna(-1).astype(int).values
        ind_gw = indices_gw[indices_gw >= 0]
        indices_g = g_idxs[{gr_dim: ib}].fillna(-1).astype(int).values
        ind_g = indices_g[indices_g >= 0]

        # 1. univariate adjustment of sim -> scen
        # the kind may differ depending on the variables
        scen_block = xr.zeros_like(sim[{"time": ind_gw}])
        for iv, v in enumerate(sim[pts_dims[0]].values):
            sl = {"time": ind_gw, pts_dims[0]: iv}
            with set_options(sdba_extra_output=False):
                ADJ = base.train(
                    ref[sl], hist[sl], **base_kws_vars[v], skip_input_checks=True
                )
                scen_block[{pts_dims[0]: iv}] = ADJ.adjust(
                    sim[sl], **adj_kws, skip_input_checks=True
                )

        # 2. npdft adjustment of sim
        npdft_block = xr.apply_ufunc(
            _npdft_adjust,
            standardize(sim[{"time": ind_gw}].copy(), dim="time")[0],
            af_q[{gr_dim: ib}],
            rot_matrices,
            quantiles,
            input_core_dims=[
                [pts_dims[0]] + dims,
                ["iterations", pts_dims[1], "quantiles"],
                ["iterations", pts_dims[1], pts_dims[0]],
                ["quantiles"],
            ],
            output_core_dims=[
                [pts_dims[0]] + dims,
            ],
            dask="parallelized",
            output_dtypes=[sim.dtype],
            kwargs={"method": interp, "extrap": extrapolation},
            vectorize=True,
        )

        # 3. reorder scen according to npdft results
        reordered = reordering(ref=npdft_block, sim=scen_block)
        if win > 1:
            # keep  central value of window (intersecting indices in gw_idxs and g_idxs)
            scen_mbcn[{"time": ind_g}] = reordered[{"time": np.in1d(ind_gw, ind_g)}]
        else:
            scen_mbcn[{"time": ind_g}] = reordered

    return scen_mbcn.to_dataset(name="scen")


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[])
def qm_adjust(
    ds: xr.Dataset, *, group: Grouper, interp: str, extrapolation: str, kind: str
) -> xr.Dataset:
    """
    QM (DQM and EQM): Adjust step on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            af : Adjustment factors
            hist_q : Quantiles over the training data
            sim : Data to adjust.
    group : Grouper
        The grouper object.
    interp : str
        The interpolation method to use.
    extrapolation : str
        The extrapolation method to use.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.

    Returns
    -------
    xr.Dataset
        The adjusted data.
    """
    af = u.interp_on_quantiles(
        ds.sim,
        ds.hist_q,
        ds.af,
        group=group,
        method=interp,
        extrapolation=extrapolation,
    )

    scen: xr.DataArray = u.apply_correction(ds.sim, af, kind).rename("scen")
    out = scen.to_dataset()
    return out


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[], trend=[])
def dqm_adjust(
    ds: xr.Dataset,
    *,
    group: Grouper,
    interp: str,
    kind: str,
    extrapolation: str,
    detrend: int | PolyDetrend,
) -> xr.Dataset:
    """
    DQM adjustment on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            scaling : Scaling factor between ref and hist
            af : Adjustment factors
            hist_q : Quantiles over the training data
            sim : Data to adjust
    group : Grouper
        The grouper object.
    interp : str
        The interpolation method to use.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.
    extrapolation : str
        The extrapolation method to use.
    detrend : int | PolyDetrend
        The degree of the polynomial detrending to apply. If 0, no detrending is applied.

    Returns
    -------
    xr.Dataset
        The adjusted data and the trend.
    """
    scaled_sim = u.apply_correction(
        ds.sim,
        u.broadcast(
            ds.scaling,
            ds.sim,
            group=group,
            interp=interp if group.prop != "dayofyear" else "nearest",
        ),
        kind,
    )

    if isinstance(detrend, int):
        detrending = PolyDetrend(degree=detrend, kind=kind, group=group)
    else:
        detrending = detrend

    detrending = detrending.fit(scaled_sim)
    ds["sim"] = detrending.detrend(scaled_sim)
    scen = qm_adjust.func(
        ds,
        group=group,
        interp=interp,
        extrapolation=extrapolation,
        kind=kind,
    ).scen
    scen = detrending.retrend(scen)

    out = xr.Dataset({"scen": scen, "trend": detrending.ds.trend})
    return out


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[], sim_q=[])
def qdm_adjust(ds: xr.Dataset, *, group, interp, extrapolation, kind) -> xr.Dataset:
    """
    QDM: Adjust process on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            af : Adjustment factors
            hist_q : Quantiles over the training data
            sim : Data to adjust.
    """
    sim_q = group.apply(u.rank, ds.sim, main_only=True, pct=True)
    af = u.interp_on_quantiles(
        sim_q,
        ds.quantiles,
        ds.af,
        group=group,
        method=interp,
        extrapolation=extrapolation,
    )
    scen = u.apply_correction(ds.sim, af, kind)
    return xr.Dataset({"scen": scen, "sim_q": sim_q})


@map_blocks(
    reduces=[Grouper.ADD_DIMS, Grouper.DIM],
    af=[Grouper.PROP],
    hist_thresh=[Grouper.PROP],
)
def loci_train(ds: xr.Dataset, *, group, thresh) -> xr.Dataset:
    """
    LOCI: Train on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    """
    s_thresh = group.apply(
        u.map_cdf, ds.rename(hist="x", ref="y"), y_value=thresh
    ).isel(x=0)
    sth = u.broadcast(s_thresh, ds.hist, group=group)
    ws = xr.where(ds.hist >= sth, ds.hist, np.nan)
    wo = xr.where(ds.ref >= thresh, ds.ref, np.nan)

    ms = group.apply("mean", ws, skipna=True)
    mo = group.apply("mean", wo, skipna=True)

    # Adjustment factor
    af = u.get_correction(ms - s_thresh, mo - thresh, u.MULTIPLICATIVE)
    return xr.Dataset({"af": af, "hist_thresh": s_thresh})


@map_blocks(reduces=[Grouper.PROP], scen=[])
def loci_adjust(ds: xr.Dataset, *, group, thresh, interp) -> xr.Dataset:
    """
    LOCI: Adjust on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            hist_thresh : Hist's equivalent thresh from ref
            sim : Data to adjust
    """
    sth = u.broadcast(ds.hist_thresh, ds.sim, group=group, interp=interp)
    factor = u.broadcast(ds.af, ds.sim, group=group, interp=interp)
    with xr.set_options(keep_attrs=True):
        scen: xr.DataArray = (
            (factor * (ds.sim - sth) + thresh).clip(min=0).rename("scen")
        )
    out = scen.to_dataset()
    return out


@map_groups(af=[Grouper.PROP])
def scaling_train(ds: xr.Dataset, *, dim, kind) -> xr.Dataset:
    """
    Scaling: Train on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    """
    mhist = ds.hist.mean(dim)
    mref = ds.ref.mean(dim)
    af: xr.DataArray = u.get_correction(mhist, mref, kind).rename("af")
    out = af.to_dataset()
    return out


@map_blocks(reduces=[Grouper.PROP], scen=[])
def scaling_adjust(ds: xr.Dataset, *, group, interp, kind) -> xr.Dataset:
    """
    Scaling: Adjust on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            af : Adjustment factors.
            sim : Data to adjust.
    """
    af = u.broadcast(ds.af, ds.sim, group=group, interp=interp)
    scen: xr.DataArray = u.apply_correction(ds.sim, af, kind).rename("scen")
    out = scen.to_dataset()
    return out


def npdf_transform(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    r"""
    N-pdf transform : Iterative univariate adjustment in random rotated spaces.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : Reference multivariate timeseries
            hist : simulated timeseries on the reference period
            sim : Simulated timeseries on the projected period.
            rot_matrices : Random rotation matrices.
    **kwargs : Any
        pts_dim : multivariate dimension name
        base : Adjustment class
        base_kws : Kwargs for initialising the adjustment object
        adj_kws : Kwargs of the `adjust` call
        n_escore : Number of elements to include in the e_score test (0 for all, < 0 to skip)

    Returns
    -------
    xr.Dataset
        Dataset with `scenh`, `scens` and `escores` DataArrays, where `scenh` and `scens` are `hist` and `sim`
        respectively after adjustment according to `ref`. If `n_escore` is negative, `escores` will be filled with nans.
    """
    ref = ds.ref.rename(time_hist="time")
    hist = ds.hist.rename(time_hist="time")
    sim = ds.sim
    dim = kwargs["pts_dim"]

    escores = []
    for i, R in enumerate(ds.rot_matrices.transpose("iterations", ...)):
        # @ operator stands for matrix multiplication (along named dimensions): x@R = R@x
        # @R rotates an array defined over dimension x unto new dimension x'. x@R = x'
        refp = ref @ R
        histp = hist @ R
        simp = sim @ R

        # Perform univariate adjustment in rotated space (x')
        ADJ = kwargs["base"].train(
            refp, histp, **kwargs["base_kws"], skip_input_checks=True
        )
        scenhp = ADJ.adjust(histp, **kwargs["adj_kws"], skip_input_checks=True)
        scensp = ADJ.adjust(simp, **kwargs["adj_kws"], skip_input_checks=True)

        # Rotate back to original dimension x'@R = x
        # Note that x'@R is a back rotation because the matrix multiplication is now done along x' due to xarray
        # operating along named dimensions.
        # In normal linear algebra, this is equivalent to taking @R.T, the back rotation.
        hist = scenhp @ R
        sim = scensp @ R

        # Compute score
        if kwargs["n_escore"] >= 0:
            escores.append(
                escore(
                    ref,
                    hist,
                    dims=(dim, "time"),
                    N=kwargs["n_escore"],
                    scale=True,
                ).expand_dims(iterations=[i])
            )

    if kwargs["n_escore"] >= 0:
        escores = xr.concat(escores, "iterations")
    else:
        # All nan, but with the proper shape.
        escores = (
            ref.isel({dim: 0, "time": 0}) * hist.isel({dim: 0, "time": 0})
        ).expand_dims(iterations=ds.iterations) * np.nan

    return xr.Dataset(
        data_vars={
            "scenh": hist.rename(time="time_hist").transpose(*ds.hist.dims),
            "scen": sim.transpose(*ds.sim.dims),
            "escores": escores,
        }
    )


def _fit_on_cluster(data, thresh, dist, cluster_thresh):
    """Extract clusters on 1D data and fit "dist" on the maximums."""
    _, _, _, maximums = u.get_clusters_1d(data, thresh, cluster_thresh)
    params = list(
        _fitfunc_1d(maximums - thresh, dist=dist, floc=0, nparams=3, method="ML")
    )
    # We forced 0, put back thresh.
    params[-2] = thresh
    return params


def _extremes_train_1d(ref, hist, ref_params, *, q_thresh, cluster_thresh, dist, N):
    """Train for method ExtremeValues, only for 1D input along time."""
    # Fast-track, do nothing for all-nan slices
    if all(np.isnan(ref)) or all(np.isnan(hist)):
        return np.full(N, np.nan), np.full(N, np.nan), np.nan

    # Find quantile q_thresh
    thresh = (
        np.nanquantile(ref[ref >= cluster_thresh], q_thresh)
        + np.nanquantile(hist[hist >= cluster_thresh], q_thresh)
    ) / 2

    # Fit genpareto on cluster maximums on ref (if needed) and hist.
    if np.isnan(ref_params).all():
        ref_params = _fit_on_cluster(ref, thresh, dist, cluster_thresh)

    hist_params = _fit_on_cluster(hist, thresh, dist, cluster_thresh)

    # Find probabilities of extremes according to fitted dist
    Px_ref = dist.cdf(ref[ref >= thresh], *ref_params)
    hist = hist[hist >= thresh]
    Px_hist = dist.cdf(hist, *hist_params)

    # Find common probabilities range.
    Pmax = min(Px_ref.max(), Px_hist.max())
    Pmin = max(Px_ref.min(), Px_hist.min())
    Pcommon = (Px_hist <= Pmax) & (Px_hist >= Pmin)
    Px_hist = Px_hist[Pcommon]

    # Find values of hist extremes if they followed ref's distribution.
    hist_in_ref = dist.ppf(Px_hist, *ref_params)

    # Adjustment factors, unsorted
    af = hist_in_ref / hist[Pcommon]
    # sort them in Px order, and pad to have N values.
    order = np.argsort(Px_hist)
    px_hist = np.pad(Px_hist[order], ((0, N - af.size),), constant_values=np.nan)
    af = np.pad(af[order], ((0, N - af.size),), constant_values=np.nan)

    return px_hist, af, thresh


@map_blocks(
    reduces=["time"], px_hist=["quantiles"], af=["quantiles"], thresh=[Grouper.PROP]
)
def extremes_train(
    ds: xr.Dataset,
    *,
    group: Grouper,
    q_thresh: float,
    cluster_thresh: float,
    dist,
    quantiles: np.ndarray,
) -> xr.Dataset:
    """
    Train extremes for a given variable series.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the reference and historical data.
    group : Grouper
        The grouper object.
    q_thresh : float
        The quantile threshold to use.
    cluster_thresh : float
        The threshold for clustering.
    dist : Any
        The distribution to fit.
    quantiles : array-like
        The quantiles to compute.

    Returns
    -------
    xr.Dataset
        The dataset containing the quantiles, the adjustment factors, and the threshold.
    """
    px_hist, af, thresh = xr.apply_ufunc(
        _extremes_train_1d,
        ds.ref,
        ds.hist,
        ds.ref_params or np.nan,
        input_core_dims=[("time",), ("time",), ()],
        output_core_dims=[("quantiles",), ("quantiles",), ()],
        vectorize=True,
        kwargs={
            "q_thresh": q_thresh,
            "cluster_thresh": cluster_thresh,
            "dist": dist,
            "N": len(quantiles),
        },
    )
    # Outputs of map_blocks must have dimensions.
    if not isinstance(thresh, xr.DataArray):
        thresh = xr.DataArray(thresh)
    thresh = thresh.expand_dims(group=[1])
    return xr.Dataset(
        {"px_hist": px_hist, "af": af, "thresh": thresh},
        coords={"quantiles": quantiles},
    )


def _fit_cluster_and_cdf(data, thresh, dist, cluster_thresh):
    """Fit 1D cluster maximums and immediately compute CDF."""
    fut_params = _fit_on_cluster(data, thresh, dist, cluster_thresh)
    return dist.cdf(data, *fut_params)


@map_blocks(reduces=["quantiles", Grouper.PROP], scen=[])
def extremes_adjust(
    ds: xr.Dataset,
    *,
    group: Grouper,
    frac: float,
    power: float,
    dist,
    interp: str,
    extrapolation: str,
    cluster_thresh: float,
) -> xr.Dataset:
    """
    Adjust extremes to reflect many distribution factors.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the reference and historical data.
    group : Grouper
        The grouper object.
    frac : float
        The fraction of the transition function.
    power : float
        The power of the transition function.
    dist : Any
        The distribution to fit.
    interp : str
        The interpolation method to use.
    extrapolation : str
        The extrapolation method to use.
    cluster_thresh : float
        The threshold for clustering.

    Returns
    -------
    xr.Dataset
        The dataset containing the adjusted data.
    """
    # Find probabilities of extremes of fut according to its own cluster-fitted dist.
    px_fut = xr.apply_ufunc(
        _fit_cluster_and_cdf,
        ds.sim,
        ds.thresh,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        kwargs={"dist": dist, "cluster_thresh": cluster_thresh},
        vectorize=True,
    )

    # Find factors by interpolating from hist probs to fut probs. apply them.
    af = u.interp_on_quantiles(
        px_fut, ds.px_hist, ds.af, method=interp, extrapolation=extrapolation
    )
    scen = u.apply_correction(ds.sim, af, "*")

    # Smooth transition function between simulation and scenario.
    transition = (
        ((ds.sim - ds.thresh) / ((ds.sim.max("time")) - ds.thresh)) / frac
    ) ** power
    transition = transition.clip(0, 1)

    adjusted: xr.DataArray = (transition * scen) + ((1 - transition) * ds.scen)
    out = adjusted.rename("scen").squeeze("group", drop=True).to_dataset()
    return out


def _otc_adjust(
    X: np.ndarray,
    Y: np.ndarray,
    bin_width: dict | float | np.ndarray | None = None,
    bin_origin: dict | float | np.ndarray | None = None,
    num_iter_max: int | None = 100_000_000,
    jitter_inside_bins: bool = True,
    normalization: str | None = "max_distance",
):
    """
    Optimal Transport Correction of the bias of X with respect to Y.

    Parameters
    ----------
    X : np.ndarray
        Historical data to be corrected.
    Y : np.ndarray
        Bias correction reference, target of optimal transport.
    bin_width : dict or float or np.ndarray, optional
        Bin widths for specified dimensions.
    bin_origin : dict or float or np.ndarray, optional
        Bin origins for specified dimensions.
    num_iter_max : int, optional
        Maximum number of iterations used in the earth mover distance algorithm.
    jitter_inside_bins : bool
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    normalization : {None, 'standardize', 'max_distance', 'max_value'}
        Per-variable transformation applied before the distances are calculated
        in the optimal transport.

    Returns
    -------
    np.ndarray
        Adjusted data

    References
    ----------
    :cite:cts:`sdba-robin_2021`
    """
    # nans are removed and put back in place at the end
    X_og = X.copy()
    mask = (~np.isnan(X)).all(axis=1)
    X = X[mask]
    Y = Y[(~np.isnan(Y)).all(axis=1)]

    # Initialize parameters
    if bin_width is None:
        bin_width = u.bin_width_estimator([Y, X])
    elif isinstance(bin_width, dict):
        _bin_width = u.bin_width_estimator([Y, X])
        for k, v in bin_width.items():
            _bin_width[k] = v
        bin_width = _bin_width
    elif isinstance(bin_width, float | int):
        bin_width = np.ones(X.shape[1]) * bin_width

    if bin_origin is None:
        bin_origin = np.zeros(X.shape[1])
    elif isinstance(bin_origin, dict):
        _bin_origin = np.zeros(X.shape[1])
        if bin_origin is not None:
            for v, k in bin_origin.items():
                _bin_origin[v] = k
        bin_origin = _bin_origin
    elif isinstance(bin_origin, float | int):
        bin_origin = np.ones(X.shape[1]) * bin_origin

    num_iter_max = 100_000_000 if num_iter_max is None else num_iter_max

    # Get the bin positions and frequencies of X and Y, and for all Xs the bin to which they belong
    gridX, muX, binX = u.histogram(X, bin_width, bin_origin)
    gridY, muY, _ = u.histogram(Y, bin_width, bin_origin)

    # Compute the optimal transportation plan
    plan = u.optimal_transport(gridX, gridY, muX, muY, num_iter_max, normalization)

    gridX = np.floor((gridX - bin_origin) / bin_width)
    gridY = np.floor((gridY - bin_origin) / bin_width)

    # regroup the indices of all the points belonging to a same bin
    binX_sort = np.lexsort(binX[:, ::-1].T)
    sorted_bins = binX[binX_sort]
    _, binX_start, binX_count = np.unique(
        sorted_bins, return_index=True, return_counts=True, axis=0
    )
    binX_start_sort = np.sort(binX_start)
    binX_groups = np.split(binX_sort, binX_start_sort[1:])

    out = np.empty(X.shape)
    rng = np.random.default_rng()
    # The plan row corresponding to a source bin indicates its probabilities to be transported to every target bin
    for i, binX_group in enumerate(binX_groups):
        # Pick as much target bins for this source bin as there are points in the source bin
        choice = rng.choice(range(muY.size), p=plan[i, :], size=binX_count[i])
        out[binX_group] = (gridY[choice] + 1 / 2) * bin_width + bin_origin

    if jitter_inside_bins:
        out += np.random.uniform(low=-bin_width / 2, high=bin_width / 2, size=out.shape)

    # reintroduce nans
    Z = X_og
    Z[mask] = out
    Z[~mask] = np.nan
    return Z


@map_groups(scen=[Grouper.DIM])
def otc_adjust(
    ds: xr.Dataset,
    dim: list,
    pts_dim: str,
    bin_width: dict | float | None = None,
    bin_origin: dict | float | None = None,
    num_iter_max: int | None = 100_000_000,
    jitter_inside_bins: bool = True,
    adapt_freq_thresh: dict | None = None,
    normalization: str | None = "max_distance",
):
    """
    Optimal Transport Correction of the bias of `hist` with respect to `ref`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    dim : list
        The dimensions defining the distribution on which optimal transport is performed.
    pts_dim : str
        The dimension defining the multivariate components of the distribution.
    bin_width : dict or float, optional
        Bin widths for specified dimensions.
    bin_origin : dict or float, optional
        Bin origins for specified dimensions.
    num_iter_max : int, optional
        Maximum number of iterations used in the earth mover distance algorithm.
    jitter_inside_bins : bool
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    adapt_freq_thresh : dict, optional
        Threshold for frequency adaptation per variable.
    normalization : {None, 'standardize', 'max_distance', 'max_value'}
        Per-variable transformation applied before the distances are calculated
        in the optimal transport.

    Returns
    -------
    xr.Dataset
        Adjusted data
    """
    ref = ds.ref
    hist = ds.hist

    if adapt_freq_thresh is not None:
        for var, thresh in adapt_freq_thresh.items():
            hist.loc[var] = _adapt_freq_hist(
                xr.Dataset(
                    {"ref": ref.sel({pts_dim: var}), "hist": hist.sel({pts_dim: var})}
                ),
                thresh,
            )

    ref_map = {d: f"ref_{d}" for d in dim}
    ref = ref.rename(ref_map).stack(dim_ref=ref_map.values())

    hist = hist.stack(dim_hist=dim)

    if isinstance(bin_width, dict):
        bin_width = {
            np.where(ref[pts_dim].values == var)[0][0]: op
            for var, op in bin_width.items()
        }
    if isinstance(bin_origin, dict):
        bin_origin = {
            np.where(ref[pts_dim].values == var)[0][0]: op
            for var, op in bin_origin.items()
        }

    scen = xr.apply_ufunc(
        _otc_adjust,
        hist,
        ref,
        kwargs={
            "bin_width": bin_width,
            "bin_origin": bin_origin,
            "num_iter_max": num_iter_max,
            "jitter_inside_bins": jitter_inside_bins,
            "normalization": normalization,
        },
        input_core_dims=[["dim_hist", pts_dim], ["dim_ref", pts_dim]],
        output_core_dims=[["dim_hist", pts_dim]],
        keep_attrs=True,
        vectorize=True,
    )

    scen = scen.unstack().rename("scen")

    return scen.to_dataset()


def _dotc_adjust(
    X1: np.ndarray,
    Y0: np.ndarray,
    X0: np.ndarray,
    bin_width: dict | float | None = None,
    bin_origin: dict | float | None = None,
    num_iter_max: int | None = 100_000_000,
    cov_factor: str | None = "std",
    jitter_inside_bins: bool = True,
    kind: dict | None = None,
    normalization: str | None = "max_distance",
):
    """
    Dynamical Optimal Transport Correction of the bias of X with respect to Y.

    Parameters
    ----------
    X1 : np.ndarray
        Simulation data to adjust.
    Y0 : np.ndarray
        Bias correction reference.
    X0 : np.ndarray
        Historical simulation data.
    bin_width : dict or float, optional
        Bin widths for specified dimensions.
    bin_origin : dict or float, optional
        Bin origins for specified dimensions.
    num_iter_max : int, optional
        Maximum number of iterations used in the earth mover distance algorithm.
    cov_factor : str, optional
        Rescaling factor.
    jitter_inside_bins : bool
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    kind : dict, optional
        Keys are variable names and values are adjustment kinds, either additive or multiplicative.
        Unspecified dimensions are treated as "+".
    normalization : {None, 'standardize', 'max_distance', 'max_value'}
        Per-variable transformation applied before the distances are calculated
        in the optimal transport.

    Returns
    -------
    np.ndarray
        Adjusted data

    References
    ----------
    :cite:cts:`sdba-robin_2021`
    """
    # nans are removed and put back in place at the end
    X1_og = X1.copy()
    mask = ~np.isnan(X1).any(axis=1)
    X1 = X1[mask]
    X0 = X0[~np.isnan(X0).any(axis=1)]
    Y0 = Y0[~np.isnan(Y0).any(axis=1)]
    # Initialize parameters
    if isinstance(bin_width, dict):
        _bin_width = u.bin_width_estimator([Y0, X0, X1])
        for v, k in bin_width.items():
            _bin_width[v] = k
        bin_width = _bin_width
    elif isinstance(bin_width, float | int):
        bin_width = np.ones(X0.shape[1]) * bin_width

    if isinstance(bin_origin, dict):
        _bin_origin = np.zeros(X0.shape[1])
        for v, k in bin_origin.items():
            _bin_origin[v] = k
        bin_origin = _bin_origin
    elif isinstance(bin_origin, float | int):
        bin_origin = np.ones(X0.shape[1]) * bin_origin

    # Map ref to hist
    yX0 = _otc_adjust(
        Y0,
        X0,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        jitter_inside_bins=False,
        normalization=normalization,
    )

    # Map hist to sim
    yX1 = _otc_adjust(
        yX0,
        X1,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        jitter_inside_bins=False,
        normalization=normalization,
    )

    # Temporal evolution
    motion = np.empty(yX0.shape)
    for j in range(yX0.shape[1]):
        if kind is not None and j in kind.keys() and kind[j] == "*":
            motion[:, j] = yX1[:, j] / yX0[:, j]
        else:
            motion[:, j] = yX1[:, j] - yX0[:, j]

    # Apply a variance dependent rescaling factor
    if cov_factor == "cholesky":
        fact0 = u.eps_cholesky(np.cov(Y0, rowvar=False))
        fact1 = u.eps_cholesky(np.cov(X0, rowvar=False))
        motion = (fact0 @ np.linalg.inv(fact1) @ motion.T).T
    elif cov_factor == "std":
        fact0 = np.std(Y0, axis=0)
        fact1 = np.std(X0, axis=0)
        motion = motion @ np.diag(fact0 / fact1)

    # Apply the evolution to ref
    Y1 = np.empty(yX0.shape)
    for j in range(yX0.shape[1]):
        if kind is not None and j in kind.keys() and kind[j] == "*":
            Y1[:, j] = Y0[:, j] * motion[:, j]
        else:
            Y1[:, j] = Y0[:, j] + motion[:, j]

    # Map sim to the evolution of ref
    out = _otc_adjust(
        X1,
        Y1,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        jitter_inside_bins=jitter_inside_bins,
        normalization=normalization,
    )
    # reintroduce nans
    Z1 = X1_og
    Z1[mask] = out
    Z1[~mask] = np.nan

    return Z1


@map_groups(scen=[Grouper.DIM])
def dotc_adjust(
    ds: xr.Dataset,
    dim: list,
    pts_dim: str,
    bin_width: dict | float | None = None,
    bin_origin: dict | float | None = None,
    num_iter_max: int | None = 100_000_000,
    cov_factor: str | None = "std",
    jitter_inside_bins: bool = True,
    kind: dict | None = None,
    adapt_freq_thresh: dict | None = None,
    normalization: str | None = "max_distance",
):
    """
    Dynamical Optimal Transport Correction of the bias of X with respect to Y.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
            sim : simulated data
    dim : list
        The dimensions defining the distribution on which optimal transport is performed.
    pts_dim : str
        The dimension defining the multivariate components of the distribution.
    bin_width : dict or float, optional
        Bin widths for specified dimensions.
    bin_origin : dict or float, optional
        Bin origins for specified dimensions.
    num_iter_max : int, optional
        Maximum number of iterations used in the earth mover distance algorithm.
    cov_factor : str, optional
        Rescaling factor.
    jitter_inside_bins : bool
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    kind : dict, optional
        Keys are variable names and values are adjustment kinds, either additive or multiplicative.
        Unspecified dimensions are treated as "+".
    adapt_freq_thresh : dict, optional
        Threshold for frequency adaptation per variable.
    normalization : {None, 'standardize', 'max_distance', 'max_value'}
        Per-variable transformation applied before the distances are calculated
        in the optimal transport.

    Returns
    -------
    xr.Dataset
        Adjusted data
    """
    hist = ds.hist
    sim = ds.sim
    ref = ds.ref

    if adapt_freq_thresh is not None:
        for var, thresh in adapt_freq_thresh.items():
            hist.loc[var] = _adapt_freq_hist(
                xr.Dataset(
                    {"ref": ref.sel({pts_dim: var}), "hist": hist.sel({pts_dim: var})}
                ),
                thresh,
            )

    # Drop data added by map_blocks and prepare for apply_ufunc
    hist_map = {d: f"hist_{d}" for d in dim}
    hist = hist.rename(hist_map).stack(dim_hist=hist_map.values())

    ref_map = {d: f"ref_{d}" for d in dim}
    ref = ref.rename(ref_map).stack(dim_ref=ref_map.values())

    sim = sim.stack(dim_sim=dim)

    if kind is not None:
        kind = {
            np.where(ref[pts_dim].values == var)[0][0]: op for var, op in kind.items()
        }
    if isinstance(bin_width, dict):
        bin_width = {
            np.where(ref[pts_dim].values == var)[0][0]: op
            for var, op in bin_width.items()
        }
    if isinstance(bin_origin, dict):
        bin_origin = {
            np.where(ref[pts_dim].values == var)[0][0]: op
            for var, op in bin_origin.items()
        }

    scen = xr.apply_ufunc(
        _dotc_adjust,
        sim,
        ref,
        hist,
        kwargs={
            "bin_width": bin_width,
            "bin_origin": bin_origin,
            "num_iter_max": num_iter_max,
            "cov_factor": cov_factor,
            "jitter_inside_bins": jitter_inside_bins,
            "kind": kind,
            "normalization": normalization,
        },
        input_core_dims=[
            ["dim_sim", pts_dim],
            ["dim_ref", pts_dim],
            ["dim_hist", pts_dim],
        ],
        output_core_dims=[["dim_sim", pts_dim]],
        keep_attrs=True,
        vectorize=True,
    )

    scen = scen.unstack().rename("scen")

    return scen.to_dataset()
