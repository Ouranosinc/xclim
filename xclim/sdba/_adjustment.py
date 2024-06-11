# pylint: disable=no-value-for-parameter
"""
Adjustment Algorithms
=====================

This file defines the different steps, to be wrapped into the Adjustment objects.
"""

from __future__ import annotations

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


def _adapt_freq_hist(ds, adapt_freq_thresh):
    """Adapt frequency of null values of `hist`    in order to match `ref`."""
    with units.context(infer_context(ds.ref.attrs.get("standard_name"))):
        thresh = convert_units_to(adapt_freq_thresh, ds.ref)
    dim = ["time"] + ["window"] * ("window" in ds.hist.dims)
    return _adapt_freq.func(
        xr.Dataset(dict(sim=ds.hist, ref=ds.ref)), thresh=thresh, dim=dim
    ).sim_ad


@map_groups(
    af=[Grouper.PROP, "quantiles"],
    hist_q=[Grouper.PROP, "quantiles"],
    scaling=[Grouper.PROP],
)
def dqm_train(
    ds, *, dim, kind, quantiles, adapt_freq_thresh, jitter_under_thresh_value
) -> xr.Dataset:
    """Train step on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training source
    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
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

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q, scaling=scaling))


@map_groups(
    af=[Grouper.PROP, "quantiles"],
    hist_q=[Grouper.PROP, "quantiles"],
)
def eqm_train(
    ds, *, dim, kind, quantiles, adapt_freq_thresh, jitter_under_thresh_value
) -> xr.Dataset:
    """EQM: Train step on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training source
    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
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

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q))


def _npdft_train(ref, hist, rots, quantiles, method, extrap, n_escore):
    r"""Npdf transform to correct a source `hist` into target `ref`.

    Perform a rotation, bias correct `hist` into `ref` with QuantileDeltaMapping, and rotate back.
    Do this iteratively over all rotations `rots` and conserve adjustment factors `af_q` in each iteration.
    """
    af_q = np.zeros((len(rots), ref.shape[0], len(quantiles)))
    escores = np.zeros(len(rots)) * np.NaN
    if n_escore > 0:
        ref_step, hist_step = (
            int(np.ceil(arr.shape[1] / n_escore)) for arr in [ref, hist]
        )
    for ii in range(len(rots)):
        rot = rots[0] if ii == 0 else rots[ii] @ rots[ii - 1].T
        ref, hist = (rot @ da for da in [ref, hist])
        # loop over variables
        for iv in range(ref.shape[0]):
            ref_q, hist_q = (
                nbu._sortquantile(da, quantiles) for da in [ref[iv], hist[iv]]
            )
            af_q[ii, iv] = u.get_correction(hist_q, ref_q, "+")
            af = u._interp_on_quantiles_1D(
                u._rank_np(hist[iv]),
                quantiles,
                af_q[ii, iv],
                method=method,
                extrap=extrap,
            )
            hist[iv] = u.apply_correction(hist[iv], af, "+")
        if n_escore > 0:
            escores[ii] = nbu._escore(ref[:, ::ref_step], hist[:, ::hist_step])
    hist = rots[-1].T @ hist
    return af_q, escores


def mbcn_train(
    ds,
    rot_matrices,
    pts_dims,
    quantiles,
    g_idxs,
    gw_idxs,
    interp,
    extrapolation,
    n_escore,
) -> xr.Dataset:
    """Npdf transform training.

    Adjusting factors obtained for each rotation in the npdf transform and conserved to be applied in
    the adjusting step in ``mcbn_adjust``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target  (stacked variables)
            hist : training source (stacked variables)
    rot_matrices : xr.DataArray
        The rotation matrices as a 3D array ('iterations', <pts_dims[0]>, <pts_dims[1]>), with shape (n_iter, <N>, <N>).
    pts_dims : str
        The name of the "multivariate" dimension and its primed counterpart. Defaults to "multivar", which
        is the normal case when using :py:func:`xclim.sdba.base.stack_variables`, and "multivar_prime",
    g_idxs : xr.DataArray
        Indices of the times in each time group
    gw_idxs : xr.DataArray
        Indices of the times in each windowed time group
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
        indices = gw_idxs[{gr_dim: ib}].astype(int).values
        ind = indices[indices >= 0]

        # npdft training : multiple rotations on standardized datasets
        # keep track of adjustment factors in each rotation for later use
        af_q, escores = xr.apply_ufunc(
            _npdft_train,
            standardize(ref[{"time": ind}])[0].copy(),
            standardize(hist[{"time": ind}])[0].copy(),
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
            kwargs={"method": interp, "extrap": extrapolation, "n_escore": n_escore},
            vectorize=True,
        )
        af_q_l.append(af_q.expand_dims({gr_dim: [ib]}))
        escores_l.append(escores.expand_dims({gr_dim: [ib]}))
    af_q = xr.concat(af_q_l, dim=gr_dim)
    escores = xr.concat(escores_l, dim=gr_dim)
    out = xr.Dataset(dict(af_q=af_q, escores=escores)).assign_coords(
        {"quantiles": quantiles, gr_dim: gw_idxs[gr_dim].values}
    )
    return out


def _npdft_adjust(sim, af_q, rots, quantiles, method, extrap):
    """Npdf transform adjusting.

    Adjusting factors `af_q` obtained in the training step are applied on the simulated data `sim` at each iterated
    rotation
    """
    # add dummy dim  if period_dim absent to uniformize the function below
    # This could be done at higher level, not sure where is best
    if dummy_dim_added := (len(sim.shape) == 2):
        sim = sim[:, np.newaxis, :]

    # adjust npdft
    for ii in range(len(rots)):
        rot = rots[0] if ii == 0 else rots[ii] @ rots[ii - 1].T
        sim = np.einsum("ij,j...->i...", rot, sim)
        # loop over variables
        for iv in range(sim.shape[0]):
            af = u._interp_on_quantiles_1D_multi(
                u._rank_np(sim[iv], axis=-1),
                quantiles,
                af_q[ii, iv],
                method=method,
                extrap=extrap,
            )
            sim[iv] = u.apply_correction(sim[iv], af, "+")

    rot = rots[-1].T
    sim = np.einsum("ij,j...->i...", rot, sim)
    if dummy_dim_added:
        sim = sim[:, 0, :]

    return sim


def mbcn_adjust(
    ref,
    hist,
    sim,
    ds,
    pts_dims,
    interp,
    extrapolation,
    base,
    base_kws_vars,
    adj_kws,
    period_dim,
) -> xr.Dataset:
    """Perform the adjustment portion MBCn multivariate bias correction technique.

    The function ``mbcn_train`` pre-computes the adjustment factors for each rotation
    in the npdf portion of the MBCn algorithm. The rest of adjustment is performed here
    in ``mbcn_adjust``.

    ref : xr.Dataset
        training target
    hist : xr.Dataset
        training source
    sim : xr.DataArray
        data to adjust (stacked with multivariate dimension)
    ds.rot_matrices : xr.DataArray
        Rotation matrices used in the training step.
    ds.af_q : xr.DataArray
        Adjustment factors obtained in the training step for the npdf transform
    ds.g_idxs : xr.DataArray
        Indices of the times in each time group
    ds.gw_idxs : xr.DataArray
        Indices of the times in each windowed time group
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
    period_dim : str | None (defaults to None)
        Name of the period dimension used when stacking time periods of `sim`  using :py:func:`xclim.core.calendar.stack_periods`.
        If specified, the interpolation of the npdf transform is performed only once and applied on all periods simultaneously.
        This should be more performant, but also more memory intensive.
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

    sim_u = sim[pts_dims[0]].attrs["_units"]
    # mbcn core
    scen_mbcn = xr.zeros_like(sim)
    for ib in range(gw_idxs[gr_dim].size):
        # indices in a given time block (with and without the window)
        indices_gw = gw_idxs[{gr_dim: ib}].astype(int).values
        ind_gw = indices_gw[indices_gw >= 0]
        indices_g = g_idxs[{gr_dim: ib}].astype(int).values
        ind_g = indices_g[indices_g >= 0]

        # 1. univariate adjustment of sim -> scen
        # the kind may be differ depending on the variables
        scen_block = xr.zeros_like(sim[{"time": ind_gw}])
        for iv, v in enumerate(sim[pts_dims[0]].values):
            sl = {"time": ind_gw, pts_dims[0]: iv}
            with set_options(sdba_extra_output=False):
                ADJ = base.train(
                    ref[v][{"time": ind_gw}],
                    hist[v][{"time": ind_gw}],
                    **base_kws_vars[v],
                )
                scen_block[{pts_dims[0]: iv}] = ADJ.adjust(
                    sim[sl].assign_attrs({"units": sim_u[iv]}), **adj_kws
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

    return scen_mbcn


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[])
def qm_adjust(ds, *, group, interp, extrapolation, kind) -> xr.Dataset:
    """QM (DQM and EQM): Adjust step on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            af : Adjustment factors
            hist_q : Quantiles over the training data
            sim : Data to adjust.
    """
    af = u.interp_on_quantiles(
        ds.sim,
        ds.hist_q,
        ds.af,
        group=group,
        method=interp,
        extrapolation=extrapolation,
    )

    scen = u.apply_correction(ds.sim, af, kind).rename("scen")
    return scen.to_dataset()


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[], trend=[])
def dqm_adjust(ds, *, group, interp, kind, extrapolation, detrend):
    """DQM adjustment on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            scaling : Scaling factor between ref and hist
            af : Adjustment factors
            hist_q : Quantiles over the training data
            sim : Data to adjust
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
        detrend = PolyDetrend(degree=detrend, kind=kind, group=group)

    detrend = detrend.fit(scaled_sim)
    ds["sim"] = detrend.detrend(scaled_sim)
    scen = qm_adjust.func(
        ds,
        group=group,
        interp=interp,
        extrapolation=extrapolation,
        kind=kind,
    ).scen
    scen = detrend.retrend(scen)

    out = xr.Dataset({"scen": scen, "trend": detrend.ds.trend})
    return out


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[], sim_q=[])
def qdm_adjust(ds, *, group, interp, extrapolation, kind) -> xr.Dataset:
    """QDM: Adjust process on one block.

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
    return xr.Dataset(dict(scen=scen, sim_q=sim_q))


@map_blocks(
    reduces=[Grouper.ADD_DIMS, Grouper.DIM],
    af=[Grouper.PROP],
    hist_thresh=[Grouper.PROP],
)
def loci_train(ds, *, group, thresh) -> xr.Dataset:
    """LOCI: Train on one block.

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
def loci_adjust(ds, *, group, thresh, interp) -> xr.Dataset:
    """LOCI: Adjust on one block.

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
        scen = (factor * (ds.sim - sth) + thresh).clip(min=0)
    return scen.rename("scen").to_dataset()


@map_groups(af=[Grouper.PROP])
def scaling_train(ds, *, dim, kind) -> xr.Dataset:
    """Scaling: Train on one group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : training target
            hist : training data
    """
    mhist = ds.hist.mean(dim)
    mref = ds.ref.mean(dim)
    af = u.get_correction(mhist, mref, kind)
    return af.rename("af").to_dataset()


@map_blocks(reduces=[Grouper.PROP], scen=[])
def scaling_adjust(ds, *, group, interp, kind) -> xr.Dataset:
    """Scaling: Adjust on one block.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            af: Adjustment factors.
            sim : Data to adjust.
    """
    af = u.broadcast(ds.af, ds.sim, group=group, interp=interp)
    scen = u.apply_correction(ds.sim, af, kind)
    return scen.rename("scen").to_dataset()


def npdf_transform(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    r"""N-pdf transform : Iterative univariate adjustment in random rotated spaces.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            ref : Reference multivariate timeseries
            hist : simulated timeseries on the reference period
            sim : Simulated timeseries on the projected period.
            rot_matrices : Random rotation matrices.
    \*\*kwargs
        pts_dim : multivariate dimension name
        base : Adjustment class
        base_kws : Kwargs for initialising the adjustment object
        adj_kws : Kwargs of the `adjust` call
        n_escore : Number of elements to include in the e_score test (0 for all, < 0 to skip)

    Returns
    -------
    xr.Dataset
        Dataset with `scenh`, `scens` and `escores` DataArrays, where `scenh` and `scens` are `hist` and `sim`
        respectively after adjustment according to `ref`. If `n_escore` is negative, `escores` will be filled with NaNs.
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
        # All NaN, but with the proper shape.
        escores = (
            ref.isel({dim: 0, "time": 0}) * hist.isel({dim: 0, "time": 0})
        ).expand_dims(iterations=ds.iterations) * np.NaN

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
    # Find quantile q_thresh
    thresh = (
        np.quantile(ref[ref >= cluster_thresh], q_thresh)
        + np.quantile(hist[hist >= cluster_thresh], q_thresh)
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
    px_hist = np.pad(Px_hist[order], ((0, N - af.size),), constant_values=np.NaN)
    af = np.pad(af[order], ((0, N - af.size),), constant_values=np.NaN)

    return px_hist, af, thresh


@map_blocks(
    reduces=["time"], px_hist=["quantiles"], af=["quantiles"], thresh=[Grouper.PROP]
)
def extremes_train(ds, *, group, q_thresh, cluster_thresh, dist, quantiles):
    """Train extremes for a given variable series."""
    px_hist, af, thresh = xr.apply_ufunc(
        _extremes_train_1d,
        ds.ref,
        ds.hist,
        ds.ref_params or np.NaN,
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
    ds, *, group, frac, power, dist, interp, extrapolation, cluster_thresh
):
    """Adjust extremes to reflect many distribution factors."""
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

    out = (transition * scen) + ((1 - transition) * ds.scen)
    return out.rename("scen").squeeze("group", drop=True).to_dataset()
