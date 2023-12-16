"""
Adjustment Algorithms
=====================

This file defines the different steps, to be wrapped into the Adjustment objects.
"""
from __future__ import annotations

import bottleneck as bn
import numpy as np
import xarray as xr
from xarray.core.utils import get_temp_dimname

from xclim.core.units import convert_units_to, infer_context, units
from xclim.indices.stats import _fitfunc_1d  # noqa

from . import nbutils as nbu
from . import utils as u
from ._processing import _adapt_freq
from .base import Grouper, map_blocks, map_groups
from .detrending import PolyDetrend
from .processing import escore, standardize


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
def dqm_train(ds, *, dim, kind, quantiles, adapt_freq_thresh) -> xr.Dataset:
    """Train step on one group.

    Notes
    -----
    Dataset must contain the following variables:
      ref : training target
      hist : training data

    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    """
    hist = _adapt_freq_hist(ds, adapt_freq_thresh) if adapt_freq_thresh else ds.hist

    refn = u.apply_correction(ds.ref, u.invert(ds.ref.mean(dim), kind), kind)
    histn = u.apply_correction(hist, u.invert(hist.mean(dim), kind), kind)

    ref_q = nbu.quantile(refn, quantiles, dim)
    hist_q = nbu.quantile(histn, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)
    mu_ref = ds.ref.mean(dim)
    mu_hist = hist.mean(dim)
    scaling = u.get_correction(mu_hist, mu_ref, kind=kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q, scaling=scaling))


@map_groups(
    af=[Grouper.PROP, "quantiles"],
    hist_q=[Grouper.PROP, "quantiles"],
)
def eqm_train(ds, *, dim, kind, quantiles, adapt_freq_thresh) -> xr.Dataset:
    """EQM: Train step on one group.

    Dataset variables:
      ref : training target
      hist : training data

    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    """
    hist = _adapt_freq_hist(ds, adapt_freq_thresh) if adapt_freq_thresh else ds.hist
    ref_q = nbu.quantile(ds.ref, quantiles, dim)
    hist_q = nbu.quantile(hist, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q))


# =======================================================================================
# general functions, should in utils or something
# =======================================================================================
def _rank(arr, axis=None):
    rnk = bn.nanrankdata(arr, axis=axis)
    rnk = rnk / np.nanmax(rnk, axis=axis, keepdims=True)
    mx, mn = 1, np.nanmin(rnk, axis=axis, keepdims=True)
    return mx * (rnk - mn) / (mx - mn)


def _get_group_complement(da, group):
    # complement of "dayofyear": "year", etc.
    gr = group.name if isinstance(group, Grouper) else group
    gr = group
    if gr == "time.dayofyear":
        return da.time.dt.year
    if gr == "time.month":
        return da.time.dt.strftime("%Y-%d")


def get_windowed_group(da, group, complement_dim=None):
    r"""Splits an input array into `group`, its complement, and expands the array along a rolling `window` dimension.

    Aims to give a faster alternative to `map_blocks` constructions.

    """
    if complement_dim is None:
        complement_dim = get_temp_dimname(da.dims, "complement_dim")
    win_dim = get_temp_dimname(da.dims, "window_dim")
    print(complement_dim)

    group = group if isinstance(group, Grouper) else Grouper(group, 1)
    # should grouper allow time & win>1? I think only win=1 makes sense... Grouper should raise error
    if group.name == "time":
        return da.rename({"time": complement_dim})
    gr, win = group.name, group.window
    gr_dim = gr.split(".")[-1]

    gr_complement_dim = get_temp_dimname(da.dims, "group_complement_dim")
    time_dims = [gr_dim, gr_complement_dim]
    complement_dims = [win_dim, gr_complement_dim]

    if win == 1:
        da = da.expand_dims({win_dim: [0]})
    else:
        da = da.rolling(time=win, center=True).construct(window_dim=win_dim)
    da = da.groupby(gr).apply(
        lambda da: da.assign_coords(time=_get_group_complement(da, gr)).rename(
            {"time": gr_complement_dim}
        )
    )
    da = da.chunk({gr_dim: -1, gr_complement_dim: -1})
    da = da.stack({complement_dim: complement_dims})
    return da.assign_attrs(
        {
            "group": (gr, win),
            "group_dim": gr_dim,
            "complement_dims": complement_dims,
            "complement_dim": complement_dim,
            "time_dims": time_dims,
            "window_dim": win_dim,
        }
    )


def ungroup(gr_da, template_time, group="time", dim="complement_dim"):
    r"""Inverse the operation done with :py:func:`get_windowed_group`. Only works if `window` is 1."""
    group = group if isinstance(group, Grouper) else Grouper(group, 1)
    # group = gr_da.attrs["group"][0]
    if group.name == "time":
        return gr_da.rename({dim: "time"})
    complement_dim, win_dim = gr_da.attrs["complement_dim"], gr_da.attrs["window_dim"]

    gr_da = gr_da.unstack(complement_dim)
    pos_center = gr_da[win_dim].size // 2
    gr_da = gr_da[{win_dim: slice(pos_center, pos_center + 1)}]
    # return gr_da
    grouped_time = get_windowed_group(
        template_time[{d: 0 for d in template_time.dims if d != "time"}], group
    )
    grouped_time = grouped_time.unstack(complement_dim)
    da = (
        gr_da.stack(time=gr_da.attrs["time_dims"])
        .drop_vars(gr_da.attrs["time_dims"])
        .assign_coords(time=grouped_time.values.ravel())
    )
    return da.where(da.time.notnull(), drop=True)[{win_dim: 0}]


# =======================================================================================
# train/adjust functions for npdf
# =======================================================================================


def npdf_train(
    ds, rots, quantiles, group=None, dim=None, standardize_inplace=False
) -> xr.Dataset:
    """EQM: Train step on one group.

    Dataset variables:
      ref : training target
      hist : training data

    I was thinking we should leave the possibility to standardize outside the function. Given it should
    really be done in each group, once a group is formed, it would not make sense to allow to standardize outside.
    The only use for this is: If we let group=None a possibility, then simply specifying the dim along which to perform the computation
    should be sufficient.
    """
    ref = ds.ref
    hist = ds.hist
    af_q_l = []
    if group is None:
        group = Grouper("time")
    gr_ref, gr_hist = (
        get_windowed_group(da, group, complement_dim=dim) for da in [ref, hist]
    )
    # dim = gr_ref.attrs["complement_dim"]
    dim = "complement_dim"
    if standardize_inplace:
        gr_ref, gr_hist = (standardize(da, dim=dim)[0] for da in [gr_ref, gr_hist])
    grouping_attrs = gr_hist.attrs
    for i_it, R in enumerate(rots.transpose("iterations", ...)):
        refp = gr_ref @ R
        histp = gr_hist @ R
        ref_q, hist_q = (
            da.quantile(dim=dim, q=quantiles).rename({"quantile": "quantiles"})
            for da in [refp, histp]
        )
        rnks = xr.apply_ufunc(
            _rank,
            histp,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
        )
        af_q = u.get_correction(hist_q, ref_q, "+")
        af_q_l.append(af_q.expand_dims({"iterations": [i_it]}))

        af = xr.apply_ufunc(
            u._interp_on_quantiles_1D,
            rnks,
            quantiles,
            af_q,
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            dask="parallelized",
            kwargs=dict(method="nearest", extrap="constant"),
            output_dtypes=[gr_ref.dtype],
            vectorize=True,
        )
        histp = u.apply_correction(histp, af, "+")
        gr_hist = histp @ R

    af_q = xr.concat(af_q_l, dim="iterations")
    af_q.attrs = grouping_attrs
    hist = ungroup(gr_hist.assign_attrs(grouping_attrs), hist.time, group=group)
    return xr.Dataset(data_vars=dict(af_q=af_q, scenh_std=hist, rotation_matrices=rots))


def npdf_adjust(
    sim, ds, quantiles, period_dim=None, standardize_inplace=False
) -> xr.Dataset:
    """Npdf adjust

    * for now, I had to keep sim separated from ds, had a weird scalar window_dim remaining in sim else
    * period_dim is used to indicate we want to compute many sim periods at once. If specified, it should be

    the dimension generated with stack_periods. unstacking will be done outside the function.



    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    """
    rots = ds.rotation_matrices
    af_q = ds.af_q
    dim = af_q.attrs["complement_dim"]
    group = Grouper(*af_q.attrs["group"])
    dims = [dim] if period_dim is None else [period_dim, dim]
    gr_sim = get_windowed_group(sim, group, complement_dim=dim)
    gr_sim = standardize(gr_sim, dim=dim)[0]
    temp_attrs = gr_sim.attrs
    for i_it, R in enumerate(rots.transpose("iterations", ...)):
        simp = gr_sim @ R
        rnks = xr.apply_ufunc(
            _rank,
            simp,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
        )
        af = xr.apply_ufunc(
            u._interp_on_quantiles_1D,
            rnks,
            quantiles,
            af_q.isel(iterations=i_it),
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            dask="parallelized",
            kwargs=dict(method="nearest", extrap="constant"),
            output_dtypes=[gr_sim.dtype],
            vectorize=True,
        )
    gr_sim = get_windowed_group(sim, group)
    gr_sim = standardize(gr_sim, dim=dim)[0]
    temp_attrs = gr_sim.attrs
    for i_it, R in enumerate(rots.transpose("iterations", ...)):
        simp = gr_sim @ R
        rnks = xr.apply_ufunc(
            _rank,
            simp,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
        )
        af = xr.apply_ufunc(
            u._interp_on_quantiles_1D_multi,
            rnks,
            quantiles,
            af_q.isel(iterations=i_it),
            input_core_dims=[dims, ["quantiles"], ["quantiles"]],
            output_core_dims=[dims],
            dask="parallelized",
            kwargs=dict(method="nearest", extrap="constant"),
            output_dtypes=[gr_sim.dtype],
            vectorize=True,
        )
        simp = u.apply_correction(simp, af, "+")
        gr_sim = simp @ R

    sim = ungroup(gr_sim.assign_attrs(temp_attrs), sim.time)
    return xr.Dataset(data_vars=dict(scens_std=sim))


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[])
def qm_adjust(ds, *, group, interp, extrapolation, kind) -> xr.Dataset:
    """QM (DQM and EQM): Adjust step on one block.

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


# =======================================================================================
# old numpy implementations, keep for now for benckmarks
# =======================================================================================


def time_group_indices(times, group):
    gr, win = group.name, group.window
    # get time indices (0,1,2,...) for each block
    timeind = xr.DataArray(np.arange(times.size), coords={"time": times})
    win_dim0, win_dim = (
        get_temp_dimname(timeind.dims, lab) for lab in ["window_dim0", "window_dim"]
    )
    if gr != "time":
        # time indices for each block with window = 1
        g_idxs = timeind.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr)).rename(
                {"time": "year"}
            )
        )
        # time indices for each block with general window
        da = timeind.rolling(time=win, center=True).construct(window_dim=win_dim0)
        gw_idxs = da.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr))
            .stack({win_dim: ["time", win_dim0]})
            .reset_index(dims_or_levels=[win_dim])
        )
        gw_idxs = gw_idxs.transpose(..., "window_dim")
    else:
        gw_idxs = timeind.rename({"time": win_dim}).expand_dims({win_dim0: [-1]})
        g_idxs = gw_idxs.copy()
    return g_idxs, gw_idxs


def _get_af(ref, hist, hista, q, kind, method, extrap):
    ref_q, hist_q = (np.nanquantile(arr, q) for arr in [ref, hist])
    af_q = u.get_correction(hist_q, ref_q, kind)
    # af_r = u._interp_on_quantiles_1D(_rank(np.arange(len(hista))), q, af_q, method, extrap)
    af_r = u._interp_on_quantiles_1D(_rank(hista, axis=-1), q, af_q, method, extrap)
    return af_r, af_q


def single_qdm_2(ref, hist, q, kind, method, extrap):
    scenh = np.zeros_like(hist)
    af_q = np.zeros((ref.shape[0], len(q)))
    # loop on multivar
    for iv in range(ref.shape[0]):
        af_r, af_q[iv, :] = _get_af(
            ref[iv],
            hist[iv],
            hist[iv],
            q,
            kind,
            method,
            extrap,
        )
        scenh[iv] = u.apply_correction(hist[iv], af_r, kind)
    return scenh, af_q


def _npdf_train_np(refs, hists, rots, *, nquantiles, interp, extrapolation):
    q, method, extrap = nquantiles, interp, extrapolation
    af_q = np.zeros((len(rots), refs.shape[0], len(q)))
    for ii in range(len(rots)):
        rot = rots[0] if ii == 0 else rots[ii] @ rots[ii - 1].T
        refs, hists = (rot @ da for da in [refs, hists])
        hists, af_q[ii, ...] = single_qdm_2(refs, hists, q, "+", method, extrap)
        # refs0, hists0 = (rots[ii].T @ da for da in [refs, hists])
        # step = int(np.ceil(refs0.shape[1]/ 1000))
        # pts = np.arange(0,refs0.shape[1], step)
        # af_q[ii, ...] = af_q[ii, ...]*0 + _escore(refs0[:,pts], hists0[:,pts])
    hists = rots[-1].T @ hists
    return hists, af_q


def npdf_train_np(
    ref,
    hist,
    n_iter=20,
    rot_matrices=None,
    pts_dim="multivar",
    base_kws=None,
    adj_kws=None,
    standardize_inplace=False,
):
    r"""N-dimensional probability density function transform.

    Parameters
    ----------
    ref  : xr.DataArray
        Reference multivariate timeseries
    hist : xr.DataArray
        simulated timeseries on the reference period
    n_iter : int
        The number of iterations to perform. Defaults to 20.
    pts_dim : str
        The name of the "multivariate" dimension. Defaults to "multivar", which is the
        normal case when using :py:func:`xclim.sdba.base.stack_variables`.
    rot_matrices : xr.DataArray, optional
        The rotation matrices as a 3D array ('iterations', <pts_dim>, <anything>), with shape (n_iter, <N>, <N>).
        If left empty, random rotation matrices will be automatically generated.
    base_kws : dict, optional
        Arguments passed to the training of the univariate adjustment.
    adj_kws : dict, optional
        Dictionary of arguments to pass to the adjust method of the univariate adjustment.
    standarize_inplace : bool
        If true, perform a standardization of ref,hist,sim. Defaults to false
    """
    # Manage train/adj keywords
    if base_kws is None:
        base_kws = {}
    if adj_kws is None:
        adj_kws = {}
    base_kws.setdefault("group", Grouper("time"))
    group = (
        base_kws["group"]
        if isinstance(base_kws["group"], Grouper)
        else Grouper(base_kws["group"], 1)
    )
    base_kws.pop("group")
    kwargs = {**base_kws, **adj_kws}
    for k, default in [
        ["nquantiles", 20],
        ["interp", "nearest"],
        ["extrapolation", "constant"],
    ]:
        kwargs.setdefault(k, default)
    kwargs.pop("kind")
    if np.isscalar(kwargs["nquantiles"]):
        kwargs["nquantiles"] = u.equally_spaced_nodes(kwargs["nquantiles"])

    refs, hists = (
        [ref, hist]
        if standardize_inplace is False
        else (standardize(arr)[0] for arr in [ref, hist])
    )

    pts_dim_pr = xr.core.utils.get_temp_dimname(
        set(refs.dims).union(hists.dims), pts_dim + "_prime"
    )
    if rot_matrices is None:
        rot_matrices = u.rand_rot_matrix(
            ref[pts_dim], num=n_iter, new_dim=pts_dim_pr
        ).rename(matrices="iterations")

    if group == "time":
        gr_refs, gr_hists = (
            da.rename({"time": "complement_dim"}) for da in [refs, hists]
        )
    else:
        gr_refs, gr_hists = (get_windowed_group(da, group) for da in [refs, hists])
    temp_attrs = gr_hists.attrs

    gr_hists, af_q = xr.apply_ufunc(
        _npdf_train_np,
        gr_refs,
        gr_hists,
        rot_matrices,
        dask="parallelized",
        kwargs=kwargs,
        input_core_dims=[
            [pts_dim] + ["complement_dim"],
            [pts_dim] + ["complement_dim"],
            ["iterations", pts_dim_pr, pts_dim],
        ],
        output_core_dims=[
            [pts_dim] + ["complement_dim"],
            ["iterations", pts_dim, "quantiles"],
        ],
        vectorize=True,
        output_dtypes=[gr_hists.dtype, gr_hists.dtype],
        output_sizes={"quantiles": len(kwargs["nquantiles"])},
    )
    if group == "time":
        hists = gr_hists.rename({"complement_dim": "time"})
    else:
        hists = ungroup(gr_hists.assign_attrs(temp_attrs), hists.time)

    af_q = af_q.assign_coords(quantiles=kwargs["nquantiles"])
    af_q.attrs["group"] = (group.name, group.window)
    for k in ["interp", "extrapolation"]:
        af_q.attrs[k] = kwargs[k]
    return hists, af_q, rot_matrices


# def _fast_npdf_adjust(sims, rots, af_q, g_idxs, *, nquantiles, interp, extrapolation):
#     q, method, extrap = nquantiles, interp, extrapolation
#     for ii in range(len(rots)):
#         rot = rots[0] if ii == 0 else rots[ii] @ rots[ii - 1].T
#         sims = np.einsum("ij,jkl->ikl", rot, sims)
#         for ib in range(g_idxs.shape[0]):
#             g_indxs = np.int64(g_idxs[ib, :][g_idxs[ib, :] >= 0])
#             for iv in range(sims.shape[0]):
#                 af0 = u._interp_on_quantiles_1D_multi(
#                     _rank(sims[iv][..., g_indxs], axis=-1),
#                     q,
#                     af_q[ii, iv, ib, :],
#                     method,
#                     extrap,
#                 )
#                 sims[iv][..., g_indxs] = u.apply_correction(
#                     sims[iv][..., g_indxs], af0, "+"
#                 )
#     sims = np.einsum("ij,jkl->ikl", rots[-1].T, sims)
#     return sims


# def fast_npdf_adjust(
#     sim,
#     af_q,
#     rot_matrices,
#     standardize_inplace=False,
# ):
#     r"""N-dimensional probability density function transform.

#     Parameters
#     ----------
#     sim  : xr.DataArray
#         Reference multivariate timeseries
#     n_iter : int
#         The number of iterations to perform. Defaults to 20.
#     pts_dim : str
#         The name of the "multivariate" dimension. Defaults to "multivar", which is the
#         normal case when using :py:func:`xclim.sdba.base.stack_variables`.
#     rot_matrices : xr.DataArray, optional
#         The rotation matrices as a 3D array ('iterations', <pts_dim>, <anything>), with shape (n_iter, <N>, <N>).
#         If left empty, random rotation matrices will be automatically generated.
#     base_kws : dict, optional
#         Arguments passed to the training of the univariate adjustment.
#     adj_kws : dict, optional
#         Dictionary of arguments to pass to the adjust method of the univariate adjustment.
#     standarize_inplace : bool
#         If true, perform a standardization of ref,hist,sim. Defaults to false
#     """
#     # =======================================================================================
#     # fast_npdf
#     # =======================================================================================
#     dummydim = False
#     if "movingwin" not in sim.dims:
#         sim = sim.expand_dims({"movingwin": [0]})
#         dummydim = True
#     if standardize_inplace:
#         sims = standardize(sim)[0]
#     else:
#         sims = sim
#     g_idxs, _ = time_group_indices(sims.time, af_q.attrs["group"])
#     multivar_dims = list(rot_matrices.transpose("iterations", ...).dims[1:])
#     pts_dim = [d for d in multivar_dims if "prime" not in d][0]
#     multivar_dims.remove(pts_dim)
#     pts_dim_pr = multivar_dims[0]
#     kwargs = {k: af_q.attrs[k] for k in ["interp", "extrapolation"]}
#     kwargs["nquantiles"] = af_q.quantiles.values
#     sims = xr.apply_ufunc(
#         _fast_npdf_adjust,
#         sims,
#         rot_matrices,
#         af_q,
#         g_idxs,
#         dask="parallelized",
#         input_core_dims=[
#             [pts_dim, "movingwin", "time"],
#             ["iterations", pts_dim_pr, pts_dim],
#             ["iterations", pts_dim, g_idxs.dims[0], "quantiles"],
#             g_idxs.dims,
#         ],
#         kwargs=kwargs,
#         output_core_dims=[[pts_dim, "movingwin", "time"]],
#         vectorize=True,
#         output_dtypes=[sims.dtype],
#     )

#     sims = sims if dummydim is False else sims.isel(movingwin=0)
#     return sims


def _npdf_adjust_np(sims, af_q, rots, *, nquantiles, interp, extrapolation):
    q, method, extrap = nquantiles, interp, extrapolation
    print("q")
    print(q)
    for ii in range(len(rots)):
        rot = rots[0] if ii == 0 else rots[ii] @ rots[ii - 1].T
        sims = np.einsum("ij,jkl->ikl", rot, sims)
        rnks = _rank(sims, axis=-1)
        for iv in range(sims.shape[0]):
            af0 = u._interp_on_quantiles_1D_multi(
                rnks[iv],
                q,
                af_q[ii, iv, :],
                method,
                extrap,
            )
            sims[iv] = u.apply_correction(sims[iv], af0, "+")
    sims = np.einsum("ij,jkl->ikl", rots[-1].T, sims)
    return sims


def npdf_adjust_np(
    sim,
    af_q,
    rot_matrices,
    standardize_inplace=False,
):
    r"""N-dimensional probability density function transform.

    Parameters
    ----------
    sim  : xr.DataArray
        Reference multivariate timeseries
    n_iter : int
        The number of iterations to perform. Defaults to 20.
    pts_dim : str
        The name of the "multivariate" dimension. Defaults to "multivar", which is the
        normal case when using :py:func:`xclim.sdba.base.stack_variables`.
    rot_matrices : xr.DataArray, optional
        The rotation matrices as a 3D array ('iterations', <pts_dim>, <anything>), with shape (n_iter, <N>, <N>).
        If left empty, random rotation matrices will be automatically generated.
    base_kws : dict, optional
        Arguments passed to the training of the univariate adjustment.
    adj_kws : dict, optional
        Dictionary of arguments to pass to the adjust method of the univariate adjustment.
    standarize_inplace : bool
        If true, perform a standardization of ref,hist,sim. Defaults to false
    """
    # =======================================================================================
    # fast_npdf
    # =======================================================================================
    # Manage train/adj keywords
    dummydim = False
    if "movingwin" not in sim.dims:
        sim = sim.expand_dims({"movingwin": [0]})
        dummydim = True
    if standardize_inplace:
        sims = standardize(sim)[0]
    else:
        sims = sim
    multivar_dims = list(rot_matrices.transpose("iterations", ...).dims[1:])
    pts_dim = [d for d in multivar_dims if "prime" not in d][0]
    multivar_dims.remove(pts_dim)
    pts_dim_pr = multivar_dims[0]
    kwargs = {k: af_q.attrs[k] for k in ["interp", "extrapolation"]}
    kwargs["nquantiles"] = af_q.quantiles.values
    group = Grouper(*af_q.attrs["group"])
    if group == "time":
        gr_sims = sims.rename({"time": "complement_dim"})
    else:
        gr_sims = get_windowed_group(sims, group)
    temp_attrs = gr_sims.attrs
    gr_sims = xr.apply_ufunc(
        _npdf_adjust_np,
        gr_sims,
        af_q,
        rot_matrices,
        kwargs=kwargs,
        input_core_dims=[
            [pts_dim, "moving_win", "complement_dim"],
            ["iterations", pts_dim, "quantiles"],
            ["iterations", pts_dim_pr, pts_dim],
        ],
        output_core_dims=[
            [pts_dim] + ["moving_win", "complement_dim"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sims.dtype],
    )
    print(gr_sims.dims)
    if group == "time":
        sims = gr_sims.rename({"complement_dim": "time"})
    else:
        sims = ungroup(gr_sims.assign_attrs(temp_attrs), sims.time)

    sims = sims if dummydim is False else sims.isel(movingwin=0)
    return sims
