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
from .processing import standardize

# from .processing import escore, standardize


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


def get_windowed_group(da, group, stack_dim=None):
    r"""Splits an input array into `group`, its complement, and expands the array along a rolling `window` dimension.

    Aims to give a faster alternative to `map_blocks` constructions.

    """
    # define dims (simplify this)
    if stack_dim is None:
        stack_dim = get_temp_dimname(da.dims, "stack_dim")
    win_dim = get_temp_dimname(da.dims, "window_dim")
    group = group if isinstance(group, Grouper) else Grouper(group, 1)
    gr, win = group.name, group.window
    gr_dim = gr.split(".")[-1]
    gr_complement_dim = get_temp_dimname(da.dims, "group_complement_dim")
    time_dims = [gr_dim, gr_complement_dim]
    complement_dims = [win_dim, gr_complement_dim]
    # should grouper allow time & win>1? I think only win=1 makes sense... Grouper should raise error
    if group.name == "time":
        da = da.rename({"time": stack_dim})
    else:
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
        da = da.stack({stack_dim: complement_dims})

    da = da.assign_attrs(
        {
            "group": (gr, win),
            "group_dim": gr_dim,
            "complement_dims": complement_dims,
            "stack_dim": stack_dim,
            "time_dims": time_dims,
            "window_dim": win_dim,
        }
    )
    return da


def ungroup(gr_da, template_time):
    r"""Inverse the operation done with :py:func:`get_windowed_group`. Only works if `window` is 1."""
    group = Grouper(*gr_da.attrs["group"])
    # group = group if isinstance(group, Grouper) else Grouper(group, 1)
    stack_dim, win_dim = gr_da.attrs["stack_dim"], gr_da.attrs["window_dim"]
    if group.name == "time":
        return gr_da.rename({stack_dim: "time"})
    gr_da = gr_da.unstack(stack_dim)
    pos_center = gr_da[win_dim].size // 2
    gr_da = gr_da[{win_dim: slice(pos_center, pos_center + 1)}]
    grouped_time = get_windowed_group(
        template_time[{d: 0 for d in template_time.dims if d != "time"}], group.name
    )
    grouped_time = grouped_time.unstack(stack_dim)
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
    ds,
    quantiles,
    method,
    extrap,
    group,
    n_escore,
) -> xr.Dataset:
    """EQM: Train step on one group.


    I was thinking we should leave the possibility to standardize outside the function. Given it should
    really be done in each group, once a group is formed, it would not make sense to allow to standardize outside.
    The only use for this is: If we let group=None a possibility, then simply specifying the dim along which to perform the computation
    should be sufficient.

    Parameters
    ----------
    Dataset variables:
        ref : training target
        hist : training data
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
    # unload data
    ref = ds.ref
    hist = ds.hist
    rot_matrices = ds.rot_matrices
    af_q_l = []

    # group and standardize
    # e.g. Grouper("time.dayofyear", 31)
    # time -> dayofyear, year, window -> dayofyear, stack_dim
    gr_ref, gr_hist = (get_windowed_group(da, group) for da in [ref, hist])
    dim = gr_ref.attrs["stack_dim"]
    grouping_attrs = gr_hist.attrs
    gr_ref, gr_hist = (standardize(da, dim=dim)[0] for da in [gr_ref, gr_hist])

    # npdf core
    for i_it, R in enumerate(rot_matrices.transpose("iterations", ...)):
        # rotate
        refp = gr_ref @ R
        histp = gr_hist @ R

        # train
        # ref_q, hist_q = (
        #     da.quantile(dim=dim, q=quantiles).rename({"quantile": "quantiles"})
        #     for da in [refp, histp]
        # )
        ref_q, hist_q = (
            xr.apply_ufunc(
                nbu._sortquantile,
                da,
                quantiles,
                input_core_dims=[[dim], ["quantiles"]],
                output_core_dims=[["quantiles"]],
                dask="parallelized",
                output_dtypes=[refp.dtype],
                vectorize=True,
            )
            for da in [refp, histp]
        )
        af_q = u.get_correction(hist_q, ref_q, "+")
        af_q_l.append(af_q.expand_dims({"iterations": [i_it]}))

        # adjust
        rnks = xr.apply_ufunc(
            _rank,
            histp,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
        )
        af = xr.apply_ufunc(
            u._interp_on_quantiles_1D,
            rnks,
            quantiles,
            af_q,
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            dask="parallelized",
            kwargs=dict(method=method, extrap=extrap),
            output_dtypes=[gr_ref.dtype],
            vectorize=True,
        )
        histp = u.apply_correction(histp, af, "+")

        # undo rotation
        gr_hist = histp @ R

    # retrieve adjustment factors and undo time grouping
    af_q = xr.concat(af_q_l, dim="iterations")
    af_q = af_q.assign_coords(quantiles=quantiles)
    af_q.attrs = grouping_attrs
    hist = ungroup(gr_hist.assign_attrs(grouping_attrs), hist.time)
    return xr.Dataset(data_vars=dict(af_q=af_q, scenh_npdft=hist))


def npdf_adjust(
    ds,
    group,
    method,
    extrap,
    period_dim,
) -> xr.Dataset:
    """Npdf adjust

    * for now, I had to keep sim separated from ds, had a weird scalar window_dim remaining in sim else
    * period_dim is used to indicate we want to compute many sim periods at once. If specified, it should be

    the dimension generated with stack_periods. unstacking will be done outside the function.



    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.
    """
    # unload training parameters
    sim = ds.sim
    rots = ds.rot_matrices
    af_q = ds.af_q
    quantiles = af_q.quantiles

    # group and standardize
    gr_sim = get_windowed_group(sim, group)
    dim = gr_sim.attrs["stack_dim"]
    dims = [dim] if period_dim is None else [period_dim, dim]
    grouping_attrs = gr_sim.attrs
    gr_sim = standardize(gr_sim, dim=dim)[0]
    # npdf core (adjust)
    for i_it, R in enumerate(rots.transpose("iterations", ...)):
        # rotate
        simp = gr_sim @ R

        # adjust
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
            kwargs=dict(method=method, extrap=extrap),
            output_dtypes=[gr_sim.dtype],
            vectorize=True,
        )
        simp = u.apply_correction(simp, af, "+")

        # undo rotation
        gr_sim = simp @ R

    # undo grouping
    sim = ungroup(gr_sim.assign_attrs(grouping_attrs), sim.time)
    return xr.Dataset(data_vars=dict(scens_npdft=sim))


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
