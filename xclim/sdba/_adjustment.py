"""
Adjustment Algorithms
=====================

This file defines the different steps, to be wrapped into the Adjustment objects.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.core.units import convert_units_to, infer_context, str2pint, units
from xclim.indices.stats import _fitfunc_1d  # noqa

from . import nbutils as nbu
from . import utils as u
from ._processing import _adapt_freq
from .base import Grouper, map_blocks, map_groups
from .detrending import PolyDetrend
from .processing import escore


def _adapt_freq_hist(ds: xr.Dataset, adapt_freq_thresh: str):
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
    ds: xr.Dataset,
    *,
    dim: str,
    kind: str,
    quantiles: np.ndarray,
    adapt_freq_thresh: str | None = None,
) -> xr.Dataset:
    """Train step on one group.

    Notes
    -----
    Dataset must contain the following variables:
      ref : training target
      hist : training data

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the training data.
    dim : str
        The dimension along which to compute the quantiles.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.
    quantiles : array-like
        The quantiles to compute.
    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.

    Returns
    -------
    xr.Dataset
        The dataset containing the adjustment factors, the quantiles over the training data, and the scaling factor.
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
def eqm_train(
    ds: xr.Dataset,
    *,
    dim: str,
    kind: str,
    quantiles: np.ndarray,
    adapt_freq_thresh: str | None = None,
) -> xr.Dataset:
    """EQM: Train step on one group.

    Notes
    -----
    Dataset variables:
      ref : training target
      hist : training data

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the training data.
    dim : str
        The dimension along which to compute the quantiles.
    kind : str
        The kind of correction to compute. See :py:func:`xclim.sdba.utils.get_correction`.
    quantiles : array-like
        The quantiles to compute.
    adapt_freq_thresh : str | None
        Threshold for frequency adaptation. See :py:class:`xclim.sdba.processing.adapt_freq` for details.
        Default is None, meaning that frequency adaptation is not performed.

    Returns
    -------
    xr.Dataset
        The dataset containing the adjustment factors and the quantiles over the training data.
    """
    hist = _adapt_freq_hist(ds, adapt_freq_thresh) if adapt_freq_thresh else ds.hist
    ref_q = nbu.quantile(ds.ref, quantiles, dim)
    hist_q = nbu.quantile(hist, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q))


@map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[])
def qm_adjust(
    ds: xr.Dataset, *, group: Grouper, interp: str, extrapolation: str, kind: str
) -> xr.Dataset:
    """QM (DQM and EQM): Adjust step on one block.

    Notes
    -----
    Dataset variables:
      af : Adjustment factors
      hist_q : Quantiles over the training data
      sim : Data to adjust.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the data to adjust.
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
    """DQM adjustment on one block.

    Notes
    -----
    Dataset variables:
      scaling : Scaling factor between ref and hist
      af : Adjustment factors
      hist_q : Quantiles over the training data
      sim : Data to adjust

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the data to adjust.
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
    """QDM: Adjust process on one block.

    Notes
    -----
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
def loci_train(ds: xr.Dataset, *, group, thresh) -> xr.Dataset:
    """LOCI: Train on one block.

    Notes
    -----
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
    """LOCI: Adjust on one block.

    Notes
    -----
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
    """Scaling: Train on one group.

    Notes
    -----
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
    """Scaling: Adjust on one block.

    Notes
    -----
    Dataset variables:
      af : Adjustment factors.
      sim : Data to adjust.
    """
    af = u.broadcast(ds.af, ds.sim, group=group, interp=interp)
    scen: xr.DataArray = u.apply_correction(ds.sim, af, kind).rename("scen")
    out = scen.to_dataset()
    return out


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
def extremes_train(
    ds: xr.Dataset,
    *,
    group: Grouper,
    q_thresh: float,
    cluster_thresh: float,
    dist,
    quantiles: np.ndarray,
) -> xr.Dataset:
    """Train extremes for a given variable series.

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
    """Adjust extremes to reflect many distribution factors.

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
    bin_width: list | None = None,
    bin_origin: list | None = None,
    num_iter_max: int | None = 100_000_000,
    spray_bins: bool = True,
):
    """Optimal Transport Correction of the bias of X with respect to Y.

    Parameters
    ----------
    X : np.ndarray
        Historical data to be corrected.
    Y : np.ndarray
        Bias correction reference, target of optimal transport.
    bin_width : list | None
        Bin widths for all dimensions.
    bin_origin : list | None
        Bin origins for all dimensions.
    num_iter_max : int | None
        Maximum number of iterations used in the earth mover distance algorithm.

    Returns
    -------
    np.ndarray
        Adjusted data

    References
    ----------
    :cite:cts:`sdba-robin_2021`
    """
    # Initialize parameters
    if bin_width is None:
        bin_width = u.bin_width_estimator([Y, X])
    elif isinstance(bin_width, list):
        bin_width = np.array(bin_width)

    if bin_origin is None:
        bin_origin = np.zeros(len(bin_width))
    elif isinstance(bin_origin, list):
        bin_origin = np.array(bin_origin)

    num_iter_max = 100_000_000 if num_iter_max is None else num_iter_max

    # Get the bin positions and frequencies of X and Y, and for all Xs the bin to which they belong
    gridX, muX, binX = u.histogram(X, bin_width, bin_origin)
    gridY, muY, _ = u.histogram(Y, bin_width, bin_origin)

    # Compute the optimal transportation plan
    plan = u.optimal_transport(gridX, gridY, muX, muY, num_iter_max)

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
        # Get the plan row of this bin
        pi = np.where((binX[binX_group[0]] == gridX).all(1))[0][0]
        # Pick as much target bins for this source bin as there are points in the source bin
        choice = rng.choice(range(muY.size), p=plan[pi, :], size=binX_count[i])
        out[binX_group] = (gridY[choice] + 1 / 2) * bin_width + bin_origin

    if spray_bins:
        out += np.random.uniform(low=-bin_width / 2, high=bin_width / 2, size=out.shape)

    return out


@map_groups(scen=[Grouper.DIM])
def otc_adjust(
    ds: xr.Dataset,
    dim: list,
    pts_dim: str,
    bin_width: list | None = None,
    bin_origin: list | None = None,
    num_iter_max: int | None = 100_000_000,
    spray_bins: bool = True,
    adapt_freq_thresh: dict | None = None,
):
    """Optimal Transport Correction of the bias of `hist` with respect to `ref`.

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
    bin_width : list | None
        Bin widths for all dimensions.
    bin_origin : list | None
        Bin origins for all dimensions.
    num_iter_max : int | None
        Maximum number of iterations used in the earth mover distance algorithm.
    spray_bins : bool = True
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    adapt_freq_thresh : dict | None = None
        Threshold for frequency adaptation per variable.

    Returns
    -------
    xr.Dataset
        Adjusted data
    """
    ref = ds.ref
    hist = ds.hist

    if adapt_freq_thresh is not None:
        for var, thresh in adapt_freq_thresh.items():
            if "units" not in ref.attrs["units"] or ref.attrs["units"] == "":
                # Try to force units
                units = str2pint(thresh).units
                ref.attrs.update(units=str(units))
            hist.loc[var] = _adapt_freq_hist(
                xr.Dataset(
                    {"ref": ref.sel({pts_dim: var}), "hist": hist.sel({pts_dim: var})}
                ),
                thresh,
            )

    ref_map = {d: f"ref_{d}" for d in dim}
    ref = ref.rename(ref_map).stack(dim_ref=ref_map.values()).dropna(dim="dim_ref")

    hist = hist.stack(dim_hist=dim).dropna(dim="dim_hist")

    scen = xr.apply_ufunc(
        _otc_adjust,
        hist,
        ref,
        kwargs=dict(
            bin_width=bin_width,
            bin_origin=bin_origin,
            num_iter_max=num_iter_max,
            spray_bins=spray_bins,
        ),
        input_core_dims=[["dim_hist", pts_dim], ["dim_ref", pts_dim]],
        output_core_dims=[["dim_hist", pts_dim]],
        keep_attrs=True,
        vectorize=True,
    )

    # Pad dim differences with NA to please map_blocks
    ref = ref.unstack().rename({v: k for k, v in ref_map.items()})
    scen = scen.unstack().rename("scen")
    for d in dim:
        full_d = xr.concat([ref[d], scen[d]], dim=d).drop_duplicates(d)
        scen = scen.reindex({d: full_d})

    return scen.to_dataset()


def _dotc_adjust(
    X1: np.ndarray,
    Y0: np.ndarray,
    X0: np.ndarray,
    bin_width: list | None = None,
    bin_origin: list | None = None,
    num_iter_max: int | None = 100_000_000,
    cov_factor: str | None = "std",
    spray_bins: bool = True,
    kind: dict | None = None,
):
    """Dynamical Optimal Transport Correction of the bias of X with respect to Y.

    Parameters
    ----------
    X1 : np.ndarray
        Simulation data to adjust.
    Y0 : np.ndarray
        Bias correction reference.
    X0 : np.ndarray
        Historical simulation data.
    bin_width : list | None
        Bin widths for all dimensions.
    bin_origin : list | None
        Bin origins for all dimensions.
    num_iter_max : int | None
        Maximum number of iterations used in the earth mover distance algorithm.
    cov_factor : str | None = "std"
        Rescaling factor.

    Returns
    -------
    np.ndarray
        Adjusted data

    References
    ----------
    :cite:cts:`sdba-robin_2021`
    """
    # Initialize parameters
    if bin_width is None:
        bin_width = u.bin_width_estimator([Y0, X0, X1])
    elif isinstance(bin_width, list):
        bin_width = np.array(bin_width)

    if cov_factor == "cholesky":
        fact0 = u.eps_cholesky(np.cov(Y0, rowvar=False))
        fact1 = u.eps_cholesky(np.cov(X0, rowvar=False))
        cov_factor = np.dot(fact0, np.linalg.inv(fact1))
    elif cov_factor == "std":
        fact0 = np.std(Y0, axis=0)
        fact1 = np.std(X0, axis=0)
        cov_factor = np.diag(fact0 / fact1)
    else:
        cov_factor = np.identity(Y0.shape[1])

    # Map ref to hist
    yX0 = _otc_adjust(
        Y0,
        X0,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        spray_bins=False,
    )

    # Map hist to sim
    yX1 = _otc_adjust(
        yX0,
        X1,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        spray_bins=False,
    )

    # Temporal evolution
    motion = np.empty(yX0.shape)
    for j in range(yX0.shape[1]):
        if kind is not None and j in kind.keys() and kind[j] == "*":
            motion[:, j] = yX1[:, j] / yX0[:, j]
        else:
            motion[:, j] = yX1[:, j] - yX0[:, j]

    # Apply a variance dependent rescaling factor
    motion = np.apply_along_axis(lambda x: np.dot(cov_factor, x), 1, motion)

    # Apply the evolution to ref
    Y1 = np.empty(yX0.shape)
    for j in range(yX0.shape[1]):
        if kind is not None and j in kind.keys() and kind[j] == "*":
            Y1[:, j] = Y0[:, j] * motion[:, j]
        else:
            Y1[:, j] = Y0[:, j] + motion[:, j]

    # Map sim to the evolution of ref
    Z1 = _otc_adjust(
        X1,
        Y1,
        bin_width=bin_width,
        bin_origin=bin_origin,
        num_iter_max=num_iter_max,
        spray_bins=spray_bins,
    )

    return Z1


@map_groups(scen=[Grouper.DIM])
def dotc_adjust(
    ds: xr.Dataset,
    dim: list,
    pts_dim: str,
    bin_width: list | None = None,
    bin_origin: list | None = None,
    num_iter_max: int | None = 100_000_000,
    cov_factor: str | None = "std",
    spray_bins: bool = True,
    kind: dict | None = None,
    adapt_freq_thresh: dict | None = None,
):
    """Dynamical Optimal Transport Correction of the bias of X with respect to Y.

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
    bin_width : list | None
        Bin widths for all dimensions.
    bin_origin : list | None
        Bin origins for all dimensions.
    num_iter_max : int | None
        Maximum number of iterations used in the earth mover distance algorithm.
    cov_factor : str | None = "std"
        Rescaling factor.
    spray_bins : bool = True
        If `False`, output points are located at the center of their bin.
        If `True`, a random location is picked uniformly inside their bin. Default is `True`.
    kind : dict | None = None
        Keys are variable names and values are adjustment kinds, either additive or multiplicative.
        Unspecified dimensions are treated as "+".
    adapt_freq_thresh : dict | None = None
        Threshold for frequency adaptation per variable.

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
            if "units" not in ref.attrs["units"] or ref.attrs["units"] == "":
                # Try to force units
                units = str2pint(thresh).units
                ref.attrs.update(units=str(units))
            hist.loc[var] = _adapt_freq_hist(
                xr.Dataset(
                    {"ref": ref.sel({pts_dim: var}), "hist": hist.sel({pts_dim: var})}
                ),
                thresh,
            )

    # Drop data added by map_blocks and prepare for apply_ufunc
    hist_map = {d: f"hist_{d}" for d in dim}
    hist = (
        hist.rename(hist_map).stack(dim_hist=hist_map.values()).dropna(dim="dim_hist")
    )

    ref_map = {d: f"ref_{d}" for d in dim}
    ref = ref.rename(ref_map).stack(dim_ref=ref_map.values()).dropna(dim="dim_ref")

    sim = sim.stack(dim_sim=dim).dropna(dim="dim_sim")

    if kind is not None:
        kind = {
            np.where(ref[pts_dim].values == var)[0][0]: op for var, op in kind.items()
        }

    scen = xr.apply_ufunc(
        _dotc_adjust,
        sim,
        ref,
        hist,
        kwargs=dict(
            bin_width=bin_width,
            bin_origin=bin_origin,
            num_iter_max=num_iter_max,
            cov_factor=cov_factor,
            spray_bins=spray_bins,
            kind=kind,
        ),
        input_core_dims=[
            ["dim_sim", pts_dim],
            ["dim_ref", pts_dim],
            ["dim_hist", pts_dim],
        ],
        output_core_dims=[["dim_sim", pts_dim]],
        keep_attrs=True,
        vectorize=True,
    )

    # Pad dim differences with NA to please map_blocks
    hist = hist.unstack().rename({v: k for k, v in hist_map.items()})
    ref = ref.unstack().rename({v: k for k, v in ref_map.items()})
    scen = scen.unstack().rename("scen")
    for d in dim:
        full_d = xr.concat([hist[d], ref[d], scen[d]], dim=d).drop_duplicates(d)
        scen = scen.reindex({d: full_d})

    return scen.to_dataset()
