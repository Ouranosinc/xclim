"""Detrended Quantile Matching (Cannon et al. 2015), code inspired from Santander's downscaleR"""
import numpy as np
import xarray as xr

from .temp import polyfit
from .temp import polyval
from .utils import ADDITIVE
from .utils import group_apply
from .utils import interp_quantiles
from .utils import jitter_under_thresh
from .utils import MULTIPLICATIVE


def train(
    obs,
    sim,
    group="time.month",
    kind=ADDITIVE,
    window=1,
    mult_thresh=None,
    nquantiles=40,
):
    """The Detrended Quantile Mapping from Cannon et al. (2015). Code based on the implementation in Santander's downscaleR"""
    if kind == MULTIPLICATIVE:
        # Replace every thing under mult_thresh by a non-zero random number under mult_thresh
        obs = jitter_under_thresh(obs, mult_thresh)
        sim = jitter_under_thresh(sim, mult_thresh)

    # Normalize : remove mean (op depends on kind)
    obsn = group_apply(normalize, obs, group, window=window, kind=kind)
    simn = group_apply(normalize, sim, group, window=window, kind=kind)

    qmf = xr.Dataset()
    qmf = qmf.assign(
        obs_mean=group_apply("mean", obs, group, window),
        sim_mean=group_apply("mean", sim, group, window),
    )

    tau = np.append(np.insert(np.arange(1, nquantiles) / (nquantiles + 1), 0, 0), 1)

    # Get quantiles for each group
    x = group_apply("quantile", simn, group, window=window, q=tau)
    y = group_apply("quantile", obsn, group, window=window, q=tau)

    return qmf.assign(
        yq=y,
        xq=x,
        simn_min=group_apply("min", simn, group, window=window),
        simn_max=group_apply("max", simn, group, window=window),
        obsn_min=group_apply("min", obsn, group, window=window),
        obsn_max=group_apply("max", obsn, group, window=window),
    )


def predict(
    qmf,
    fut,
    group="time.month",
    kind=ADDITIVE,
    window=1,
    mult_thresh=None,
    detrend=True,
):

    if kind == MULTIPLICATIVE:
        fut = jitter_under_thresh(fut, mult_thresh)

    def prescale(gr, sim_mean, obs_mean, dim="time"):
        if kind == MULTIPLICATIVE:
            return fut * obs_mean / sim_mean
        return fut - sim_mean + obs_mean

    fut = (
        group_apply(
            prescale,
            fut,
            group,
            window=window,
            grouped_args=(qmf.sim_mean, qmf.obs_mean),
        )
        .sortby("time")
        .drop_vars("month")
    )

    if detrend:
        coeffs = polyfit(fut, deg=1, dim="time")
        fut_mean = polyval(fut.time, coeffs)
    else:

        fut_mean = qmf.obs_mean.sel(month=fut.time.dt.month)

    # Normalize with trend instead of own mean
    if kind == MULTIPLICATIVE:
        futn = fut / fut_mean
    else:
        futn = fut - fut_mean

    # xq and yq form a mapping function, use interpolation to get new values of fut
    yout = group_apply(
        interp_quantiles, futn, group, window=window, grouped_args=(qmf.xq, qmf.yq)
    )

    # Extrapolation for values outside range
    def extrapolate(dsgr, obsn_min, obsn_max, simn_min, simn_max, dim="time"):
        yout = dsgr.yout.where(dsgr.futn > simn_min, dsgr.futn * obsn_min / simn_min)
        return yout.where(dsgr.futn < simn_max, dsgr.futn * obsn_max / simn_max)

    ds = xr.Dataset({"yout": yout, "futn": futn})
    yout = group_apply(
        extrapolate,
        ds,
        group,
        window=window,
        grouped_args=(qmf.obsn_min, qmf.obsn_max, qmf.simn_min, qmf.simn_max),
    )

    if kind == MULTIPLICATIVE:
        return yout * fut_mean
    return yout + fut_mean


def normalize(gr, dim="time", kind=ADDITIVE):
    if kind == MULTIPLICATIVE:
        return gr / gr.mean(dim)
    return gr - gr.mean(dim)


def min_factor(gr, dim="time", kind=ADDITIVE):
    if kind == MULTIPLICATIVE:
        return gr.obsn.min(dim) / gr.simn.min(dim)
    return gr.obsn.min(dim) / gr.simn.min(dim)


def max_factor(gr, dim="time", kind=ADDITIVE):
    if kind == MULTIPLICATIVE:
        return gr.obsn.max(dim) / gr.simn.max(dim)
    return gr.obsn.max(dim) / gr.simn.max(dim)
