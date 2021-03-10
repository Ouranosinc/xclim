"""
Quantile Mapping algorithms.

This file defines the different QM steps, to be wrapped into the Adjustment objects.
"""
import numpy as np
import xarray as xr

import xclim.sdba.nbutils as nbu
import xclim.sdba.utils as u

from .base import map_blocks, map_groups


@map_groups(
    af=["<PROP>", "quantiles"], hist_q=["<PROP>", "quantiles"], scaling=["<PROP>"]
)
def dqm_train(ds, *, dim="time", kind="+", quantiles=None):
    """DQM: Train step: Element on one group of a 1D timeseries"""
    refn = u.apply_correction(ds.ref, u.invert(ds.ref.mean(dim), kind), kind)
    histn = u.apply_correction(ds.hist, u.invert(ds.hist.mean(dim), kind), kind)

    ref_q = nbu.quantile(refn, quantiles, dim)
    hist_q = nbu.quantile(histn, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)
    mu_ref = ds.ref.mean(dim)
    mu_hist = ds.hist.mean(dim)
    scaling = u.get_correction(mu_hist, mu_ref, kind=kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q, scaling=scaling))


def dqm_train_main(ref, hist, group, nquantiles=15, kind="+"):
    """DQM: Train step: Entry point."""
    quantiles = np.array(u.equally_spaced_nodes(nquantiles, eps=1e-6), dtype="float32")
    ds = xr.Dataset({"ref": ref, "hist": hist})
    return dqm_train(ds, group=group, quantiles=quantiles, kind=kind)


@map_blocks(refvar="sim", scen=["<DIM>"])
def dqm_adjust(ds, *, group, interp="nearest", extrapolation="constant", kind="+"):
    """DQM: Adjust step: Atomic on a 1D timeseries."""
    af, hist_q = u.extrapolate_qm(ds.af, ds.hist_q, method=extrapolation)
    af = u.interp_on_quantiles(ds.sim, hist_q, af, group=group, method=interp)

    scen = u.apply_correction(ds.sim, af, kind).rename("scen")
    return scen.to_dataset()


@map_blocks(refvar="sim", sim=["<DIM>"])
def dqm_scale_sim(ds, *, group, interp="nearest", kind="+"):
    """DQM: Sim preprocessing: Atomic a 1D timeseries."""
    sim = u.apply_correction(
        ds.sim,
        u.broadcast(
            ds.scaling,
            ds.sim,
            group=group,
            interp=interp if group.prop != "dayofyear" else "nearest",
        ),
        kind,
    )
    return sim.rename("sim").to_dataset()


@map_groups(main_only=True, trend=["<DIM>"])
def polydetrend_get_trend(da, *, dim, deg):
    """Polydetrend, atomic func on 1 group of a 1D timeseries."""
    pfc = da.polyfit(dim=dim, deg=deg)
    trend = xr.polyval(coord=da[dim], coeffs=pfc.polyfit_coefficients)
    return trend.rename("trend").to_dataset()


def dqm_adjust_main(
    ds, sim, group, kind="+", interp="nearest", extrapolation="constant"
):
    """DQM: Adjust step: Main."""

    scaled_sim = dqm_scale_sim(
        xr.Dataset({"scaling": ds.scaling, "sim": sim}),
        group=group,
        kind=kind,
        interp=interp,
    ).sim

    trend = polydetrend_get_trend(scaled_sim, group=group, deg=1).trend
    sim_detrended = u.apply_correction(scaled_sim, u.invert(trend, kind), kind)

    scen = dqm_adjust(
        xr.Dataset({"af": ds.af, "hist_q": ds.hist_q, "sim": sim_detrended}),
        group=group,
        interp=interp,
        extrapolation=extrapolation,
        kind=kind,
    ).scen

    return u.apply_correction(scen, trend, kind)
