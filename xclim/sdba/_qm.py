"""
Quantile Mapping algorithms.

This file defines the different QM steps, to be wrapped into the Adjustment objects.
"""
import xarray as xr

from . import nbutils as nbu
from . import utils as u
from .base import map_blocks, map_groups


@map_groups(
    af=["<PROP>", "quantiles"], hist_q=["<PROP>", "quantiles"], scaling=["<PROP>"]
)
def dqm_train(ds, *, dim, kind, quantiles):
    """DQM: Train step on one group."""
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
    af=["<PROP>", "quantiles"],
    hist_q=["<PROP>", "quantiles"],
)
def eqm_train(ds, *, dim, kind, quantiles):
    """EQM: Train step on one group."""
    ref_q = nbu.quantile(ds.ref, quantiles, dim)
    hist_q = nbu.quantile(ds.hist, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q))


@map_blocks(reduces=["<PROP>", "quantiles"], scen=[])
def qm_adjust(ds, *, group, interp, extrapolation, kind):
    """QM (DQM and EQM): Adjust step on one block."""
    af, hist_q = u.extrapolate_qm(ds.af, ds.hist_q, method=extrapolation)
    af = u.interp_on_quantiles(ds.sim, hist_q, af, group=group, method=interp)

    scen = u.apply_correction(ds.sim, af, kind).rename("scen")
    return scen.to_dataset()


@map_blocks(reduces=["<PROP>"], sim=[])
def dqm_scale_sim(ds, *, group, interp, kind):
    """DQM: Sim preprocessing on one block"""
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


@map_blocks(reduces=["<PROP>", "quantiles"], scen=[])
def qdm_adjust(ds, *, group, interp, extrapolation, kind):
    """QDM: Adjust process on one block."""
    af, _ = u.extrapolate_qm(ds.af, ds.hist_q, method=extrapolation)

    sim_q = group.apply(u.rank, ds.sim, main_only=True, pct=True)
    sel = {dim: sim_q[dim] for dim in set(af.dims).intersection(set(sim_q.dims))}
    sel["quantiles"] = sim_q
    af = u.broadcast(af, ds.sim, group=group, interp=interp, sel=sel)

    scen = u.apply_correction(ds.sim, af, kind)
    return scen.rename("scen").to_dataset()
