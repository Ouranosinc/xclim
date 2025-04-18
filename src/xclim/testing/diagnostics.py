# pylint: disable=no-member,missing-kwoa
"""
SDBA Diagnostic Testing Module
==============================

This module is meant to compare results with those expected from papers, or create figures illustrating the
behavior of sdba methods and utilities.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import gaussian_kde, scoreatpercentile

from xclim.sdba.adjustment import (
    DetrendedQuantileMapping,
    EmpiricalQuantileMapping,
    QuantileDeltaMapping,
)
from xclim.sdba.processing import adapt_freq
from xclim.testing.sdba_utils import cannon_2015_rvs, series

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    warnings.warn("Matplotlib not found, plot-generating functions will not work.")
    plt = None


__all__ = ["adapt_freq_graph", "cannon_2015_figure_2", "synth_rainfall"]


def synth_rainfall(shape: float, scale: float = 1.0, wet_freq: float = 0.25, size: int = 1) -> np.ndarray:
    r"""
    Return gamma distributed rainfall values for wet days.

    The returned values are zero for dry days.

    Parameters
    ----------
    shape : float
        The shape parameter of the gamma distribution.
    scale : float
        The scale parameter of the gamma distribution.
    wet_freq : float
        The frequency of wet days.
    size : int
        The number of values to generate.

    Returns
    -------
    np.ndarray
        The generated values.

    Notes
    -----
    The probability density for the Gamma distribution is:

    .. math::

       p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

    where :math:`k` is the shape and :math:`\theta` the scale, and :math:`\Gamma` is the Gamma function.
    """
    is_wet = np.random.binomial(1, p=wet_freq, size=size)
    wet_intensity = np.random.gamma(shape, scale, size)
    return np.where(is_wet, wet_intensity, 0)


def cannon_2015_figure_2() -> plt.Figure:
    """
    Create a graphic similar to figure 2 of Cannon et al. 2015.

    The figure shows the distributions of the reference, historical and simulated data, as well as the future

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    # noqa: D103
    if plt is None:
        raise ModuleNotFoundError("Matplotlib not found.")

    n = 10000
    ref, hist, sim = cannon_2015_rvs(n, random=False)
    QM = EmpiricalQuantileMapping(kind="*", group="time", interp="linear")
    QM.train(ref, hist)
    sim_eqm = QM.predict(sim)

    DQM = DetrendedQuantileMapping(kind="*", group="time", interp="linear")
    DQM.train(ref, hist)
    sim_dqm = DQM.predict(sim, degree=0)

    QDM = QuantileDeltaMapping(kind="*", group="time", interp="linear")
    QDM.train(ref, hist)
    sim_qdm = QDM.predict(sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.linspace(0, 105, 50)
    ax1.plot(x, gaussian_kde(ref)(x), color="r", label="Obs hist")
    ax1.plot(x, gaussian_kde(hist)(x), color="k", label="GCM hist")
    ax1.plot(x, gaussian_kde(sim)(x), color="blue", label="GCM simure")

    ax1.plot(x, gaussian_kde(sim_qdm)(x), color="lime", label="QDM future")
    ax1.plot(x, gaussian_kde(sim_eqm)(x), color="darkgreen", ls="--", label="QM future")
    ax1.plot(x, gaussian_kde(sim_dqm)(x), color="lime", ls=":", label="DQM future")
    ax1.legend(frameon=False)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")

    tau = np.array([0.25, 0.5, 0.75, 0.95, 0.99]) * 100
    bc_gcm = (scoreatpercentile(sim, tau) - scoreatpercentile(hist, tau)) / scoreatpercentile(hist, tau)
    bc_qdm = (scoreatpercentile(sim_qdm, tau) - scoreatpercentile(ref, tau)) / scoreatpercentile(ref, tau)
    bc_eqm = (scoreatpercentile(sim_eqm, tau) - scoreatpercentile(ref, tau)) / scoreatpercentile(ref, tau)
    bc_dqm = (scoreatpercentile(sim_dqm, tau) - scoreatpercentile(ref, tau)) / scoreatpercentile(ref, tau)

    ax2.plot([0, 1], [0, 1], ls=":", color="blue")
    ax2.plot(bc_gcm, bc_gcm, "-", color="blue", label="GCM")
    ax2.plot(bc_gcm, bc_qdm, marker="o", mfc="lime", label="QDM")
    ax2.plot(
        bc_gcm,
        bc_eqm,
        marker="o",
        mfc="darkgreen",
        ls=":",
        color="darkgreen",
        label="QM",
    )
    ax2.plot(
        bc_gcm,
        bc_dqm,
        marker="s",
        mec="lime",
        mfc="w",
        ls="--",
        color="lime",
        label="DQM",
    )

    for i, s in enumerate(tau / 100):
        ax2.text(bc_gcm[i], bc_eqm[i], f"{s}  ", ha="right", va="center", fontsize=9)
    ax2.set_xlabel("GCM relative change")
    ax2.set_ylabel("Bias adjusted relative change")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_aspect("equal")
    plt.tight_layout()
    return fig


def adapt_freq_graph():
    """
    Create a graphic with the additive adjustment factors estimated after applying the adapt_freq method.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    if plt is None:
        raise ModuleNotFoundError("Matplotlib not found.")

    n = 10000
    x = series(synth_rainfall(2, 2, wet_freq=0.25, size=n), "pr")  # sim
    y = series(synth_rainfall(2, 2, wet_freq=0.5, size=n), "pr")  # ref

    xp = adapt_freq(x, y, thresh=0).sim_ad  # noqa

    fig, (ax1, ax2) = plt.subplots(2, 1)
    sx = x.sortby(x)
    sy = y.sortby(y)
    sxp = xp.sortby(xp)

    # Original and corrected series
    ax1.plot(sx.values, color="blue", lw=1.5, label="x : sim")
    ax1.plot(sxp.values, color="pink", label="xp : sim corrected")
    ax1.plot(sy.values, color="k", label="y : ref")
    ax1.legend()

    # Compute qm factors
    qm_add = QuantileDeltaMapping(kind="+", group="time").train(y, x).ds
    qm_mul = QuantileDeltaMapping(kind="*", group="time").train(y, x).ds

    qm_add_p = QuantileDeltaMapping(kind="+", group="time").train(y, xp).ds
    qm_mul_p = QuantileDeltaMapping(kind="*", group="time").train(y, xp).ds

    qm_add.cf.plot(ax=ax2, color="cyan", ls="--", label="+: y-x")
    qm_add_p.cf.plot(ax=ax2, color="cyan", label="+: y-xp")

    qm_mul.cf.plot(ax=ax2, color="brown", ls="--", label="*: y/x")
    qm_mul_p.cf.plot(ax=ax2, color="brown", label="*: y/xp")

    ax2.legend(loc="upper left", frameon=False)
    return fig
