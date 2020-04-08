"""
This module is meant to compare results with those expected from papers, or create figures illustrating the
behavior of downscaling methods and utilities.
"""
from importlib import reload

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import scoreatpercentile
from scipy.stats.kde import gaussian_kde

from . import utils as tu
from xclim.downscaling.correction import QuantileDeltaMapping
from xclim.downscaling.examples import dqm
from xclim.downscaling.examples import eqm
from xclim.downscaling.examples import qdm
from xclim.downscaling.processing import adapt_freq


def synth_rainfall(shape, scale=1, wet_freq=0.25, size=1):
    """Return gamma distributed rainfall values for wet days.

    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\\Gamma(k)},

    where :math:`k` is the shape and :math:`\theta` the scale,
    and :math:`\\Gamma` is the Gamma function.


    """
    is_wet = np.random.binomial(1, p=wet_freq, size=size)
    wet_intensity = np.random.gamma(shape, scale, size)
    return np.where(is_wet, wet_intensity, 0)


def cannon_2015_figure_2():
    n = 50000
    obs, hist, fut = tu.cannon_2015_rvs(n, True)
    fut_qdm, qdm_tf = qdm(obs, hist, fut, "*", "time", interp="linear")

    fut_eqm, eqm_tf = eqm(obs, hist, fut, "*", "time", interp="linear")

    fut_dqm, dqm_tf = dqm(obs, hist, fut, "*", "time", interp="linear")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.linspace(0, 105, 50)
    ax1.plot(x, gaussian_kde(obs)(x), color="r", label="Obs hist")
    ax1.plot(x, gaussian_kde(hist)(x), color="k", label="GCM hist")
    ax1.plot(x, gaussian_kde(fut)(x), color="blue", label="GCM future")

    ax1.plot(x, gaussian_kde(fut_qdm)(x), color="lime", label="QDM future")
    ax1.plot(x, gaussian_kde(fut_eqm)(x), color="darkgreen", ls="--", label="QM future")
    ax1.plot(x, gaussian_kde(fut_dqm)(x), color="lime", ls=":", label="DQM future")
    ax1.legend(frameon=False)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")

    tau = np.array([0.25, 0.5, 0.75, 0.95, 0.99]) * 100
    bc_gcm = (
        scoreatpercentile(fut, tau) - scoreatpercentile(hist, tau)
    ) / scoreatpercentile(hist, tau)
    bc_qdm = (
        scoreatpercentile(fut_qdm, tau) - scoreatpercentile(obs, tau)
    ) / scoreatpercentile(obs, tau)
    bc_eqm = (
        scoreatpercentile(fut_eqm, tau) - scoreatpercentile(obs, tau)
    ) / scoreatpercentile(obs, tau)
    bc_dqm = (
        scoreatpercentile(fut_dqm, tau) - scoreatpercentile(obs, tau)
    ) / scoreatpercentile(obs, tau)

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
    ax2.set_ylabel("Bias corrected relative change")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_aspect("equal")
    plt.tight_layout()
    return fig


def adapt_freq_graph():
    """
    Create a graphic with the additive correction factors estimated after applying the adapt_freq method.
    """
    n = 10000
    x = tu.series(synth_rainfall(2, 2, wet_freq=0.25, size=n), "pr")  # sim
    y = tu.series(synth_rainfall(2, 2, wet_freq=0.5, size=n), "pr")  # obs

    xp = adapt_freq(x, y, thresh=0).sim_ad

    fig, (ax1, ax2) = plt.subplots(2, 1)
    sx = x.sortby(x)
    sy = y.sortby(y)
    sxp = xp.sortby(xp)

    # Original and corrected series
    ax1.plot(sx.values, color="blue", lw=1.5, label="x : sim")
    ax1.plot(sxp.values, color="pink", label="xp : sim corrected")
    ax1.plot(sy.values, color="k", label="y : obs")
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
