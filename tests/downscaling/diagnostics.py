"""
This module is meant to compare results with those expected from papers, or create figures illustrating the
behavior of downscaling methods and utilities.
"""
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from .conftest import _series as series
from xclim.downscaling.qdm import train
from xclim.downscaling.utils import adapt_freq


def synth_rainfall(shape, scale=1, wet_freq=0.25, size=1):
    """Return gamma distributed rainfall values for wet days."""
    is_wet = np.random.binomial(1, p=wet_freq, size=size)
    wet_intensity = np.random.gamma(shape, scale, size)
    return np.where(is_wet, wet_intensity, 0)


def adapt_freq_graph():
    """
    Create a graphic with the additive correction factors estimated after applying the adapt_freq method.
    """
    n = 10000
    x = series(synth_rainfall(2, 2, wet_freq=0.25, size=n), "pr")  # sim
    y = series(synth_rainfall(2, 2, wet_freq=0.5, size=n), "pr")  # obs

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
    qm_add = train(x, y, kind="+", group="time")
    qm_mul = train(x, y, kind="*", group="time")

    qm_add_p = train(xp, y, kind="+", group="time")
    qm_mul_p = train(xp, y, kind="*", group="time")

    qm_add.qf.plot(ax=ax2, color="cyan", ls="--", label="+: y-x")
    qm_add_p.qf.plot(ax=ax2, color="cyan", label="+: y-xp")

    qm_mul.qf.plot(ax=ax2, color="brown", ls="--", label="*: y/x")
    qm_mul_p.qf.plot(ax=ax2, color="brown", label="*: y/xp")

    ax2.legend()
    return fig
