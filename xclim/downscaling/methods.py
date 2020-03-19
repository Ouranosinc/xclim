"""Methods. Preliminary step for xclim.downscaling: simple methods implemented in xarray."""
import numpy as np

from .temp import polyfit
from .temp import polyval
from .utils import ADDITIVE
from .utils import interp_quantiles
from .utils import MULTIPLICATIVE


# TODO: Split into train / predict


def delta_quantile_mapping(
    obs, sim, fut, kind=ADDITIVE, mult_thresh=None, nquantiles=40, detrend=True
):
    """The DQM from Cannon et al. (2015). Code based on the implementation in Santander's downscaleR"""

    # TODO: use utils.jitter_under_thresh
    if kind == MULTIPLICATIVE:
        # Replace every thing under mult_thresh by a non-zero random number under mult_thresh
        epsilon = np.nextafter(obs.dtype.type(0.0), 1)
        obs = obs.where(
            obs < mult_thresh & obs.notnull(),
            np.random.rand(*obs.shape) * (mult_thresh - epsilon) + epsilon,
        )
        sim = sim.where(
            sim < mult_thresh & sim.notnull(),
            np.random.rand(*obs.shape) * (mult_thresh - epsilon) + epsilon,
        )
        fut = fut.where(
            fut < mult_thresh & fut.notnull(),
            np.random.rand(*obs.shape) * (mult_thresh - epsilon) + epsilon,
        )

    obs_mean = obs.mean("time")
    sim_mean = sim.mean("time")

    if kind == MULTIPLICATIVE:
        fut = fut / sim_mean * obs_mean
    else:
        fut = fut - sim_mean + obs_mean

    if detrend:
        coeffs = polyfit(fut, deg=1, dim="time")
        fut_mean = polyval(coeffs, fut.time)
    else:
        fut_mean = obs_mean

    tau = np.append(np.insert(np.arange(1, nquantiles) / (nquantiles + 1), 0, 0), 1)

    # Je pense qu'on peut encapsuler cette logique dans un array qui contient les facteurs de correction et qu'on
    # utilise pour réindexer.
    if kind == MULTIPLICATIVE:
        x = (sim / sim_mean).quantile(tau, dim="time")
        y = (obs / obs_mean).quantile(tau, dim="time")

        yout = interp_quantiles(x, y, fut / fut_mean)

        yout = yout.where(
            (fut / fut_mean) < (sim / sim_mean).min(dim="time"),
            (obs / obs_mean).min(dim="time")
            * (fut / fut_mean)
            / (sim / sim_mean).min(dim="time"),
        )
        yout = yout.where(
            (fut / fut_mean) > (sim / sim_mean).max(dim="time"),
            (obs / obs_mean).max(dim="time")
            * (fut / fut_mean)
            / (sim / sim_mean).max(dim="time"),
        )

        yout = yout * fut_mean

    else:
        x = (sim - sim_mean).quantile(tau, dim="time")
        y = (obs - obs_mean).quantile(tau, dim="time")

        yout = interp_quantiles(x, y, fut - fut_mean)

        yout = yout.where(
            (fut - fut_mean) < (sim - sim_mean).min(dim="time"),
            (obs - obs_mean).min(dim="time")
            * (fut - fut_mean)
            / (sim - sim_mean).min(dim="time"),
        )
        yout = yout.where(
            (fut - fut_mean) > (sim - sim_mean).max(dim="time"),
            (obs - obs_mean).max(dim="time")
            * (fut - fut_mean)
            / (sim - sim_mean).max(dim="time"),
        )

        yout = yout + fut_mean

    return yout
