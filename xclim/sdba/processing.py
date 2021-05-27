"""Pre and post processing for bias adjustment."""
import dask.array as dsk
import numpy as np
import xarray as xr

from xclim.core.utils import uses_dask

from .base import Grouper, map_groups
from .nbutils import vecquantiles
from .utils import ADDITIVE, apply_correction, ecdf, invert


@map_groups(sim_ad=[Grouper.DIM], pth=[Grouper.PROP], dP0=[Grouper.PROP])
def adapt_freq(
    ds: xr.Dataset,
    *,
    dim,
    thresh: float = 0,
):
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on [Themessl2012]_.

    Parameters
    ----------
    ds : xr.Dataset
      With variables :  "ref", Target/reference data, usually observed data.
      and  "sim", Simulated data.
    group : Union[str, Grouper]
      Grouping information, see base.Grouper
    thresh : float
      Threshold below which values are considered zero.

    Returns
    -------
    xr.Dataset wth the following variables:

      - `sim_adj`: Simulated data with the same frequency of values under threshold than ref.
        Adjustement is made group-wise.
      - `pth` : For each group, the smallest value of sim that was not frequency-adjusted. All values smaller were
        either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
      - `dP0` : For each group, the percentage of values that were corrected in sim.

    Notes
    -----
    With :math:`P_0^r` the frequency of values under threshold :math:`T_0` in the reference (ref) and
    :math:`P_0^s` the same for the simulated values, :math:`\\Delta P_0 = \\frac{P_0^s - P_0^r}{P_0^s}`,
    when positive, represents the proportion of values under :math:`T_0` that need to be corrected.

    The correction replaces a proportion :math:`\\Delta P_0` of the values under :math:`T_0` in sim by a uniform random
    number between :math:`T_0` and :math:`P_{th}`, where :math:`P_{th} = F_{ref}^{-1}( F_{sim}( T_0 ) )` and
    `F(x)` is the empirical cumulative distribution function (CDF).


    References
    ----------
    .. [Themessl2012] Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4.
    """
    # Compute the probability of finding a value <= thresh
    # This is the "dry-day frequency" in the precipitation case
    P0_sim = ecdf(ds.sim, thresh, dim=dim)
    P0_ref = ecdf(ds.ref, thresh, dim=dim)

    # The proportion of values <= thresh in sim that need to be corrected, compared to ref
    dP0 = (P0_sim - P0_ref) / P0_sim

    # Compute : ecdf_ref^-1( ecdf_sim( thresh ) )
    # The value in ref with the same rank as the first non zero value in sim.
    # pth is meaningless when freq. adaptation is not needed
    pth = vecquantiles(ds.ref, P0_sim, dim).where(dP0 > 0)

    if "window" in ds.sim.dims:
        # P0_sim was computed using the window, but only the original timeseries is corrected.
        sim = ds.sim.isel(window=(ds.sim.window.size - 1) // 2)
        dim = [dim[0]]
    else:
        sim = ds.sim

    # Get the percentile rank of each value in sim.
    rank = sim.rank(dim[0], pct=True)

    # Frequency-adapted sim
    sim_ad = sim.where(
        dP0 < 0,  # dP0 < 0 means no-adaptation.
        sim.where(
            (rank < P0_ref) | (rank > P0_sim),  # Preserve current values
            # Generate random numbers ~ U[T0, Pth]
            (pth.broadcast_like(sim) - thresh) * np.random.random_sample(size=sim.shape)
            + thresh,
        ),
    )

    # Set some metadata
    sim_ad.attrs.update(ds.sim.attrs)
    pth.attrs[
        "long_name"
    ] = "Smallest value of the timeseries not corrected by frequency adaptation."
    dP0.attrs[
        "long_name"
    ] = "Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation"

    # Tell group_apply that these will need reshaping (regrouping)
    # This is needed since if any variable comes out a groupby with the original group axis, the whole output is broadcasted back to the original dims.
    pth.attrs["_group_apply_reshape"] = True
    dP0.attrs["_group_apply_reshape"] = True
    return xr.Dataset(data_vars={"pth": pth, "dP0": dP0, "sim_ad": sim_ad})


def jitter_under_thresh(x: xr.DataArray, thresh: float):
    """Replace values smaller than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
      Values.
    thresh : float
      Threshold under which to add uniform random noise to values.

    Returns
    -------
    array

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    epsilon = np.finfo(x.dtype).eps
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=epsilon, high=thresh, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=epsilon, high=thresh, size=x.shape)
    return x.where(~((x < thresh) & (x.notnull())), jitter)


def jitter_over_thresh(x: xr.DataArray, thresh: float, upper_bnd: float):
    """Replace values greater than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
      Values.
    thresh : float
      Threshold over which to add uniform random noise to values.
    upper_bnd : float
      Maximum possible value for the random noise
    Returns
    -------
    array

    Notes
    -----
    If thresh is low, this will change the mean value of x.
    """
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=thresh, high=upper_bnd, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=thresh, high=upper_bnd, size=x.shape)
    return x.where(~((x > thresh) & (x.notnull())), jitter)


@map_groups(reduces=[Grouper.PROP], data=[])
def normalize(
    ds,
    *,
    dim,
    kind: str = ADDITIVE,
):
    """Normalize an array by removing its mean.
    Normalization if performed group-wise.

    Parameters
    ----------
    ds: Dataset
      The variable `data` is normalized.
      If a `norm` variable is present, is uses this one instead of computing the norm again.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    kind : {'+', '*'}
      How to apply the adjustment, either additively or multiplicatively.
    Returns
    -------
    xr.Dataset
      Group-wise anomaly of x
    """

    if "norm" in ds:
        norm = invert(ds.norm, kind)
    else:
        norm = invert(ds.data.mean(dim=dim), kind)

    return xr.Dataset(dict(data=apply_correction(ds.data, norm, kind)))


def uniform_noise_like(da: xr.DataArray, low: float = 1e-6, high: float = 1e-3):
    """Return an unform noise array of the same shape as da.

    Noise is uniformly distributed between low and high.
    Alternative method to `jitter_under_thresh` for avoiding zeroes.
    """
    if uses_dask(da):
        mod = dsk
        kw = {"chunks": da.chunks}
    else:
        mod = np
        kw = {}

    return da.copy(
        data=(high - low) * mod.random.random_sample(size=da.shape, **kw) + low
    )
