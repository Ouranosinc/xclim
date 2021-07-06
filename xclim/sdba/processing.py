"""Pre and post processing for bias adjustment."""
from typing import Optional, Sequence

import dask.array as dsk
import numpy as np
import xarray as xr

from xclim.core.utils import uses_dask

from . import nbutils as nbu
from .base import Grouper, map_groups
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
    dim : str
      Dimension name.
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
    pth = nbu.vecquantiles(ds.ref, P0_sim, dim).where(dP0 > 0)

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
    return x.where(~((x < thresh) & (x.notnull())), jitter.astype(x.dtype))


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
    return x.where(~((x > thresh) & (x.notnull())), jitter.astype(x.dtype))


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


def standardize(
    da: xr.DataArray,
    mean: Optional[xr.DataArray] = None,
    std: Optional[xr.DataArray] = None,
    dim: str = "time",
):
    """Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns the standardized data, the mean and the standard deviation.
    """
    if mean is None:
        mean = da.mean(dim)
    if std is None:
        std = da.std(dim)
    with xr.set_options(keep_attrs=True):
        return (da - mean) / std, mean, std


def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    return (std * da) + mean


@map_groups(reordered=[Grouper.DIM], main_only=True)
def _reordering_group(ds, *, dim):
    """Group-wise reordering."""

    def _reordering_1d(data, ordr):
        return np.sort(data)[np.argsort(np.argsort(ordr))]

    return (
        xr.apply_ufunc(
            _reordering_1d,
            ds.sim,
            ds.ref,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ds.sim.dtype],
        )
        .rename("reordered")
        .to_dataset()
    )


def reordering(sim, ref, group="time"):
    """Reorders data in `sim` following the order of ref.

    The rank structure of `ref` is used to reorder the elements of `sim` along dimension "time",
    optionally doing the operation group-wise.

    Parameters
    ----------
    sim : xr.DataArray
      Array to reorder.
    ref : xr.DataArray
      Array whose rank order sim should replicate.
    group : str
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.

    Returns
    -------
    xr.Dataset
      sim reordered according to ref's rank order.

    Reference
    ---------
    Cannon, A. J. (2018). Multivariate quantile mapping bias correction: An N-dimensional probability density function
    transform for climate model simulations of multiple variables. Climate Dynamics, 50(1), 31–49.
    https://doi.org/10.1007/s00382-017-3580-6
    """
    ds = xr.Dataset({"sim": sim, "ref": ref})
    return _reordering_group(ds, group=group).reordered


def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,
    scale: bool = False,
):
    r"""Energy score, or energy dissimilarity metric, based on [SkezelyRizzo]_ and [Cannon18]_.

    Parameters
    ----------
    tgt: DataArray
      Target observations.
    sim: DataArray
      Candidate observations. Must have the same dimensions as `tgt`.
    dims: sequence of 2 strings
      The name of the dimensions along which the variables and observation points are listed.
      `tgt` and `sim` can have different length along the second one, but must be equal along the first one.
      The result will keep all other dimensions.
    N : int
      If larger than 0, the number of observations to use in the score computation. The points are taken
      evenly distributed along `obs_dim`.
    scale: boolean
      Whether to scale the data before computing the score. If True, both arrays as scaled according
      to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
      statistics excluding NaN values.

    Returns
    -------
    e-score
        xr.DataArray with dimensions not in `dims`.

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,,n_j`
    proposed by Szekely and Rizzo (2005) is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of [Cannon18]_ to
    the metric. With that factor, the test becomes identical to the one defined by [BaringhausFranz]_.

    References
    ----------
    .. [SkezelyRizzo] Szekely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
    .. [BaringhausFranz] Baringhaus, L. and Franz, C. (2004) On a new multivariate two-sample test, Journal of Multivariate Analysis, 88(1), 190–206. https://doi.org/10.1016/s0047-259x(03)00079-4
    """

    pts_dim, obs_dim = dims

    if N > 0:
        # If N non-zero we only take around N points, evenly distributed
        sim_step = int(np.ceil(sim[obs_dim].size / N))
        sim = sim.isel({obs_dim: slice(None, None, sim_step)})
        tgt_step = int(np.ceil(tgt[obs_dim].size / N))
        tgt = tgt.isel({obs_dim: slice(None, None, tgt_step)})

    if scale:
        tgt, avg, std = standardize(tgt)
        sim, _, _ = standardize(sim, avg, std)

    # The dimension renaming is to allow different coordinates.
    # Otherwise, apply_ufunc tries to align both obs_dim together.
    new_dim = xr.core.utils.get_temp_dimname(tgt.dims, obs_dim)
    sim = sim.rename({obs_dim: new_dim})
    return xr.apply_ufunc(
        nbu._escore,
        tgt,
        sim,
        input_core_dims=[[pts_dim, obs_dim], [pts_dim, new_dim]],
        output_dtypes=[sim.dtype],
        dask="parallelized",
    )
