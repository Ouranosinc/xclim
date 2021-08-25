"""Pre and post processing for bias adjustment."""
from typing import Optional, Sequence, Tuple, Union

import dask.array as dsk
import numpy as np
import xarray as xr
from xarray.core.utils import get_temp_dimname

from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to
from xclim.core.utils import uses_dask

from ._processing import _adapt_freq, _normalize, _reordering
from .base import Grouper
from .nbutils import _escore
from .utils import ADDITIVE


@update_xclim_history
def adapt_freq(
    ref: xr.DataArray,
    sim: xr.DataArray,
    *,
    group: Union[Grouper, str],
    thresh: str = "0 mm d-1",
) -> xr.Dataset:
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
    thresh : str
      Threshold below which values are considered zero, a quantity with units.

    Returns
    -------
    sim_adj : xr.DataArray
      Simulated data with the same frequency of values under threshold than ref.
      Adjustment is made group-wise.
    pth : xr.DataArray
      For each group, the smallest value of sim that was not frequency-adjusted. All values smaller were
      either left as zero values or given a random value between thresh and pth.
      NaN where frequency adaptation wasn't needed.
    dP0 : xr.DataArray
      For each group, the percentage of values that were corrected in sim.

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
    sim = convert_units_to(sim, ref)
    thresh = convert_units_to(thresh, ref)

    out = _adapt_freq(xr.Dataset(dict(sim=sim, ref=ref)), group=group, thresh=thresh)

    # Set some metadata
    out.sim_ad.attrs.update(sim.attrs)
    out.sim_ad.attrs.update(
        references="Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4."
    )
    out.pth.attrs.update(
        long_name="Smallest value of the timeseries not corrected by frequency adaptation.",
        units=sim.units,
    )
    out.dP0.attrs.update(
        long_name=f"Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation",
    )

    return out.sim_ad, out.pth, out.dP0


@update_xclim_history
def jitter_under_thresh(x: xr.DataArray, thresh: str):
    """Replace values smaller than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
      Values.
    thresh : str
      Threshold under which to add uniform random noise to values, a quantity with units.

    Returns
    -------
    array

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    thresh = convert_units_to(thresh, x)
    epsilon = np.finfo(x.dtype).eps
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=epsilon, high=thresh, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=epsilon, high=thresh, size=x.shape)
    out = x.where(~((x < thresh) & (x.notnull())), jitter.astype(x.dtype))
    out.attrs.update(x.attrs)  # copy attrs and same units
    return out


@update_xclim_history
def jitter_over_thresh(x: xr.DataArray, thresh: str, upper_bnd: str) -> xr.Dataset:
    """Replace values greater than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
      Values.
    thresh : str
      Threshold over which to add uniform random noise to values, a quantity with units.
    upper_bnd : str
      Maximum possible value for the random noise, a quantity with units.
    Returns
    -------
    xr.Dataset

    Notes
    -----
    If thresh is low, this will change the mean value of x.
    """
    thresh = convert_units_to(thresh, x)
    upper_bnd = convert_units_to(upper_bnd, x)
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=thresh, high=upper_bnd, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=thresh, high=upper_bnd, size=x.shape)
    out = x.where(~((x > thresh) & (x.notnull())), jitter.astype(x.dtype))
    out.attrs.update(x.attrs)  # copy attrs and same units
    return out


@update_xclim_history
def normalize(
    data: xr.DataArray,
    norm: Optional[xr.DataArray] = None,
    *,
    group: Union[Grouper, str],
    kind: str = ADDITIVE,
) -> xr.Dataset:
    """Normalize an array by removing its mean.

    Normalization if performed group-wise and according to `kind`.

    Parameters
    ----------
    data: xr.DataArray
      The variable to normalize.
    norm : xr.DataArray, optional
      If present, it is used instead of computing the norm again.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details..
    kind : {'+', '*'}
      If `kind` is "+", the mean is subtracted from the mean and if it is '*', it is divided from the data.

    Returns
    -------
    xr.DataArray
      Groupwise anomaly
    """
    ds = xr.Dataset(dict(data=data))

    if norm is not None:
        norm = convert_units_to(norm, data)
        ds = ds.assign(norm=norm)

    out = _normalize(ds, group=group, kind=kind)
    out.attrs.update(data.attrs)
    return out.data.rename(data.name)


def uniform_noise_like(
    da: xr.DataArray, low: float = 1e-6, high: float = 1e-3
) -> xr.DataArray:
    """Return a uniform noise array of the same shape as da.

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


@update_xclim_history
def standardize(
    da: xr.DataArray,
    mean: Optional[xr.DataArray] = None,
    std: Optional[xr.DataArray] = None,
    dim: str = "time",
) -> Tuple[Union[xr.DataArray, xr.Dataset], xr.DataArray, xr.DataArray]:
    """Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns the standardized data, the mean and the standard deviation.
    """
    if mean is None:
        mean = da.mean(dim, keep_attrs=True)
    if std is None:
        std = da.std(dim, keep_attrs=True)
    with xr.set_options(keep_attrs=True):
        return (da - mean) / std, mean, std


@update_xclim_history
def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    with xr.set_options(keep_attrs=True):
        return (std * da) + mean


@update_xclim_history
def reordering(ref: xr.DataArray, sim: xr.DataArray, group: str = "time") -> xr.Dataset:
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
    out = _reordering(ds, group=group).reordered
    out.attrs.update(sim.attrs)
    return out


@update_xclim_history
def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,
    scale: bool = False,
) -> xr.DataArray:
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
    scale: bool
      Whether to scale the data before computing the score. If True, both arrays as scaled according
      to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
      statistics excluding NaN values.

    Returns
    -------
    xr.DataArray
        e-score with dimensions not in `dims`.

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
    .. [SkezelyRizzo] Skezely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
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
    new_dim = get_temp_dimname(tgt.dims, obs_dim)
    sim = sim.rename({obs_dim: new_dim})
    out = xr.apply_ufunc(
        _escore,
        tgt,
        sim,
        input_core_dims=[[pts_dim, obs_dim], [pts_dim, new_dim]],
        output_dtypes=[sim.dtype],
        dask="parallelized",
    )

    out.name = "escores"
    out.attrs.update(
        long_name="Energy dissimilarity metric",
        description=f"Escores computed from {N or 'all'} points.",
        references="Skezely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)",
    )
    return out
