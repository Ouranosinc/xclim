"""Pre and post processing for bias adjustment."""
from typing import Optional, Union

import numpy as np
import xarray as xr
from dask import array as dsk

from .base import Grouper, parse_group
from .utils import ADDITIVE, apply_correction, broadcast, ecdf, invert


@parse_group
def adapt_freq(
    sim: xr.DataArray,
    ref: xr.DataArray,
    thresh: float = 0,
    *,
    group: Union[str, Grouper] = "time",
):
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on [Themessl2012]_.

    Parameters
    ----------
    ref : xr.DataArray
      Target/reference data, usually observed data.
    sim : xr.DataArray
      Simulated data.
    thresh : float
      Threshold below which values are considered zero.
    group : Union[str, Grouper]
      Grouping information, see base.Grouper

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

    def _adapt_freq_group(ds, dim=["time"]):
        if isinstance(ds.sim.data, dsk.Array):
            # In order to be efficient and lazy, some classical numpy ops will be replaced by dask's version
            mod = dsk
            kws = {"chunks": ds.sim.chunks}
        else:
            mod = np
            kws = {}

        # Compute the probability of finding a value <= thresh
        # This is the "dry-day frequency" in the precipitation case
        P0_sim = ecdf(ds.sim, thresh, dim=dim)
        P0_ref = ecdf(ds.ref, thresh, dim=dim)

        # The proportion of values <= thresh in sim that need to be corrected, compared to ref
        dP0 = (P0_sim - P0_ref) / P0_sim

        # Compute : ecdf_ref^-1( ecdf_sim( thresh ) )
        # The value in ref with the same rank as the first non zero value in sim.
        pth = xr.apply_ufunc(
            np.nanpercentile,
            ds.ref,
            P0_sim
            * 100,  # np.percentile takes values in [0, 100], ecdf outputs in [0, 1]
            input_core_dims=[dim, []],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[ds.ref.dtype],
        ).where(
            dP0 > 0
        )  # pth is meaningless when freq. adaptation is not needed

        if "window" in ds.sim.dims:
            # P0_sim was computed using the window, but only the original timeseries is corrected.
            sim = ds.sim.isel(window=(ds.sim.window.size - 1) // 2)
            dim = [dim[0]]
        else:
            sim = ds.sim

        # Get the percentile rank of each value in sim.
        # da.rank() doesn't work with dask arrays.
        rank = (
            xr.apply_ufunc(
                lambda da: np.argsort(np.argsort(da, axis=-1), axis=-1),
                sim,
                input_core_dims=[dim],
                output_core_dims=[dim],
                dask="parallelized",
                output_dtypes=[sim.dtype],
            )
            / sim.notnull().sum(dim=dim)
        )

        # Frequency-adapted sim
        sim_ad = sim.where(
            dP0 < 0,  # dP0 < 0 means no-adaptation.
            sim.where(
                (rank < P0_ref) | (rank > P0_sim),  # Preserve current values
                # Generate random numbers ~ U[T0, Pth]
                (pth.broadcast_like(sim) - thresh)
                * mod.random.random_sample(size=sim.shape, **kws)
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

    return group.apply(_adapt_freq_group, {"sim": sim, "ref": ref})


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
    if isinstance(x.data, dsk.Array):
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
    if isinstance(x.data, dsk.Array):
        jitter = dsk.random.uniform(
            low=thresh, high=upper_bnd, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=thresh, high=upper_bnd, size=x.shape)
    return x.where(~((x > thresh) & (x.notnull())), jitter)


@parse_group
def normalize(
    x: xr.DataArray,
    *,
    group: Union[str, Grouper] = "time",
    kind: str = ADDITIVE,
    norm: Optional[xr.DataArray] = None,
):
    """Normalize an array by removing its mean.

    Normalization if performed group-wise.

    Parameters
    ----------
    x : xr.DataArray
      Array to be normalized.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    kind : {'+', '*'}
      How to apply the adjustment, either additively or multiplicatively.
    norm : xr.DataArray
      If the norm was already computed (for example with `group.apply("mean", x)`), skip the
        computation step. The array should have the same dimensions as `x` except for "time" that should
        be replaced by `group.prop`.

    Returns
    -------
    xr.DataArray or xr.Dataset
      Group-wise anomaly of x
    """

    def _normalize_group(grp, dim=["time"]):
        return apply_correction(grp, invert(grp.mean(dim=dim), kind), kind)

    if norm is None:
        return group.apply(_normalize_group, x)

    return apply_correction(
        x,
        broadcast(invert(norm, kind), x, group=group, interp="nearest"),
        kind,
    )
