"""Compute functions of processing.py.

Here are defined the functions wrapped by map_blocks or map_groups,
user-facing, metadata-handling functions should be defined in processing.py.
"""
from typing import Sequence

import numpy as np
import xarray as xr

from . import nbutils as nbu
from .base import Grouper, map_groups
from .utils import ADDITIVE, apply_correction, ecdf, invert


@map_groups(
    sim_ad=[Grouper.ADD_DIMS, Grouper.DIM], pth=[Grouper.PROP], dP0=[Grouper.PROP]
)
def _adapt_freq(
    ds: xr.Dataset,
    *,
    dim: Sequence[str],
    thresh: float = 0,
) -> xr.Dataset:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is the compute function, see :py:func:`xclim.sdba.processing.adapt_freq` for the user-facing function.

    Parameters
    ----------
    ds : xr.Dataset
      With variables :  "ref", Target/reference data, usually observed data.
      and  "sim", Simulated data.
    dim : str, or seqence of strings
      Dimension name(s). If more than one, the probabilities and quantiles are computed within all the dimensions.
      If  `window` is in the names, it is removed before the correction and the final timeseries is corrected along dim[0] only.
    group : Union[str, Grouper]
      Grouping information, see base.Grouper
    thresh : float
      Threshold below which values are considered zero.

    Returns
    -------
    xr.Dataset, wth the following variables:

      - `sim_adj`: Simulated data with the same frequency of values under threshold than ref.
        Adjustment is made group-wise.
      - `pth` : For each group, the smallest value of sim that was not frequency-adjusted. All values smaller were
        either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
      - `dP0` : For each group, the percentage of values that were corrected in sim.
    """
    # Compute the probability of finding a value <= thresh
    # This is the "dry-day frequency" in the precipitation case
    P0_sim = ecdf(ds.sim, thresh, dim=dim)
    P0_ref = ecdf(ds.ref, thresh, dim=dim)

    # The proportion of values <= thresh in sim that need to be corrected, compared to ref
    dP0 = (P0_sim - P0_ref) / P0_sim

    if dP0.isnull().all():
        # All NaN slice.
        pth = dP0.copy()
        sim_ad = ds.sim.copy()
    else:

        # Compute : ecdf_ref^-1( ecdf_sim( thresh ) )
        # The value in ref with the same rank as the first non zero value in sim.
        # pth is meaningless when freq. adaptation is not needed
        pth = nbu.vecquantiles(ds.ref, P0_sim, dim).where(dP0 > 0)

        # Probabilites and quantiles computed within all dims, but correction along the first one only.
        if "window" in dim:
            # P0_sim was computed using the window, but only the original time series is corrected.
            # Grouper.apply does this step, but if done here it makes the code faster.
            sim = ds.sim.isel(window=(ds.sim.window.size - 1) // 2)
        else:
            sim = ds.sim
        dim = dim[0]

        # Get the percentile rank of each value in sim.
        rank = sim.rank(dim, pct=True)

        # Frequency-adapted sim
        sim_ad = sim.where(
            dP0 < 0,  # dP0 < 0 means no-adaptation.
            sim.where(
                (rank < P0_ref) | (rank > P0_sim),  # Preserve current values
                # Generate random numbers ~ U[T0, Pth]
                (pth.broadcast_like(sim) - thresh)
                * np.random.random_sample(size=sim.shape)
                + thresh,
            ),
        )

    # Tell group_apply that these will need reshaping (regrouping)
    # This is needed since if any variable comes out a groupby with the original group axis,
    # the whole output is broadcasted back to the original dims.
    pth.attrs["_group_apply_reshape"] = True
    dP0.attrs["_group_apply_reshape"] = True
    return xr.Dataset(data_vars={"pth": pth, "dP0": dP0, "sim_ad": sim_ad})


@map_groups(reduces=[Grouper.PROP], data=[])
def _normalize(
    ds: xr.Dataset,
    *,
    dim: Sequence[str],
    kind: str = ADDITIVE,
) -> xr.Dataset:
    """Normalize an array by removing its mean.
    Normalization is performed group-wise.

    Parameters
    ----------
    ds: xr.Dataset
      The variable `data` is normalized.
      If a `norm` variable is present, is uses this one instead of computing the norm again.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    dim : sequence of strings
      Dimension name(s).
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


@map_groups(reordered=[Grouper.DIM], main_only=True)
def _reordering(ds, *, dim):
    """Group-wise reordering.

    Parameters
    ----------
    ds: xr.Dataset
      With variables:
        - sim : The timeseries to reorder.
        - ref : The timeseries whose rank to use.
    dim: str
      The dimension along which to reorder.
    """

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
