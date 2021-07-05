from typing import Callable

import numpy as np
from xarray.core.dataarray import DataArray

from xclim.core.calendar import percentile_doy
from xclim.indices import (
    cold_spell_duration_index,
    days_over_precip_thresh,
    fraction_over_precip_thresh,
    tg10p,
    tg90p,
    tn10p,
    tn90p,
    tx10p,
    tx90p,
    warm_spell_duration_index,
)
from xclim.testing import open_dataset


def ar1(alpha, n, positive_values=False):
    """Return random AR1 DataArray."""

    # White noise
    wn = np.random.randn(n - 1) * np.sqrt(1 - alpha ** 2)

    # Autoregressive series of order 1
    out = np.empty(n)
    out[0] = np.random.randn()
    for i, w in enumerate(wn):
        if positive_values:
            out[i + 1] = np.abs(alpha * out[i] + w)
        else:
            out[i + 1] = alpha * out[i] + w

    return out


def bootstrap_testor(
    serie: DataArray,
    per: int,
    compute_indice: Callable[[DataArray, DataArray], DataArray],
    positive_values=False,
):
    # GIVEN
    n = int(60 * 365.25)
    alpha = 0.8
    climat_variable = serie(ar1(alpha, n, positive_values), start="2000-01-01")
    in_base_slice = slice("2000-01-01", "2029-12-31")
    out_base_slice = slice("2030-01-01", "2059-12-31")
    per = percentile_doy(climat_variable.sel(time=in_base_slice), per=per)
    # WHEN
    no_bootstrap = compute_indice(climat_variable, per, False)
    no_bs_in_base = no_bootstrap.sel(time=(in_base_slice))
    no_bs_out_base = no_bootstrap.sel(time=(out_base_slice))
    bootstrap = compute_indice(climat_variable, per, True)
    bootstrapped_in_base = bootstrap.sel(time=(in_base_slice))
    bs_out_base = bootstrap.sel(time=(out_base_slice))
    # THEN
    # bootstrapping should increase the indices values within the in_base
    # will not work on unrealistic values such as a constant temperature
    assert np.count_nonzero(bootstrapped_in_base > no_bs_in_base) > np.count_nonzero(
        bootstrapped_in_base < no_bs_in_base
    )
    # bootstrapping should let the out of base unchanged
    assert np.count_nonzero(no_bs_out_base != bs_out_base) == 0


def test_bootstrap(tas_series, tasmax_series, tasmin_series, pr_series):
    # The closer the targetted percentile is to the median the less bootstrapping makes sense to use.
    # The tests may even fail if the chosen percentile is close to 50
    # temperatures
    bootstrap_testor(
        tas_series, 98, lambda x, y, z: tg90p(x, y, freq="MS", bootstrap=z)
    )
    bootstrap_testor(
        tasmin_series, 98, lambda x, y, z: tn90p(x, y, freq="YS", bootstrap=z)
    )
    bootstrap_testor(
        tasmax_series, 98, lambda x, y, z: tx90p(x, y, freq="MS", bootstrap=z)
    )
    bootstrap_testor(
        tasmin_series, 2, lambda x, y, z: tn10p(x, y, freq="MS", bootstrap=z)
    )
    bootstrap_testor(
        tasmax_series, 2, lambda x, y, z: tx10p(x, y, freq="YS", bootstrap=z)
    )
    bootstrap_testor(tas_series, 2, lambda x, y, z: tg10p(x, y, freq="MS", bootstrap=z))
    bootstrap_testor(
        tasmax_series,
        98,
        lambda x, y, z: warm_spell_duration_index(
            x, y, window=6, freq="MS", bootstrap=z
        ),
    )
    bootstrap_testor(
        tasmin_series,
        2,
        lambda x, y, z: cold_spell_duration_index(
            x, y, window=6, freq="MS", bootstrap=z
        ),
    )
    # precipitations
    bootstrap_testor(
        pr_series,
        99,
        lambda x, y, z: days_over_precip_thresh(x, y, freq="MS", bootstrap=z),
        positive_values=True,
    )
    bootstrap_testor(
        pr_series,
        98,
        lambda x, y, z: fraction_over_precip_thresh(x, y, freq="MS", bootstrap=z),
        positive_values=True,
    )


def test_doctest_ndims():
    """Replicates doctest to facilitate debugging."""
    tas = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tas
    t90 = percentile_doy(tas, window=5, per=90)
    tg90p(tas=tas, t90=t90, freq="YS", bootstrap=True)

    tg90p(tas=tas, t90=t90.isel(percentiles=0), freq="YS", bootstrap=True)
