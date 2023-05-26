r"""
McArthur Forest Fire Danger (Mark 5) System
===========================================

This submodule defines indices related to the McArthur Forest Fire Danger Index Mark 5. Currently
implemented are the :py:func:`xclim.indices.fire.keetch_byram_drought_index`,
:py:func:`xclim.indices.fire.griffiths_drought_factor` and :py:func:`xclim.indices.fire.mcarthur_forest_fire_danger_index`
indices, which are used by the eponym indicators. The implementation of these indices follows :cite:t:`ffdi-finkele_2006` and :cite:t:`ffdi-noble_1980`, with any differences described in the
documentation for each index. Users are encouraged to read this module's documentation and consult
:cite:t:`ffdi-finkele_2006` for a full description of the methods used to calculate each index.

"""
# This file is structured in the following way:
# Section 1: individual codes, numba-accelerated and vectorized functions.
# Section 2: Exposed methods and indices.
#
# Methods starting with a "_" are not usable with xarray objects, whereas the others are.
from __future__ import annotations

import numpy as np
import xarray as xr
from numba import float64, guvectorize, int64

from xclim.core.units import convert_units_to, declare_units

__all__ = [
    "griffiths_drought_factor",
    "keetch_byram_drought_index",
    "mcarthur_forest_fire_danger_index",
]

# SECTION 1 - Codes - Numba accelerated and vectorized functions


@guvectorize(
    [(float64[:], float64[:], float64, float64, float64[:])],
    "(n),(n),(),()->(n)",
    nopython=True,
    cache=True,
)
def _keetch_byram_drought_index(p, t, pa, kbdi0, kbdi: float):  # pragma: no cover
    """Compute the Keetch-Byram drought (KBDI) index.

    Parameters
    ----------
    p : array_like
        Total rainfall over previous 24 hours [mm].
    t : array_like
        Maximum temperature near the surface over previous 24 hours [C].
    pa : float
        Mean annual accumulated rainfall.
    kbdi0 : float
        Previous value of the Keetch-Byram drought index used to initialise the KBDI calculation.

    Returns
    -------
    array_like
        Keetch-Byram drought index.
    """
    no_p = 0.0  # Where to define zero rainfall
    rr = 5.0  # Initialise remaining runoff

    for d in range(len(p)):
        # Calculate the runoff and remaining runoff for this timestep
        if p[d] <= no_p:
            r = p[d]
            rr = 5.0
        else:
            r = min(p[d], rr)
            rr -= r

        Peff = p[d] - r
        ET = (
            1e-3
            * (203.2 - kbdi0)
            * (0.968 * np.exp(0.0875 * t[d] + 1.5552) - 8.3)
            / (1 + 10.88 * np.exp(-0.00173 * pa))
        )

        kbdi0 += ET - Peff

        # Limit kbdi to between 0 and 200 mm
        if kbdi0 < 0.0:
            kbdi0 = 0.0

        if kbdi0 > 203.2:
            kbdi0 = 203.2

        kbdi[d] = kbdi0


@guvectorize(
    [(float64[:], float64[:], int64, float64[:])],
    "(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def _griffiths_drought_factor(p, smd, lim, df):  # pragma: no cover
    """Compute the Griffiths drought factor.

    Parameters
    ----------
    p : array_like
        Total rainfall over previous 24 hours [mm].
    smd : array_like
        Soil moisture deficit (e.g. KBDI).
    lim : int
        How to limit the drought factor.
        If 0, use equation (14) in :cite:t:`ffdi-finkele_2006`.
        If 1, use equation (13) in :cite:t:`ffdi-finkele_2006`.

    Returns
    -------
    df : array_like
        The limited Griffiths drought factor
    """
    wl = 20  # 20-day window length

    for d in range(wl - 1, len(p)):
        pw = p[d - wl + 1 : d + 1]

        # Calculate the x-function from significant rainfall
        # events
        conseq = 0
        pmax = 0.0
        P = 0.0
        x = 1.0
        for iw in range(wl):
            event = pw[iw] > 2.0
            event_end = ~event & (conseq != 0)
            final_event = event & (iw == (wl - 1))

            if event:
                conseq = conseq + 1
                P = P + pw[iw]
                if pw[iw] >= pmax:
                    N = wl - iw
                    pmax = pw[iw]

            if event_end | final_event:
                # N = 0 defines a rainfall event since 9am today,
                # so doesn't apply here, where p is the rainfall
                # over previous 24 hours.
                x_ = N**1.3 / (N**1.3 + P - 2.0)
                x = min(x_, x)

                conseq = 0
                P = 0.0
                pmax = 0.0

        if lim == 0:
            if smd[d] < 20:
                xlim = 1 / (1 + 0.1135 * smd[d])
            else:
                xlim = 75 / (270.525 - 1.267 * smd[d])
            if x > xlim:
                x = xlim

        dfw = (
            10.5
            * (1 - np.exp(-(smd[d] + 30) / 40))
            * (41 * x**2 + x)
            / (40 * x**2 + x + 1)
        )

        if lim == 1:
            if smd[d] < 25.0:
                dflim = 6.0
            elif (smd[d] >= 25.0) & (smd[d] < 42.0):
                dflim = 7.0
            elif (smd[d] >= 42.0) & (smd[d] < 65.0):
                dflim = 8.0
            elif (smd[d] >= 65.0) & (smd[d] < 100.0):
                dflim = 9.0
            else:
                dflim = 10.0
            if dfw > dflim:
                dfw = dflim

        if dfw > 10.0:
            dfw = 10.0

        df[d] = dfw


# SECTION 2 - Public methods and indices


@declare_units(
    pr="[precipitation]",
    tasmax="[temperature]",
    pr_annual="[precipitation]",
    kbdi0="[precipitation]",
)
def keetch_byram_drought_index(
    pr: xr.DataArray,
    tasmax: xr.DataArray,
    pr_annual: xr.DataArray,
    kbdi0: xr.DataArray | None = None,
) -> xr.DataArray:
    """Keetch-Byram drought index (KBDI) for soil moisture deficit.

    The KBDI indicates the amount of water necessary to bring the soil moisture content back to
    field capacity. It is often used in the calculation of the McArthur Forest Fire Danger
    Index. The method implemented here follows :cite:t:`ffdi-finkele_2006` but limits the
    maximum KBDI to 203.2 mm, rather than 200 mm, in order to align best with the majority of
    the literature.

    Parameters
    ----------
    pr : xr.DataArray
        Total rainfall over previous 24 hours [mm/day].
    tasmax : xr.DataArray
        Maximum temperature near the surface over previous 24 hours [degC].
    pr_annual: xr.DataArray
        Mean (over years) annual accumulated rainfall [mm/year].
    kbdi0 : xr.DataArray, optional
        Previous KBDI values used to initialise the KBDI calculation [mm/day]. Defaults to 0.

    Returns
    -------
    xr.DataArray
        Keetch-Byram drought index.

    Notes
    -----
    This method implements the method described in :cite:t:`ffdi-finkele_2006` (section 2.1.1) for
    calculating the KBDI with one small difference: in :cite:t:`ffdi-finkele_2006` the maximum
    KBDI is limited to 200 mm to represent the maximum field capacity of the soil (8 inches
    according to :cite:t:`ffdi-keetch_1968`). However, it is more common in the literature to limit
    the KBDI to 203.2 mm which is a more accurate conversion from inches to mm. In this function,
    the KBDI is limited to 203.2 mm.

    References
    ----------
    :cite:cts:`ffdi-keetch_1968,ffdi-finkele_2006,ffdi-holgate_2017,ffdi-dolling_2005`
    """

    def _keetch_byram_drought_index_pass(pr, tasmax, pr_annual, kbdi0):
        """Pass inputs on to guvectorized function `_keetch_byram_drought_index`. DO NOT CALL DIRECTLY, use `keetch_byram_drought_index` instead."""
        # This function is actually only required as xr.apply_ufunc will not receive
        # a guvectorized function which has the output(s) in its function signature
        return _keetch_byram_drought_index(pr, tasmax, pr_annual, kbdi0)

    pr = convert_units_to(pr, "mm/day", context="hydro")
    tasmax = convert_units_to(tasmax, "C")
    pr_annual = convert_units_to(pr_annual, "mm/year", context="hydro")
    if kbdi0 is not None:
        kbdi0 = convert_units_to(kbdi0, "mm/day", context="hydro")
    else:
        kbdi0 = xr.full_like(pr.isel(time=0), 0)

    kbdi = xr.apply_ufunc(
        _keetch_byram_drought_index_pass,
        pr,
        tasmax,
        pr_annual,
        kbdi0,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[pr.dtype],
    )
    kbdi.attrs["units"] = "mm/day"
    return kbdi


@declare_units(
    pr="[precipitation]",
    smd="[precipitation]",
)
def griffiths_drought_factor(
    pr: xr.DataArray,
    smd: xr.DataArray,
    limiting_func: str = "xlim",
) -> xr.DataArray:
    """Griffiths drought factor based on the soil moisture deficit.

    The drought factor is a numeric indicator of the forest fire fuel availability in the
    deep litter bed. It is often used in the calculation of the McArthur Forest Fire Danger
    Index. The method implemented here follows :cite:t:`ffdi-finkele_2006`.

    Parameters
    ----------
    pr : xr.DataArray
        Total rainfall over previous 24 hours [mm/day].
    smd : xarray DataArray
        Daily soil moisture deficit (often KBDI) [mm/day].
    limiting_func : {"xlim", "discrete"}
        How to limit the values of the drought factor.
        If "xlim" (default), use equation (14) in :cite:t:`ffdi-finkele_2006`.
        If "discrete", use equation Eq (13) in :cite:t:`ffdi-finkele_2006`, but with the lower
        limit of each category bound adjusted to match the upper limit of the previous bound.

    Returns
    -------
    df : xr.DataArray
        The limited Griffiths drought factor.

    Notes
    -----
    Calculation of the Griffiths drought factor depends on the rainfall over the previous 20 days.
    Thus, the first non-NaN time point in the drought factor returned by this function
    corresponds to the 20th day of the input data.

    References
    ----------
    :cite:cts:`ffdi-griffiths_1999,ffdi-finkele_2006,ffdi-holgate_2017`
    """

    def _griffiths_drought_factor_pass(pr, smd, lim):
        """Pass inputs on to guvectorized function `_griffiths_drought_factor`. DO NOT CALL DIRECTLY, use `griffiths_drought_factor` instead."""
        # This function is actually only required as xr.apply_ufunc will not receive
        # a guvectorized function which has the output(s) in its function signature
        return _griffiths_drought_factor(pr, smd, lim)

    pr = convert_units_to(pr, "mm/day", context="hydro")
    smd = convert_units_to(smd, "mm/day")

    if limiting_func == "xlim":
        lim = 0
    elif limiting_func == "discrete":
        lim = 1
    else:
        raise ValueError(f"{limiting_func} is not a valid input for `limiting_func`")

    df = xr.apply_ufunc(
        _griffiths_drought_factor_pass,
        pr,
        smd,
        kwargs=dict(lim=lim),
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[pr.dtype],
    )
    df.attrs["units"] = ""

    # First non-zero entry is at the 19th time point since df is calculated
    # from a 20-day rolling window. Make prior points NaNs.
    return df.where(df.time >= df.time.isel(time=19))


@declare_units(
    drought_factor="[]",
    tasmax="[temperature]",
    hurs="[]",
    sfcWind="[speed]",
)
def mcarthur_forest_fire_danger_index(
    drought_factor: xr.DataArray,
    tasmax: xr.DataArray,
    hurs: xr.DataArray,
    sfcWind: xr.DataArray,
):
    """McArthur forest fire danger index (FFDI) Mark 5.

    The FFDI is a numeric indicator of the potential danger of a forest fire.

    Parameters
    ----------
    drought_factor : xr.DataArray
        The drought factor, often the daily Griffiths drought factor (see :py:func:`griffiths_drought_factor`).
    tasmax : xr.DataArray
        The daily maximum temperature near the surface, or similar. Different applications have used
        different inputs here, including the previous/current day's maximum daily temperature at a height of
        2m, and the daily mean temperature at a height of 2m.
    hurs : xr.DataArray
        The relative humidity near the surface and near the time of the maximum daily temperature, or similar.
        Different applications have used different inputs here, including the mid-afternoon relative humidity
        at a height of 2m, and the daily mean relative humidity at a height of 2m.
    sfcWind : xr.DataArray
        The wind speed near the surface and near the time of the maximum daily temperature, or similar.
        Different applications have used different inputs here, including the mid-afternoon wind speed at a
        height of 10m, and the daily mean wind speed at a height of 10m.

    Returns
    -------
    xr.DataArray
        The McArthur forest fire danger index.

    References
    ----------
    :cite:cts:`ffdi-noble_1980,ffdi-dowdy_2018,ffdi-holgate_2017`
    """
    tasmax = convert_units_to(tasmax, "C")
    hurs = convert_units_to(hurs, "pct")
    sfcWind = convert_units_to(sfcWind, "km/h")

    ffdi = drought_factor**0.987 * np.exp(
        0.0338 * tasmax - 0.0345 * hurs + 0.0234 * sfcWind + 0.243147
    )
    ffdi.attrs["units"] = ""
    return ffdi
