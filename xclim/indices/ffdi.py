# noqa: D205,D400
r"""
=============================================
McArthur Forest Fire Danger Indices Submodule
=============================================

This submodule defines the :py:func:`xclim.indices.Keetch_Byram_drought_index`,
:py:func:`xclim.indices.Griffiths_drought_factor` and
:py:func:`xclim.indices.McArthur_forest_fire_danger_index` indices, which are used by the eponym indicators.
Users should read this module's documentation and consult :cite:t:`fire-finkele_2006` which provides details
of the methods used to calculate each index.
"""
# This file is structured in the following way:
# Section 1: individual codes, numba-accelerated and vectorized functions.
# Section 2: Larger computing functions (the KBDI iterator)
# Section 3: Exposed methods and indices.
#
# Methods starting with a "_" are not usable with xarray objects, whereas the others are.
import numpy as np
import xarray as xr
from numba import float64, guvectorize, int64

from xclim.core.units import convert_units_to, declare_units

# SECTION 1 - Codes - Numba accelerated and vectorized functions


@guvectorize(
    [
        (
            float64,
            float64,
            float64,
            float64,
            float64,
            float64[:],
            float64[:],
        )
    ],
    "(),(),(),(),()->(),()",
)
def _Keetch_Byram_drought_index(p, t, pa, rr0, kbdi0, rr, kbdi):  # pragma: no cover
    """
    Compute the Keetch-Byram drought index (KBDI) over one time step.

    Parameters
    ----------
    p : float
        Total rainfall over previous 24 hours [mm].
    t: float
        Maximum temperature near the surface over previous 24 hours [C].
    pa: float
        Mean annual accumulated rainfall
    rr0: float
        Remaining rainfall to be assigned to runoff from previous iteration.
        Runoff is approximated as the first 5 mm of rain within consecutive
        days with nonzero rainfall.
    kbdi0 : float
        Previous value of the Keetch-Byram drought index.

    Returns
    -------
    rr : array_like
        Remaining rainfall to be assigned to runoff.
    kbdi : array_like
        Keetch-Byram drought index.
    """
    # Reset remaining runoff if there is zero rainfall
    if p == 0.0:
        rr0 = 5.0

    # Calculate the runoff for this timestep
    if p < rr0:
        r = p
    else:
        r = rr0

    Peff = p - r
    ET = (
        1e-3
        * (203.2 - kbdi0)
        * (0.968 * np.exp(0.0875 * t + 1.5552) - 8.3)
        / (1 + 10.88 * np.exp(-0.00173 * pa))
    )
    kbdi_curr = kbdi0 - Peff + ET

    # Limit kbdi to between 0 and 200 mm
    if kbdi_curr < 0.0:
        kbdi_curr = 0.0

    if kbdi_curr > 203.2:
        kbdi_curr = 203.2

    rr[0] = rr0 - r
    kbdi[0] = kbdi_curr


@guvectorize(
    [
        (
            float64[:],
            float64[:],
            int64,
            float64[:],
        )
    ],
    "(n),(n),()->(n)",
)
def _Griffiths_drought_factor(p, smd, lim, df):  # pragma: no cover
    """
    Compute the Griffiths drought factor.

    Parameters
    ----------
    p : array_like
        Total rainfall over previous 24 hours [mm].
    smd : array_like
        Soil moisture deficit (e.g. KBDI).
    lim : integer
        How to limit the drought factor. If 0, use equation (14) in
        :cite:t:`fire-finkele_2006`. If 1, use equation Eq (13) in
        :cite:t:`fire-finkele_2006`.

    Returns
    -------
    df : array_like
        The limited Griffiths drought factor
    """
    wl = 20  # 20 day window length

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


# SECTION 2 : Iterators


def _Keetch_Byram_drought_index_calc(pr, tasmax, pr_annual, kbdi0):
    """Primary function computing the Keetch-Byram drought index. DO NOT CALL DIRECTLY, use `Keetch_Byram_drought_index` instead."""
    kbdi = np.zeros_like(pr)
    runoff_remain = 5.0
    kbdi_prev = kbdi0

    for it in range(pr.shape[-1]):
        runoff_remain, kbdi[..., it] = _Keetch_Byram_drought_index(
            pr[..., it],
            tasmax[..., it],
            pr_annual,
            runoff_remain,
            kbdi_prev,
        )
        kbdi_prev = kbdi[..., it]

    return kbdi


def _Griffiths_drought_factor_calc(pr, smd, lim):
    """Primary function computing the Griffiths drought factor. DO NOT CALL DIRECTLY, use `Griffiths_drought_factor` instead."""
    # This function is actually only required as xr.apply_ufunc will not allow
    # `func=_Griffiths_drought_factor` since this is guvectorized and has the
    # output in its function signature
    return _Griffiths_drought_factor(pr, smd, lim)


# SECTION 3 - Public methods and indices


# @declare_units(
#     pr="[precipitation]",
#     tasmax="[temperature]",
#     pr_annual="[precipitation]",
# )
def Keetch_Byram_drought_index(
    pr: xr.DataArray,
    tasmax: xr.DataArray,
    pr_annual: xr.DataArray,
    kbdi0: xr.DataArray | None = None,
):
    """
    Calculate the Keetch-Byram drought index (KBDI).

    This method implements the methodology and formula described in :cite:t:`fire-finkele_2006`
    (section 2.1.1) for calculating the KBDI. See Notes below.

    Parameters
    ----------
    pr : xr.DataArray
        Total rainfall over previous 24 hours.
    tasmax : xr.DataArray
        Maximum temperature near the surface over previous 24 hours.
    pr_annual: xr.DataArray
        Mean (over years) annual accumulated rainfall
    kbdi0 : xr.DataArray, optional
        Previous KBDI map used to initialise the KBDI calculation. Defaults to 0.

    Returns
    -------
    kbdi : xr.DataArray
        Keetch-Byram drought index.

    Notes
    -----
    :cite:t:`fire-finkele_2006` limit the maximum KBDI to 200 mm to represent the
    maximum field capacity of the soil (8 in according to :cite:t:`fire-keetch_1968`).
    However, it is more common in the literature to limit the KBDI to 203.2 mm which
    is a more accurate conversion from in to mm. In the function, the KBDI is limited
    to 203.2 mm.

    References
    ----------
    :cite:t:`fire-keetch_1968,fire-finkele_2006,fire-holgate_2017,fire-dolling_2005`
    """
    # pr = convert_units_to(pr, "mm/day")
    # tasmax = convert_units_to(tasmax, "C")
    # pr_annual = convert_units_to(pr_annual, "mm/day")

    if kbdi0 is None:
        kbdi0 = xr.full_like(pr.isel(time=0), 0)

    return xr.apply_ufunc(
        _Keetch_Byram_drought_index_calc,
        pr,
        tasmax,
        pr_annual,
        kbdi0,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[pr.dtype],
    )


# @declare_units(
#     pr="[precipitation]",
#     smd="[precipitation]",
# )
def Griffiths_drought_factor(
    pr: xr.DataArray,
    smd: xr.DataArray,
    limiting_func: str = "xlim",
):
    """
    Calculate the Griffiths drought factor based on the soil moisture deficit.

    This method implements the methodology and formula described in :cite:t:`fire-finkele_2006`
    (section 2.2) for calculating the Griffiths drought factor.

    Parameters
    ----------
    pr : xr.DataArray
        Total rainfall over previous 24 hours.
    smd : xarray DataArray
        Daily soil moisture deficit (often KBDI).
    limiting_func : {"xlim", "discrete"}
        How to limit the values of the drought factor. If "xlim" (default), use equation (14) in
        :cite:t:`fire-finkele_2006`. If "discrete", use equation Eq (13) in
        :cite:t:`fire-finkele_2006`.

    Returns
    -------
    df : xr.DataArray
        The limited Griffiths drought factor.

    Notes
    -----
    Calculation of the Griffiths drought factor depends on the rainfall over the previous 20 days.
    Thus, the first available time point in the drought factor returned by this function
    corresponds to the 20th day of the input data.

    References
    ----------
    :cite:t:`fire-griffiths_1999,fire-finkele_2006,fire-holgate_2017`
    """
    # pr = convert_units_to(pr, "mm/day")
    # smd = convert_units_to(smd, "mm/day")

    if limiting_func == "xlim":
        lim = 0
    elif limiting_func == "discrete":
        lim = 1
    else:
        raise ValueError(f"{limiting_func} is not a valid input for `limiting_func`")

    df = xr.apply_ufunc(
        _Griffiths_drought_factor_calc,
        pr,
        smd,
        kwargs=dict(lim=lim),
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[pr.dtype],
    )

    # First non-zero entry is at the 19th time point since df is calculated
    # from a 20 day rolling window
    return df.isel(time=slice(19, None))


# @declare_units(
#     D="[precipitation]",
#     T="[temperature]",
#     H="[humidity]",
#     V="[wind]",
# )
def McArthur_forest_fire_danger_index(
    D: xr.DataArray,
    T: xr.DataArray,
    H: xr.DataArray,
    V: xr.DataArray,
):
    """
    Calculate the McArthur forest fire danger index (FFDI) Mark 5.

    This method calculates the FFDI using the formula in :cite:t:`fire-noble_1980`.

    Parameters
    ----------
    D : xr.DataArray
        The drought factor to use in the FFDI calculation. Often the daily Griffiths
        drought factor (see `Griffiths_drought_factor`).
    T : xr.DataArray
        The temperature to use in the FFDI calculation. Often the current or previous
        day's maximum daily temperature near the surface.
    H : xr.DataArray
        The relative humidity to use in the FFDI calculation. Often the relative humidity
        near the time of the maximum temperature used for T.
    V : xr.DataArray
        The wind velocity to use in the FFDI calculation. Often mid-afternoon wind speed
        at a height of 10 m.

    Returns
    -------
    ffdi : xr.DataArray
        The McArthur forest fire danger index.

    References
    ----------
    :cite:t:`fire-noble_1980,fire-dowdy_2018,fire-holgate_2017`
    """
    # D = convert_units_to(D, "mm")
    # T = convert_units_to(T, "C")
    # H = convert_units_to(H, "%")
    # V = convert_units_to(V, "km/h")

    return D**0.987 * np.exp(0.0338 * T - 0.0345 * H + 0.0234 * V + 0.243147)
