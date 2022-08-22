# noqa: D205,D400
r"""
=============================================
McArthur Forest Fire Danger Indices Submodule
=============================================

This submodule defines the :py:func:`xclim.indices.Keech_Byram_drought_index`, 
:py:func:`xclim.indices.Griffiths_drought_factor` and
:py:func:`xclim.indices.McArthur_forest_fire_danger_index` indices, which are used by the eponym indicators.
Users should read this module's documentation and consult the :cite:t:`fire-finkele_2006` which provides
details of the methods used to calculate each index.
"""
# This file is structured in the following way:
# Section 1: individual codes, numba-accelerated and vectorized functions.
# Section 2: Larger computing functions (the KBDI iterator)
# Section 3: Exposed methods and indices.
#
# Methods starting with a "_" are not usable with xarray objects, whereas the others are.
import numpy as np
import xarray as xr
from numba import guvectorize, float64

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
def _Keech_Byram_drought_index(p, t, pa, rr0, kbdi0, rr, kbdi):  # pragma: no cover
    """
    Compute the Keech-Byram drought index over one time step
    following :cite:p:`fire-finkele_2006`

    Parameters
    ----------
    p : float
        Total rainfall over previous 24 hours, at 9am [mm].
    t: float
        Maximum temperature over previous 24 hours, at 9am [C].
    pa: float
        Mean annual accumulated rainfall
    rr0: float
        Remaining rainfall to be assigned to runoff from previous iteration.
        Runoff is approximated as the first 5 mm of rain within consecutive
        days with nonzero rainfall.
    kbdi0 : float
        Previous value of the Keech-Byram drought index.

    Returns
    -------
    rr : array_like
        Remaining rainfall to be assigned to runoff.
    kbdi : array_like
        Keech-Byram drought index at 9am.
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

    if kbdi_curr > 200.0:
        kbdi_curr = 200.0

    rr[0] = rr0 - r
    kbdi[0] = kbdi_curr


# SECTION 2 : Iterators


def _Keech_Byram_drought_index_calc(pr, tasmax, pr_annual, kbdi0):
    """
    Primary function computing the Keech-Byram drought index.
    DO NOT CALL DIRECTLY, use `Keech_Byram_drought_index` instead.
    """
    kbdi = np.zeros_like(pr)
    runoff_remain = 5.0
    kbdi_prev = kbdi0

    for it in range(pr.shape[-1]):
        runoff_remain, kbdi[..., it] = _Keech_Byram_drought_index(
            pr[..., it],
            tasmax[..., it],
            pr_annual,
            runoff_remain,
            kbdi_prev,
        )
        kbdi_prev = kbdi[..., it]

    return kbdi


# SECTION 3 - Public methods and indices


# @declare_units(
#     pr="[precipitation]",
#     tasmax="[temperature]",
#     pr_annual="[precipitation]",
# )
def Keech_Byram_drought_index(
    pr: xr.DataArray,
    tasmax: xr.DataArray,
    pr_annual: xr.DataArray,
    kbdi0: xr.DataArray | None = None,
):
    """
    Calculate the Keetch-Byram drought index (KBDI), defined as:

        KBDI_n = KBDI_n-1 − Peff + ET

    Peff is the previous 24-hour rainfall amount, pr_n-1, decreased by an amount to allow
    for interception and/or runoff:

        Peff = pr_n-1 - (interception/runoff)

    where the interception and/or runoff is approximated as the first 5 mm within consecutive
    days with nonzero rainfall.

    ET is the evapotransporation, estimated as:

        ET = (203.2 - KBDI_n-1) * (0.968 * exp(0.0875 * tasmax_n-1 + 1.5552) - 8.3)
             ---------------------------------------------------------------------- * 10 ** (-3)
                          1 + 10.88 * exp(-0.00173 * pr_annual)

    where tasmax_n-1 is the previous day's max temperature and pr_annual is the annual accumulated
    rainfall averaged over a number of years.

    Parameters
    ----------
    pr : xr.DataArray
        Total rainfall over previous 24 hours, at 9am.
    tasmax : xr.DataArray
        Maximum temperature over previous 24 hours, at 9am.
    pr_annual: xr.DataArray
        Mean annual accumulated rainfall
    kbdi0 : xr.DataArray, optional
        Previous KBDI map used to initialise the KBDI calculation. Defaults to 0.

    References
    ----------
    Keetch & Byram 1968 (on calculation):
        https://www.srs.fs.usda.gov/pubs/viewpub.php?index=40
    Finkele et al. 2006 (on calculation):
        https://webarchive.nla.gov.au/awa/20060903105143/http://www.bom.gov.au/bmrc/pubs/researchreports/RR119.pdf
    Holgate et al. 2017 (on calculation):
        https://www.publish.csiro.au/wf/WF16217
    Dolling et al. 2005 (on initialisation):
        https://www.sciencedirect.com/science/article/pii/S0168192305001802#bib5
    """
    # pr = convert_units_to(pr, "mm/day")
    # tasmax = convert_units_to(tasmax, "C")
    # pr_annual = convert_units_to(pr_annual, "mm/day")

    if kbdi0 is None:
        kbdi0 = xr.full_like(pr.isel(time=0), 0)

    kbdi = xr.apply_ufunc(
        _Keech_Byram_drought_index_calc,
        pr,
        tasmax,
        pr_annual,
        kbdi0,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[pr.dtype],
    )

    return kbdi