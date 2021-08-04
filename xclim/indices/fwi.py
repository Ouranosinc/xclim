# noqa: D205,D400
"""
==============================
Fire Weather Indices Submodule
==============================

This submodule defines the :py:func:`xclim.indices.fire_season`, :py::func:`xclim.indices.drought_code`
and :py:func:`xclim.indices.fire_weather_indexes` indices, which are used by the eponym indicators.
Users should read this module's documentation and the one of `fire_weather_ufunc`.

First adapted from Matlab code `CalcFWITimeSeriesWithStartup.m` from GFWED made for using
MERRA2 data, which was a translation of FWI.vba of the Canadian Fire Weather Index system.
Then, updated and synchronized with the R code of the cffdrs package. When given the correct parameters,
the current code has an error below 3% when compared with the [GFWED]_ data.

Parts of the code and of the documentation in this submodule are directly taken from [cffdrs] which was published with the GPL-2 license.

Fire season
-----------
Fire weather indexes are iteratively computed, each day's value depending on the previous day indexes.
Additionally and optionally, the codes are "shut down" (set to NaN) in winter. There are a few ways of computing this
shut down and the subsequent spring start up. The `fire_season` function allows for full control of that,
replicating the `fireSeason` method in the R package. It produces a mask to be given a `season_mask` in the
indicators. However, the `fire_weather_ufunc` and the indicators also accept a `season_method` parameter so the
fire season can be computed inside the iterator. Passing `season_method=None` switches to an "always on" mode
replicating the `fwi` method of the R package.

The fire season determination is based on three consecutive daily maximum temperature thresholds ([WF93]_ , [LA08]_).
A "GFWED" method is also implemented. There, the 12h LST temperature is used instead of the daily maximum.
The current implementation is slightly different from the description in [GFWED]_, but it replicates the Matlab code
when `temp_start_thresh` and `temp_end_thresh` are both set to 6 degC.
In xclim, the number of consecutive days, the start and end temperature thresholds and the snow depth threshold
can all be modified.

Overwintering
-------------
Additionnaly, overwintering of the drought code is also directly implemented in :py:func:`fire_weather_ufunc`.
The last drought_code of the season is kept in "winter" (where the fire season mask is False) and the precipitation
is accumulated until the start of the next season. The first drought code is computed as a function of these instead
of using the default DCStart value. Parameters to :py:func:`_overwintering_drought_code` are listed below.
The code for the overwintering is based on [ME19]_.

Finally, a mechanism for dry spring starts is implemented. For now, it is slightly different from what the GFWED, uses, but
seems to agree with the state of the science of the CFS. When activated, the drought code and Duff-moisture codes are started
in spring with a value that is function of the number of days since the last significative precipitation event.
The conventionnal start value increased by that number of days times a "dry start" factor. Parameters are controlled in
the call of the indices and :py:func:`fire_weather_ufunc`. Overwintering of the drought code overrides this mechanism if both are activated.
GFWED use a more complex approach with an added check on the previous day's snow cover for determining "dry" points. Moreover,
there, the start values are only the multiplication of a factor to the number of dry days, the conventionnal

Examples
--------
The current litterature seems to agree that climate-oriented series of the fire weather indexes should be computed
using only the longest fire season of each year and activatting the overwintering of the drought code and
the "dry start" for the duff-moisture code. The following example uses reasonable parameters when computing over all of Canada.


**Note:** here the example snippets use the _indices_ defined in this very module, but we always recommend using the _indicators_
defined in the `xc.atmos` module.

>>> ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
>>> ds = ds.assign(
        hurs=xclim.atmos.relative_humidity_from_dewpoint(ds=ds),
        tas=xclim.core.units.convert_units_to(ds.tas, "degC"),
        pr=xclim.core.units.convert_units_to(ds.pr, "mm/d"),
        sfcWind=xclim.atmos.wind_speed_from_vector(ds=ds)[0]
    )
>>> season_mask = fire_season(
        tas=ds.tas,
        method="WF93",
        freq="YS",
        # Parameters below are at their default values, but listed here for explicitness.
        temp_start_thresh="12 degC",
        temp_end_thresh="5 degC",
        temp_condition_days=3,
    )
>>> out_fwi = fire_weather_indexes(
        tas=ds.tas,
        pr=ds.pr,
        hurs=ds.hurs,
        sfcWind=ds.sfcWind,
        lat=ds.lat,
        season_mask=season_mask,
        overwintering=True,
        dry_start="CFS",
        prec_thresh="1.5 mm/d",
        dmc_dry_factor=1.2,
        # Parameters below are at their default values, but listed here for explicitness.
        carry_over_fraction=0.75,
        wetting_efficiency_fraction=0.75,
        dc_start=15,
        dmc_start=6,
        ffmc_start=85,
    )

Similarly, the next lines calculate the fire weather indexes, but according to the parameters and options
used in NASA's GFWED datasets. Here, no need to split the fire season mask from the rest of the computation
as _all_ seasons are used, even the very short shoulder seasons.

>>> ds = open_dataset("FWI/GFWED_sample_2017.nc")
>>> out_fwi = fire_weather_indexes(
        tas=ds.tas,
        pr=ds.prbc,
        snd=ds.snow_depth,
        hurs=ds.rh,
        sfcWind=ds.sfcwind,
        lat=ds.lat,
        season_method="GFWED",
        overwintering=False,
        dry_start="GFWED",
        temp_start_thresh="6 degC",
        temp_end_thresh="6 degC",
        # Parameters below are at their default values, but listed here for explicitness.
        temp_condition_days=3,
        snow_condition_days=3,
        dc_start=15,
        dmc_start=6,
        ffmc_start=85,
        dmc_dry_factor=2,
    )

References
----------
Codes:

.. [CFS2015] Updated source code for calculating fire danger indexes in the Canadian Forest Fire Weather Index System, Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.
.. [cffdrs] Cantin, A., Wang, X., Parisien M-A., Wotton, M., Anderson, K., Moore, B., Schiks, T., Flannigan, M., Canadian Forest Fire Danger Rating System, R package, CRAN, https://cran.r-project.org/package=cffdrs

https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi

Matlab code of the GFWED obtained through personal communication.

Fire season determination methods:

.. [WF93] Wotton, B.M. and Flannigan, M.D. (1993). Length of the fire season in a changing climate. ForestryChronicle, 69, 187-192.
.. [LA08] Lawson B.D. and Armitage O.B. 2008. Weather Guide for the Canadian Forest Fire Danger RatingSystem. Natural Resources Canada, Canadian Forest Service, Northern Forestry Centre, Edmonton,Alberta. 84 p.http://cfs.nrcan.gc.ca/pubwarehouse/pdfs/29152.pdf
.. [GFWED] Field, R. D., Spessa, A. C., Aziz, N. A., Camia, A., Cantin, A., Carr, R., de Groot, W. J., Dowdy, A. J., Flannigan, M. D., Manomaiphiboon, K., Pappenberger, F., Tanpipat, V., and Wang, X. (2015) Development of a Global Fire Weather Database, Nat. Hazards Earth Syst. Sci., 15, 1407–1423, https://doi.org/10.5194/nhess-15-1407-2015

Drought Code overwintering:

.. [VW85] Van Wagner, C.E. 1985. Drought, timelag and fire danger rating. Pages 178-185 in L.R. Donoghueand R.E. Martin, eds.  Proc.  8th Conf.  Fire For.  Meteorol., 29 Apr.-3 May 1985, Detroit, MI. Soc.Am. For., Bethesda, MD.http://cfs.nrcan.gc.ca/pubwarehouse/pdfs/23550.pd
.. [ME19] McElhinny, M., Beckers, J. F., Hanes, C., Flannigan, M., and Jain, P.: A high-resolution reanalysis of global fire weather from 1979 to 2018 – overwintering the Drought Code, Earth Syst. Sci. Data, 12, 1823–1833, https://doi.org/10.5194/essd-12-1823-2020, 2020.
"""
# This file is structured in the following way:
# Section 1: individual codes, numba-accelerated and vectorized functions.
# Section 2: Larger computing functons (the FWI iterator and the fire_season iterator)
# Section 3: Exposed methods and indices.
#
# Methods starting with a "_" are not usable with xarray objects, whereas the others are.
from collections import OrderedDict
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr
from numba import jit, vectorize

from xclim.core.units import convert_units_to, declare_units

from . import run_length as rl

default_params = dict(
    temp_start_thresh=(12, "degC"),
    temp_end_thresh=(5, "degC"),
    snow_thresh=(0.01, "m"),
    temp_condition_days=3,
    snow_condition_days=3,
    carry_over_fraction=0.75,
    wetting_efficiency_fraction=0.75,
    dc_start=15,
    dmc_start=6,
    ffmc_start=85,
    prec_thresh=(1.0, "mm/d"),
    dc_dry_factor=5,
    dmc_dry_factor=2,
    snow_cover_days=60,
    snow_min_cover_frac=0.75,
    snow_min_mean_depth=(0.1, "m"),
)
"""
Default values for numerical parameters of fire_weather_ufunc.

Parameters with units are given as a tuple of default value and units.
A more complete explanation of these parameters is given in the doc of :py:func:`fire_weather_ufunc`.
"""

# SECTION 1 - Codes - Numba accelerated and vectorized functions

# Values taken from GFWED code
DAY_LENGTHS = np.array(
    [
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8],
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
        12 * [9],
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
        [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6],
    ]
)

DAY_LENGTH_FACTORS = np.array(
    [
        [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
        12 * [1.39],
        [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
    ]
)


@jit
def _day_length(lat: Union[int, float], mth: int):  # pragma: no cover
    """Return the average day length for a month within latitudinal bounds."""
    if -30 > lat >= -90:
        dl = DAY_LENGTHS[0, :]
    elif -15 > lat >= -30:
        dl = DAY_LENGTHS[1, :]
    elif 15 > lat >= -15:
        return 9
    elif 30 > lat >= 15:
        dl = DAY_LENGTHS[3, :]
    elif 90 >= lat >= 30:
        dl = DAY_LENGTHS[4, :]
    elif lat > 90 or lat < -90:
        raise ValueError("Invalid lat specified.")
    else:
        raise ValueError
    return dl[mth - 1]


@jit
def _day_length_factor(lat: float, mth: int):  # pragma: no cover
    """Return the day length factor."""
    if -15 > lat >= -90:
        dlf = DAY_LENGTH_FACTORS[0, :]
    elif 15 > lat >= -15:
        return 1.39
    elif 90 >= lat >= 15:
        dlf = DAY_LENGTH_FACTORS[2, :]
    elif lat > 90 or lat < -90:
        raise ValueError("Invalid lat specified.")
    else:
        raise ValueError
    return dlf[mth - 1]


@vectorize
def _fine_fuel_moisture_code(t, p, w, h, ffmc0):  # pragma: no cover
    """Compute the fine fuel moisture code over one time step.

    Parameters
    ----------
    t: array
      Noon temperature [C].
    p : array
      Rain fall in open over previous 24 hours, at noon [mm].
    w : array
      Noon wind speed [km/h].
    h : array
      Noon relative humidity [%].
    ffmc0 : array
      Previous value of the fine fuel moisture code.

    Returns
    -------
    array
      Fine fuel moisture code at the current timestep.
    """
    mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0)  # *Eq.1*#
    if p > 0.5:
        rf = p - 0.5  # *Eq.2*#
        if mo > 150.0:
            mo = (
                mo
                + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            ) + (0.0015 * (mo - 150.0) ** 2) * np.sqrt(rf)
            # *Eq.3b*#
        elif mo <= 150.0:
            mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (
                1.0 - np.exp(-6.93 / rf)
            )
            # *Eq.3a*#
        if mo > 250.0:
            mo = 250.0

    ed = (
        0.942 * (h ** 0.679)
        + (11.0 * np.exp((h - 100.0) / 10.0))
        + 0.18 * (21.1 - t) * (1.0 - 1.0 / np.exp(0.1150 * h))
    )  # *Eq.4*#

    if mo < ed:
        ew = (
            0.618 * (h ** 0.753)
            + (10.0 * np.exp((h - 100.0) / 10.0))
            + 0.18 * (21.1 - t) * (1.0 - 1.0 / np.exp(0.115 * h))
        )  # *Eq.5*#
        if mo < ew:
            # *Eq.7a*#
            kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + (
                0.0694 * np.sqrt(w)
            ) * (1.0 - ((100.0 - h) / 100.0) ** 8)
            kw = kl * (0.581 * np.exp(0.0365 * t))  # *Eq.7b*#
            m = ew - (ew - mo) / 10.0 ** kw  # *Eq.9*#
        elif mo > ew:
            m = mo
    elif mo == ed:
        m = mo
    else:
        kl = 0.424 * (1.0 - (h / 100.0) ** 1.7) + (0.0694 * np.sqrt(w)) * (
            1.0 - (h / 100.0) ** 8
        )  # *Eq.6a*#
        kw = kl * (0.581 * np.exp(0.0365 * t))  # *Eq.6b*#
        m = ed + (mo - ed) / 10.0 ** kw  # *Eq.8*#

    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)  # *Eq.10*#
    if ffmc > 101.0:
        ffmc = 101.0
    elif ffmc <= 0.0:
        ffmc = 0.0

    return ffmc


@vectorize
def _duff_moisture_code(t, p, h, mth: int, lat: float, dmc0: float):  # pragma: no cover
    """Compute the Duff moisture code over one time step.

    Parameters
    ----------
    t: array
      Noon temperature [C].
    p : array
      Rain fall in open over previous 24 hours, at noon [mm].
    h : array
      Noon relative humidity [%].
    mth : integer array
      Month of the year [1-12].
    lat : float
      Latitude.
    dmc0 : float
      Previous value of the Duff moisture code.

    Returns
    -------
    array
      Duff moisture code at the current timestep
    """
    if np.isnan(dmc0):
        return np.nan

    dl = _day_length(lat, mth)

    if t < -1.1:
        rk = 0
    else:
        rk = 1.894 * (t + 1.1) * (100.0 - h) * dl * 0.0001  # *Eqs.16 and 17*#

    if p > 1.5:
        ra = p
        rw = 0.92 * ra - 1.27  # *Eq.11*#
        wmi = 20.0 + 280.0 / np.exp(
            0.023 * dmc0
        )  # *Eq.12*#  This line replicates cffdrs (R code from CFS)
        # wmi = 20.0 + np.exp(5.6348 - dmc0 / 43.43)  # *Eq.12*# This line repliacates GFWED (Matlab code)
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)  # *Eq.13a*#
        else:
            if dmc0 <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc0)  # *Eq.13b*#
            else:
                b = 6.2 * np.log(dmc0) - 17.2  # *Eq.13c*#
        wmr = wmi + (1000 * rw) / (48.77 + b * rw)  # *Eq.14*#
        pr = 43.43 * (5.6348 - np.log(wmr - 20.0))  # *Eq.15*# cffdrs R cfs
        # pr = 244.72 - 43.43 * np.log(wmr - 20.0)  # *Eq.15*# GFWED Matlab
    else:  # p <= 1.5
        pr = dmc0

    if pr < 0.0:
        pr = 0.0
    dmc = pr + rk
    if dmc < 0:
        dmc = 0.0
    return dmc


@vectorize
def _drought_code(t, p, mth, lat, dc0):  # pragma: no cover
    """Compute the drought code over one time step.

    Parameters
    ----------
    t: array
      Noon temperature [C].
    p : array
      Rain fall in open over previous 24 hours, at noon [mm].
    mth : integer array
      Month of the year [1-12].
    lat : float
      Latitude.
    dc0 : float
      Previous value of the drought code.

    Returns
    -------
    array
      Drought code at the current timestep
    """
    fl = _day_length_factor(lat, mth)

    if t < -2.8:
        t = -2.8
    pe = (0.36 * (t + 2.8) + fl) / 2  # *Eq.22*#
    if pe < 0.0:
        pe = 0.0

    if p > 2.8:
        ra = p
        rw = 0.83 * ra - 1.27  # *Eq.18*#  Rd
        smi = 800.0 * np.exp(-dc0 / 400.0)  # *Eq.19*# Qo
        dr = dc0 - 400.0 * np.log(1.0 + ((3.937 * rw) / smi))  # *Eqs. 20 and 21*#
        if dr > 0.0:
            dc = dr + pe
        elif np.isnan(dc0):
            dc = np.NaN
        else:
            dc = pe
    else:  # f p <= 2.8:
        dc = dc0 + pe
    return dc


def initial_spread_index(ws, ffmc):
    """Initialize spread index.

    Parameters
    ----------
    ws : array
      Noon wind speed [km/h].
    ffmc : array
      Fine fuel moisture code.

    Returns
    -------
    array
      Initial spread index.
    """
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)  # *Eq.1*#
    ff = 19.1152 * np.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)  # *Eq.25*#
    isi = ff * np.exp(0.05039 * ws)  # *Eq.26*#
    return isi


def build_up_index(dmc, dc):
    """Build-up index.

    Parameters
    ----------
    dmc : array
      Duff moisture code.
    dc : array
      Drought code.

    Returns
    -------
    array
      Build up index.
    """
    bui = np.where(
        dmc <= 0.4 * dc,
        (0.8 * dc * dmc) / (dmc + 0.4 * dc),  # *Eq.27a*#
        dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7),
    )  # *Eq.27b*#
    return np.clip(bui, 0, None)


def fire_weather_index(isi, bui):
    """Fire weather index.

    Parameters
    ----------
    isi : array
      Initial spread index
    bui : array
      Build up index.

    Returns
    -------
    array
      Build up index.
    """
    fwi = np.where(
        bui <= 80.0,
        0.1 * isi * (0.626 * bui ** 0.809 + 2.0),  # *Eq.28a*#
        0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))),
    )  # *Eq.28b*#
    fwi[fwi > 1] = np.exp(2.72 * (0.434 * np.log(fwi[fwi > 1])) ** 0.647)  # *Eq.30b*#
    return fwi


def daily_severity_rating(fwi):
    """Daily severity rating.

    Parameters
    ----------
    fwi : array
      Fire weather index

    Returns
    -------
    array
      Daily severity rating.
    """
    return 0.0272 * fwi ** 1.77


@vectorize
def _overwintering_drought_code(DCf, wpr, a, b, minDC):  # pragma: no cover
    """Compute the season-starting drought code based on the previous season's last drought code and the total winter precipitation.

    Parameters
    ----------
    DCf : ndarray
      The previous season's last drought code
    wpr : ndarray
      The accumulated precipitation since the end of the fire season.
    carry_over_fraction : float
    wetting_efficiency_fraction: float
    minDC : int
      The overwintered DC cannot be below this value, usually the normal "dc_start" value.
    """
    if np.isnan(DCf) or np.isnan(wpr):
        return np.nan
    Qf = 800 * np.exp(-DCf / 400)
    Qs = a * Qf + b * (3.94 * wpr)
    DCs = 400 * np.log(800 / Qs)
    if DCs < minDC:
        DCs = minDC
    return DCs


# SECTION 2 : Iterators


def _fire_season(
    tas: np.ndarray,
    snd: Optional[np.ndarray] = None,
    method: str = "WF93",
    temp_start_thresh: float = default_params["temp_start_thresh"][0],
    temp_end_thresh: float = default_params["temp_end_thresh"][0],
    temp_condition_days: int = default_params["temp_condition_days"],
    snow_condition_days: int = default_params["snow_condition_days"],
    snow_thresh: float = default_params["snow_thresh"][0],
):
    """Compute the active fire season mask.

    Parameters
    ----------
    tas : ndarray
      Temperature [degC], the time axis on the last position.
    snd : ndarray, optional
      Snow depth [m], time axis on the last position, used with method == 'LA08'.
    method : {"WF93", "LA08", "GFWED"}
      Which method to use.
    temp_start_thresh : float
    temp_end_thresh : float
    temp_condition_days : int
    snow_condition_days : int
    snow_thresh : float
      Numerical parameters of the methods.

    Returns
    -------
    season_mask : ndarray (bool)
      True where the fire season is active, same shape as tas.
    """
    season_mask = np.full_like(tas, False, dtype=bool)

    if method == "WF93":
        # In WF93, the check is done the N last days, EXCLUDING the current one.
        start_index = temp_condition_days + 1
    elif method in ["LA08", "GFWED"]:
        # In LA08, the check INCLUDES the current day,
        start_index = max(temp_condition_days, snow_condition_days)

    for it in range(start_index, tas.shape[-1]):
        if method == "WF93":
            temp = tas[..., it - temp_condition_days : it]

            # Start up when the last X days were all above a threshold.
            start_up = np.all(temp > temp_start_thresh, axis=-1)
            # Shut down when the last X days were all below a threshold
            shut_down = np.all(temp < temp_end_thresh, axis=-1)

        elif method == "LA08":
            snow = snd[..., it - snow_condition_days + 1 : it + 1]
            temp = tas[..., it - temp_condition_days + 1 : it + 1]

            # Start up when the last X days including today have no snow on the ground.
            start_up = np.all(snow <= snow_thresh, axis=-1)
            # Shut down when today has snow OR the last X days (including today) were all below a threshold.
            shut_down = (snd[..., it] > snow_thresh) | np.all(
                temp < temp_end_thresh, axis=-1
            )

        elif method == "GFWED":
            msnow = np.mean(snd[..., it - snow_condition_days + 1 : it + 1], axis=-1)
            mtemp = np.mean(tas[..., it - temp_condition_days + 1 : it + 1], axis=-1)

            # Start up when the last X days including today have no snow on the ground.
            start_up = (mtemp > temp_start_thresh) & (msnow < snow_thresh)

            # Shut down when mean snow OR mean temp are over/under threshold
            shut_down = (msnow >= snow_thresh) | (mtemp < temp_end_thresh)

        # Mask is on if the previous days was on OR is there is a start up,  AND if it's not a shut down,
        # Aka is off if either the previous day was or it is a shut down.
        season_mask[..., it] = (season_mask[..., it - 1] | start_up) & ~shut_down

    return season_mask


def _fire_weather_calc(
    tas, pr, rh, ws, snd, mth, lat, season_mask, dc0, dmc0, ffmc0, winter_pr, **params
):
    """Primary function computing all Fire Weather Indexes. DO NOT CALL DIRECTLY, use `fire_weather_ufunc` instead."""
    # Dear code reader, sorry.
    outputs = params["outputs"]
    ind_prevs = {"DC": dc0.copy(), "DMC": dmc0.copy(), "FFMC": ffmc0.copy()}

    season_method = params.get("season_method")
    if season_method is None:
        # None means "always on"
        season_mask = np.full_like(tas, True, dtype=bool)
        # Start with default value
        ind_prevs["DC"][np.isnan(dc0)] = params["dc_start"]
        ind_prevs["DMC"][np.isnan(dmc0)] = params["dmc_start"]
        ind_prevs["FFMC"][np.isnan(ffmc0)] = params["ffmc_start"]
    elif season_method != "mask":
        # "mask" means it was passed as an arg. Other values are methods so we compute.
        season_mask = _fire_season(
            tas,
            snd,
            method=season_method,
            temp_start_thresh=params["temp_start_thresh"],
            temp_end_thresh=params["temp_end_thresh"],
            snow_thresh=params["snow_thresh"],
            temp_condition_days=params["temp_condition_days"],
            snow_condition_days=params["snow_condition_days"],
        )

    # Codes are only computed if they are in "outputs"
    for ind in list(ind_prevs.keys()):
        if ind not in outputs:
            ind_prevs.pop(ind)

    # Outputs as a dict for easier access, but order is important in the return
    out = OrderedDict()
    for name in outputs:
        if name == "winter_pr":
            # If winter_pr was requested, it should have been given.
            out[name] = winter_pr.copy()
        elif name == "season_mask":
            # If the mask was requested as output, put the one given or computed.
            out[name] = season_mask
        else:
            # Start with NaNs
            out[name] = np.full_like(tas, np.nan)

    # Cast the mask as integers, use smallest dtype for memory purposes. (maybe this is not impact on performance?)
    season_mask = season_mask.astype(np.int16)

    overwintering = params["overwintering"]
    dry_start = params["dry_start"]
    if overwintering and "DC" in ind_prevs:
        # In overwintering, dc0 is understood as the previous season's last DC code.
        ow_DC = dc0.copy()
        ind_prevs["DC"] = np.full_like(dc0, np.nan)

    if dry_start:
        ow_DC = dc0.copy()
        ow_DMC = dmc0.copy()
        start_up_wet = np.zeros_like(
            dmc0, dtype=bool
        )  # Pre allocate to avoid "unboundlocalerror"

    # Iterate on all days.
    for it in range(tas.shape[-1]):
        if season_method is not None:
            # Not in the always on mode, thus we must care about start up and shut downs of the fire season.
            if it == 0:
                if params["initial_start_up"]:
                    # As if the previous iteration was all 0s
                    delta = season_mask[..., it]
                else:
                    # Continue the previous state
                    # Meant for special corner cases when we use the season mask but some points are already "on" on the first day AND we know previous DC DMC and FFMC.
                    delta = 0 * season_mask[..., it]
            else:
                delta = season_mask[..., it] - season_mask[..., it - 1]

            # In [ME19], there are the 4 cases (in order), no need for the last one, it is implicit.
            shut_down = delta == -1
            winter = (delta == 0) & (season_mask[..., it] == 0)
            start_up = delta == 1
            # active_season = (delta == 0) & (season_mask[it] == 1)

            if dry_start:
                # When we use special start values for dry cells
                # Cells where the current precipitation is significant
                wetpts = pr[..., it] > params["prec_thresh"]

                if "SNOW" in dry_start and it >= params["snow_cover_days"]:
                    # This is for the GFWED mode with snow
                    snow_cover_history = snd[
                        ..., it - params["snow_cover_days"] + 1 : it + 1
                    ]
                    snow_days = np.count_nonzero(
                        snow_cover_history > params["snow_thresh"], axis=-1
                    )

                    # Points where the snow cover is enough to trigger a "wet" start up.
                    start_up_wet = (
                        start_up
                        & (
                            snow_days / params["snow_cover_days"]
                            >= params["snow_min_cover_frac"]
                        )
                        & (
                            snow_cover_history.mean(axis=-1)
                            >= params["snow_min_mean_depth"]
                        )
                    )

            if "DC" in ind_prevs:
                if overwintering:
                    # Store end of season DC.
                    ow_DC[shut_down] = ind_prevs["DC"][shut_down]
                    # Fist day of winter, put current precip.
                    out["winter_pr"][shut_down] = pr[shut_down, it]
                    # Winter, add current precip.
                    out["winter_pr"][winter] = out["winter_pr"][winter] + pr[winter, it]

                    dc0 = ow_DC[start_up]
                    # Where ow_DC was NaN (happens at the start of the first season when no ow_DC was given in input),
                    # put the default start,
                    ind_prevs["DC"][start_up] = np.where(
                        np.isnan(dc0),
                        params["dc_start"],
                        _overwintering_drought_code(
                            dc0,
                            out["winter_pr"][start_up],
                            params["carry_over_fraction"],
                            params["wetting_efficiency_fraction"],
                            params["dc_start"],
                        ),
                    )
                    # Put NaN to be explicit.
                    ow_DC[start_up] = np.nan
                    out["winter_pr"][start_up] = np.nan
                elif dry_start:
                    # Dry start up for DC is overrided by overwintering.
                    ow_DC[shut_down] = params["dc_start"]

                    if "GFWED" in dry_start:
                        # The GFWED includes the current day in the "wet points" check.
                        ow_DC[(start_up | winter) & wetpts] = 0
                        ow_DC[(start_up | winter) & ~wetpts] = (
                            ow_DC[(start_up | winter) & ~wetpts]
                            + params["dc_dry_factor"]
                        )
                    else:  # "CFS"
                        ow_DC[winter & wetpts] = params["dc_start"]
                        ow_DC[winter & ~wetpts] = (
                            ow_DC[winter & ~wetpts] + params["dc_dry_factor"]
                        )

                    if "SNOW" in dry_start:
                        # Points where we have start up and where snow cover was enough
                        # We cancel dry dc accumulation and switch to conventionnal
                        ow_DC[start_up_wet] = params["dc_start"]
                    ind_prevs["DC"][start_up] = ow_DC[start_up]
                    ow_DC[start_up] = np.nan
                else:
                    ind_prevs["DC"][start_up] = params["dc_start"]
                ind_prevs["DC"][shut_down] = np.nan

            if "DMC" in ind_prevs:
                if dry_start:
                    ow_DMC[shut_down] = params["dmc_start"]

                    if "GFWED" in dry_start:
                        # The GFWED includes the current day in the "wet points" check.
                        ow_DMC[(start_up | winter) & wetpts] = 0
                        ow_DMC[(start_up | winter) & ~wetpts] = (
                            ow_DMC[(start_up | winter) & ~wetpts]
                            + params["dmc_dry_factor"]
                        )
                    else:  # "CFS"
                        ow_DMC[winter & wetpts] = params["dmc_start"]
                        ow_DMC[winter & ~wetpts] = (
                            ow_DMC[winter & ~wetpts] + params["dmc_dry_factor"]
                        )

                    if "SNOW" in dry_start:
                        # Points where we have start up and where snow cover was enough
                        # We cancel dry dc accumulation and switch to conventionnal
                        ow_DMC[start_up_wet] = params["dmc_start"]
                    ind_prevs["DMC"][start_up] = ow_DMC[start_up]
                    ow_DMC[start_up] = np.nan
                else:
                    ind_prevs["DMC"][start_up] = params["dmc_start"]
                ind_prevs["DMC"][shut_down] = np.nan

            if "FFMC" in ind_prevs:
                ind_prevs["FFMC"][start_up] = params["ffmc_start"]
                ind_prevs["FFMC"][shut_down] = np.nan

        # Main computation
        if "DC" in outputs:
            out["DC"][..., it] = _drought_code(
                tas[..., it], pr[..., it], mth[..., it], lat, ind_prevs["DC"]
            )
        if "DMC" in outputs:
            out["DMC"][..., it] = _duff_moisture_code(
                tas[..., it],
                pr[..., it],
                rh[..., it],
                mth[..., it],
                lat,
                ind_prevs["DMC"],
            )
        if "FFMC" in outputs:
            out["FFMC"][..., it] = _fine_fuel_moisture_code(
                tas[..., it], pr[..., it], ws[..., it], rh[..., it], ind_prevs["FFMC"]
            )
        if "ISI" in outputs:
            out["ISI"][..., it] = initial_spread_index(
                ws[..., it], out["FFMC"][..., it]
            )
        if "BUI" in outputs:
            out["BUI"][..., it] = build_up_index(
                out["DMC"][..., it], out["DC"][..., it]
            )
        if "FWI" in outputs:
            out["FWI"][..., it] = fire_weather_index(
                out["ISI"][..., it], out["BUI"][..., it]
            )

        if "DSR" in outputs:
            out["DSR"][..., it] = daily_severity_rating(out["FWI"][..., it])

        # Set the previous values
        for ind, ind_prev in ind_prevs.items():
            ind_prev[...] = out[ind][..., it]

    if len(outputs) == 1:
        return out[outputs[0]]

    return tuple(out.values())


# SECTION 3 - Public methods and indices


def fire_weather_ufunc(
    *,
    tas: xr.DataArray,
    pr: xr.DataArray,
    hurs: Optional[xr.DataArray] = None,
    sfcWind: Optional[xr.DataArray] = None,
    snd: Optional[xr.DataArray] = None,
    lat: Optional[xr.DataArray] = None,
    dc0: Optional[xr.DataArray] = None,
    dmc0: Optional[xr.DataArray] = None,
    ffmc0: Optional[xr.DataArray] = None,
    winter_pr: Optional[xr.DataArray] = None,
    season_mask: Optional[xr.DataArray] = None,
    start_dates: Optional[Union[str, xr.DataArray]] = None,
    indexes: Sequence[str] = None,
    season_method: Optional[str] = None,
    overwintering: bool = False,
    dry_start: Optional[str] = None,
    initial_start_up: bool = True,
    **params,
):
    """Fire Weather Indexes computation using xarray's apply_ufunc.

    No unit handling. Meant to be used by power users only. Please prefer using the
    :py:indicator:`DC` and :py:indicator:`FWI` indicators or
    the :py:func:`drought_code` and :py:func:`fire_weather_indexes` indices defined in the same
    submodule.

    Dask arrays must have only one chunk along the "time" dimension.
    User can control which indexes are computed with the `indexes` argument.

    Parameters
    ----------
    tas : xr.DataArray
        Noon surface temperature in °C
    pr : xr.DataArray
        Rainfall over previous 24h, at noon in mm/day
    hurs : xr.DataArray, optional
        Noon surface relative humidity in %, not needed for DC
    sfcWind : xr.DataArray, optional
        Noon surface wind speed in km/h, not needed for DC, DMC or BUI
    snd : xr.DataArray, optional
        Noon snow depth in m, only needed if `season_method` is "LA08"
    lat : xr.DataArray, optional
        Latitude in °N, not needed for FFMC or ISI
    dc0 : xr.DataArray, optional
        Previous DC map, see Notes. Defaults to NaN.
    dmc0 : xr.DataArray, optional
        Previous DMC map, see Notes. Defaults to NaN.
    ffmc0 : xr.DataArray, optional
        Previous FFMC map, see Notes. Defaults to NaN.
    winter_pr : xr.DataArray, optional
        Accumulated precipitation since the end of the last season, until the beginning of the current data, mm/day.
        Only used if `overwintering` is True, defaults to 0.
    season_mask : xr.DataArray, optional
        Boolean mask, True where/when the fire season is active.
    indexes : Sequence[str], optional
        Which indexes to compute. If intermediate indexes are needed, they will be added to the list and output.
    season_method : {None, "WF93", "LA08", "GFWED"}
        How to compute the start up and shut down of the fire season.
        If "None", no start ups or shut downs are computed, similar to the R fwi function.
        Ignored if `season_mask` is given.
    overwintering: bool
        Whether to activate DC overwintering or not. If True, either season_method or season_mask must be given.
    dry_start: {None, 'CFS', 'GFWED'}
        Whether to activate the DC and DMC "dry start" mechanism and which method to use. See Notes.
        If overwintering is activated, it overrides this parameter : only DMC is handled through the dry start mechanism.
    initial_start_up : bool
        If True (default), gridpoints where the fire season is active on the first timestep go through a start up phase
        for that time step. Otherwise, previous codes must be given as a continuing fire season is assumed for those points.
    carry_over_fraction: float
    wetting_efficiency_fraction: float
        Drought code overwintering parameters, see :py:func:`overwintering_drought_code`.
    temp_condition_days: int
    temp_start_thresh: float
    temp_end_thresh: float
    snow_thresh: float
    snow_condition_days: int
        Parameters for the fire season determination. See :py:func:`fire_season`. Temperature is in degC, snow in m.
        The `snow_thresh` parameters is also used when `dry_start` is set to "GFWED", see Notes.
    dc_start: float
    dmc_start: float
    ffmc_start: float
        Default starting values for the three base codes.
    prec_thresh: float
        If the "dry start" is activated, this is the "wet" day precipitation threshold, see Notes. In mm/d.
    dc_dry_factor : float
        DC's start up values for the "dry start" mechanism, see Notes.
    dmc_dry_factor : float
        DMC's start up values for the "dry start" mechanism, see Notes.
    snow_cover_days: int
    snow_min_cover_frac : float
    snow_min_mean_depth : float
        Additionnal parameters for GFWED's version of the "dry start" mechanism. See Notes. Snow depth is in m.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing the computed indexes as prescribed in `indexes`, including the intermediate
        ones needed, even if they were not explicitly listed in `indexes`. When overwintering is
        activated, `winter_pr` is added. If `season_method` is not None and `season_mask` was not given,
        `season_mask` is computed on-the-fly and added to the output.

    Notes
    -----
    When overwintering is activated, the argument `dc0` is understood as last season's
    last DC map and will be used to compute the overwintered DC at the beginning of the
    next season.

    If overwintering is not activated and neither is fire season computation (`season_method`
    and `season_mask` are `None`), `dc0`, `dmc0` and `ffmc0` are understood as the codes
    on the day before the first day of FWI computation. They will default to their respective start values.
    This "always on" mode replicates the R "fwi" code.

    If the "dry start" mechanism is set to "CFS" (but there is no overwintering), the arguments `dc0` and `dmc0` are
    understood as the potential start up values from last season. With :math:`DC_{start}` the conventionnal start up value,
    :math:`F_{dry-dc}` the `dc_dry_factor` and  :math:`N_{dry}` the number of days since the last significative precipitation event,
    the start up value :math:`DC_0` is computed as:

    .. math::

        DC_0 = DC_{start} +  F_{dry-dc} * N_{dry}

    The last significative precipitation event is the last day where precipitation was greater or equal to "prec_thresh".
    The same happens for the DMC, with corresponding parameters. If overwintering is activated, this mechnanism is only used for the DMC.

    Alternatively, `dry_start` can be set to "GFWED". In this mode, the start up values are computed as:

    .. math::

        DC_0 = F_{dry-dc} * N_{dry}

    Where the current day is also included in the determination of :math:`N_{dry}` (:math:`DC_0` can thus be 0). Finally, for this "GFWED" mode,
    if snow cover is provided, a second check is performed: the dry start procedure is skipped and conventionnal start up values are used for cells
    where the snow cover of the last `snow_cover_days` was above `snow_thresh` for at least `snow_cover_days` * `snow_min_cover_frac` days and
    where the mean snow cover over the same period was greater of equal to `snow_min_mean_depth`.
    """
    indexes = set(indexes or ["DC", "DMC", "FFMC", "ISI", "BUI", "FWI", "DSR"])

    if "DSR" in indexes:
        indexes.update({"FWI"})
    if "FWI" in indexes:
        indexes.update({"ISI", "BUI"})
    if "BUI" in indexes:
        indexes.update({"DC", "DMC"})
    if "ISI" in indexes:
        indexes.update({"FFMC"})
    indexes = sorted(
        list(indexes),
        key=["DC", "DMC", "FFMC", "ISI", "BUI", "FWI", "DSR"].index,
    )

    # Whether each argument is needed in _fire_weather_calc
    # Same order as _fire_weather_calc, Assumes the list of indexes is complete.
    # (name, list of indexes + start_up/shut_down modes, has_time_dim)
    needed_args = (
        (tas, "tas", ["DC", "DMC", "FFMC", "WF93", "LA08"], True),
        (pr, "pr", ["DC", "DMC", "FFMC"], True),
        (hurs, "hurs", ["DMC", "FFMC"], True),
        (sfcWind, "sfcWind", ["FFMC"], True),
        (snd, "snd", ["LA08"], True),
        (tas.time.dt.month, "month", ["DC", "DMC"], True),
        (lat, "lat", ["DC", "DMC"], False),
    )
    # Arg order : tas, pr, hurs, sfcWind, snd, mth, lat, season_mask, dc0, dmc0, ffmc0, winter_pr
    #              0   1    2      3        4   5    6    7             8    9     10    11
    args = [None] * 12
    input_core_dims = [[]] * 12

    # Verification of all arguments
    for i, (arg, name, usedby, has_time_dim) in enumerate(needed_args):
        if any([ind in indexes + [season_method] for ind in usedby]):
            if arg is None:
                raise TypeError(
                    f"Missing input argument {name} for index combination {indexes} with fire season method '{season_method}'"
                )
            args[i] = arg
            input_core_dims[i] = ["time"] if has_time_dim else []

    # For the GFWED dry start mode, we include snow depth is available
    if snd is not None and dry_start == "GFWED":
        args[4] = snd
        input_core_dims[4] = ["time"]
        dry_start = "GFWED+SNOW"
    elif dry_start not in [None, "CFS", "GFWED"]:
        raise ValueError("'dry_start' must be one of None, 'CFS' or 'GFWED'.")

    # Always pass the previous codes.
    if dc0 is None:
        dc0 = xr.full_like(tas.isel(time=0), np.nan)
    if dmc0 is None:
        dmc0 = xr.full_like(tas.isel(time=0), np.nan)
    if ffmc0 is None:
        ffmc0 = xr.full_like(tas.isel(time=0), np.nan)
    args[8:11] = [dc0, dmc0, ffmc0]

    # Output config from the current indexes list
    outputs = indexes
    output_dtypes = [tas.dtype] * len(indexes)
    output_core_dims = len(indexes) * [("time",)]

    if season_mask is not None:
        # A mask was passed, ignore passed method and tell the ufunc to use it.
        args[7] = season_mask
        input_core_dims[7] = ["time"]
        season_method = "mask"
    elif season_method is not None:
        # Season mask not given and a method chosen : we output the computed mask.
        outputs.append("season_mask")
        output_core_dims.append(("time",))
        output_dtypes.append(bool)

    if overwintering:
        # Overwintering code activated
        if season_method is None and season_mask is None:
            raise ValueError(
                "If overwintering is activated, either `season_method` or `season_mask` must be given."
            )

        # Last winter PR is 0 by default
        if winter_pr is None:
            winter_pr = xr.zeros_like(pr.isel(time=0))

        args[11] = winter_pr

        # Activating overwintering will produce an extra output, that has no "time" dimension.
        outputs.append("winter_pr")
        output_core_dims.append([])
        output_dtypes.append(pr.dtype)

    # Kwargs from default parameters. take the value when it is a tuple.
    kwargs = {
        k: v if not isinstance(v, tuple) else v[0] for k, v in default_params.items()
    }
    kwargs.update(**params)
    kwargs.update(
        season_method=season_method,
        overwintering=overwintering,
        dry_start=dry_start,
        initial_start_up=initial_start_up,
        outputs=outputs,
    )

    if tas.ndim == 1:
        dummy_dim = xr.core.utils.get_temp_dimname(tas.dims, "dummy")
        # When arrays only have the 'time' dimension, non-temporal inputs of the wrapped ufunc
        # become scalars. We add a dummy dimension so we don't have to deal with that.
        for i in range(len(args)):
            if isinstance(args[i], xr.DataArray):
                args[i] = args[i].expand_dims({dummy_dim: [1]})

    das = xr.apply_ufunc(
        _fire_weather_calc,
        *args,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        dask_gufunc_kwargs={
            "meta": tuple(np.array((), dtype=dtype) for dtype in output_dtypes)
        },
    )

    if tas.ndim == 1:
        if len(outputs) == 1:
            das = das.squeeze(dummy_dim, drop=True)
        else:
            das = [da.squeeze(dummy_dim, drop=True) for da in das]

    if len(outputs) == 1:
        return {outputs[0]: das}
    return {name: da for name, da in zip(outputs, das)}


@declare_units(winter_pr="[length]")
def overwintering_drought_code(
    last_dc: xr.DataArray,
    winter_pr: xr.DataArray,
    carry_over_fraction: Union[xr.DataArray, float] = default_params[
        "carry_over_fraction"
    ],
    wetting_efficiency_fraction: Union[xr.DataArray, float] = default_params[
        "wetting_efficiency_fraction"
    ],
    min_dc: Union[xr.DataArray, float] = default_params["dc_start"],
) -> xr.DataArray:
    """Compute the season-starting drought code based on the previous season's last drought code and the total winter precipitation.

    This method replicates the "wDC" method of the [cffdrs]_ R package, with an added control on the "minimum" DC.

    Parameters
    ----------
    last_dc : xr.DataArray
      The previous season's last drought code.
    winter_pr : xr.DataArray
      The accumulated precipitation since the end of the fire season.
    carry_over_fraction : xr.DataArray or float
      Carry-over fraction of last fall’s moisture
    wetting_efficiency_fraction : xr.DataArray or float
      Effectiveness of winter precipitation in recharging moisture reserves in spring
    min_dc : xr.DataArray or float
      Minimum drought code starting value.

    Returns
    -------
    wDC : xr.DataArray
       Overwintered drought code.

    Notes
    -----
    Details taken from the R package documentation ([cffdrs]_):
    Of the three fuel moisture codes (i.e.  FFMC, DMC and DC) making up the FWI System,
    only the DC needs to be considered in terms of its values carrying over from one fire season to the next.
    In Canada both the FFMC and the DMC are assumed to reach moisture saturation from overwinter
    precipitation at or before spring melt; this is a reasonable assumption and any error in these
    assumed starting conditions quickly disappears.  If snowfall (or other overwinter precipitation)
    is not large enough however, the fuel layer tracked by the Drought Code may not fully reach saturation
    afterspring snow melt; because of the long response time in this fuel layer (53 days in standard
    conditions) a large error in this spring starting condition can affect the DC for a significant
    portion of the fire season. In areas where overwinter precipitation is 200 mm or more, full moisture
    recharge occurs and DC overwintering is usually unnecessary.  More discussion of overwintering and
    fuel drying time lag can be found in [LA08]_ and Van Wagner (1985)

    Carry-over fraction of last fall's moisture:
        - 1.0, Daily DC calculated up to 1 November; continuous snow cover, or freeze-up, whichever comes first
        - 0.75, Daily DC calculations stopped before any of the above conditions met or the area is subject to occasional winter chinook conditions, leaving the ground bare and subject to moisture depletion
        - 0.5,  Forested areas subject to long periods in fall or winter that favor depletion of soil moisture

    Effectiveness of winter precipitation in recharging moisture reserves in spring:
        - 0.9, Poorly drained, boggy sites with deep organic layers
        - 0.75, Deep ground frost does not occur until late fall, if at all; moderately drained sites that allow infiltration of most of the melting snowpack
        - 0.5, Chinook-prone areas and areas subject to early and deep ground frost; well-drained soils favoring rapid percolation or topography favoring rapid runoff before melting of ground frost

    Source: [LA08]_ - Table 9.
    """
    winter_pr = convert_units_to(winter_pr, "mm")

    wDC = xr.apply_ufunc(  # noqa
        _overwintering_drought_code,
        last_dc,
        winter_pr,
        carry_over_fraction,
        wetting_efficiency_fraction,
        min_dc,
        input_core_dims=[[]] * 5,
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[last_dc.dtype],
    )
    wDC.attrs["units"] = ""
    return wDC


def _convert_parameters(
    params: Mapping[str, Union[int, float]]
) -> Mapping[str, Union[int, float]]:
    for param, value in params.copy().items():
        if param not in default_params:
            raise ValueError(
                f"{param} is not a valid parameter for fire weather indices. See list in xc.indices.fwi.default_params."
            )
        elif isinstance(default_params[param], tuple):
            params[param] = convert_units_to(value, default_params[param][1])
    return params


@declare_units(
    tas="[temperature]",
    pr="[precipitation]",
    sfcWind="[speed]",
    hurs="[]",
    snd="[length]",
)
def fire_weather_indexes(
    tas: xr.DataArray,
    pr: xr.DataArray,
    sfcWind: xr.DataArray,
    hurs: xr.DataArray,
    lat: xr.DataArray,
    snd: Optional[xr.DataArray] = None,
    ffmc0: Optional[xr.DataArray] = None,
    dmc0: Optional[xr.DataArray] = None,
    dc0: Optional[xr.DataArray] = None,
    season_mask: Optional[xr.DataArray] = None,
    season_method: Optional[str] = None,
    overwintering: bool = False,
    dry_start: Optional[str] = None,
    initial_start_up: bool = True,
    **params,
):
    """Fire weather indexes.

    Computes the 6 fire weather indexes as defined by the Canadian Forest Service:
    the Drought Code, the Duff-Moisture Code, the Fine Fuel Moisture Code,
    the Initial Spread Index, the Build Up Index and the Fire Weather Index.

    Parameters
    ----------
    tas : xr.DataArray
      Noon temperature.
    pr : xr.DataArray
      Rain fall in open over previous 24 hours, at noon.
    sfcWind : xr.DataArray
      Noon wind speed.
    hurs : xr.DataArray
      Noon relative humidity.
    lat : xr.DataArray
      Latitude coordinate
    snd : xr.DataArray
      Noon snow depth, only used if `season_method='LA08'` is passed.
    ffmc0 : xr.DataArray
      Initial values of the fine fuel moisture code.
    dmc0 : xr.DataArray
      Initial values of the Duff moisture code.
    dc0 : xr.DataArray
      Initial values of the drought code.
    season_mask : xr.DataArray, optional
        Boolean mask, True where/when the fire season is active.
    season_method : {None, "WF93", "LA08", "GFWED"}
        How to compute the start-up and shutdown of the fire season.
        If "None", no start-ups or shutdowns are computed, similar to the R fwi function.
        Ignored if `season_mask` is given.
    overwintering: bool
        Whether to activate DC overwintering or not. If True, either season_method or season_mask must be given.
    dry_start: {None, 'CFS', 'GFWED'}
        Whether to activate the DC and DMC "dry start" mechanism or not, see :py:func:`fire_weather_ufunc`.
    initial_start_up : bool
        If True (default), gridpoints where the fire season is active on the first timestep go through a start_up phase
        for that time step. Otherwise, previous codes must be given as a continuing fire season is assumed for those points.
    params :
      Any other keyword parameters as defined in :py:func:`fire_weather_ufunc` and in :py:data:`default_params`.

    Returns
    -------
    DC: xr.DataArray, [dimensionless]
    DMC: xr.DataArray, [dimensionless]
    FFMC: xr.DataArray, [dimensionless]
    ISI: xr.DataArray, [dimensionless]
    BUI: xr.DataArray, [dimensionless]
    FWI: xr.DataArray, [dimensionless]

    Notes
    -----
    See https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi, the module's doc and
    doc of :py:func:`fire_weather_ufunc` for more information.

    References
    ----------
    Updated source code for calculating fire danger indexes in the Canadian Forest Fire Weather Index System, Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.
    """
    tas = convert_units_to(tas, "C")
    pr = convert_units_to(pr, "mm/day")
    sfcWind = convert_units_to(sfcWind, "km/h")
    hurs = convert_units_to(hurs, "pct")
    if snd is not None:
        snd = convert_units_to(snd, "m")

    out = fire_weather_ufunc(
        tas=tas,
        pr=pr,
        hurs=hurs,
        sfcWind=sfcWind,
        lat=lat,
        dc0=dc0,
        dmc0=dmc0,
        ffmc0=ffmc0,
        snd=snd,
        indexes=["DC", "DMC", "FFMC", "ISI", "BUI", "FWI"],
        season_mask=season_mask,
        season_method=season_method,
        overwintering=overwintering,
        dry_start=dry_start,
        initial_start_up=initial_start_up,
        **_convert_parameters(params),
    )
    for outd in out.values():
        outd.attrs["units"] = ""
    return out["DC"], out["DMC"], out["FFMC"], out["ISI"], out["BUI"], out["FWI"]


@declare_units(tas="[temperature]", pr="[precipitation]", snd="[length]")
def drought_code(
    tas: xr.DataArray,
    pr: xr.DataArray,
    lat: xr.DataArray,
    snd: Optional[xr.DataArray] = None,
    dc0: Optional[xr.DataArray] = None,
    season_mask: Optional[xr.DataArray] = None,
    season_method: Optional[str] = None,
    overwintering: bool = False,
    dry_start: Optional[str] = None,
    initial_start_up: bool = True,
    **params,
):
    r"""Drought code (FWI component).

    The drought code is part of the Canadian Forest Fire Weather Index System.
    It is a numeric rating of the average moisture content of organic layers.

    Parameters
    ----------
    tas : xr.DataArray
      Noon temperature.
    pr : xr.DataArray
      Rain fall in open over previous 24 hours, at noon.
    lat : xr.DataArray
      Latitude coordinate
    snd : xr.DataArray
      Noon snow depth.
    dc0 : xr.DataArray
      Initial values of the drought code.
    season_mask : xr.DataArray, optional
      Boolean mask, True where/when the fire season is active.
    season_method : {None, "WF93", "LA08", "GFWED"}
      How to compute the start up and shut down of the fire season.
      If "None", no start ups or shud downs are computed, similar to the R fwi function.
      Ignored if `season_mask` is given.
    overwintering: bool
      Whether to activate DC overwintering or not. If True, either season_method or season_mask must be given.
    dry_start: {None, "CFS", 'GFWED'}
        Whether to activate the DC and DMC "dry start" mechanism and which method to use.
        , see :py:func:`fire_weather_ufunc`.
    initial_start_up : bool
        If True (default), gridpoints where the fire season is active on the first timestep go through a start_up phase
        for that time step. Otherwise, previous codes must be given as a continuing fire season is assumed for those points.
    params :
      Any other keyword parameters as defined in `xclim.indices.fwi.fire_weather_ufunc` and in :py:data:`default_params`.

    Returns
    -------
    xr.DataArray, [dimensionless]
       Drought code

    Notes
    -----
    See https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi, the module's doc and
    doc of :py:func:`fire_weather_ufunc` for more information.

    References
    ----------
    Updated source code for calculating fire danger indexes in the Canadian Forest Fire Weather Index System, Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.
    """
    tas = convert_units_to(tas, "C")
    pr = convert_units_to(pr, "mm/day")
    if snd is not None:
        snd = convert_units_to(snd, "m")

    out = fire_weather_ufunc(
        tas=tas,
        pr=pr,
        lat=lat,
        dc0=dc0,
        snd=snd,
        indexes=["DC"],
        season_mask=season_mask,
        season_method=season_method,
        overwintering=overwintering,
        dry_start=dry_start,
        initial_start_up=initial_start_up,
        **_convert_parameters(params),
    )
    out["DC"].attrs["units"] = ""
    return out["DC"]


@declare_units(
    tas="[temperature]",
    snd="[length]",
)
def fire_season(
    tas: xr.DataArray,
    snd: Optional[xr.DataArray] = None,
    method: str = "WF93",
    freq: Optional[str] = None,
    **params,
):
    """Fire season mask.

    Binary mask of the active fire season, defined by conditions on consecutive daily temperatures and, optionally, snow depths.

    Parameters
    ----------
    tas : xr.DataArray
      Daily surface temperature, cffdrs recommends using maximum daily temperature.
    snd : xr.DataArray, optional
      Snow depth, used with method == 'LA08'.
    method : {"WF93", "LA08", "GFWED"}
      Which method to use. "LA08"  and "GFWED" need the snow depth.
    freq : str, optional
      If given only the longest fire season for each period defined by this frequency,
      Every "seasons" are returned if None, including the short shoulder seasons.
    temp_start_thresh: str
      Minimal temperature needed to start the season.
    temp_end_thresh : str
      Maximal temperature needed to end the season.
    temp_condition_days: int
      Number of days with temperature above or below the thresholds to trigger a start or an end of the fire season.
    snow_condition_days: int
      Number of days with snow depth above or below the threshold to trigger a start or an end of the fire season, only used with method "LA08".
    snow_thresh: str
      Minimal snow depth level to end a fire season, only used with method "LA08".

    Returns
    -------
    fire_season : xr.DataArray
      Fire season mask

    References
    ----------
    Wotton, B.M. and Flannigan, M.D. (1993). Length of the fire season in a changing climate. ForestryChronicle, 69, 187-192.
    Lawson, B.D. and O.B. Armitage. 2008. Weather guide for the Canadian Forest Fire Danger Rating System. NRCAN, CFS, Edmonton, AB
    """

    def _apply_fire_season(ds):
        season_mask = ds.tas.copy(
            data=_fire_season(
                tas=ds.tas.values,
                snd=None if method == "WF93" else ds.snd.values,
                method=method,
                **_convert_parameters(params),
            )
        )
        season_mask.attrs = {}

        if freq is not None:
            time = season_mask.time
            season_mask = season_mask.resample(time=freq).map(rl.keep_longest_run)
            season_mask["time"] = time

        return season_mask

    ds = convert_units_to(tas, "degC").rename("tas").to_dataset()
    if snd is not None:
        ds["snd"] = convert_units_to(snd, "m")
        ds = ds.unify_chunks()
    ds = ds.transpose(..., "time")

    tmpl = xr.full_like(tas, np.nan)
    out = ds.map_blocks(_apply_fire_season, template=tmpl)
    out.attrs["units"] = ""
    return out
