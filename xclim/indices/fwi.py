# noqa: D205,D400
"""
==============================
Fire Weather Indices Submodule
==============================

Methods defined here are used by :func:`xclim.indices.fire_season`, :func:`xclim.indices.drought_code` and :func:`xclim.indices.fire_weather_indexes`, and corresponding indicators.

First adapted from Matlab code `CalcFWITimeSeriesWithStartup.m` from GFWED made for using
MERRA2 data, which was a translation of FWI.vba of the Canadian Fire Weather Index system.
Then, updated and synchronized with the R code of the cffdrs package.

Fire season
-----------
Fire weather indexes are iteratively computed, each day's value depending on the previous day indexes.
Additionally, the codes are "shut down" (set to NaN) in winter. There are a few ways of computing this
shut down and the subsequent spring start up. The `fire_season` function allows for full control of that,
replication the `fireSeason` method in the R package. It produces a mask to be given a `season_mask` in the
indicators. However, the `fire_weather_ufunc` and the indicators accept a `season_method` parameter so the
fire season is computed inside the iterator. Passing `season_method=None` switched to an "always_on" mode
replicating the `fwi` method of the R package.

Additionnaly, overwintering of the drought code is activated by default with the `overwintering` keyword.
When activated, the last drought_code of the season is kept in "winter" and the precipitation is accumulated
durring the same period. The first drought code is computed as a function of these instead of using the default
DCStart value.


Parameters
----------
Default values for the following parameters are stored in the DEFAULT_PARAMS dict.

    #snowCoverDaysCalc : int
    #    Number of days prior to spring over which to determine if winter had substantial snow cover
    #minWinterSnoD : float
    #    Minimum mean depth (m) during past snowCoverDaysCalc days for winter to be considered having had substantial snow cover
    #snoDThresh : float
    #    Minimum depth (m) for there to be considered snow on ground at any given time
    #minSnowDayFrac : float
    #    Minimum fraction of days during snowCoverDaysCalc where snow cover was greater than snoDThresh for winter to be considered having had substantial snow cover
    #startShutDays : int
    #    Number of previous days over which to consider start or end of winter
    #tempThresh : float
    #    Temp thresh (C) to define start and end of winter
    #precThresh : float
    #    Min precip (mm/day) when determining if last three days had any precip
    #DCDryStartFactor : float
    #    DC number of days since precip mult factor for dry start.
    #DMCDryStartFactor : float
    #    DMC number of days since precip mult factor for dry start.
    DCStart : float
        DC starting value after wet winter
    DMCStart : float
        DMC starting value after wet winter
    FFMCStart : float
        FFMC starting value after any winter

References
----------
Updated source code for calculating fire danger indexes in the Canadian Forest Fire Weather Index System, Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.

https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi

Fire season determination methods:

Wotton, B.M. and Flannigan, M.D. (1993). Length of the fire season in a changing climate. ForestryChronicle, 69, 187-192.
Lawson, B.D. and O.B. Armitage. 2008. Weather guide for the Canadian Forest Fire Danger Rating System. NRCAN, CFS, Edmonton, AB

Drought Code overwintering:

McElhinny, M., Beckers, J. F., Hanes, C., Flannigan, M., and Jain, P.: A high-resolution reanalysis of global fire weather from 1979 to 2018 – overwintering the Drought Code, Earth Syst. Sci. Data, 12, 1823–1833, https://doi.org/10.5194/essd-12-1823-2020, 2020.

.. todo::

    Allow computation of DC/DMC/FFMC independently,
"""
from collections import OrderedDict
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from numba import jit, vectorize

DEFAULT_PARAMS = dict(
    # min_lat=-58,
    # max_lat=75,
    # minLandFrac=0.1,
    # minT=-10,
    # minPrec=0.25,
    # snowCoverDaysCalc=60,
    # minWinterSnoD=0.1,
    # snoDThresh=0.01,
    # minSnowDayFrac=0.75,
    # startShutDays=2,
    # tempThresh=6,
    # precThresh=1.0,
    # DCDryStartFactor=5,
    # DMCDryStartFactor=2,
    DCStart=15.0,
    FFMCStart=85.0,
    DMCStart=6.0,
    carry_over_fraction=0.75,
    wetting_efficiency_fraction=0.75,
    temp_condition_days=3,
    temp_start_thresh=12,
    temp_end_thresh=5,
    snow_thresh=0,
    snow_condition_days=3,
)


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
def _overwintering_drought_code(DCf, wpr, a=0.75, b=0.75, minDC=15):
    """Compute the season-starting drought code based on the previous season's last drought code and the total winter precipitation.

    Parameters
    ----------
    DCf : ndarray
      The previous season's last drought code
    wpr : ndarray
      The accumulated precipitation since the end of the fire season.
    carry_over_fraction : int
    wetting_efficiency_fraction
    """
    if np.isnan(DCf) or np.isnan(wpr):
        return np.nan
    Qf = 800 * np.exp(-DCf / 400)
    Qs = a * Qf + b * (3.94 * wpr)
    DCs = 400 * np.log(800 / Qs)
    if DCs < minDC:
        DCs = minDC
    return DCs


def _fire_season(
    tas,
    snd=None,
    method="WF93",
    temp_start_thresh=12,
    temp_end_thresh=5,
    temp_condition_days=3,
    snow_condition_days=3,
    snow_thresh=0,
):
    """Compute the active fire season mask.

    Parameters
    ----------
    tas : ndarray
      Temperature [degC], the time axis on the last position.
    snd : ndarray, optional
      Snow depth [m], time axis on the last position, used with method == 'LA08'.
    method : {"WF93", "LA08"}
      Which method to use.
    params
      Other parameters.

    Returns
    -------
    active : ndarray (bool)
      True where the fire season is active, same shape as tas.
    """
    season_mask = np.full_like(tas, False, dtype=bool)

    if method == "WF93":
        start_index = temp_condition_days + 1
    elif method == "LA08":
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

        # Mask is on if the previous days was on OR is there is a start up,  AND if it's not a shut down,
        # Aka is off if either the previous day was or it is a shut down.
        season_mask[..., it] = (season_mask[..., it - 1] | start_up) & ~shut_down

    return season_mask


def _fire_weather_calc(
    tas, pr, rh, ws, snd, mth, lat, season_mask, dc0, dmc0, ffmc0, winter_pr, **params
):
    """Primary function computing all Fire Weather Indexes. DO NOT CALL DIRECTLY, use `fire_weather_ufunc` instead."""
    outputs = params["outputs"]
    ind_prevs = {"DC": dc0.copy(), "DMC": dmc0.copy(), "FFMC": ffmc0.copy()}

    season_method = params.get("season_method")
    if season_method is None:
        # None means "always on"
        season_mask = np.full_like(tas, True, dtype=bool)
        # Start with default value
        ind_prevs["DC"][np.isnan(dc0)] = params["DCStart"]
        ind_prevs["DMC"][np.isnan(dmc0)] = params["DMCStart"]
        ind_prevs["FFMC"][np.isnan(ffmc0)] = params["FFMCStart"]
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

    for ind in list(ind_prevs.keys()):
        if ind not in outputs:
            ind_prevs.pop(ind)

    out = OrderedDict()
    for name in outputs:
        if name == "winter_pr":
            out[name] = winter_pr.copy()
        elif name == "season_mask":
            out[name] = season_mask
        else:
            out[name] = np.full_like(tas, np.nan)

    # Cast the mask as integers, use smallest dtype for memory purposes.
    season_mask = season_mask.astype(np.int16)

    overwintering = params["overwintering"]
    if overwintering:
        last_DC = dc0.copy()
        ind_prevs["DC"] = np.full_like(dc0, np.nan)

    for it in range(tas.shape[-1]):

        if season_method is not None:
            # Not in the always-on mode
            if it == 0:
                delta = season_mask[..., it]
            else:
                delta = season_mask[..., it] - season_mask[..., it - 1]

            shut_down = delta == -1
            # winter = (delta == 0) & (season_mask[..., it] == 0)
            start_up = delta == 1
            # case4 = (delta == 0) & (season_mask[it] == 1)

            if "DC" in ind_prevs:
                if overwintering:
                    last_DC[shut_down] = ind_prevs["DC"][shut_down]
                    out["winter_pr"][shut_down] = 0
                    out["winter_pr"] = out["winter_pr"] + pr[..., it]

                    dc0 = last_DC[start_up]
                    ind_prevs["DC"][start_up] = np.where(
                        np.isnan(dc0),
                        params["DCStart"],
                        _overwintering_drought_code(
                            dc0,
                            out["winter_pr"][start_up],
                            params["carry_over_fraction"],
                            params["wetting_efficiency_fraction"],
                            params["DCStart"],
                        ),
                    )
                    last_DC[start_up] = np.nan
                else:
                    ind_prevs["DC"][start_up] = params["DCStart"]
                ind_prevs["DC"][shut_down] = np.nan

            if "DMC" in ind_prevs:
                ind_prevs["DMC"][start_up] = params["DMCStart"]
                ind_prevs["DMC"][shut_down] = np.nan

            if "FFMC" in ind_prevs:
                ind_prevs["FFMC"][start_up] = params["FFMCStart"]
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

    if "winter_pr" in outputs:
        out["winter_pr"][season_mask[..., -1] == 1] = np.nan
    return tuple(out.values())


def fire_weather_ufunc(
    *,
    tas: xr.DataArray,
    pr: xr.DataArray,
    rh: Optional[xr.DataArray] = None,
    ws: Optional[xr.DataArray] = None,
    snd: Optional[xr.DataArray] = None,
    lat: Optional[xr.DataArray] = None,
    dc0: Optional[xr.DataArray] = None,
    dmc0: Optional[xr.DataArray] = None,
    ffmc0: Optional[xr.DataArray] = None,
    winter_pr: Optional[xr.DataArray] = None,
    season_mask: Optional[xr.DataArray] = None,
    start_dates: Optional[Union[str, xr.DataArray]] = None,
    indexes: Sequence[str] = None,
    season_method: Optional[str] = "WF93",
    overwintering: bool = True,
    **params,
):
    """Fire Weather Indexes computation using xarray's apply_ufunc.

    Parameters
    ----------
    tas : xr.DataArray
        Noon surface temperature in °C
    pr : xr.DataArray
        Rainfall over previous 24h, at noon in mm/day
    rh : xr.DataArray, optional
        Noon surface relative humidity in %, not needed for DC
    ws : xr.DataArray, optional
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
    start_date : str, optional
        Date at which to start the computation.
        Defaults to `snowCoverDaysCalc` after the beginning of tas.
    season_method : {None, "WF93", "LA08"}
        How to compute the start up and shut down of the fire season.
        If "None", no start ups or shud downs are computed, similar to the R fwi function.
        Ignored if `season_mask` is given.
    overwintering: bool
        Whether to activate DC overwintering or not. If True, either season_method or season_mask must be given.
    **params :
        Other keyword arguments for the Fire Weather Indexes computation.
        Default values of those are stored in `xclim.indices.fwi.DEFAULT_PARAMS`
        See this `xclim.indices.fwi`'s doc for details.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing the computed indexes as prescribed in `indexes`

    Notes
    -----
    When overwintering is activated, the argument `dc0` is understood as last season's
    last DC map and will be used to compute the overwintered DC at the beginning of the
    next season.

    If overwintering is not activated and neither is fire season computation (`season_method`
    and `season_mask` are `None`), `dc0`, `dmc0` and `ffmc0` are understood as the codes
    on the day before the first day of FWI computation.
    """
    for k, v in DEFAULT_PARAMS.items():
        params.setdefault(k, v)

    indexes = set(
        params.setdefault(
            "indexes",
            indexes or ["DC", "DMC", "FFMC", "ISI", "BUI", "FWI", "DSR"],
        )
    )
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
        (rh, "rh", ["DMC", "FFMC"], True),
        (ws, "ws", ["FFMC"], True),
        (snd, "snd", ["LA08"], True),
        (tas.time.dt.month, "month", ["DC", "DMC"], True),
        (lat, "lat", ["DC", "DMC"], False),
    )
    # Arg order : tas, pr, rh, ws, snd, mth, lat, season_mask, dc0, dmc0, ffmc0, winter_pr
    #              0   1   2   3    4   5    6    7             8    9     10    11
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

    params["season_method"] = season_method
    params["overwintering"] = overwintering
    params["outputs"] = outputs

    if tas.ndim == 1:
        dummy_dim = xr.core.utils.get_temp_dimname(tas.dims, "dummy")
        # The 1D problem. With dask 0D values are scalars, without they are 0D arrays.
        # We add a dummy dimension
        for i in range(len(args)):
            if isinstance(args[i], xr.DataArray):
                args[i] = args[i].expand_dims({dummy_dim: [1]})

    das = xr.apply_ufunc(
        _fire_weather_calc,
        *args,
        kwargs=params,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs={
            "meta": tuple(np.array((), dtype=dtype) for dtype in output_dtypes)
        },
    )

    if tas.ndim == 1:
        das = [da.squeeze(dummy_dim, drop=True) for da in das]

    if len(outputs) == 1:
        return {outputs[0]: das}
    return {name: da for name, da in zip(outputs, das)}
