# noqa: D205,D400
"""
==============================
Fire Weather Indices Submodule
==============================

Methods defined here are used by :func:`xclim.indices.drought_code` and :func:`xclim.indices.fire_weather_indexes`.

Adapted from Matlab code `CalcFWITimeSeriesWithStartup.m` from GFWED made for using
MERRA2 data, which was a translation of FWI.vba of the Canadian Fire Weather Index system.

Shut down and start up
----------------------
Fire weather indexes are iteratively computed, each day's value depending on the previous day indexes.
Additionally, the codes are "shut down" (set to NaN) in winter. There are a few ways of computing this
shut down and the subsequent spring start up. The principal method (`'snow_depth'`) uses temperature, precipitation
and snow depth, a variable which is not always available. xclim implements less restrictive options usable with model
climate data using only temperature. In the list below, parameters between "" refer to those listed in the next section.

Shut down methods:

    `'temperature'`
        Grid points where the average temperature of the last "startShutDays" is below "tempThresh" are shut down.
    `'snow_depth'`
        In addition to the `'temperature'` condition, pixels where the average snow depth of the last "startShutDays"
        is greater or equal to "snoDThresh" are also shut down.

Start up methods:

    `None`
        Grid points that were shut down in the previous timestep but not in the current are set to the *wet* start values. ("DCStart", "DMCStart" and "FFMCStart")
    `snow_depth`
        Same as above, but the *wet* start is only used on grid points where: 1) the average snow depth of the last "snowCoverDaysCalc" is above "minWinterSnoD" and
        2) at least "minSnowDayFrac" % of the last "snowCoverDaysCalc" had a snow depth above "snoDThresh". For all other grid points, the *dry* start is used,
        where DC and DMC are started with their "DryStartFactor" multiplied by the smallest number between "snowCoverDaysCalc" and the number of days since
        the last rain event of at least "precThresh" mm.


Parameters
----------
Default values for the following parameters are stored in the DEFAULT_PARAMS dict. The current implementation doesn't use all those parameters, so it might be useless to modify them.

    snowCoverDaysCalc : int
        Number of days prior to spring over which to determine if winter had substantial snow cover
    minWinterSnoD : float
        Minimum mean depth (m) during past snowCoverDaysCalc days for winter to be considered having had substantial snow cover
    snoDThresh : float
        Minimum depth (m) for there to be considered snow on ground at any given time
    minSnowDayFrac : float
        Minimum fraction of days during snowCoverDaysCalc where snow cover was greater than snoDThresh for winter to be considered having had substantial snow cover
    startShutDays : int
        Number of previous days over which to consider start or end of winter
    tempThresh : float
        Temp thresh (C) to define start and end of winter
    precThresh : float
        Min precip (mm/day) when determining if last three days had any precip
    DCDryStartFactor : float
        DC number of days since precip mult factor for dry start.
    DMCDryStartFactor : float
        DMC number of days since precip mult factor for dry start.
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

.. todo::

    Skip computations over the ocean and where Tg_annual < -10 and where Pr_annual < 0.25,
    Add references,
    Allow computation of DC/DMC/FFMC independently,
"""
from collections import OrderedDict
from typing import Optional, Sequence, Union
from warnings import warn

import dask.array as dsk
import numpy as np
import xarray as xr
from numba import jit, vectorize

from xclim.core.units import convert_units_to, declare_units

from .run_length import rle

DEFAULT_PARAMS = dict(
    # min_lat=-58,
    # max_lat=75,
    # minLandFrac=0.1,
    # minT=-10,
    # minPrec=0.25,
    snowCoverDaysCalc=60,
    minWinterSnoD=0.1,
    snoDThresh=0.01,
    minSnowDayFrac=0.75,
    startShutDays=2,
    tempThresh=6,
    precThresh=1.0,
    DCDryStartFactor=5,
    DMCDryStartFactor=2,
    DCStart=15.0,
    FFMCStart=85.0,
    DMCStart=6.0,
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
            ) + (0.0015 * (mo - 150.0) ** 2) * np.sqrt(
                rf
            )  # *Eq.3b*#
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
        if mo <= ew:
            kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + (
                0.0694 * np.sqrt(w)
            ) * (
                1.0 - ((100.0 - h) / 100.0) ** 8
            )  # *Eq.7a*#
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
    dl = _day_length(lat, mth)

    if t < -1.1:
        rk = 0
    else:
        rk = 1.894 * (t + 1.1) * (100.0 - h) * dl * 0.0001  # *Eqs.16 and 17*#

    if p > 1.5:
        ra = p
        rw = 0.92 * ra - 1.27  # *Eq.11*#
        # wmi = 20.0 + 280.0 / math.exp(0.023 * dmc0)  # *Eq.12*#
        wmi = 20.0 + np.exp(5.6348 - dmc0 / 43.43)  # *Eq.12*#
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)  # *Eq.13a*#
        else:
            if dmc0 <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc0)  # *Eq.13b*#
            else:
                b = 6.2 * np.log(dmc0) - 17.2  # *Eq.13c*#
        wmr = wmi + (1000 * rw) / (48.77 + b * rw)  # *Eq.14*#
        # pr = 43.43 * (5.6348 - math.log(wmr - 20.0))  # *Eq.15*#
        pr = 244.72 - 43.43 * np.log(wmr - 20.0)  # *Eq.15*#
    else:  # p <= 1.5
        pr = dmc0

    if pr < 0.0:
        pr = 0.0
    dmc = pr + rk
    # if dmc <= 1.0:
    #    dmc = 1.0
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
    ff = 19.115 * np.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)  # *Eq.25*#
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
    bb = np.where(
        bui <= 80.0,
        0.1 * isi * (0.626 * bui ** 0.809 + 2.0),  # *Eq.28a*#
        0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))),
    )  # *Eq.28b*#

    fwi = np.where(
        bb <= 1.0, bb, np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)  # *Eq.30b*#
    )  # *Eq.30a*#

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
def _overwintering_drought_code(DCf, wpr, a=0.75, b=0.75):
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
    Qf = 800 * np.exp(-DCf / 400)
    Qs = a * Qf + b * (3.94 * wpr)
    DCs = 400 * np.log(800 / Qs)
    if DCs < 15:
        DCs = 15
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


@declare_units(
    tas="[temperature]",
    snd="[length]",
    temp_end_thresh="[temperature]",
    temp_start_thresh="[temperature]",
    snow_thresh="[length]",
)
def fire_season(
    tas,
    snd=None,
    method="WF93",
    keep_longest: Union[bool, str] = False,
    temp_start_thresh="12 degC",
    temp_end_thresh="5 degC",
    temp_condition_days=3,
    snow_condition_days=3,
    snow_thresh="0 cm",
):
    """Compute the active fire season mask.

    Parameters
    ----------
    tas : xr.DataArray
      Daily surface temperature, cffdrs recommends using maximum daily temperature.
    snd : xr.DataArray, optional
      Snow depth, used with method == 'LA08'.
    method : {"WF93", "LA08"}
      Which method to use.
    keep_longest : bool or str
      If True, only keeps the longest fire season in the mask. If a str, it is understood
      as a frequnecy and only the longest fire season for each of these period is kept.
      Every "seasons" are returned if False, including the short shoulder seasons.
    params
      Other parameters.

    Returns
    -------
    active : ndarray (bool)
      True where the fire season is active, same shape as tas.
    """
    kwargs = {
        "temp_start_thresh": convert_units_to(temp_start_thresh, "degC"),
        "temp_end_thresh": convert_units_to(temp_end_thresh, "degC"),
        "snow_thresh": convert_units_to(snow_thresh, "m"),
        "temp_condition_days": temp_condition_days,
        "snow_condition_days": snow_condition_days,
    }

    def _keep_longest_run(mask):
        # Get run lengths, fill NaNs with the lengths, remove added end element.
        lengths = rle(mask, "time").ffill("time").isel(time=slice(None, -1))
        # Remove shoulders
        return mask.where(lengths == lengths.max("time"), False)

    def _apply_fire_season(ds):
        season_mask = ds.tas.copy(
            data=_fire_season(
                tas=ds.tas.transpose(..., "time").values,
                snd=None if method == "WF93" else ds.snd.transpose(..., "time").values,
                method=method,
                **kwargs,
            )
        )
        season_mask.attrs = {}

        if keep_longest is not False:
            if isinstance(keep_longest, str):
                time = season_mask.time
                season_mask = season_mask.resample(time=keep_longest).map(
                    _keep_longest_run
                )
                season_mask["time"] = time
            else:
                season_mask = _keep_longest_run(season_mask)

        return season_mask

    ds = convert_units_to(tas, "degC").rename("tas").to_dataset()
    if snd is not None:
        ds["snd"] = convert_units_to(snd, "m")
        ds = ds.unify_chunks()

    if isinstance(tas.data, dsk.Array):
        tmpl = tas.copy(data=dsk.empty_like(ds.tas.data))
    else:
        tmpl = tas.copy(data=np.empty_like(ds.tas.data))

    out = ds.map_blocks(_apply_fire_season, template=tmpl)
    out.attrs["units"] = ""
    return out


def _fire_weather_calc(
    tas, pr, rh, ws, snd, mth, lat, dcprev, dmcprev, ffmcprev, season_mask, **params
):
    """Primary function computing all Fire Weather Indexes. DO NOT CALL DIRECTLY, use `fire_weather_ufunc` instead."""
    indexes = params["indexes"]

    ind_prevs = {"DC": dcprev, "DMC": dmcprev, "FFMC": ffmcprev}
    for name, ind_prev in ind_prevs.copy().items():
        if ind_prev is None:
            ind_prevs.pop(name)
        else:
            ind_prevs[name] = ind_prev.copy()

    ind_data = OrderedDict()
    for indice in indexes:
        ind_data[indice] = np.full_like(tas, np.nan)

    if season_mask is None:
        season_mask = _fire_season(tas, snd, method=params.get("season_method", "WF93"))

    # Cast the mask as integers, use smallest dtype for memory purposes.
    season_mask = season_mask.astype(np.int16)

    overwintering = params.get("overwintering", False)
    if overwintering:
        acc_precip = np.full_like(tas[..., 0], np.nan, dtype=float)
        last_DC = np.full_like(tas[..., 0], np.nan, dtype=float)

    for it in range(tas.shape[-1]):

        if it == 0:
            delta = season_mask[it]
        else:
            delta = season_mask[it] - season_mask[it - 1]

        shut_down = delta == -1
        winter = (delta == 0) & (season_mask[it] == 0)
        start_up = delta == 1
        # case4 = (delta == 0) & (season_mask[it] == 1)

        if "DC" in ind_prevs:
            if overwintering:
                last_DC[shut_down] = ind_prevs["DC"][shut_down]
                acc_precip[shut_down] = pr[shut_down, it]

                acc_precip[winter] = acc_precip[winter] + pr[winter, it]

                ind_prevs["DC"][start_up] = _overwintering_drought_code(
                    last_DC[start_up], acc_precip[start_up]
                )
                last_DC[start_up] = np.nan
                acc_precip[start_up] = np.nan
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
        if "DC" in indexes:
            ind_data["DC"][..., it] = _drought_code(
                tas[..., it], pr[..., it], mth[..., it], lat, ind_prevs["DC"]
            )
        if "DMC" in indexes:
            ind_data["DMC"][..., it] = _duff_moisture_code(
                tas[..., it],
                pr[..., it],
                rh[..., it],
                mth[..., it],
                lat,
                ind_prevs["DMC"],
            )
        if "FFMC" in indexes:
            ind_data["FFMC"][..., it] = _fine_fuel_moisture_code(
                tas[..., it], pr[..., it], ws[..., it], rh[..., it], ind_prevs["FFMC"]
            )
        if "ISI" in indexes:
            ind_data["ISI"][..., it] = initial_spread_index(
                ws[..., it], ind_data["FFMC"][..., it]
            )
        if "BUI" in indexes:
            ind_data["BUI"][..., it] = build_up_index(
                ind_data["DMC"][..., it], ind_data["DC"][..., it]
            )
        if "FWI" in indexes:
            ind_data["FWI"][..., it] = fire_weather_index(
                ind_data["ISI"][..., it], ind_data["BUI"][..., it]
            )

        if "DSR" in indexes:
            ind_data["DSR"][..., it] = daily_severity_rating(ind_data["FWI"][..., it])

        # Set the previous values
        for ind, ind_prev in ind_prevs.items():
            ind_prev[...] = ind_data[ind][..., it]

    if len(indexes) == 1:
        return ind_data[indexes[0]]
    return tuple(ind_data.values())


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
    season_mask: Optional[xr.DataArray] = None,
    start_dates: Optional[Union[str, xr.DataArray]] = None,
    indexes: Sequence[str] = None,
    season_method: str = "WF93",
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
        Noon snow depth in m, only needed if `start_up_mode` is "snow_depth"
    lat : xr.DataArray, optional
        Latitude in °N, not needed for FFMC or ISI
    dc0 : xr.DataArray, optional
        DC the day before `start_date`, defaults to NaN.
    dmc0 : xr.DataArray, optional
        DMC the day before `start_date`, defaults to NaN.
    ffmc0 : xr.DataArray, optional
        FFMC the day before `start_date`, defaults to NaN.
    season_mask : xr.DataArray, optional
        Boolean mask, True where the fire season is active. Same shape as tas.
    indexes : Sequence[str], optional
        Which indexes to compute. If intermediate indexes are needed, they will be added to the list and output.
    start_date : str, optional
        Date at which to start the computation.
        Defaults to `snowCoverDaysCalc` after the beginning of tas.
    season_method : {"WF93", "LA08"}
        How to compute the start up and shut down of the fire season. See module doc for valid values.
        Ignored if `season_mask` is given.
    **params :
        Other keyword arguments for the Fire Weather Indexes computation.
        Default values of those are stored in `xclim.indices.fwi.DEFAULT_PARAMS`
        See this `xclim.indices.fwi`'s doc for details.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing the computed indexes as prescribed in `indexes`
    """
    for k, v in DEFAULT_PARAMS.items():
        params.setdefault(k, v)

    indexes = set(
        params.setdefault(
            "indexes",
            indexes or ["DC", "DMC", "FFMC", "ISI", "BUI", "FWI", "DSR", "fireSeason"],
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
        key=["DC", "DMC", "FFMC", "ISI", "BUI", "FWI", "DSR", "fireSeason"].index,
    )

    # Whether each argument is needed in _fire_weather_calc
    # Same order as _fire_weather_calc, Assumes the list of indexes is complete.
    # (name, list of indexes + start_up/shut_down modes, has_time_dim)
    needed_args = (
        (tas, "tas", ["DC", "DMC", "FFMC", "fireSeason"], True),
        (pr, "pr", ["DC", "DMC", "FFMC"], True),
        (rh, "rh", ["DMC", "FFMC"], True),
        (ws, "ws", ["FFMC"], True),
        (snd, "snd", ["LA08"], True),
        (tas.time.dt.month, "month", ["DC", "DMC"], True),
        (lat, "lat", ["DC", "DMC"], False),
        (dc0, "dc0", ["DC"], False),
        (dmc0, "dmc0", ["DMC"], False),
        (ffmc0, "ffmc0", ["FFMC"], False),
    )
    args = []
    input_core_dims = []
    # Verification of all arguments
    for i, (arg, name, usedby, has_time_dim) in enumerate(needed_args):
        if any([ind in indexes + [season_method] for ind in usedby]):
            if arg is None:
                raise TypeError(
                    f"Missing input argument {name} for index combination {indexes} with fire season method '{season_method}'"
                )
            if hasattr(arg, "data") and isinstance(arg.data, dsk.Array):
                # TODO remove this when xarray supports multiple dask outputs in apply_ufunc
                warn(
                    "Dask arrays have been detected in the input of the Fire Weather calculation but they are not supported yet. Data will be loaded."
                )
                args.append(arg.load())
            else:
                args.append(arg)
            input_core_dims.append(["time"] if has_time_dim else [])
        else:
            args.append(None)
            input_core_dims.append([])

    if season_mask is not None:
        args.append(season_mask)
        input_core_dims.append(["time"])
    else:
        args.append(None)
        input_core_dims.append([])

    params["season_method"] = season_method
    params["indexes"] = indexes

    das = xr.apply_ufunc(
        _fire_weather_calc,
        *args,
        kwargs=params,
        input_core_dims=input_core_dims,  # nargs[0] * (("time",),) + nargs[1] * ((),) + snowdims,
        output_core_dims=len(indexes) * (("time",),),
        dask="forbidden",
    )
    if len(indexes) == 1:
        return {indexes[0]: das}
    return {ind: da for ind, da in zip(indexes, das)}
