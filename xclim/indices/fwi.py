"""
Adapted from:
Matlab code CalcFWITimeSeriesWithStartup.m from GFWED made for using MERRA2 data.
This was a translation of FWI.vba of the Canadian Fire Weather Index system.
Updated source code for calculating fire danger indices in the Canadian Forest Fire Weather Index System
Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.

See https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi


Notes
-----
TODO: Skip computations over the ocean and where Tg_annual < -10 and where Pr_annual < 0.25
TODO: Vectorization over spatial chunks: replace math.expression by np.expression AND/OR Use numba to vectorize said functions
TODO: Add references
TODO: Alternative computation of start up / shut down without snow_depth
TODO: Allow computation of DC/DMC/FFMC independently
"""
import math

import numpy as np
import xarray as xr

# from xclim import generic
# from xclim import run_length as rl
# from xclim import utils


DEFAULT_PARAMS = dict(
    # snd_thresh=0.1,
    min_lat=-58,
    max_lat=75,
    minLandFrac=0.1,
    minT=-10,
    minPrec=0.25,
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
"""
paramSets(currParam).History = cellstr(datestr(now)); paramDesc(currDesc) = cellstr('history'); currDesc = currDesc + 1;
    paramSets(currParam).Source = cellstr('Robert Field'); paramDesc(currDesc) = cellstr('source'); currDesc = currDesc + 1;
    paramSets(currParam).Title = cellstr('Global Fire Weather Database'); paramDesc(currDesc) = cellstr('title'); currDesc = currDesc + 1;
    paramSets(currParam).Center = cellstr('NASA GISS / Columbia University'); paramDesc(currDesc) = cellstr('center'); currDesc = currDesc + 1;

    paramSets(currParam).Name = cellstr('Default'); paramDesc(currDesc) = cellstr('Descriptive name for configuration'); currDesc = currDesc + 1;
    paramSets(currParam).minLat = -58;              paramDesc(currDesc) = cellstr('Min latitude for analysis'); currDesc = currDesc + 1;
    paramSets(currParam).maxLat = 75;               paramDesc(currDesc) = cellstr('Max latitude for analysis'); currDesc = currDesc + 1;
    paramSets(currParam).minLandFrac = 0.1;         paramDesc(currDesc) = cellstr('Minimum grid cell land fraction for analysis'); currDesc = currDesc + 1;
    paramSets(currParam).minT = -10;                paramDesc(currDesc) = cellstr('Mask out anything with mean annual Tsurf less than this'); currDesc = currDesc + 1;
    paramSets(currParam).minPrec = 0.25;            paramDesc(currDesc) = cellstr('Mask out anything with mean annual prec less than this'); currDesc = currDesc + 1;

    %These are what would be tested for sensitivity
    paramSets(currParam).snoDThresh = 0.01;         paramDesc(currDesc) = cellstr('Minimum depth (m) for there to be considered snow on ground at any given time'); currDesc = currDesc + 1;
    paramSets(currParam).snowCoverDaysCalc = 60;    paramDesc(currDesc) = cellstr('Number of days prior to spring over which to determine if winter had substantial snow cover'); currDesc = currDesc + 1;
    paramSets(currParam).minWinterSnoD = 0.1;       paramDesc(currDesc) = cellstr('Minimum mean depth (m) during past snowCoverDaysCalc days for winter to be considered having had substantial snow cover'); currDesc = currDesc + 1;
    paramSets(currParam).minSnowDayFrac = 0.75;     paramDesc(currDesc) = cellstr('Minimum fraction of days during snowCoverDaysCalc where snow cover was greater than snoDThresh for winter to be considered having had substantial snow cover');  currDesc = currDesc + 1;
    paramSets(currParam).startShutDays = 2;         paramDesc(currDesc) = cellstr('Number of previous days over which to consider start or end of winter'); currDesc = currDesc + 1;
    paramSets(currParam).tempThresh = 6;            paramDesc(currDesc) = cellstr('Temp thresh (C) to define start and end of winter'); currDesc = currDesc + 1;
    paramSets(currParam).precThresh =1.0;           paramDesc(currDesc) = cellstr('Min precip (mm/day) when determining if last three days had any precip'); currDesc = currDesc + 1;
    paramSets(currParam).DCStart = 15;              paramDesc(currDesc) = cellstr('DC starting value after wet winter'); currDesc = currDesc + 1;
    paramSets(currParam).DMCStart = 6;              paramDesc(currDesc) = cellstr('DMC starting value after wet winter'); currDesc = currDesc + 1;
    paramSets(currParam).FFMCStart = 85;            paramDesc(currDesc) = cellstr('FFMC starting value after any winter'); currDesc = currDesc + 1;
    paramSets(currParam).DCDryStartFactor=5;        paramDesc(currDesc) = cellstr('DC number of days since precip mult factor for dry start.'); currDesc = currDesc + 1;
    paramSets(currParam).DMCDryStartFactor=2;       paramDesc(currDesc) = cellstr('DMC number of days since precip mult factor for dry start.'); currDesc = currDesc + 1;
    paramSets(currParam).nClimSkipYears = 1;
"""

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


def day_length(lat):
    """Return the average day length by month within latitudinal bounds."""
    lat_bnds = (-90, -30, -10, 10, 30, 90)
    i = np.digitize(lat, lat_bnds) - 1
    return DAY_LENGTHS[i]


def day_length_factor(lat, kind="gfwed"):
    """Return the day length factor.

    Note
    ----
    Taken from GFWED code.
    """
    lat_bnds = (-90, -15, 15, 90)
    i = np.digitize(lat, lat_bnds) - 1
    return DAY_LENGTH_FACTORS[i]


# def significant_snow_cover(snd, min_snow_days=0.75, min_snd="10 cm", snd_thresh="1 cm"):
#     """Return whether or not a site report significant snow cover for each year.

#     Snow cover is considered significant if the number of days with snow is above a given fraction and if the mean
#     snow depth is above a threshold.
#     """
#     msnd = utils.convert_units_to(min_snd, snd)
#     sndt = utils.convert_units_to(snd_thresh, snd)

#     north = (
#         snd.sel(lat=slice(0, None))
#         .sel(time=snd.time.dt.month.isin([1, 2]))
#         .resample(time="A")
#     )
#     south = (
#         snd.sel(lat=slice(None, -1e-6))
#         .sel(time=snd.time.dt.month.isin([7, 8]))
#         .resample(time="A")
#     )

#     def condition(winter_snow):
#         """Return if the winter snow mean depth is above threshold and snow covered the ground for at least a
#         fraction of the winter."""
#         c1 = winter_snow.mean() > msnd
#         c2 = winter_snow.apply(lambda x: np.sum((x > sndt) * 1)) >= (
#             winter_snow.count() * min_snow_days
#         )
#         return c1 * c2

#     out = [condition(hemi) for hemi in [south, north]]
#     return xr.concat(out, dim="lat")


# def start_up_snow(snd, snd_thresh="1 cm"):
#     """Return day of year at which snow depth is below threshold for three consecutive days."""
#     w = 3
#     sndt = utils.convert_units_to(snd_thresh, snd)
#     under = snd < sndt

#     north = under.sel(lat=slice(0, None)).resample(time="A")
#     south = (
#         under.sel(lat=slice(None, -1e-6))
#         .sel(time=snd.time.dt.month.isin([7, 8, 9, 10, 11, 12]))
#         .resample(time="A")
#     )

#     out = [
#         g.apply(rl.first_run_ufunc, window=w, index="dayofyear") for g in [south, north]
#     ]
#     return xr.concat(out, dim="lat") + w - 1


# def start_up_temp(tas, thresh="6 C"):
#     """Return the date at which the mean temperature is above threshold for three consecutive days."""
#     w = 3
#     t = utils.convert_units_to(thresh, tas)
#     over = tas > t

#     north = over.sel(lat=slice(0, None)).resample(time="A")
#     south = (
#         over.sel(lat=slice(None, -1e-6))
#         .sel(time=tas.time.dt.month.isin([7, 8, 9, 10, 11, 12]))
#         .resample(time="A")
#     )

#     out = [
#         g.apply(rl.first_run_ufunc, window=w, index="dayofyear") for g in [south, north]
#     ]
#     return xr.concat(out, dim="lat") + w - 1


def _fine_fuel_moisture_code(t, p, w, h, ffmc0):
    """Scalar computation of the fine fuel moisture code."""
    if np.isnan(ffmc0):
        return np.nan
    mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0)  # *Eq.1*#
    if p > 0.5:
        rf = p - 0.5  # *Eq.2*#
        if mo > 150.0:
            mo = (
                mo
                + 42.5
                * rf
                * math.exp(-100.0 / (251.0 - mo))
                * (1.0 - math.exp(-6.93 / rf))
            ) + (0.0015 * (mo - 150.0) ** 2) * math.sqrt(
                rf
            )  # *Eq.3b*#
        elif mo <= 150.0:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (
                1.0 - math.exp(-6.93 / rf)
            )
            # *Eq.3a*#
        if mo > 250.0:
            mo = 250.0

    ed = (
        0.942 * (h ** 0.679)
        + (11.0 * math.exp((h - 100.0) / 10.0))
        + 0.18 * (21.1 - t) * (1.0 - 1.0 / math.exp(0.1150 * h))
    )  # *Eq.4*#

    if mo < ed:
        ew = (
            0.618 * (h ** 0.753)
            + (10.0 * math.exp((h - 100.0) / 10.0))
            + 0.18 * (21.1 - t) * (1.0 - 1.0 / math.exp(0.115 * h))
        )  # *Eq.5*#
        if mo <= ew:
            kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + (
                0.0694 * math.sqrt(w)
            ) * (
                1.0 - ((100.0 - h) / 100.0) ** 8
            )  # *Eq.7a*#
            kw = kl * (0.581 * math.exp(0.0365 * t))  # *Eq.7b*#
            m = ew - (ew - mo) / 10.0 ** kw  # *Eq.9*#
        elif mo > ew:
            m = mo
    elif mo == ed:
        m = mo
    else:
        kl = 0.424 * (1.0 - (h / 100.0) ** 1.7) + (0.0694 * math.sqrt(w)) * (
            1.0 - (h / 100.0) ** 8
        )  # *Eq.6a*#
        kw = kl * (0.581 * math.exp(0.0365 * t))  # *Eq.6b*#
        m = ed + (mo - ed) / 10.0 ** kw  # *Eq.8*#

    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)  # *Eq.10*#
    if ffmc > 101.0:
        ffmc = 101.0
    elif ffmc <= 0.0:
        ffmc = 0.0

    return ffmc


def fine_fuel_moisture_code(tas, pr, ws, rh, ffmc0):
    """Fine fuel moisture code

    Parameters
    ----------
    tas: array
      Noon temperature [C].
    pr : array
      Rain fall in open over previous 24 hours, at noon [mm].
    ws : array
      Noon wind speed [km/h].
    rh : array
      Noon relative humidity [%].
    ffmc0 : float
      Initial value of the fine fuel moisture code.

    Returns
    -------
    array
      Fine fuel moisture code
    """

    it = np.nditer(
        [tas, pr, ws, rh, ffmc0, None],
        [],
        5 * [["readonly"]] + [["writeonly", "allocate"]],  # add no_broadcast?
    )

    with it:
        for (t, p, w, h, ffmc, out) in it:
            it[5] = _fine_fuel_moisture_code(t, p, w, h, ffmc)

        return it.operands[5]


def _duff_moisture_code(t, p, h, mth, lat, dmc0):
    """Scalar computation of the Duff moisture code."""
    if np.isnan(dmc0):
        return np.nan
    el = day_length(lat)

    if t < -1.1:
        t = -1.1
    rk = 1.894 * (t + 1.1) * (100.0 - h) * (el[mth - 1] * 0.0001)  # *Eqs.16 and 17*#

    if p > 1.5:
        ra = p
        rw = 0.92 * ra - 1.27  # *Eq.11*#
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc0)  # *Eq.12*#
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)  # *Eq.13a*#
        else:
            if dmc0 <= 65.0:
                b = 14.0 - 1.3 * math.log(dmc0)  # *Eq.13b*#
            else:
                b = 6.2 * math.log(dmc0) - 17.2  # *Eq.13c*#
        wmr = wmi + (1000 * rw) / (48.77 + b * rw)  # *Eq.14*#
        pr = 43.43 * (5.6348 - math.log(wmr - 20.0))  # *Eq.15*#
    else:  # p <= 1.5
        pr = dmc0
    if pr < 0.0:
        pr = 0.0
    dmc = pr + rk
    if dmc <= 1.0:
        dmc = 1.0
    return dmc


def duff_moisture_code(tas, pr, rh, mth, lat, dmc0):
    """Duff moisture code

    Parameters
    ----------
    tas: array
      Noon temperature [C].
    pr : array
      Rain fall in open over previous 24 hours, at noon [mm].
    rh : array
      Noon relative humidity [%].
    mth : integer array
      Month of the year [1-12].
    lat : float
      Latitude.
    dmc0 : float
      Initial value of the Duff moisture code.

    Returns
    -------
    array
      Duff moisture code
    """
    it = np.nditer(
        [tas, pr, rh, mth, lat, dmc0, None],
        [],
        6 * [["readonly"]] + [["writeonly", "allocate"]],
    )

    with it:
        for (t, p, h, m, l, dmc, out) in it:
            it[6] = _duff_moisture_code(t, p, h, m, l, dmc)

        return it.operands[6]


def _drought_code(t, p, mth, lat, dc0):
    """Scalar computation of the drought code."""
    if np.isnan(dc0):
        return np.nan

    fl = day_length_factor(lat)

    if t < -2.8:
        t = -2.8
    pe = (0.36 * (t + 2.8) + fl[mth - 1]) / 2  # *Eq.22*#
    if pe < 0.0:
        pe = 0.0

    if p > 2.8:
        ra = p
        rw = 0.83 * ra - 1.27  # *Eq.18*#  Rd
        smi = 800.0 * math.exp(-dc0 / 400.0)  # *Eq.19*# Qo
        dr = dc0 - 400.0 * math.log(1.0 + ((3.937 * rw) / smi))  # *Eqs. 20 and 21*#
        if dr > 0.0:
            dc = dr + pe
        else:
            dc = pe
    else:  # f p <= 2.8:
        dc = dc0 + pe
    return dc


def drought_code(tas, pr, mth, lat, dc0):
    """Drought code

    Parameters
    ----------
    tas: array
      Noon temperature [C].
    pr : array
      Rain fall in open over previous 24 hours, at noon [mm].
    mth : integer array
      Month of the year [1-12].
    lat : float
      Latitude.
    dc0 : float
      Initial value of the drought code.

    Returns
    -------
    array
      Drought code
    """
    it = np.nditer(
        [tas, pr, mth, lat, dc0, None],
        [],
        5 * [["readonly"]] + [["writeonly", "allocate"]],
    )

    with it:
        for (t, p, m, l, dc, out) in it:
            it[5] = _drought_code(t, p, m, l, dc)

        return it.operands[5]


def initial_spread_index(ws, ffmc):
    """Initial spread index

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
    """Build up index

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
    """Fire weather index

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
    """Daily severity rating

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


# def ffmc_ufunc(tas, pr, ws, rh, ffmc0):
#     return xr.apply_ufunc(
#         fine_fuel_moisture_code,
#         tas,
#         pr,
#         ws,
#         rh,
#         ffmc0,
#         input_core_dims=4 * (("time",),) + ((),),
#         output_core_dims=(("time",),),
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[np.float],
#         # keep_attrs=True,
#     )


# def dmc_ufunc(tas, pr, rh, mth, lat, dmc0):
#     return xr.apply_ufunc(
#         duff_moisture_code,
#         tas,
#         pr,
#         rh,
#         mth,
#         lat,
#         dmc0,
#         input_core_dims=4 * (("time",),) + 2 * ((),),
#         output_core_dims=(("time",),),
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[np.float],
#         # keep_attrs=True,
#     )


# def dc_ufunc(tas, pr, mth, lat, dc0):
#     return xr.apply_ufunc(
#         drought_code,
#         tas,
#         pr,
#         mth,
#         lat,
#         dc0,
#         input_core_dims=3 * (("time",),) + 2 * ((),),
#         output_core_dims=(("time",),),
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[np.float],
#         # keep_attrs=True,
#     )


def calc_indices(tas, pr, rh, ws, snow, mth, lat, dcprev, dmcprev, ffmcprev, **params):
    """Big iterator, iterating in time, vectorized in space."""
    dc = np.zeros_like(tas) * np.nan
    dmc = np.zeros_like(tas) * np.nan
    ffmc = np.zeros_like(tas) * np.nan
    isi = np.zeros_like(tas) * np.nan
    bui = np.zeros_like(tas) * np.nan
    fwi = np.zeros_like(tas) * np.nan
    # dsr = np.zeros_like(tas) * np.nan

    for it in range(params.get("start", params["snowCoverDaysCalc"]), tas.shape[-1]):
        snow_cover_recent = snow[..., it - params["startShutDays"] : it + 1].mean(
            axis=-1
        )
        snow_cover_history = snow[..., it - params["snowCoverDaysCalc"] + 1 : it + 1]
        snow_days = np.count_nonzero(snow_cover_history > params["snoDThresh"], axis=-1)
        temp_recent = tas[..., it - params["startShutDays"] : it + 1].mean(axis=-1)

        prec_history = np.flip(pr[..., : it + 1], axis=-1)
        days_with_prec = prec_history >= params["precThresh"]
        days_since_last_prec = days_with_prec.argmax(axis=-1)
        days_since_last_prec = np.where(
            np.any(days_with_prec, axis=-1),
            days_since_last_prec,
            params["snowCoverDaysCalc"],
        )

        shut_down = (temp_recent < params["tempThresh"]) | (
            snow_cover_recent >= params["snoDThresh"]
        )

        dcprev[shut_down] = np.nan
        dmcprev[shut_down] = np.nan
        ffmcprev[shut_down] = np.nan

        start_up = np.isnan(dcprev) & ~shut_down
        start_up_wet = (
            start_up
            & (snow_days / params["snowCoverDaysCalc"] >= params["minSnowDayFrac"])
            & (snow_cover_history.mean(axis=-1) >= params["minWinterSnoD"])
        )
        start_up_dry = start_up & ~start_up_wet

        dcprev[start_up_wet] = params["DCStart"]
        dmcprev[start_up_wet] = params["DMCStart"]
        dcprev[start_up_dry] = (
            params["DCDryStartFactor"] * days_since_last_prec[start_up_dry]
        )
        dmcprev[start_up_dry] = (
            params["DMCDryStartFactor"] * days_since_last_prec[start_up_dry]
        )
        ffmcprev[start_up] = params["FFMCStart"]

        dc[..., it] = drought_code(tas[..., it], pr[..., it], mth[..., it], lat, dcprev)
        dmc[..., it] = duff_moisture_code(
            tas[..., it], pr[..., it], rh[..., it], mth[..., it], lat, dmcprev
        )
        ffmc[..., it] = fine_fuel_moisture_code(
            tas[..., it], pr[..., it], ws[..., it], rh[..., it], ffmcprev
        )

        isi[..., it] = initial_spread_index(ws[..., it], ffmc[..., it])
        bui[..., it] = build_up_index(dmc[..., it], dc[..., it])

        fwi[..., it] = fire_weather_index(isi[..., it], bui[..., it])
        # dsr[..., it] = daily_severity_rating(fwi[..., it])

        dcprev[...] = dc[..., it]
        dmcprev[...] = dmc[..., it]
        ffmcprev[...] = ffmc[..., it]

    return dc, dmc, ffmc, isi, bui, fwi  # , dsr


def all_ufunc(tas, pr, rh, ws, snow, lat, **params):
    for k, v in DEFAULT_PARAMS.items():
        params.setdefault(k, v)

    if "start_date" in params:
        params["start"] = int(
            abs(snow.time - np.datetime64(params["start_date"])).argmin("time")
        )
        if params["start"] < params["snowCoverDaysCalc"]:
            raise ValueError(
                "Input data must start at least {} days before the specified start date.".format(
                    params["snowCoverDaysCalc"]
                )
            )
        elif not all(var0 in params for var0 in ["dc0", "dmc0", "ffmc0"]):
            raise ValueError(
                "If a start date is specified, initial maps dc0, dmc0 and ffmc0 must also be given."
            )

    return xr.apply_ufunc(
        calc_indices,
        tas,
        pr,
        rh,
        ws,
        snow,
        tas.time.dt.month,
        lat,
        params.pop("dc0", xr.zeros_like(tas.isel(time=0)) * np.nan),
        params.pop("dmc0", xr.zeros_like(tas.isel(time=0)) * np.nan),
        params.pop("ffmc0", xr.zeros_like(tas.isel(time=0)) * np.nan),
        kwargs=params,
        input_core_dims=6 * (("time",),) + 4 * ((),),
        output_core_dims=6 * (("time",),),
        dask="allowed",
    )
