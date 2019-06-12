"""
Adapted from:
Updated source code for calculating fire danger indices in the Canadian Forest Fire Weather Index System
Y. Wang, K.R. Anderson, and R.M. Suddaby, INFORMATION REPORT NOR-X-424, 2015.

See https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi

D. Huard
"""

import math
import numpy as np
import xarray as xr


def _fine_fuel_moisture_code(t, p, w, h, ffmc0):
    """Scalar computation of the fine fuel moisture code."""

    mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0)  # *Eq.1*#
    if p > 0.5:
        rf = p - 0.5  # *Eq.2*#
        if (mo > 150.0):
            mo = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))) \
                + (.0015 * (mo - 150.0) ** 2) * math.sqrt(rf)  # *Eq.3b*#
        elif mo <= 150.0:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
            # *Eq.3a*#
        if (mo > 250.0):
            mo = 250.0

    ed = .942 * (h ** .679) + (11.0 * math.exp((h - 100.0) / 10.0)) + 0.18 * (21.1 - t) \
        * (1.0 - 1.0 / math.exp(.1150 * h))  # *Eq.4*#

    if (mo < ed):
        ew = .618 * (h ** .753) + (10.0 * math.exp((h - 100.0) / 10.0)) \
            + .18 * (21.1 - t) * (1.0 - 1.0 / math.exp(.115 * h))  # *Eq.5*#
        if (mo <= ew):
            kl = .424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + (.0694 * math.sqrt(w)) \
                * (1.0 - ((100.0 - h) / 100.0) ** 8)  # *Eq.7a*#
            kw = kl * (.581 * math.exp(.0365 * t))  # *Eq.7b*#
            m = ew - (ew - mo) / 10.0 ** kw  # *Eq.9*#
        elif mo > ew:
            m = mo
    elif (mo == ed):
        m = mo
    elif (mo > ed):
        kl = .424 * (1.0 - (h / 100.0) ** 1.7) + (.0694 * math.sqrt(w)) * \
            (1.0 - (h / 100.0) ** 8)  # *Eq.6a*#
        kw = kl * (.581 * math.exp(.0365 * t))  # *Eq.6b*#
        m = ed + (mo - ed) / 10.0 ** kw  # *Eq.8*#

    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)  # *Eq.10*#
    if (ffmc > 101.0):
        ffmc = 101.0
    if (ffmc <= 0.0):
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

    it = np.nditer([tas, pr, ws, rh, None], [], 4 * [['readonly'], ] + [['writeonly', 'allocate']])

    with it:
        for (t, p, w, h, out) in it:
            it[4] = _fine_fuel_moisture_code(t, p, w, h, ffmc0)
            ffmc0 = it[4]

        return it.operands[4]


def _duff_moisture_code(t, p, h, mth, dmc0):
    """Scalar computation of the Duff moisture code."""
    el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]

    t = t
    if (t < -1.1):
        t = -1.1
    rk = 1.894 * (t + 1.1) * (100.0 - h) * (el[mth - 1] * 0.0001)  # *Eqs.16 and 17*#

    if p > 1.5:
        ra = p
        rw = 0.92 * ra - 1.27  # *Eq.11*#
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc0)  # *Eq.12*#
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)  # *Eq.13a*#
        elif dmc0 > 33.0:
            if dmc0 <= 65.0:
                b = 14.0 - 1.3 * math.log(dmc0)  # *Eq.13b*#
            elif dmc0 > 65.0:
                b = 6.2 * math.log(dmc0) - 17.2  # *Eq.13c*#
        wmr = wmi + (1000 * rw) / (48.77 + b * rw)  # *Eq.14*#
        pr = 43.43 * (5.6348 - math.log(wmr - 20.0))  # *Eq.15*#
    elif p <= 1.5:
        pr = dmc0
    if (pr < 0.0):
        pr = 0.0
    dmc = pr + rk
    if (dmc <= 1.0):
        dmc = 1.0
    return dmc


def duff_moisture_code(tas, pr, rh, mth, dmc0):
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
    dmc0 : float
      Initial value of the Duff moisture code.

    Returns
    -------
    array
      Duff moisture code
    """
    it = np.nditer([tas, pr, rh, mth, None], [], 4 * [['readonly'], ] + [['writeonly', 'allocate']])

    with it:
        for (t, p, h, m, out) in it:
            it[4] = _duff_moisture_code(t, p, h, m, dmc0)
            dmc0 = it[4]

        return it.operands[4]


def _drought_code(t, p, mth, dc0):
    """Scalar computation of the drought code."""
    fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    t = t

    if (t < -2.8):
        t = -2.8
    pe = (0.36 * (t + 2.8) + fl[mth - 1]) / 2  # *Eq.22*#
    if pe <= 0.0:
        pe = 0.0

    if (p > 2.8):
        ra = p
        rw = 0.83 * ra - 1.27  # *Eq.18*#
        smi = 800.0 * math.exp(-dc0 / 400.0)  # *Eq.19*#
        dr = dc0 - 400.0 * math.log(1.0 + ((3.937 * rw) / smi))  # *Eqs. 20 and 21*#
        if (dr > 0.0):
            dc = dr + pe
    elif p <= 2.8:
        dc = dc0 + pe
    return dc


def drought_code(tas, pr, mth, dc0):
    """Drought code

    Parameters
    ----------
    tas: array
      Noon temperature [C].
    pr : array
      Rain fall in open over previous 24 hours, at noon [mm].
    mth : integer array
      Month of the year [1-12].
    dc0 : float
      Initial value of the drought code.

    Returns
    -------
    array
      Drought code
    """
    it = np.nditer([tas, pr, mth, None], [], 3 * [['readonly'], ] + [['writeonly', 'allocate']])

    with it:
        for (t, p, m, out) in it:
            it[3] = _drought_code(t, p, m, dc0)
            dc0 = it[3]

        return it.operands[3]


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
    bui = xr.where(dmc <= 0.4 * dc,
                   (0.8 * dc * dmc) / (dmc + 0.4 * dc),   # *Eq.27a*#
                   dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7))  # *Eq.27b*#
    return bui.clip(0, np.inf)


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
    bb = xr.where(bui <= 80.0,
                  0.1 * isi * (0.626 * bui ** 0.809 + 2.0),  # *Eq.28a*#
                  0.1 * isi * (1000.0 / (25. + 108.64 / np.exp(0.023 * bui))))  # *Eq.28b*#

    fwi = xr.where(bb <= 1.0,
                   bb,  # *Eq.30b*#
                   np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647))  # *Eq.30a*#

    return fwi


def ffmc_ufunc(tas, pr, ws, rh, ffmc0):
    return xr.apply_ufunc(fine_fuel_moisture_code,
                          tas, pr, ws, rh,
                          input_core_dims=4 * (('time', ),),
                          output_core_dims=(('time',),),
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float, ],
                          keep_attrs=True,
                          kwargs={'ffmc0': ffmc0})


def dmc_ufunc(tas, pr, rh, mth, dmc0):
    return xr.apply_ufunc(duff_moisture_code,
                          tas, pr, rh, mth,
                          input_core_dims=4 * (('time', ),),
                          output_core_dims=(('time', ), ),
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float, ],
                          keep_attrs=True,
                          kwargs={'dmc0': dmc0})


def dc_ufunc(tas, pr, mth, dc0):
    return xr.apply_ufunc(drought_code,
                          tas, pr, mth,
                          input_core_dims=3 * (('time', ),),
                          output_core_dims=(('time', ), ),
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float, ],
                          keep_attrs=True,
                          kwargs={'dc0': dc0})
