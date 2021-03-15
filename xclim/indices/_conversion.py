# noqa: D100
from typing import Tuple

import numpy as np
import xarray as xr

from xclim.core.units import convert_units_to, declare_units

__all__ = [
    "tas",
    "uas_vas_2_sfcwind",
    "sfcwind_2_uas_vas",
    "saturation_vapor_pressure",
    "relative_humidity",
    "specific_humidity",
    "snowfall_approximation",
    "rain_approximation",
]


@declare_units(tasmin="[temperature]", tasmax="[temperature]")
def tas(tasmin: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """Average temperature from minimum and maximum temperatures.

    We assume a symmetrical distribution for the temperature and retrieve the average value as Tg = (Tx + Tn) / 2

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum (daily) temperature
    tasmax : xarray.DataArray
        Maximum (daily) temperature

    Returns
    -------
    xarray.DataArray
        Mean (daily) temperature [same units as tasmin]
    """
    tasmax = convert_units_to(tasmax, tasmin)
    tas = (tasmax + tasmin) / 2
    tas.attrs["units"] = tasmin.attrs["units"]
    return tas


@declare_units(uas="[speed]", vas="[speed]", calm_wind_thresh="[speed]")
def uas_vas_2_sfcwind(
    uas: xr.DataArray, vas: xr.DataArray, calm_wind_thresh: str = "0.5 m/s"
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Wind speed and direction from the eastward and northward wind components.

    Computes the magnitude and angle of the wind vector from its northward and eastward components,
    following the meteorological convention that sets calm wind to a direction of 0° and northerly wind to 360°.

    Parameters
    ----------
    uas : xr.DataArray
      Eastward wind velocity
    vas : xr.DataArray
      Northward wind velocity
    calm_wind_thresh : str
      The threshold under which winds are considered "calm" and for which the direction
      is set to 0. On the Beaufort scale, calm winds are defined as < 0.5 m/s.

    Returns
    -------
    wind : xr.DataArray, [m s-1]
      Wind velocity
    windfromdir : xr.DataArray, [°]
      Direction from which the wind blows, following the meteorological convention where
      360 stands for North and 0 for calm winds.

    Notes
    -----
    Winds with a velocity less than `calm_wind_thresh` are given a wind direction of 0°,
    while stronger northerly winds are set to 360°.
    """
    # Converts the wind speed to m s-1
    uas = convert_units_to(uas, "m/s")
    vas = convert_units_to(vas, "m/s")
    wind_thresh = convert_units_to(calm_wind_thresh, "m/s")

    # Wind speed is the hypotenuse of "uas" and "vas"
    wind = np.hypot(uas, vas)
    wind.attrs["units"] = "m s-1"

    # Calculate the angle
    windfromdir_math = np.degrees(np.arctan2(vas, uas))

    # Convert the angle from the mathematical standard to the meteorological standard
    windfromdir = (270 - windfromdir_math) % 360.0

    # According to the meteorological standard, calm winds must have a direction of 0°
    # while northerly winds have a direction of 360°
    # On the Beaufort scale, calm winds are defined as < 0.5 m/s
    windfromdir = xr.where(windfromdir.round() == 0, 360, windfromdir)
    windfromdir = xr.where(wind < wind_thresh, 0, windfromdir)
    windfromdir.attrs["units"] = "degree"
    return wind, windfromdir


@declare_units(sfcWind="[speed]", sfcWindfromdir="[]")
def sfcwind_2_uas_vas(
    sfcWind: xr.DataArray, sfcWindfromdir: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Eastward and northward wind components from the wind speed and direction.

    Compute the eastward and northward wind components from the wind speed and direction.

    Parameters
    ----------
    sfcWind : xr.DataArray
      Wind velocity
    sfcWindfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention
      where 360 stands for North.

    Returns
    -------
    uas : xr.DataArray, [m s-1]
      Eastward wind velocity
    vas : xr.DataArray, [m s-1]
      Northward wind velocity

    """
    # Converts the wind speed to m s-1
    sfcWind = convert_units_to(sfcWind, "m/s")

    # Converts the wind direction from the meteorological standard to the mathematical standard
    windfromdir_math = (-sfcWindfromdir + 270) % 360.0

    # TODO: This commented part should allow us to resample subdaily wind, but needs to be cleaned up and put elsewhere
    # if resample is not None:
    #     wind = wind.resample(time=resample).mean(dim='time', keep_attrs=True)
    #
    #     # nb_per_day is the number of values each day. This should be calculated
    #     windfromdir_math_per_day = windfromdir_math.reshape((len(wind.time), nb_per_day))
    #     # Averages the subdaily angles around a circle, i.e. mean([0, 360]) = 0, not 180
    #     windfromdir_math = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
    #                                       for angles in windfromdir_math_per_day])

    uas = sfcWind * np.cos(np.radians(windfromdir_math))
    vas = sfcWind * np.sin(np.radians(windfromdir_math))
    uas.attrs["units"] = "m s-1"
    vas.attrs["units"] = "m s-1"
    return uas, vas


@declare_units(tas="[temperature]", ice_thresh="[temperature]")
def saturation_vapor_pressure(
    tas: xr.DataArray, ice_thresh: str = None, method: str = "sonntag90"
) -> xr.DataArray:
    """Saturation vapor pressure from temperature.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array
    ice_thresh : str
      Threshold temperature under which to switch to equations in reference to ice instead of water.
      If None (default) everything is computed with reference to water.
    method : {"dewpoint", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
      Which method to use, see notes.

    Returns
    -------
    xarray.DataArray, [Pa]
      Saturation vapor pressure

    Notes
    -----
    In all cases implemented here :math:`log(e_{sat})` is an empirically fitted function (usually a polynomial)
    where coefficients can be different when ice is taken as reference instead of water. Available methods are:

    - "goffgratch46" or "GG46", based on [goffgratch46]_, values and equation taken from [voemel]_.
    - "sonntag90" or "SO90", taken from [sonntag90]_.
    - "tetens30" or "TE30", based on [tetens30]_, values and equation taken from [voemel]_.
    - "wmo08" or "WMO08", taken from [wmo08]_.


    References
    ----------
    .. [goffgratch46] Goff, J. A., and S. Gratch (1946) Low-pressure properties of water from -160 to 212 °F, in Transactions of the American Society of Heating and Ventilating Engineers, pp 95-122, presented at the 52nd annual meeting of the American Society of Heating and Ventilating Engineers, New York, 1946.
    .. [sonntag90] Sonntag, D. (1990). Important new values of the physical constants of 1986, vapour pressure formulations based on the ITS-90, and psychrometer formulae. Zeitschrift für Meteorologie, 40(5), 340-344.
    .. [tetens30] Tetens, O. 1930. Über einige meteorologische Begriffe. Z. Geophys 6: 207-309.
    .. [voemel] http://cires1.colorado.edu/~voemel/vp.html
    .. [wmo08] World Meteorological Organization. (2008). Guide to meteorological instruments and methods of observation. Geneva, Switzerland: World Meteorological Organization. https://www.weather.gov/media/epz/mesonet/CWOP-WMO8.pdf
    """
    if ice_thresh is not None:
        thresh = convert_units_to(ice_thresh, "degK")
    else:
        thresh = convert_units_to("0 K", "degK")
    ref_is_water = tas > thresh

    if method in ["sonntag90", "SO90"]:
        e_sat = xr.where(
            ref_is_water,
            100
            * np.exp(  # Where ref_is_water is True, x100 is to convert hPa to Pa
                -6096.9385 / tas
                + 16.635794
                + -2.711193e-2 * tas
                + 1.673952e-5 * tas ** 2
                + 2.433502 * np.log(tas)  # numpy's log is ln
            ),
            100
            * np.exp(  # Where ref_is_water is False (thus ref is ice)
                -6024.5282 / tas
                + 24.7219
                + 1.0613868e-2 * tas
                + -1.3198825e-5 * tas ** 2
                + -0.49382577 * np.log(tas)
            ),
        )
    elif method in ["tetens30", "TE30"]:
        e_sat = xr.where(
            ref_is_water,
            610.78 * np.exp(17.269388 * (tas - 273.16) / (tas - 35.86)),
            610.78 * np.exp(21.8745584 * (tas - 273.16) / (tas - 7.66)),
        )
    elif method in ["goffgratch46", "GG46"]:
        Tb = 373.16  # Water boiling temp [K]
        eb = 101325  # e_sat at Tb [Pa]
        Tp = 273.16  # Triple-point temperature [K]
        ep = 611.73  # e_sat at Tp [Pa]
        e_sat = xr.where(
            ref_is_water,
            eb
            * 10
            ** (
                -7.90298 * ((Tb / tas) - 1)
                + 5.02808 * np.log10(Tb / tas)
                + -1.3817e-7 * (10 ** (11.344 * (1 - tas / Tb)) - 1)
                + 8.1328e-3 * (10 ** (-3.49149 * ((Tb / tas) - 1)) - 1)
            ),
            ep
            * 10
            ** (
                -9.09718 * ((Tp / tas) - 1)
                + -3.56654 * np.log10(Tp / tas)
                + 0.876793 * (1 - tas / Tp)
            ),
        )
    elif method in ["wmo08", "WMO08"]:
        e_sat = xr.where(
            ref_is_water,
            611.2 * np.exp(17.62 * (tas - 273.16) / (tas - 30.04)),
            611.2 * np.exp(22.46 * (tas - 273.16) / (tas - 0.54)),
        )
    else:
        raise ValueError(
            f"Method {method} is not in ['sonntag90', 'tetens30', 'goffgratch46', 'wmo08']"
        )

    e_sat.attrs["units"] = "Pa"
    return e_sat


@declare_units(
    tas="[temperature]",
    dtas="[temperature]",
    huss="[]",
    ps="[pressure]",
    ice_thresh="[temperature]",
)
def relative_humidity(
    tas: xr.DataArray,
    dtas: xr.DataArray = None,
    huss: xr.DataArray = None,
    ps: xr.DataArray = None,
    ice_thresh: str = None,
    method: str = "sonntag90",
    invalid_values: str = "clip",
) -> xr.DataArray:
    r"""
    Relative humidity.

    Compute relative humidity from temperature and either dewpoint temperature or specific humidity and pressure through the saturation vapor pressure.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array
    dtas : xr.DataArray
      Dewpoint temperature, if specified, overrides huss and ps.
    huss : xr.DataArray
      Specific Humidity
    ps : xr.DataArray
      Air Pressure
    ice_thresh : str
      Threshold temperature under which to switch to equations in reference to ice instead of water.
      If None (default) everything is computed with reference to water. Does nothing if 'method' is "bohren98".
    method : {"bohren98", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
      Which method to use, see notes of this function and of `saturation_vapor_pressure`.
    invalid_values : {"clip", "mask", None}
      What to do with values outside the 0-100 range. If "clip" (default), clips everything to 0 - 100,
      if "mask", replaces values outside the range by np.nan, and if `None` , does nothing.

    Returns
    -------
    xr.DataArray, [%]
      Relative humidity

    Notes
    -----
    In the following, let :math:`T`, :math:`T_d`, :math:`q` and :math:`p` be the temperature,
    the dew point temperature, the specific humidity and the air pressure.

    **For the "bohren98" method** : This method does not use the saturation vapor pressure directly,
    but rather uses an approximation of the ratio of :math:`\frac{e_{sat}(T_d)}{e_{sat}(T)}`.
    With :math:`L` the enthalpy of vaporization of water and :math:`R_w` the gas constant for water vapor,
    the relative humidity is computed as:

    .. math::

        RH = e^{\frac{-L (T - T_d)}{R_wTT_d}}

    From [BohrenAlbrecht1998]_, formula taken from [Lawrence2005]_. :math:`L = 2.5\times 10^{-6}` J kg-1, exact for :math:`T = 273.15` K, is used.

    **Other methods**: With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapor pressure.
    If the dewpoint temperature is given, relative humidity is computed as:

    .. math::

        RH = 100\frac{e_{sat}(T_d)}{e_{sat}(T)}

    Otherwise, the specific humidity and the air pressure must be given so relative humidity can be computed as:

    .. math::

        RH = 100\frac{w}{w_{sat}}
        w = \frac{q}{1-q}
        w_{sat} = 0.622\frac{e_{sat}}{P - e_{sat}}

    The methods differ by how :math:`e_{sat}` is computed. See the doc of :py:meth:`xclim.core.utils.saturation_vapor_pressure`.

    References
    ----------
    .. [Lawrence2005] Lawrence, M.G. (2005). The Relationship between Relative Humidity and the Dewpoint Temperature in Moist Air: A Simple Conversion and Applications. Bull. Amer. Meteor. Soc., 86, 225–234, https://doi.org/10.1175/BAMS-86-2-225
    .. [BohrenAlbrecht1998] Craig F. Bohren, Bruce A. Albrecht. Atmospheric Thermodynamics. Oxford University Press, 1998.
    """
    if method in ("bohren98", "BA90"):
        if dtas is None:
            raise ValueError("To use method 'bohren98' (BA98), dewpoint must be given.")
        dtas = convert_units_to(dtas, "degK")
        tas = convert_units_to(tas, "degK")
        L = 2.501e6
        Rw = (461.5,)
        rh = 100 * np.exp(-L * (tas - dtas) / (Rw * tas * dtas))  # type: ignore
    elif dtas is not None:
        e_sat_dt = saturation_vapor_pressure(
            tas=dtas, ice_thresh=ice_thresh, method=method
        )
        e_sat_t = saturation_vapor_pressure(
            tas=tas, ice_thresh=ice_thresh, method=method
        )
        rh = 100 * e_sat_dt / e_sat_t  # type: ignore
    else:
        ps = convert_units_to(ps, "Pa")
        huss = convert_units_to(huss, "")
        tas = convert_units_to(tas, "degK")

        e_sat = saturation_vapor_pressure(tas=tas, ice_thresh=ice_thresh, method=method)

        w = huss / (1 - huss)
        w_sat = 0.62198 * e_sat / (ps - e_sat)  # type: ignore
        rh = 100 * w / w_sat

    if invalid_values == "clip":
        rh = rh.clip(0, 100)
    elif invalid_values == "mask":
        rh = rh.where((rh <= 100) & (rh >= 0))
    rh.attrs["units"] = "%"
    return rh


@declare_units(
    tas="[temperature]",
    rh="[]",
    ps="[pressure]",
    ice_thresh="[temperature]",
)
def specific_humidity(
    tas: xr.DataArray,
    rh: xr.DataArray,
    ps: xr.DataArray = None,
    ice_thresh: str = None,
    method: str = "sonntag90",
    invalid_values: str = None,
) -> xr.DataArray:
    r"""Specific humidity from temperature, relative humidity and pressure.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array
    rh : xr.DataArrsay
    ps : xr.DataArray
      Air Pressure
    ice_thresh : str
      Threshold temperature under which to switch to equations in reference to ice instead of water.
      If None (default) everything is computed with reference to water.
    method : {"goffgratch46", "sonntag90", "tetens30", "wmo08"}
      Which method to use, see notes of this function and of `saturation_vapor_pressure`.
    invalid_values : {"clip", "mask", None}
      What to do with values larger than the saturation specific humidity and lower than 0.
      If "clip" (default), clips everything to 0 - q_sat
      if "mask", replaces values outside the range by np.nan,
      if None, does nothing.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Specific humidity

    Notes
    -----
    In the following, let :math:`T`, :math:`rh` (in %) and :math:`p` be the temperature,
    the relative humidity and the air pressure. With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapor pressure, specific humidity :math:`q` is computed as:

    .. math::

        w_{sat} = 0.622\frac{e_{sat}}{P - e_{sat}}
        w = w_{sat} * rh / 100
        q = w / (1 + w)

    The methods differ by how :math:`e_{sat}` is computed. See the doc of `xclim.core.utils.saturation_vapor_pressure`.

    If `invalid_values` is not `None`, the saturation specific humidity :math:`q_{sat}` is computed as:

    .. math::

        q_{sat} = w_{sat} / (1 + w_{sat})
    """
    ps = convert_units_to(ps, "Pa")
    rh = convert_units_to(rh, "")
    tas = convert_units_to(tas, "degK")

    e_sat = saturation_vapor_pressure(tas=tas, ice_thresh=ice_thresh, method=method)

    w_sat = 0.62198 * e_sat / (ps - e_sat)
    w = w_sat * rh
    q = w / (1 + w)

    if invalid_values is not None:
        q_sat = w_sat / (1 + w_sat)
        if invalid_values == "clip":
            q = q.clip(0, q_sat)
        elif invalid_values == "mask":
            q = q.where((q <= q_sat) & (q >= 0))
    q.attrs["units"] = ""
    return q


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def snowfall_approximation(
    pr: xr.DataArray,
    tas: xr.DataArray,
    thresh: str = "0 degC",
    method: str = "binary",
):
    """Snowfall approximation from total precipitation and temperature.

    Solid precipitation estimated from precipitation and temperature according to a given method.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum or minimum daily temperature.
    thresh : str,
      Threshold temperature, used by method "binary".
    method : {"binary"}
      Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
      Solid precipitation flux

    Notes
    -----
    The following methods are available to approximate snowfall:

    - "binary" : When the given temperature is under a given threshold, precipitation
        is assumed to be solid. The method is agnostic to the type of temperature used
        (mean, maximum or minimum).

    """
    thresh = convert_units_to(thresh, tas)
    if method == "binary":
        prsn = pr.where(tas < thresh, 0)
    else:
        raise ValueError(f"Method {method} not in ['binary'].")

    prsn.attrs["units"] = pr.attrs["units"]
    return prsn


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def rain_approximation(
    pr: xr.DataArray,
    tas: xr.DataArray,
    thresh: str = "0 degC",
    method: str = "binary",
):
    """Rainfall approximation from total precipitation and temperature.

    Liquid precipitation estimated from precipitation and temperature according to a given method.
    This is a convenience method based on `snowfall_approximation`, see the latter for details.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum or minimum daily temperature.
    thresh : str,
      Threshold temperature, used by method "binary".
    method : {"binary"}
      Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
      Liquid precipitation rate

    Notes
    -----
    See the documentation of `snowfall_approximation` for details. This method computes
    the snowfall approximation and subtracts it from the total precipitation to estimate
    the liquid rain precipitation.

    """
    prlp = pr - snowfall_approximation(pr, tas, thresh=thresh, method=method)
    prlp.attrs["units"] = pr.attrs["units"]
    return prlp
