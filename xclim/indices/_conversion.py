import numpy as np
import xarray as xr

from xclim.core.units import convert_units_to
from xclim.core.units import declare_units


__all__ = [
    "tas",
    "uas_vas_2_sfcwind",
    "sfcwind_2_uas_vas",
    "saturation_vapor_pressure",
    "relative_humidity",
    "specific_humidity",
]


@declare_units("[temperature]", tasmin="[temperature]", tasmax="[temperature]")
def tas(tasmin: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """Average temperature from minimum and maximum temperatures.

    We assume a symmetrical distribution for the temperature and retrieve the average value as Tg = (Tx + Tn) / 2

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum (daily) temperature [℃] or [K]
    tasmax : xarray.DataArray
        Maximum (daily) temperature [℃] or [K]

    Returns
    -------
    xarray.DataArray
        Mean (daily) temperature [same units as tasmin]
    """
    tasmax = convert_units_to(tasmax, tasmin)
    tas = (tasmax + tasmin) / 2
    tas.attrs["units"] = tasmin.attrs["units"]
    return tas


@declare_units(None, check_output=False, uas="[speed]", vas="[speed]")
def uas_vas_2_sfcwind(
    uas: xr.DataArray = None, vas: xr.DataArray = None, return_direction=True
):
    """Converts eastward and northward wind components to wind speed and direction.

    Parameters
    ----------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    Returns
    -------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Notes
    -----
    Northerly winds with a velocity less than 0.5 m/s are given a wind direction of 0°,
    while stronger winds are set to 360°.
    """
    # Converts the wind speed to m s-1
    uas = convert_units_to(uas, "m/s")
    vas = convert_units_to(vas, "m/s")

    # Wind speed is the hypotenuse of "uas" and "vas"
    wind = np.hypot(uas, vas)

    if not return_direction:
        wind.attrs["units"] = "m s-1"
        return wind

    # TODO Attributes should be set by the indicator, but there are no multi-output indicators, so we set them anyway if return_direction is True,
    # Add attributes to wind. This is done by copying uas' attributes and overwriting a few of them
    wind.attrs = uas.attrs
    wind.name = "sfcWind"
    wind.attrs["units"] = "m s-1"
    wind.attrs["standard_name"] = "wind_speed"
    wind.attrs["long_name"] = "Near-Surface Wind Speed"
    # Calculate the angle
    windfromdir_math = np.degrees(np.arctan2(vas, uas))

    # Convert the angle from the mathematical standard to the meteorological standard
    windfromdir = (270 - windfromdir_math) % 360.0

    # According to the meteorological standard, calm winds must have a direction of 0°
    # while northerly winds have a direction of 360°
    # On the Beaufort scale, calm winds are defined as < 0.5 m/s
    windfromdir = xr.where((windfromdir.round() == 0) & (wind >= 0.5), 360, windfromdir)
    windfromdir = xr.where(wind < 0.5, 0, windfromdir)

    # Add attributes to winddir. This is done by copying uas' attributes and overwriting a few of them
    windfromdir.attrs = uas.attrs
    windfromdir.name = "sfcWindfromdir"
    windfromdir.attrs["standard_name"] = "wind_from_direction"
    windfromdir.attrs["long_name"] = "Near-Surface Wind from Direction"
    windfromdir.attrs["units"] = "degree"

    return wind, windfromdir


@declare_units(None, check_output=False, wind="[speed]", windfromdir="[]")
def sfcwind_2_uas_vas(wind: xr.DataArray = None, windfromdir: xr.DataArray = None):
    """Converts wind speed and direction to eastward and northward wind components.

    Parameters
    ----------
    wind : xr.DataArray
      Wind velocity (m s-1)
    windfromdir : xr.DataArray
      Direction from which the wind blows, following the meteorological convention where 360 stands for North.

    Returns
    -------
    uas : xr.DataArray
      Eastward wind velocity (m s-1)
    vas : xr.DataArray
      Northward wind velocity (m s-1)

    """
    # Converts the wind speed to m s-1
    wind = convert_units_to(wind, "m/s")

    # Converts the wind direction from the meteorological standard to the mathematical standard
    windfromdir_math = (-windfromdir + 270) % 360.0

    # TODO: This commented part should allow us to resample subdaily wind, but needs to be cleaned up and put elsewhere
    # if resample is not None:
    #     wind = wind.resample(time=resample).mean(dim='time', keep_attrs=True)
    #
    #     # nb_per_day is the number of values each day. This should be calculated
    #     windfromdir_math_per_day = windfromdir_math.reshape((len(wind.time), nb_per_day))
    #     # Averages the subdaily angles around a circle, i.e. mean([0, 360]) = 0, not 180
    #     windfromdir_math = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
    #                                       for angles in windfromdir_math_per_day])

    uas = wind * np.cos(np.radians(windfromdir_math))
    vas = wind * np.sin(np.radians(windfromdir_math))

    # Add attributes to uas and vas. This is done by copying wind' attributes and overwriting a few of them
    uas.attrs = wind.attrs
    uas.name = "uas"
    uas.attrs["standard_name"] = "eastward_wind"
    uas.attrs["long_name"] = "Near-Surface Eastward Wind"
    wind.attrs["units"] = "m s-1"

    vas.attrs = wind.attrs
    vas.name = "vas"
    vas.attrs["standard_name"] = "northward_wind"
    vas.attrs["long_name"] = "Near-Surface Northward Wind"
    wind.attrs["units"] = "m s-1"

    return uas, vas


@declare_units("Pa", tas="[temperature]", ice_thresh="[temperature]")
def saturation_vapor_pressure(
    tas: xr.DataArray, ice_thresh: str = None, method: str = "sonntag90"
) -> xr.DataArray:
    """Compute saturation vapor pressure (e_sat) from the temperature.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature array
    ice_thresh : str
        Threshold temperature under which to switch to equations in reference to ice instead of water.
        If None (default) everything is computed with reference to water.
    method : {"dewpoint", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Which method to use, see notes.

    Notes
    -----
    In all cases implemented here :math:`log(e_{sat})` is an empirically fitted function (usually a polynomial)
    where coefficients can be different when ice is taken as reference instead of water. Available methods are:

    - "goffgratch46" or "GG46", based on [goffgratch46]_, values and equation taken from [voemel]_.
    - "sonntag90" or "SO90", taken from [sonntag90]_.
    - "tetens30" or "TE30", based on [tetens30], values and equation taken from [voemel]_.
    - "wmo08" or "WMO08", taken from [wmo08]_.


    References
    ----------
    .. [goffgratch46] Goff, J. A., and S. Gratch (1946) Low-pressure properties of water from -160 to 212 °F, in Transactions of the American Society of Heating and Ventilating Engineers, pp 95-122, presented at the 52nd annual meeting of the American Society of Heating and Ventilating Engineers, New York, 1946.
    .. [sonntag90] Sonntag, D. (1990). Important new values of the physical constants of 1986, vapour pressure formulations based on the ITS-90, and psychrometer formulae. Zeitschrift für Meteorologie, 40(5), 340-344.
    .. [tetens30] Tetens, O. 1930. Über einige meteorologische Begriffe. Z. Geophys 6: 207-309.
    .. [voemel] http://cires1.colorado.edu/~voemel/vp.html
    .. [wmo2008] World Meteorological Organization. (2008). Guide to meteorological instruments and methods of observation. Geneva, Switzerland: World Meteorological Organization. https://www.weather.gov/media/epz/mesonet/CWOP-WMO8.pdf
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

    return e_sat


@declare_units(
    "%",
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
    """Compute relative humidity from temperature and either dewpoint temperature
    or from specific humidity and pressure (through the saturation vapor pressure)

    Parameters
    ----------
    tas : xr.DataArray
        Temperature array
    dtas : xr.DataArray
        Dewpoint temperature, if specified, "method" must be set to "dewpoint".
    huss : xr.DataArray
        Specific Humidity
    ps : xr.DataArray
        Air Pressure
    ice_thresh : str
        Threshold temperature under which to switch to equations in reference to ice instead of water.
        If None (default) everything is computed with reference to water. Does nothing if 'method' is "dewpoint,"
    method : {"dewpoint", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Which method to use, see notes of this function and of `saturation_vapor_pressure`.
    invalid_values : {"clip", "mask", None}
        What to do with values outside the 0-100 range.
        If "clip" (default), clips everything to 0 - 100,
        if "mask", replaces values outside the range by np.nan,
        if None, does nothing.

    Notes
    -----

    In the following, let :math:`T`, :math:`T_d`, :math:`q` and :math:`p` be the temperature,
    the dew point temperature, the specific humidity and the air pressure.

    **For the "dewpoint" method** : With :math:`L` the Enthalpy of vaporization of water
    and :math:`R_w` the gas constant for water vapor, the relative humidity is computed as:

    .. math::

        RH = e^{\\frac{-L (T - T_d)}{R_wTT_d}}

    Formula taken from [Lawrence_2005]_.

    **Other methods**: With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapor pressure, relative humidity is computed as:

        ... math::

            RH = 100\\frac{w}{w_{sat}}
            w = \\frac{q}{1-q}
            w_{sat} = 0.622\\frac{e_{sat}}{P - e_{sat}}

    The methods differ by how :math:`e_{sat}` is computed. See the doc of `xclim.core.utils.saturation_vapor_pressure`.

    References
    ----------
    .. [Lawrence_2005] Lawrence, M.G. (2005). The Relationship between Relative Humidity and the Dewpoint Temperature in Moist Air: A Simple Conversion and Applications. Bull. Amer. Meteor. Soc., 86, 225–234, https://doi.org/10.1175/BAMS-86-2-225
    """
    if dtas is not None and method != "dewpoint":
        raise ValueError(
            "If the dewpoint temperature (dtas) is passed, method must be set to 'dewpoint'"
        )

    if method == "dewpoint":
        dtas = convert_units_to(dtas, "degK")
        tas = convert_units_to(tas, "degK")
        L = 2.501e6
        Rw = (461.5,)
        rh = 100 * np.exp(-L * (tas - dtas) / (Rw * tas * dtas))
    else:
        ps = convert_units_to(ps, "Pa")
        huss = convert_units_to(huss, "")
        tas = convert_units_to(tas, "degK")

        e_sat = saturation_vapor_pressure(tas=tas, ice_thresh=ice_thresh, method=method)

        w = huss / (1 - huss)
        w_sat = 0.62198 * e_sat / (ps - e_sat)
        rh = 100 * w / w_sat

    if invalid_values == "clip":
        rh = rh.clip(0, 100)
    elif invalid_values == "mask":
        rh = rh.where((rh <= 100) & (rh >= 0))

    return rh


@declare_units(
    "", tas="[temperature]", rh="[]", ps="[pressure]", ice_thresh="[temperature]",
)
def specific_humidity(
    tas: xr.DataArray,
    rh: xr.DataArray,
    ps: xr.DataArray = None,
    ice_thresh: str = None,
    method: str = "sonntag90",
    invalid_values: str = None,
) -> xr.DataArray:
    """Compute specific humidity from temperature, relative humidity and pressure (through the saturation vapor pressure)

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
    method : {"dewpoint", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Which method to use, see notes of this function and of `saturation_vapor_pressure`.
    invalid_values : {"clip", "mask", None}
        What to do with values larger than the saturation specific humidity and lower than 0.
        If "clip" (default), clips everything to 0 - q_sat
        if "mask", replaces values outside the range by np.nan,
        if None, does nothing.

    Notes
    -----

    In the following, let :math:`T`, :math:`rh` (in %) and :math:`p` be the temperature,
    the relative humidity and the air pressure. With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapor pressure, specific humidity :math:`q` is computed as:

        ... math::

            w_{sat} = 0.622\\frac{e_{sat}}{P - e_{sat}}
            w = w_{sat} * rh / 100
            q = w / (1 + w)

    The methods differ by how :math:`e_{sat}` is computed. See the doc of `xclim.core.utils.saturation_vapor_pressure`.

    If `invalid_values` is not `None`, the saturation specific humidity :math:`q_{sat}` is computed as:

        ... math::

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

    return q
