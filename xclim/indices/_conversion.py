# noqa: D100
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from xclim.core.calendar import date_range, datetime_to_decimal_year
from xclim.core.units import amount2rate, convert_units_to, declare_units, units2pint

__all__ = [
    "humidex",
    "tas",
    "uas_vas_2_sfcwind",
    "sfcwind_2_uas_vas",
    "saturation_vapor_pressure",
    "relative_humidity",
    "specific_humidity",
    "snowfall_approximation",
    "rain_approximation",
    "wind_chill_index",
    "clausius_clapeyron_scaled_precipitation",
    "potential_evapotranspiration",
]


@declare_units(tas="[temperature]", tdps="[temperature]", hurs="[]")
def humidex(
    tas: xr.DataArray,
    tdps: Optional[xr.DataArray] = None,
    hurs: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    r"""Humidex index.

    The humidex indicates how hot the air feels to an average person, accounting for the effect of humidity. It
    can be loosely interpreted as the equivalent perceived temperature when the air is dry.

    Parameters
    ----------
    tas : xarray.DataArray
      Air temperature.
    tdps : xarray.DataArray,
      Dewpoint temperature.
    hurs : xarray.DataArray
      Relative humidity.

    Returns
    -------
    xarray.DataArray, [temperature]
      The humidex index.

    Notes
    -----
    The humidex is usually computed using hourly observations of dry bulb and dewpoint temperatures. It is computed
    using the formula based on [masterton79]_:

    .. math::

       T + {\frac {5}{9}}\left[e - 10\right]

    where :math:`T` is the dry bulb air temperature (°C). The term :math:`e` can be computed from the dewpoint
    temperature :math:`T_{dewpoint}` in °K:

    .. math::

       e = 6.112 \times \exp(5417.7530\left({\frac {1}{273.16}}-{\frac {1}{T_{\text{dewpoint}}}}\right)

    where the constant 5417.753 reflects the molecular weight of water, latent heat of vaporization,
    and the universal gas constant ([mekis15]_). Alternatively, the term :math:`e` can also be computed from
    the relative humidity `h` expressed in percent using [sirangelo20]_:

    .. math::

      e = \frac{h}{100} \times 6.112 * 10^{7.5 T/(T + 237.7)}.

    The humidex *comfort scale* ([eccc]_) can be interpreted as follows:

    - 20 to 29 : no discomfort;
    - 30 to 39 : some discomfort;
    - 40 to 45 : great discomfort, avoid exertion;
    - 46 and over : dangerous, possible heat stroke;

    References
    ----------
    .. [masterton79] Masterton, J. M., & Richardson, F. A. (1979). HUMIDEX, A method of quantifying human discomfort due to excessive heat and humidity, CLI 1-79. Downsview, Ontario: Environment Canada, Atmospheric Environment Service.
    .. [mekis15] Éva Mekis, Lucie A. Vincent, Mark W. Shephard & Xuebin Zhang (2015) Observed Trends in Severe Weather Conditions Based on Humidex, Wind Chill, and Heavy Rainfall Events in Canada for 1953–2012, Atmosphere-Ocean, 53:4, 383-397, DOI: 10.1080/07055900.2015.1086970
    .. [sirangelo20] Sirangelo, B., Caloiero, T., Coscarelli, R. et al. Combining stochastic models of air temperature and vapour pressure for the analysis of the bioclimatic comfort through the Humidex. Sci Rep 10, 11395 (2020). https://doi.org/10.1038/s41598-020-68297-4
    .. [eccc] https://climate.weather.gc.ca/glossary_e.html
    """
    if (tdps is None) == (hurs is None):
        raise ValueError(
            "At least one of `tdps` or `hurs` must be given, and not both."
        )

    # Vapour pressure in hPa
    if tdps is not None:
        # Convert dewpoint temperature to Kelvins
        tdps = convert_units_to(tdps, "kelvin")
        e = 6.112 * np.exp(5417.7530 * (1 / 273.16 - 1.0 / tdps))

    elif hurs is not None:
        # Convert dry bulb temperature to Celsius
        tasC = convert_units_to(tas, "celsius")
        e = hurs / 100 * 6.112 * 10 ** (7.5 * tasC / (tasC + 237.7))

    # Temperature delta due to humidity in delta_degC
    h = 5 / 9 * (e - 10)
    h.attrs["units"] = "delta_degree_Celsius"

    # Get delta_units for output
    du = (1 * units2pint(tas) - 0 * units2pint(tas)).units
    h = convert_units_to(h, du)

    # Add the delta to the input temperature
    out = h + tas
    out.attrs["units"] = tas.units
    return out


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
    wind_from_dir : xr.DataArray, [°]
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
    wind_from_dir_math = np.degrees(np.arctan2(vas, uas))

    # Convert the angle from the mathematical standard to the meteorological standard
    wind_from_dir = (270 - wind_from_dir_math) % 360.0

    # According to the meteorological standard, calm winds must have a direction of 0°
    # while northerly winds have a direction of 360°
    # On the Beaufort scale, calm winds are defined as < 0.5 m/s
    wind_from_dir = xr.where(wind_from_dir.round() == 0, 360, wind_from_dir)
    wind_from_dir = xr.where(wind < wind_thresh, 0, wind_from_dir)
    wind_from_dir.attrs["units"] = "degree"
    return wind, wind_from_dir


@declare_units(sfcWind="[speed]", sfcWindfromdir="[]")
def sfcwind_2_uas_vas(
    sfcWind: xr.DataArray, sfcWindfromdir: xr.DataArray  # noqa
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
      Eastward wind velocity.
    vas : xr.DataArray, [m s-1]
      Northward wind velocity.

    """
    # Converts the wind speed to m s-1
    sfcWind = convert_units_to(sfcWind, "m/s")  # noqa

    # Converts the wind direction from the meteorological standard to the mathematical standard
    wind_from_dir_math = (-sfcWindfromdir + 270) % 360.0

    # TODO: This commented part should allow us to resample subdaily wind, but needs to be cleaned up and put elsewhere.
    # if resample is not None:
    #     wind = wind.resample(time=resample).mean(dim='time', keep_attrs=True)
    #
    #     # nb_per_day is the number of values each day. This should be calculated
    #     wind_from_dir_math_per_day = wind_from_dir_math.reshape((len(wind.time), nb_per_day))
    #     # Averages the subdaily angles around a circle, i.e. mean([0, 360]) = 0, not 180
    #     wind_from_dir_math = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
    #                                       for angles in wind_from_dir_math_per_day])

    uas = sfcWind * np.cos(np.radians(wind_from_dir_math))
    vas = sfcWind * np.sin(np.radians(wind_from_dir_math))
    uas.attrs["units"] = "m s-1"
    vas.attrs["units"] = "m s-1"
    return uas, vas


@declare_units(tas="[temperature]", ice_thresh="[temperature]")
def saturation_vapor_pressure(
    tas: xr.DataArray, ice_thresh: str = None, method: str = "sonntag90"  # noqa
) -> xr.DataArray:
    """Saturation vapor pressure from temperature.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array.
    ice_thresh : str
      Threshold temperature under which to switch to equations in reference to ice instead of water.
      If None (default) everything is computed with reference to water.
    method : {"dewpoint", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
      Which method to use, see notes.

    Returns
    -------
    xarray.DataArray, [Pa]
      Saturation vapor pressure.

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
    .. [voemel] https://cires1.colorado.edu/~voemel/vp.html
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
                -6096.9385 / tas  # type: ignore
                + 16.635794
                + -2.711193e-2 * tas  # type: ignore
                + 1.673952e-5 * tas ** 2
                + 2.433502 * np.log(tas)  # numpy's log is ln
            ),
            100
            * np.exp(  # Where ref_is_water is False (thus ref is ice)
                -6024.5282 / tas  # type: ignore
                + 24.7219
                + 1.0613868e-2 * tas  # type: ignore
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
                -7.90298 * ((Tb / tas) - 1)  # type: ignore
                + 5.02808 * np.log10(Tb / tas)  # type: ignore
                + -1.3817e-7 * (10 ** (11.344 * (1 - tas / Tb)) - 1)
                + 8.1328e-3 * (10 ** (-3.49149 * ((Tb / tas) - 1)) - 1)  # type: ignore
            ),
            ep
            * 10
            ** (
                -9.09718 * ((Tp / tas) - 1)  # type: ignore
                + -3.56654 * np.log10(Tp / tas)  # type: ignore
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
    tdps="[temperature]",
    huss="[]",
    ps="[pressure]",
    ice_thresh="[temperature]",
)
def relative_humidity(
    tas: xr.DataArray,
    tdps: xr.DataArray = None,
    huss: xr.DataArray = None,
    ps: xr.DataArray = None,
    ice_thresh: str = None,
    method: str = "sonntag90",
    invalid_values: str = "clip",
) -> xr.DataArray:
    r"""Relative humidity.

    Compute relative humidity from temperature and either dewpoint temperature or specific humidity and pressure through
    the saturation vapor pressure.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array
    tdps : xr.DataArray
      Dewpoint temperature, if specified, overrides huss and ps.
    huss : xr.DataArray
      Specific humidity.
    ps : xr.DataArray
      Air Pressure.
    ice_thresh : str
      Threshold temperature under which to switch to equations in reference to ice instead of water.
      If None (default) everything is computed with reference to water. Does nothing if 'method' is "bohren98".
    method : {"bohren98", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
      Which method to use, see notes of this function and of `saturation_vapor_pressure`.
    invalid_values : {"clip", "mask", None}
      What to do with values outside the 0-100 range. If "clip" (default), clips everything to 0 - 100,
      if "mask", replaces values outside the range by np.nan, and if `None`, does nothing.

    Returns
    -------
    xr.DataArray, [%]
      Relative humidity.

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
        if tdps is None:
            raise ValueError("To use method 'bohren98' (BA98), dewpoint must be given.")
        tdps = convert_units_to(tdps, "degK")
        tas = convert_units_to(tas, "degK")
        L = 2.501e6
        Rw = (461.5,)
        hurs = 100 * np.exp(-L * (tas - tdps) / (Rw * tas * tdps))  # type: ignore
    elif tdps is not None:
        e_sat_dt = saturation_vapor_pressure(
            tas=tdps, ice_thresh=ice_thresh, method=method
        )
        e_sat_t = saturation_vapor_pressure(
            tas=tas, ice_thresh=ice_thresh, method=method
        )
        hurs = 100 * e_sat_dt / e_sat_t  # type: ignore
    else:
        ps = convert_units_to(ps, "Pa")
        huss = convert_units_to(huss, "")
        tas = convert_units_to(tas, "degK")

        e_sat = saturation_vapor_pressure(tas=tas, ice_thresh=ice_thresh, method=method)

        w = huss / (1 - huss)
        w_sat = 0.62198 * e_sat / (ps - e_sat)  # type: ignore
        hurs = 100 * w / w_sat

    if invalid_values == "clip":
        hurs = hurs.clip(0, 100)
    elif invalid_values == "mask":
        hurs = hurs.where((hurs <= 100) & (hurs >= 0))
    hurs.attrs["units"] = "%"
    return hurs


@declare_units(
    tas="[temperature]",
    hurs="[]",
    ps="[pressure]",
    ice_thresh="[temperature]",
)
def specific_humidity(
    tas: xr.DataArray,
    hurs: xr.DataArray,
    ps: xr.DataArray,
    ice_thresh: str = None,
    method: str = "sonntag90",
    invalid_values: str = None,
) -> xr.DataArray:
    r"""Specific humidity from temperature, relative humidity and pressure.

    Parameters
    ----------
    tas : xr.DataArray
      Temperature array
    hurs : xr.DataArray
      Relative Humidity.
    ps : xr.DataArray
      Air Pressure.
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
      Specific humidity.

    Notes
    -----
    In the following, let :math:`T`, :math:`hurs` (in %) and :math:`p` be the temperature,
    the relative humidity and the air pressure. With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapor pressure, specific humidity :math:`q` is computed as:

    .. math::

        w_{sat} = 0.622\frac{e_{sat}}{P - e_{sat}}
        w = w_{sat} * hurs / 100
        q = w / (1 + w)

    The methods differ by how :math:`e_{sat}` is computed. See the doc of `xclim.core.utils.saturation_vapor_pressure`.

    If `invalid_values` is not `None`, the saturation specific humidity :math:`q_{sat}` is computed as:

    .. math::

        q_{sat} = w_{sat} / (1 + w_{sat})
    """
    ps = convert_units_to(ps, "Pa")
    hurs = convert_units_to(hurs, "")
    tas = convert_units_to(tas, "degK")

    e_sat = saturation_vapor_pressure(tas=tas, ice_thresh=ice_thresh, method=method)

    w_sat = 0.62198 * e_sat / (ps - e_sat)  # type: ignore
    w = w_sat * hurs
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
) -> xr.DataArray:
    """Snowfall approximation from total precipitation and temperature.

    Solid precipitation estimated from precipitation and temperature according to a given method.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum, or minimum daily temperature.
    thresh : str,
      Threshold temperature, used by method "binary".
    method : {"binary", "brown", "auer"}
      Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
      Solid precipitation flux.

    Notes
    -----
    The following methods are available to approximate snowfall and are drawn from the
    Canadian Land Surface Scheme (CLASS, [Verseghy09]_).

    - "binary" : When the temperature is under the freezing threshold, precipitation
        is assumed to be solid. The method is agnostic to the type of temperature used
        (mean, maximum or minimum).
    - "brown" : The phase between the freezing threshold goes from solid to liquid linearly
        over a range of 2°C over the freezing point.
    - "auer" : The phase between the freezing threshold goes from solid to liquid as a degree six
        polynomial over a range of 6°C over the freezing point.

    .. [Verseghy09]: Diana Verseghy (2009), CLASS – The Canadian Land Surface Scheme (Version 3.4), Technical
    Documentation (Version 1.1), Environment Canada, Climate Research Division, Science and Technology Branch.
    """
    # https://gitlab.com/cccma/classic/-/blob/master/src/atmosphericVarsCalc.f90

    if method == "binary":
        thresh = convert_units_to(thresh, tas)
        prsn = pr.where(tas <= thresh, 0)

    elif method == "brown":
        # Freezing point + 2C in the native units
        upper = convert_units_to(convert_units_to(thresh, "degC") + 2, tas)
        thresh = convert_units_to(thresh, tas)

        # Interpolate fraction over temperature (in units of tas)
        t = xr.DataArray(
            [-np.inf, thresh, upper, np.inf], dims=("tas",), attrs={"units": "degC"}
        )
        fraction = xr.DataArray([1.0, 1.0, 0.0, 0.0], dims=("tas",), coords={"tas": t})

        # Multiply precip by snowfall fraction
        prsn = pr * fraction.interp(tas=tas, method="linear")

    elif method == "auer":
        dtas = convert_units_to(tas, "degK") - convert_units_to(thresh, "degK")

        # Create nodes for the snowfall fraction: -inf, thresh, ..., thresh+6, inf [degC]
        t = np.concatenate(
            [[-273.15], np.linspace(0, 6, 100, endpoint=False), [6, 1e10]]
        )
        t = xr.DataArray(t, dims="tas", name="tas", coords={"tas": t})

        # The polynomial coefficients, valid between thresh and thresh + 6 (defined in CLASS)
        coeffs = xr.DataArray(
            [100, 4.6664, -15.038, -1.5089, 2.0399, -0.366, 0.0202],
            dims=("degree",),
            coords={"degree": range(7)},
        )

        fraction = xr.polyval(t.tas, coeffs).clip(0, 100) / 100
        fraction[0] = 1
        fraction[-2:] = 0

        # Convert snowfall fraction coordinates to native tas units
        prsn = pr * fraction.interp(tas=dtas, method="linear")

    else:
        raise ValueError(f"Method {method} not one of 'binary', 'brown' or 'auer'.")

    prsn.attrs["units"] = pr.attrs["units"]
    return prsn


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def rain_approximation(
    pr: xr.DataArray,
    tas: xr.DataArray,
    thresh: str = "0 degC",
    method: str = "binary",
) -> xr.DataArray:
    """Rainfall approximation from total precipitation and temperature.

    Liquid precipitation estimated from precipitation and temperature according to a given method.
    This is a convenience method based on `snowfall_approximation`, see the latter for details.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum, or minimum daily temperature.
    thresh : str,
      Threshold temperature, used by method "binary".
    method : {"binary", "brown", "auer"}
      Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
      Liquid precipitation rate.

    Notes
    -----
    See the documentation of `snowfall_approximation` for details. This method computes
    the snowfall approximation and subtracts it from the total precipitation to estimate
    the liquid rain precipitation.

    """
    prra = pr - snowfall_approximation(pr, tas, thresh=thresh, method=method)
    prra.attrs["units"] = pr.attrs["units"]
    return prra


@declare_units(
    tas="[temperature]",
    sfcWind="[speed]",
)
def wind_chill_index(
    tas: xr.DataArray,
    sfcWind: xr.DataArray,
    method: str = "CAN",
    mask_invalid: bool = True,
):
    """Wind chill index.

    The Wind Chill Index is an estimation of how cold the weather feels to the average person.
    It is computed from the air temperature and the 10-m wind. As defined by the Environment and Climate Change Canada ([MVSZ15]_),
    two equations exist, the conventional one and one for slow winds (usually < 5 km/h), see Notes.

    Parameters
    ----------
    tas : xarray.DataArray
      Surface air temperature.
    sfcwind : xarray.DataArray
      Surface wind speed (10 m).
    method : {'CAN', 'US'}
      If "CAN" (default), a "slow wind" equation is used where winds are slower than 5 km/h, see Notes.
    mask_invalid : bool
      Whether to mask values when the inputs are outside their validity range. or not.
      If True (default), points where the temperature is above a threshold are masked.
      The threshold is 0°C for the canadian method and 50°F for the american one.
      With the latter method, points where sfcWind < 3 mph are also masked.

    Returns
    -------
    xarray.DataArray, [degC]
      Wind Chill Index.

    Notes
    -----
    Following the calculations of Environment and Climate Change Canada, this function switches from the standardized index
    to another one for slow winds. The standard index is the same as used by the National Weather Service of the USA. Given
    a temperature at surface :math:`T` (in °C) and 10-m wind speed :math:`V` (in km/h), the Wind Chill Index :math:`W` (dimensionless)
    is computed as:

    .. math::

        W = 13.12 + 0.6125*T - 11.37*V^0.16 + 0.3965*T*V^0.16

    Under slow winds (:math:`V < 5` km/h), and using the canadian method, it becomes:

    .. math::

        W = T + \frac{-1.59 + 0.1345 * T}{5} * V


    Both equations are invalid for temperature over 0°C in the canadian method.

    The american Wind Chill Temperature index (WCT), as defined by USA's National Weather Service, is computed when
    `method='US'`. In that case, the maximal valid temperature is 50°F (10 °C) and minimal wind speed is 3 mph (4.8 km/h).

    References
    ----------
    .. [MVSZ15] Éva Mekis, Lucie A. Vincent, Mark W. Shephard & Xuebin Zhang (2015) Observed Trends in Severe Weather Conditions Based on Humidex, Wind Chill, and Heavy Rainfall Events in Canada for 1953–2012, Atmosphere-Ocean, 53:4, 383-397, DOI: 10.1080/07055900.2015.1086970
    Osczevski, R., & Bluestein, M. (2005). The New Wind Chill Equivalent Temperature Chart. Bulletin of the American Meteorological Society, 86(10), 1453–1458. https://doi.org/10.1175/BAMS-86-10-1453
    .. [NWS] Wind Chill Questions, Cold Resources, National Weather Service, retrieved 25-05-21. https://www.weather.gov/safety/cold-faqs
    """
    tas = convert_units_to(tas, "degC")
    sfcWind = convert_units_to(sfcWind, "km/h")

    V = sfcWind ** 0.16
    W = 13.12 + 0.6215 * tas - 11.37 * V + 0.3965 * tas * V

    if method.upper() == "CAN":
        W = xr.where(sfcWind < 5, tas + sfcWind * (-1.59 + 0.1345 * tas) / 5, W)
    elif method.upper() != "US":
        raise ValueError(f"`method` must be one of 'US' and 'CAN'. Got '{method}'.")

    if mask_invalid:
        mask = {"CAN": tas <= 0, "US": (sfcWind > 4.828032) & (tas <= 10)}
        W = W.where(mask[method.upper()])

    W.attrs["units"] = "degC"
    return W


@declare_units(
    delta_tas="[temperature]",
    pr_baseline="[precipitation]",
)
def clausius_clapeyron_scaled_precipitation(
    delta_tas: xr.DataArray,
    pr_baseline: xr.DataArray,
    cc_scale_factor: float = 1.07,
) -> xr.DataArray:
    """Scale precipitation according to the Clausius-Clapeyron relation.

    Parameters
    ----------
    delta_tas : xarray.DataArray
      Difference in temperature between a baseline climatology and another climatology.
    pr_baseline : xarray.DataArray
      Baseline precipitation to adjust with Clausius-Clapeyron.
    cc_scale_factor : float (default  = 1.07)
      Clausius Clapeyron scale factor.

    Returns
    -------
    DataArray
        Baseline precipitation scaled to other climatology using Clausius-Clapeyron relationship.

    Notes
    -----
    The Clausius-Clapeyron equation for water vapor under typical atmospheric conditions states that the saturation
    water vapor pressure :math:`e_s` changes approximately exponentially with temperature

        .. math::

        \frac{\\mathrm{d}e_s(T)}{\\mathrm{d}T} \approx 1.07 e_s(T)

    This function assumes that precipitation can be scaled by the same factor.

    Warnings
    --------
    Make sure that `delta_tas` is computed over a baseline compatible with `pr_baseline`. So for example,
    if `delta_tas` is the climatological difference between a baseline and a future period, then `pr_baseline`
    should be precipitations over a period within the same baseline.
    """

    # Get difference in temperature.  Time-invariant baseline temperature (from above) is broadcast.
    delta_tas = convert_units_to(delta_tas, "delta_degreeC")

    # Calculate scaled precipitation.
    pr_out = pr_baseline * (cc_scale_factor ** delta_tas)
    pr_out.attrs["units"] = pr_baseline.attrs["units"]

    return pr_out


@declare_units(tasmin="[temperature]", tasmax="[temperature]", tas="[temperature]")
def potential_evapotranspiration(
    tasmin: Optional[xr.DataArray] = None,
    tasmax: Optional[xr.DataArray] = None,
    tas: Optional[xr.DataArray] = None,
    method: str = "BR65",
) -> xr.DataArray:
    """Potential evapotranspiration.

    The potential for water evaporation from soil and transpiration by plants if the water supply is
    sufficient, according to a given method.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tas : xarray.DataArray
      Mean daily temperature.
    method : {"baierrobertson65", "BR65", "hargreaves85", "HG85", "thornthwaite48", "TW48"}
      Which method to use, see notes.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    Available methods are:

    - "baierrobertson65" or "BR65", based on [baierrobertson65]_. Requires tasmin and tasmax, daily [D] freq.
    - "hargreaves85" or "HG85", based on [hargreaves85]_. Requires tasmin and tasmax, daily [D] freq. (optional: tas can be given in addition of tasmin and tasmax).
    - "thornthwaite48" or "TW48", based on [thornthwaite48]_. Requires tasmin and tasmax, monthly [MS] or daily [D] freq. (optional: tas can be given instead of tasmin and tasmax).

    References
    ----------
    .. [baierrobertson65] Baier, W., & Robertson, G. W. (1965). Estimation of latent evaporation from simple weather observations. Canadian journal of plant science, 45(3), 276-284.
    .. [hargreaves85] Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied engineering in agriculture, 1(2), 96-99.
    .. [thornthwaite48] Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. Geographical review, 38(1), 55-94.
    """

    if method in ["baierrobertson65", "BR65"]:
        tasmin = convert_units_to(tasmin, "degF")
        tasmax = convert_units_to(tasmax, "degF")

        latr = (tasmin.lat * np.pi) / 180
        gsc = 0.082  # MJ/m2/min

        # julian day fraction
        jd_frac = (datetime_to_decimal_year(tasmin.time) % 1) * 2 * np.pi

        ds = 0.409 * np.sin(jd_frac - 1.39)
        dr = 1 + 0.033 * np.cos(jd_frac)
        omega = np.arccos(-np.tan(latr) * np.tan(ds))
        re = (
            (24 * 60 / np.pi)
            * gsc
            * dr
            * (
                omega * np.sin(latr) * np.sin(ds)
                + np.cos(latr) * np.cos(ds) * np.sin(omega)
            )
        )  # MJ/m2/day
        re = re / 4.1864e-2  # cal/cm2/day

        # Baier et Robertson(1965) formula
        out = 0.094 * (
            -87.03 + 0.928 * tasmax + 0.933 * (tasmax - tasmin) + 0.0486 * re
        )
        out = out.clip(0)

    elif method in ["hargreaves85", "HG85"]:
        tasmin = convert_units_to(tasmin, "degC")
        tasmax = convert_units_to(tasmax, "degC")
        if tas is None:
            tas = (tasmin + tasmax) / 2
        else:
            tas = convert_units_to(tas, "degC")

        latr = (tasmin.lat * np.pi) / 180
        gsc = 0.082  # MJ/m2/min
        lv = 2.5  # MJ/kg

        # julian day fraction
        jd_frac = (datetime_to_decimal_year(tasmin.time) % 1) * 2 * np.pi

        ds = 0.409 * np.sin(jd_frac - 1.39)
        dr = 1 + 0.033 * np.cos(jd_frac)
        omega = np.arccos(-np.tan(latr) * np.tan(ds))
        ra = (
            (24 * 60 / np.pi)
            * gsc
            * dr
            * (
                omega * np.sin(latr) * np.sin(ds)
                + np.cos(latr) * np.cos(ds) * np.sin(omega)
            )
        )  # MJ/m2/day

        # Hargreaves and Samani(1985) formula
        out = (0.0023 * ra * (tas + 17.8) * (tasmax - tasmin) ** 0.5) / lv
        out = out.clip(0)

    elif method in ["thornthwaite48", "TW48"]:
        if tas is None:
            tasmin = convert_units_to(tasmin, "degC")
            tasmax = convert_units_to(tasmax, "degC")
            tas = (tasmin + tasmax) / 2
        else:
            tas = convert_units_to(tas, "degC")
        tas = tas.clip(0)
        tas = tas.resample(time="MS").mean(dim="time")

        latr = (tas.lat * np.pi) / 180  # rad

        start = "-".join(
            [
                str(tas.time[0].dt.year.values),
                "{:02d}".format(tas.time[0].dt.month.values),
                "01",
            ]
        )

        end = "-".join(
            [
                str(tas.time[-1].dt.year.values),
                "{:02d}".format(tas.time[-1].dt.month.values),
                str(tas.time[-1].dt.daysinmonth.values),
            ]
        )

        time_v = xr.DataArray(
            date_range(start, end, freq="D", calendar="standard"),
            dims="time",
            name="time",
        )

        # julian day fraction
        jd_frac = (datetime_to_decimal_year(time_v) % 1) * 2 * np.pi

        ds = 0.409 * np.sin(jd_frac - 1.39)
        omega = np.arccos(-np.tan(latr) * np.tan(ds)) * 180 / np.pi  # degrees

        # monthly-mean daytime length (multiples of 12 hours)
        dl = 2 * omega / (15 * 12)
        dl_m = dl.resample(time="MS").mean(dim="time")

        # annual heat index
        id_m = (tas / 5) ** 1.514
        id_y = id_m.resample(time="YS").sum(dim="time")

        tas_idy_a = []
        for base_time, indexes in tas.resample(time="YS").groups.items():
            tas_y = tas.isel(time=indexes)
            id_v = id_y.sel(time=base_time)
            a = 6.75e-7 * id_v ** 3 - 7.71e-5 * id_v ** 2 + 0.01791 * id_v + 0.49239

            frac = (10 * tas_y / id_v) ** a
            tas_idy_a.append(frac)

        tas_idy_a = xr.concat(tas_idy_a, dim="time")

        # Thornthwaite(1948) formula
        out = 1.6 * dl_m * tas_idy_a  # cm/month
        out = 10 * out  # mm/month

    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")

    out.attrs["units"] = "mm"
    return amount2rate(out, out_units="kg m-2 s-1")
