# noqa: D100
from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from numba import float32, float64, vectorize  # noqa

from xclim.core.calendar import date_range, datetime_to_decimal_year
from xclim.core.units import (
    amount2rate,
    convert_units_to,
    declare_units,
    flux2rate,
    rate2flux,
    units,
    units2pint,
)
from xclim.core.utils import Quantified
from xclim.indices.helpers import (
    _gather_lat,
    _gather_lon,
    cosine_of_solar_zenith_angle,
    day_lengths,
    distance_from_sun,
    extraterrestrial_solar_radiation,
    solar_declination,
    time_correction_for_solar_angle,
    wind_speed_height_conversion,
)

__all__ = [
    "clausius_clapeyron_scaled_precipitation",
    "heat_index",
    "humidex",
    "longwave_upwelling_radiation_from_net_downwelling",
    "mean_radiant_temperature",
    "potential_evapotranspiration",
    "prsn_to_prsnd",
    "prsnd_to_prsn",
    "rain_approximation",
    "relative_humidity",
    "saturation_vapor_pressure",
    "sfcwind_2_uas_vas",
    "shortwave_upwelling_radiation_from_net_downwelling",
    "snd_to_snw",
    "snowfall_approximation",
    "snw_to_snd",
    "specific_humidity",
    "specific_humidity_from_dewpoint",
    "tas",
    "uas_vas_2_sfcwind",
    "universal_thermal_climate_index",
    "wind_chill_index",
]


def _deaccumulate(ds: xr.DataArray) -> xr.DataArray:
    """Deaccumulate units."""


@declare_units(tas="[temperature]", tdps="[temperature]", hurs="[]")
def humidex(
    tas: xr.DataArray,
    tdps: xr.DataArray | None = None,
    hurs: xr.DataArray | None = None,
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
    using the formula based on :cite:t:`masterton_humidex_1979`:

    .. math::

       T + {\frac {5}{9}}\left[e - 10\right]

    where :math:`T` is the dry bulb air temperature (°C). The term :math:`e` can be computed from the dewpoint
    temperature :math:`T_{dewpoint}` in °K:

    .. math::

       e = 6.112 \times \exp(5417.7530\left({\frac {1}{273.16}}-{\frac {1}{T_{\text{dewpoint}}}}\right)

    where the constant 5417.753 reflects the molecular weight of water, latent heat of vaporization,
    and the universal gas constant :cite:p:`mekis_observed_2015`. Alternatively, the term :math:`e` can also be computed
    from the relative humidity `h` expressed in percent using :cite:t:`sirangelo_combining_2020`:

    .. math::

      e = \frac{h}{100} \times 6.112 * 10^{7.5 T/(T + 237.7)}.

    The humidex *comfort scale* :cite:p:`canada_glossary_2011` can be interpreted as follows:

    - 20 to 29 : no discomfort;
    - 30 to 39 : some discomfort;
    - 40 to 45 : great discomfort, avoid exertion;
    - 46 and over : dangerous, possible heat stroke;

    Please note that while both the humidex and the heat index are calculated using dew point, the humidex uses
    a dew point of 7 °C (45 °F) as a base, whereas the heat index uses a dew point base of 14 °C (57 °F). Further,
    the heat index uses heat balance equations which account for many variables other than vapour pressure,
    which is used exclusively in the humidex calculation.

    References
    ----------
    :cite:cts:`canada_glossary_2011,masterton_humidex_1979,mekis_observed_2015,sirangelo_combining_2020`
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


@declare_units(tas="[temperature]", hurs="[]")
def heat_index(tas: xr.DataArray, hurs: xr.DataArray) -> xr.DataArray:
    r"""Heat index.

    Perceived temperature after relative humidity is taken into account :cite:p:`blazejczyk_comparison_2012`.
    The index is only valid for temperatures above 20°C.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature. The equation assumes an instantaneous value.
    hurs : xr.DataArray
        Relative humidity. The equation assumes an instantaneous value.

    Returns
    -------
    xr.DataArray, [temperature]
        Heat index for moments with temperature above 20°C.

    References
    ----------
    :cite:cts:`blazejczyk_comparison_2012`

    Notes
    -----
    While both the humidex and the heat index are calculated using dew point the humidex uses a dew point of 7 °C
    (45 °F) as a base, whereas the heat index uses a dew point base of 14 °C (57 °F). Further, the heat index uses
    heat balance equations which account for many variables other than vapour pressure, which is used exclusively in the
    humidex calculation.
    """
    thresh = 20  # degC
    t = convert_units_to(tas, "degC")
    t = t.where(t > thresh)
    r = convert_units_to(hurs, "%")

    out = (
        -8.78469475556
        + 1.61139411 * t
        + 2.33854883889 * r
        - 0.14611605 * t * r
        - 0.012308094 * t * t
        - 0.0164248277778 * r * r
        + 0.002211732 * t * t * r
        + 0.00072546 * t * r * r
        - 0.000003582 * t * t * r * r
    )
    out = out.assign_attrs(units="degC")
    return convert_units_to(out, tas.units)


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

    Examples
    --------
    >>> from xclim.indices import tas
    >>> tas = tas(tasmin_dataset, tasmax_dataset)
    """
    tasmax = convert_units_to(tasmax, tasmin)
    tas = (tasmax + tasmin) / 2
    tas.attrs["units"] = tasmin.attrs["units"]
    return tas


@declare_units(uas="[speed]", vas="[speed]", calm_wind_thresh="[speed]")
def uas_vas_2_sfcwind(
    uas: xr.DataArray, vas: xr.DataArray, calm_wind_thresh: Quantified = "0.5 m/s"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Wind speed and direction from the eastward and northward wind components.

    Computes the magnitude and angle of the wind vector from its northward and eastward components,
    following the meteorological convention that sets calm wind to a direction of 0° and northerly wind to 360°.

    Parameters
    ----------
    uas : xr.DataArray
        Eastward wind velocity
    vas : xr.DataArray
        Northward wind velocity
    calm_wind_thresh : Quantified
        The threshold under which winds are considered "calm" and for which the direction
        is set to 0. On the Beaufort scale, calm winds are defined as < 0.5 m/s.

    Returns
    -------
    wind : xr.DataArray, [m s-1]
        Wind velocity
    wind_from_dir : xr.DataArray, [°]
        Direction from which the wind blows, following the meteorological convention where
        360 stands for North and 0 for calm winds.

    Examples
    --------
    >>> from xclim.indices import uas_vas_2_sfcwind
    >>> sfcWind = uas_vas_2_sfcwind(
    ...     uas=uas_dataset, vas=vas_dataset, calm_wind_thresh="0.5 m/s"
    ... )

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
) -> tuple[xr.DataArray, xr.DataArray]:
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

    Examples
    --------
    >>> from xclim.indices import sfcwind_2_uas_vas
    >>> uas, vas = sfcwind_2_uas_vas(
    ...     sfcWind=sfcWind_dataset, sfcWindfromdir=sfcWindfromdir_dataset
    ... )
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
    tas: xr.DataArray,
    ice_thresh: Quantified | None = None,
    method: str = "sonntag90",  # noqa
) -> xr.DataArray:
    """Saturation vapour pressure from temperature.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature array.
    ice_thresh : Quantified, optional
        Threshold temperature under which to switch to equations in reference to ice instead of water.
        If None (default) everything is computed with reference to water.
    method : {"goffgratch46", "sonntag90", "tetens30", "wmo08", "its90"}
        Which method to use, see notes.

    Returns
    -------
    xarray.DataArray, [Pa]
        Saturation vapour pressure.

    Notes
    -----
    In all cases implemented here :math:`log(e_{sat})` is an empirically fitted function (usually a polynomial)
    where coefficients can be different when ice is taken as reference instead of water. Available methods are:

    - "goffgratch46" or "GG46", based on :cite:t:`goff_low-pressure_1946`, values and equation taken from :cite:t:`vomel_saturation_2016`.
    - "sonntag90" or "SO90", taken from :cite:t:`sonntag_important_1990`.
    - "tetens30" or "TE30", based on :cite:t:`tetens_uber_1930`, values and equation taken from :cite:t:`vomel_saturation_2016`.
    - "wmo08" or "WMO08", taken from :cite:t:`world_meteorological_organization_guide_2008`.
    - "its90" or "ITS90", taken from :cite:t:`hardy_its-90_1998`.

    Examples
    --------
    >>> from xclim.indices import saturation_vapor_pressure
    >>> rh = saturation_vapor_pressure(
    ...     tas=tas_dataset, ice_thresh="0 degC", method="wmo08"
    ... )

    References
    ----------
    :cite:cts:`goff_low-pressure_1946,hardy_its-90_1998,sonntag_important_1990,tetens_uber_1930,vomel_saturation_2016,world_meteorological_organization_guide_2008`
    """
    if ice_thresh is not None:
        thresh = convert_units_to(ice_thresh, "degK")
    else:
        thresh = convert_units_to("0 K", "degK")
    tas = convert_units_to(tas, "K")
    ref_is_water = tas > thresh
    if method in ["sonntag90", "SO90"]:
        e_sat = xr.where(
            ref_is_water,
            100
            * np.exp(  # Where ref_is_water is True, x100 is to convert hPa to Pa
                -6096.9385 / tas  # type: ignore
                + 16.635794
                + -2.711193e-2 * tas  # type: ignore
                + 1.673952e-5 * tas**2
                + 2.433502 * np.log(tas)  # numpy's log is ln
            ),
            100
            * np.exp(  # Where ref_is_water is False (thus ref is ice)
                -6024.5282 / tas  # type: ignore
                + 24.7219
                + 1.0613868e-2 * tas  # type: ignore
                + -1.3198825e-5 * tas**2
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
    elif method in ["its90", "ITS90"]:
        e_sat = xr.where(
            ref_is_water,
            np.exp(
                -2836.5744 / tas**2
                + -6028.076559 / tas
                + 19.54263612
                + -2.737830188e-2 * tas
                + 1.6261698e-5 * tas**2
                + 7.0229056e-10 * tas**3
                + -1.8680009e-13 * tas**4
                + 2.7150305 * np.log(tas)
            ),
            np.exp(
                -5866.6426 / tas
                + 22.32870244
                + 1.39387003e-2 * tas
                + -3.4262402e-5 * tas**2
                + 2.7040955e-8 * tas**3
                + 6.7063522e-1 * np.log(tas)
            ),
        )
    else:
        raise ValueError(
            f"Method {method} is not in ['sonntag90', 'tetens30', 'goffgratch46', 'wmo08', 'its90']"
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
    tdps: xr.DataArray | None = None,
    huss: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    ice_thresh: Quantified | None = None,
    method: str = "sonntag90",
    invalid_values: str = "clip",
) -> xr.DataArray:
    r"""Relative humidity.

    Compute relative humidity from temperature and either dewpoint temperature or specific humidity and pressure through
    the saturation vapour pressure.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature array
    tdps : xr.DataArray, optional
        Dewpoint temperature, if specified, overrides huss and ps.
    huss : xr.DataArray, optional
        Specific humidity. Must be given if tdps is not given.
    ps : xr.DataArray, optional
        Air Pressure. Must be given if tdps is not given.
    ice_thresh : Quantified, optional
        Threshold temperature under which to switch to equations in reference to ice instead of water.
        If None (default) everything is computed with reference to water. Does nothing if 'method' is "bohren98".
    method : {"bohren98", "goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Which method to use, see notes of this function and of :py:func:`saturation_vapor_pressure`.
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

    **For the "bohren98" method** : This method does not use the saturation vapour pressure directly,
    but rather uses an approximation of the ratio of :math:`\frac{e_{sat}(T_d)}{e_{sat}(T)}`.
    With :math:`L` the enthalpy of vaporization of water and :math:`R_w` the gas constant for water vapour,
    the relative humidity is computed as:

    .. math::

        RH = e^{\frac{-L (T - T_d)}{R_wTT_d}}

    From :cite:t:`bohren_atmospheric_1998`, formula taken from :cite:t:`lawrence_relationship_2005`. :math:`L = 2.5\times 10^{-6}` J kg-1, exact for :math:`T = 273.15` K, is used.

    **Other methods**: With :math:`w`, :math:`w_{sat}`, :math:`e_{sat}` the mixing ratio,
    the saturation mixing ratio and the saturation vapour pressure.
    If the dewpoint temperature is given, relative humidity is computed as:

    .. math::

        RH = 100\frac{e_{sat}(T_d)}{e_{sat}(T)}

    Otherwise, the specific humidity and the air pressure must be given so relative humidity can be computed as:

    .. math::

        RH = 100\frac{w}{w_{sat}}
        w = \frac{q}{1-q}
        w_{sat} = 0.622\frac{e_{sat}}{P - e_{sat}}

    The methods differ by how :math:`e_{sat}` is computed. See the doc of :py:func:`xclim.core.utils.saturation_vapor_pressure`.

    Examples
    --------
    >>> from xclim.indices import relative_humidity
    >>> rh = relative_humidity(
    ...     tas=tas_dataset,
    ...     tdps=tdps_dataset,
    ...     huss=huss_dataset,
    ...     ps=ps_dataset,
    ...     ice_thresh="0 degC",
    ...     method="wmo08",
    ...     invalid_values="clip",
    ... )

    References
    ----------
    :cite:cts:`bohren_atmospheric_1998,lawrence_relationship_2005`
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
    ice_thresh: Quantified | None = None,
    method: str = "sonntag90",
    invalid_values: str = None,
) -> xr.DataArray:
    r"""Specific humidity from temperature, relative humidity and pressure.

    Specific humidity is the ratio between the mass of water vapour
    and the mass of moist air :cite:p:`world_meteorological_organization_guide_2008`.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature array
    hurs : xr.DataArray
        Relative Humidity.
    ps : xr.DataArray
        Air Pressure.
    ice_thresh : Quantified, optional
        Threshold temperature under which to switch to equations in reference to ice instead of water.
        If None (default) everything is computed with reference to water.
    method : {"goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Which method to use, see notes of this function and of :py:func:`saturation_vapor_pressure`.
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
    the saturation mixing ratio and the saturation vapour pressure, specific humidity :math:`q` is computed as:

    .. math::

        w_{sat} = 0.622\frac{e_{sat}}{P - e_{sat}}
        w = w_{sat} * hurs / 100
        q = w / (1 + w)

    The methods differ by how :math:`e_{sat}` is computed. See :py:func:`xclim.core.utils.saturation_vapor_pressure`.

    If `invalid_values` is not `None`, the saturation specific humidity :math:`q_{sat}` is computed as:

    .. math::

        q_{sat} = w_{sat} / (1 + w_{sat})

    Examples
    --------
    >>> from xclim.indices import specific_humidity
    >>> rh = specific_humidity(
    ...     tas=tas_dataset,
    ...     hurs=hurs_dataset,
    ...     ps=ps_dataset,
    ...     ice_thresh="0 degC",
    ...     method="wmo08",
    ...     invalid_values="mask",
    ... )

    References
    ----------
    :cite:cts:`world_meteorological_organization_guide_2008`
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


@declare_units(
    tdps="[temperature]",
    ps="[pressure]",
)
def specific_humidity_from_dewpoint(
    tdps: xr.DataArray,
    ps: xr.DataArray,
    method: str = "sonntag90",
) -> xr.DataArray:
    r"""Specific humidity from dewpoint temperature and air pressure.

    Specific humidity is the ratio between the mass of water vapour
    and the mass of moist air :cite:p:`world_meteorological_organization_guide_2008`.

    Parameters
    ----------
    tdps : xr.DataArray
        Dewpoint temperature array.
    ps : xr.DataArray
        Air pressure array.
    method : {"goffgratch46", "sonntag90", "tetens30", "wmo08"}
        Method to compute the saturation vapour pressure.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Specific humidity.

    Notes
    -----
    If :math:`e` is the water vapour pressure, and :math:`p` the total air pressure, then specific humidity is given by

    .. math::

       q = m_w e / ( m_a (p - e) + m_w e )

    where :math:`m_w` and :math:`m_a` are the molecular weights of water and dry air respectively. This formula is often
    written with :math:`ε = m_w / m_a`, which simplifies to :math:`q = ε e / (p - e (1 - ε))`.

    Examples
    --------
    >>> from xclim.indices import specific_humidity_from_dewpoint
    >>> rh = specific_humidity_from_dewpoint(
    ...     tdps=tas_dataset,
    ...     ps=ps_dataset,
    ...     method="wmo08",
    ... )

    References
    ----------
    :cite:cts:`world_meteorological_organization_guide_2008`
    """
    ε = 0.6219569  # weight of water vs dry air []
    e = saturation_vapor_pressure(tas=tdps, method=method)  # vapour pressure [Pa]
    ps = convert_units_to(ps, "Pa")  # total air pressure

    q = ε * e / (ps - e * (1 - ε))
    q.attrs["units"] = ""
    return q


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def snowfall_approximation(
    pr: xr.DataArray,
    tas: xr.DataArray,
    thresh: Quantified = "0 degC",
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
    thresh : Quantified
        Freezing point temperature. Non-scalar values are not allowed with method "brown".
    method : {"binary", "brown", "auer"}
        Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
        Solid precipitation flux.

    Notes
    -----
    The following methods are available to approximate snowfall and are drawn from the
    Canadian Land Surface Scheme :cite:p:`verseghy_class_2009,melton_atmosphericvarscalcf90_2019`.

    - ``'binary'`` : When the temperature is under the freezing threshold, precipitation
      is assumed to be solid. The method is agnostic to the type of temperature used
      (mean, maximum or minimum).
    - ``'brown'`` : The phase between the freezing threshold goes from solid to liquid linearly
      over a range of 2°C over the freezing point.
    - ``'auer'`` : The phase between the freezing threshold goes from solid to liquid as a degree six
      polynomial over a range of 6°C over the freezing point.

    References
    ----------
    :cite:cts:`verseghy_class_2009,melton_atmosphericvarscalcf90_2019`
    """
    if method == "binary":
        thresh = convert_units_to(thresh, tas)
        prsn = pr.where(tas <= thresh, 0)

    elif method == "brown":
        if not np.isscalar(thresh):
            raise ValueError("Non-scalar `thresh` are not allowed with method `brown`.")

        # Freezing point + 2C in the native units
        thresh_plus_2 = convert_units_to(thresh, "degC") + 2
        upper = convert_units_to(f"{thresh_plus_2} degC", tas)
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
    thresh: Quantified = "0 degC",
    method: str = "binary",
) -> xr.DataArray:
    """Rainfall approximation from total precipitation and temperature.

    Liquid precipitation estimated from precipitation and temperature according to a given method.
    This is a convenience method based on :py:func:`snowfall_approximation`, see the latter for details.

    Parameters
    ----------
    pr : xarray.DataArray
        Mean daily precipitation flux.
    tas : xarray.DataArray, optional
        Mean, maximum, or minimum daily temperature.
    thresh : Quantified
        Freezing point temperature. Non-scalar values are not allowed with method 'brown'.
    method : {"binary", "brown", "auer"}
        Which method to use when approximating snowfall from total precipitation. See notes.

    Returns
    -------
    xarray.DataArray, [same units as pr]
        Liquid precipitation rate.

    Notes
    -----
    This method computes the snowfall approximation and subtracts it from the total
    precipitation to estimate the liquid rain precipitation.

    See Also
    --------
    snowfall_approximation
    """
    prra = pr - snowfall_approximation(pr, tas, thresh=thresh, method=method)
    prra.attrs["units"] = pr.attrs["units"]
    return prra


@declare_units(snd="[length]", snr="[mass]/[volume]", const="[mass]/[volume]")
def snd_to_snw(
    snd: xr.DataArray,
    snr: Quantified | None = None,
    const: Quantified = "312 kg m-3",
    out_units: str = None,
) -> xr.DataArray:
    """Snow amount from snow depth and density.

    Parameters
    ----------
    snd : xr.DataArray
        Snow depth.
    snr : Quantified, optional
        Snow density.
    const: Quantified
        Constant snow density
        `const` is only used if `snr` is None.
    out_units: str, optional
        Desired units of the snow amount output. If `None`, output units simply follow from `snd * snr`.

    Returns
    -------
    xr.DataArray
        Snow amount

    Notes
    -----
    The estimated mean snow density value of 312 kg m-3 is taken from :cite:t:`sturm_swe_2010`.

    References
    ----------
    :cite:cts:`sturm_swe_2010`
    """
    density = snr if (snr is not None) else const
    snw = rate2flux(snd, density=density, out_units=out_units).rename("snw")
    # TODO: Leave this operation to rate2flux? Maybe also the variable renaming above?
    snw.attrs["standard_name"] = "surface_snow_amount"
    return snw


@declare_units(snw="[mass]/[area]", snr="[mass]/[volume]", const="[mass]/[volume]")
def snw_to_snd(
    snw: xr.DataArray,
    snr: Quantified | None = None,
    const: Quantified = "312 kg m-3",
    out_units: str | None = None,
) -> xr.DataArray:
    """Snow depth from snow amount and density.

    Parameters
    ----------
    snw : xr.DataArray
        Snow amount.
    snr : Quantified, optional
        Snow density.
    const: Quantified
        Constant snow density
        `const` is only used if `snr` is None.
    out_units: str, optional
        Desired units of the snow depth output. If `None`, output units simply follow from `snw / snr`.

    Returns
    -------
    xr.DataArray
        Snow depth

    Notes
    -----
    The estimated mean snow density value of 312 kg m-3 is taken from :cite:t:`sturm_swe_2010`.

    References
    ----------
    :cite:cts:`sturm_swe_2010`
    """
    density = snr if (snr is not None) else const
    snd = flux2rate(snw, density=density, out_units=out_units).rename("snd")
    snd.attrs["standard_name"] = "surface_snow_thickness"
    return snd


@declare_units(
    prsn="[mass]/[area]/[time]", snr="[mass]/[volume]", const="[mass]/[volume]"
)
def prsn_to_prsnd(
    prsn: xr.DataArray,
    snr: xr.DataArray | None = None,
    const: Quantified = "100 kg m-3",
    out_units: str = None,
) -> xr.DataArray:
    """Snowfall rate from snowfall flux and density.

    Parameters
    ----------
    prsn : xr.DataArray
        Snowfall flux.
    snr : xr.DataArray, optional
        Snow density.
    const: Quantified
        Constant snow density.
        `const` is only used if `snr` is None.
    out_units: str, optional
        Desired units of the snowfall rate. If `None`, output units simply follow from `snd * snr`.

    Returns
    -------
    xr.DataArray
        Snowfall rate.

    Notes
    -----
    The estimated mean snow density value of 100 kg m-3 is taken from
    :cite:cts:`frei_snowfall_2018, cbcl_climate_2020`.

    References
    ----------
    :cite:cts:`frei_snowfall_2018, cbcl_climate_2020`
    """
    density = snr if snr else const
    prsnd = flux2rate(prsn, density=density, out_units=out_units).rename("prsnd")
    return prsnd


@declare_units(prsnd="[length]/[time]", snr="[mass]/[volume]", const="[mass]/[volume]")
def prsnd_to_prsn(
    prsnd: xr.DataArray,
    snr: xr.DataArray | None = None,
    const: Quantified = "100 kg m-3",
    out_units: str = None,
) -> xr.DataArray:
    """Snowfall flux from snowfall rate and density.

    Parameters
    ----------
    prsnd : xr.DataArray
        Snowfall rate.
    snr : xr.DataArray, optional
        Snow density.
    const: Quantified
        Constant snow density.
        `const` is only used if `snr` is None.
    out_units: str, optional
        Desired units of the snowfall rate. If `None`, output units simply follow from `snd * snr`.

    Returns
    -------
    xr.DataArray
        Snowfall flux.

    Notes
    -----
    The estimated mean snow density value of 100 kg m-3 is taken from
    :cite:cts:`frei_snowfall_2018, cbcl_climate_2020`.

    References
    ----------
    :cite:cts:`frei_snowfall_2018, cbcl_climate_2020`
    """
    density = snr if snr else const
    prsn = rate2flux(prsnd, density=density, out_units=out_units).rename("prsn")
    prsn.attrs["standard_name"] = "snowfall_flux"
    return prsn


@declare_units(rls="[radiation]", rlds="[radiation]")
def longwave_upwelling_radiation_from_net_downwelling(
    rls: xr.DataArray, rlds: xr.DataArray
) -> xr.DataArray:
    """Calculate upwelling thermal radiation from net thermal radiation and downwelling thermal radiation.

    Parameters
    ----------
    rls : xr.DataArray
        Surface net thermal radiation.
    rlds : xr.DataArray
        Surface downwelling thermal radiation.

    Returns
    -------
    xr.DataArray, [same units as rlds]
        Surface upwelling thermal radiation (rlus).
    """
    rls = convert_units_to(rls, rlds)

    rlus = rlds - rls

    rlus.attrs["units"] = rlds.units
    return rlus


@declare_units(rss="[radiation]", rsds="[radiation]")
def shortwave_upwelling_radiation_from_net_downwelling(
    rss: xr.DataArray, rsds: xr.DataArray
) -> xr.DataArray:
    """Calculate upwelling solar radiation from net solar radiation and downwelling solar radiation.

    Parameters
    ----------
    rss : xr.DataArray
        Surface net solar radiation.
    rsds : xr.DataArray
        Surface downwelling solar radiation.

    Returns
    -------
    xr.DataArray, [same units as rsds]
        Surface upwelling solar radiation (rsus).
    """
    rss = convert_units_to(rss, rsds)

    rsus = rsds - rss

    rsus.attrs["units"] = rsds.units
    return rsus


@declare_units(
    tas="[temperature]",
    sfcWind="[speed]",
)
def wind_chill_index(
    tas: xr.DataArray,
    sfcWind: xr.DataArray,
    method: str = "CAN",
    mask_invalid: bool = True,
) -> xr.DataArray:
    r"""Wind chill index.

    The Wind Chill Index is an estimation of how cold the weather feels to the average person.
    It is computed from the air temperature and the 10-m wind. As defined by the Environment and Climate Change Canada
    (:cite:cts:`mekis_observed_2015`), two equations exist, the conventional one and one for slow winds
    (usually < 5 km/h), see Notes.

    Parameters
    ----------
    tas : xarray.DataArray
        Surface air temperature.
    sfcWind : xarray.DataArray
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
    Following the calculations of Environment and Climate Change Canada, this function switches from the standardized
    index to another one for slow winds. The standard index is the same as used by the National Weather Service of the
    USA :cite:p:`us_department_of_commerce_wind_nodate`. Given a temperature at surface :math:`T` (in °C) and 10-m
    wind speed :math:`V` (in km/h), the Wind Chill Index :math:`W` (dimensionless) is computed as:

    .. math::

        W = 13.12 + 0.6125*T - 11.37*V^0.16 + 0.3965*T*V^0.16

    Under slow winds (:math:`V < 5` km/h), and using the canadian method, it becomes:

    .. math::

        W = T + \frac{-1.59 + 0.1345 * T}{5} * V


    Both equations are invalid for temperature over 0°C in the canadian method.

    The american Wind Chill Temperature index (WCT), as defined by USA's National Weather Service, is computed when
    `method='US'`. In that case, the maximal valid temperature is 50°F (10 °C) and minimal wind speed is 3 mph
    (4.8 km/h).

    For more information, see:

    - National Weather Service FAQ: :cite:p:`us_department_of_commerce_wind_nodate`.
    - The New Wind Chill Equivalent Temperature Chart: :cite:p:`osczevski_new_2005`.

    References
    ----------
    :cite:cts:`mekis_observed_2015,us_department_of_commerce_wind_nodate`
    """
    tas = convert_units_to(tas, "degC")
    sfcWind = convert_units_to(sfcWind, "km/h")

    V = sfcWind**0.16
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
    r"""Scale precipitation according to the Clausius-Clapeyron relation.

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
    The Clausius-Clapeyron equation for water vapour under typical atmospheric conditions states that the saturation
    water vapour pressure :math:`e_s` changes approximately exponentially with temperature

    .. math::
        \frac{\mathrm{d}e_s(T)}{\mathrm{d}T} \approx 1.07 e_s(T)

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
    pr_out = pr_baseline * (cc_scale_factor**delta_tas)
    pr_out.attrs["units"] = pr_baseline.attrs["units"]

    return pr_out


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    tas="[temperature]",
    lat="[]",
    hurs="[]",
    rsds="[radiation]",
    rsus="[radiation]",
    rlds="[radiation]",
    rlus="[radiation]",
    sfcWind="[speed]",
)
def potential_evapotranspiration(
    tasmin: xr.DataArray | None = None,
    tasmax: xr.DataArray | None = None,
    tas: xr.DataArray | None = None,
    lat: xr.DataArray | None = None,
    hurs: xr.DataArray | None = None,
    rsds: xr.DataArray | None = None,
    rsus: xr.DataArray | None = None,
    rlds: xr.DataArray | None = None,
    rlus: xr.DataArray | None = None,
    sfcWind: xr.DataArray | None = None,
    method: str = "BR65",
    peta: float = 0.00516409319477,
    petb: float = 0.0874972822289,
) -> xr.DataArray:
    r"""Potential evapotranspiration.

    The potential for water evaporation from soil and transpiration by plants if the water supply is sufficient,
    according to a given method.

    Parameters
    ----------
    tasmin : xarray.DataArray, optional
        Minimum daily temperature.
    tasmax : xarray.DataArray, optional
        Maximum daily temperature.
    tas : xarray.DataArray, optional
        Mean daily temperature.
    lat : xarray.DataArray, optional
        Latitude. If not given, it is sought on tasmin or tas using cf-xarray accessors.
    hurs : xarray.DataArray, optional
        Relative humidity.
    rsds : xarray.DataArray, optional
        Surface Downwelling Shortwave Radiation
    rsus : xarray.DataArray, optional
        Surface Upwelling Shortwave Radiation
    rlds : xarray.DataArray, optional
        Surface Downwelling Longwave Radiation
    rlus : xarray.DataArray, optional
        Surface Upwelling Longwave Radiation
    sfcWind : xarray.DataArray, optional
        Surface wind velocity (at 10 m)
    method : {"baierrobertson65", "BR65", "hargreaves85", "HG85", "thornthwaite48", "TW48", "mcguinnessbordne05", "MB05", "allen98", "FAO_PM98"}
        Which method to use, see notes.
    peta : float
        Used only with method MB05 as :math:`a` for calculation of PET, see Notes section.
        Default value resulted from calibration of PET over the UK.
    petb : float
        Used only with method MB05 as :math:`b` for calculation of PET, see Notes section.
        Default value resulted from calibration of PET over the UK.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    Available methods are:

    - "baierrobertson65" or "BR65", based on :cite:t:`baier_estimation_1965`. Requires tasmin and tasmax, daily [D] freq.
    - "hargreaves85" or "HG85", based on :cite:t:`george_h_hargreaves_reference_1985`. Requires tasmin and tasmax, daily [D] freq. (optional: tas can be given in addition of tasmin and tasmax).
    - "mcguinnessbordne05" or "MB05", based on :cite:t:`tanguy_historical_2018`. Requires tas, daily [D] freq, with latitudes 'lat'.
    - "thornthwaite48" or "TW48", based on :cite:t:`thornthwaite_approach_1948`. Requires tasmin and tasmax, monthly [MS] or daily [D] freq. (optional: tas can be given instead of tasmin and tasmax).
    - "allen98" or "FAO_PM98", based on :cite:t:`allen_crop_1998`. Modification of Penman-Monteith method. Requires tasmin and tasmax, relative humidity, radiation flux and wind speed (10 m wind will be converted to 2 m).

    The McGuinness-Bordne :cite:p:`mcguinness_comparison_1972` equation is:

    .. math::
        PET[mm day^{-1}] = a * \frac{S_0}{\lambda}T_a + b *\frsc{S_0}{\lambda}

    where :math:`a` and :math:`b` are empirical parameters; :math:`S_0` is the extraterrestrial radiation [MJ m-2 day-1],
    assuming a solar constant of 1367 W m-2; :math:`\\lambda` is the latent heat of vaporisation [MJ kg-1]
    and :math:`T_a` is the air temperature [°C]. The equation was originally derived for the USA,
    with :math:`a=0.0147` and :math:`b=0.07353`. The default parameters used here are calibrated for the UK,
    using the method described in :cite:t:`tanguy_historical_2018`.

    Methods "BR65", "HG85" and "MB05" use an approximation of the extraterrestrial radiation.
    See :py:func:`~xclim.indices._helpers.extraterrestrial_solar_radiation`.

    References
    ----------
    :cite:cts:`baier_estimation_1965,george_h_hargreaves_reference_1985,tanguy_historical_2018,thornthwaite_approach_1948,mcguinness_comparison_1972,allen_crop_1998`
    """
    if lat is None:
        lat = _gather_lat(tasmin if tas is None else tas)

    if method in ["baierrobertson65", "BR65"]:
        tasmin = convert_units_to(tasmin, "degF")
        tasmax = convert_units_to(tasmax, "degF")

        re = extraterrestrial_solar_radiation(tasmin.time, lat)
        re = convert_units_to(re, "cal cm-2 day-1")

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

        lv = 2.5  # MJ/kg

        ra = extraterrestrial_solar_radiation(tasmin.time, lat)
        ra = convert_units_to(ra, "MJ m-2 d-1")

        # Hargreaves and Samani (1985) formula
        out = (0.0023 * ra * (tas + 17.8) * (tasmax - tasmin) ** 0.5) / lv
        out = out.clip(0)

    elif method in ["mcguinnessbordne05", "MB05"]:
        if tas is None:
            tasmin = convert_units_to(tasmin, "degC")
            tasmax = convert_units_to(tasmax, "degC")
            tas = (tasmin + tasmax) / 2
            tas.attrs["units"] = "degC"

        tas = convert_units_to(tas, "degC")
        tasK = convert_units_to(tas, "K")

        ext_rad = extraterrestrial_solar_radiation(
            tas.time, lat, solar_constant="1367 W m-2"
        )
        latentH = 4185.5 * (751.78 - 0.5655 * tasK)
        radDIVlat = ext_rad / latentH

        # parameters from calibration provided by Dr Maliko Tanguy @ CEH
        # (calibrated for PET over the UK)
        a = peta
        b = petb

        out = radDIVlat * a * tas + radDIVlat * b

    elif method in ["thornthwaite48", "TW48"]:
        if tas is None:
            tasmin = convert_units_to(tasmin, "degC")
            tasmax = convert_units_to(tasmax, "degC")
            tas = (tasmin + tasmax) / 2
        else:
            tas = convert_units_to(tas, "degC")
        tas = tas.clip(0)
        tas = tas.resample(time="MS").mean(dim="time")

        start = "-".join(
            [
                str(tas.time[0].dt.year.values),
                f"{tas.time[0].dt.month.values:02d}",
                "01",
            ]
        )

        end = "-".join(
            [
                str(tas.time[-1].dt.year.values),
                f"{tas.time[-1].dt.month.values:02d}",
                str(tas.time[-1].dt.daysinmonth.values),
            ]
        )

        time_v = xr.DataArray(
            date_range(start, end, freq="D", calendar="standard"),
            dims="time",
            name="time",
        )

        # Thornthwaite measures half-days
        dl = day_lengths(time_v, lat) / 12
        dl_m = dl.resample(time="MS").mean(dim="time")

        # annual heat index
        id_m = (tas / 5) ** 1.514
        id_y = id_m.resample(time="YS").sum(dim="time")

        tas_idy_a = []
        for base_time, indexes in tas.resample(time="YS").groups.items():
            tas_y = tas.isel(time=indexes)
            id_v = id_y.sel(time=base_time)
            a = 6.75e-7 * id_v**3 - 7.71e-5 * id_v**2 + 0.01791 * id_v + 0.49239

            frac = (10 * tas_y / id_v) ** a
            tas_idy_a.append(frac)

        tas_idy_a = xr.concat(tas_idy_a, dim="time")

        # Thornthwaite(1948) formula
        out = 1.6 * dl_m * tas_idy_a  # cm/month
        out = 10 * out  # mm/month

    elif method in ["allen98", "FAO_PM98"]:
        tasmax = convert_units_to(tasmax, "degC")
        tasmin = convert_units_to(tasmin, "degC")

        # wind speed at two meters
        wa2 = wind_speed_height_conversion(sfcWind, h_source="10 m", h_target="2 m")
        wa2 = convert_units_to(wa2, "m s-1")

        with xr.set_options(keep_attrs=True):
            # mean temperature [degC]
            tas_m = (tasmax + tasmin) / 2
            # mean saturation vapour pressure [kPa]
            es = (1 / 2) * (
                saturation_vapor_pressure(tasmax) + saturation_vapor_pressure(tasmin)
            )
            es = convert_units_to(es, "kPa")
            # mean actual vapour pressure [kPa]
            ea = hurs * es

            # slope of saturation vapour pressure curve  [kPa degC-1]
            delta = 4098 * es / (tas_m + 237.3) ** 2
            # net radiation
            Rn = convert_units_to(rsds - rsus - (rlus - rlds), "MJ m-2 d-1")

            G = 0  # Daily soil heat flux density [MJ m-2 d-1]
            P = 101.325  # Atmospheric pressure [kPa]
            gamma = 0.665e-03 * P  # psychrometric const = C_p*P/(eps*lam) [kPa degC-1]

            # Penman-Monteith formula with reference grass:
            # height = 0.12m, surface resistance = 70 s m-1, albedo  = 0.23
            # Surface resistance implies a ``moderately dry soil surface resulting from
            # about a weekly irrigation frequency''
            out = (
                0.408 * delta * (Rn - G)
                + gamma * (900 / (tas_m + 273)) * wa2 * (es - ea)
            ) / (delta + gamma * (1 + 0.34 * wa2))

    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")

    out.attrs["units"] = "mm"
    rate = amount2rate(out, out_units="mm/d")
    return convert_units_to(rate, "kg m-2 s-1", context="hydro")


@vectorize(
    # [
    #     float64(float64, float64, float64, float64),
    #     float32(float32, float32, float32, float32),
    # ],
)
def _utci(tas, sfcWind, dt, wvp):
    """Return the empirical polynomial function for UTCI. See :py:func:`universal_thermal_climate_index`."""
    # Taken directly from the original Fortran code by Peter Bröde.
    # http://www.utci.org/public/UTCI%20Program%20Code/UTCI_a002.f90
    # tas -> Ta (surface temperature, °C)
    # sfcWind -> va (surface wind speed, m/s)
    # dt -> D_Tmrt (tas - t_mrt, K)
    # wvp -> Pa (water vapour partial pressure, kPa)
    return (
        tas
        + 6.07562052e-1
        + -2.27712343e-2 * tas
        + 8.06470249e-4 * tas * tas
        + -1.54271372e-4 * tas * tas * tas
        + -3.24651735e-6 * tas * tas * tas * tas
        + 7.32602852e-8 * tas * tas * tas * tas * tas
        + 1.35959073e-9 * tas * tas * tas * tas * tas * tas
        + -2.25836520e0 * sfcWind
        + 8.80326035e-2 * tas * sfcWind
        + 2.16844454e-3 * tas * tas * sfcWind
        + -1.53347087e-5 * tas * tas * tas * sfcWind
        + -5.72983704e-7 * tas * tas * tas * tas * sfcWind
        + -2.55090145e-9 * tas * tas * tas * tas * tas * sfcWind
        + -7.51269505e-1 * sfcWind * sfcWind
        + -4.08350271e-3 * tas * sfcWind * sfcWind
        + -5.21670675e-5 * tas * tas * sfcWind * sfcWind
        + 1.94544667e-6 * tas * tas * tas * sfcWind * sfcWind
        + 1.14099531e-8 * tas * tas * tas * tas * sfcWind * sfcWind
        + 1.58137256e-1 * sfcWind * sfcWind * sfcWind
        + -6.57263143e-5 * tas * sfcWind * sfcWind * sfcWind
        + 2.22697524e-7 * tas * tas * sfcWind * sfcWind * sfcWind
        + -4.16117031e-8 * tas * tas * tas * sfcWind * sfcWind * sfcWind
        + -1.27762753e-2 * sfcWind * sfcWind * sfcWind * sfcWind
        + 9.66891875e-6 * tas * sfcWind * sfcWind * sfcWind * sfcWind
        + 2.52785852e-9 * tas * tas * sfcWind * sfcWind * sfcWind * sfcWind
        + 4.56306672e-4 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + -1.74202546e-7 * tas * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + -5.91491269e-6 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + 3.98374029e-1 * dt
        + 1.83945314e-4 * tas * dt
        + -1.73754510e-4 * tas * tas * dt
        + -7.60781159e-7 * tas * tas * tas * dt
        + 3.77830287e-8 * tas * tas * tas * tas * dt
        + 5.43079673e-10 * tas * tas * tas * tas * tas * dt
        + -2.00518269e-2 * sfcWind * dt
        + 8.92859837e-4 * tas * sfcWind * dt
        + 3.45433048e-6 * tas * tas * sfcWind * dt
        + -3.77925774e-7 * tas * tas * tas * sfcWind * dt
        + -1.69699377e-9 * tas * tas * tas * tas * sfcWind * dt
        + 1.69992415e-4 * sfcWind * sfcWind * dt
        + -4.99204314e-5 * tas * sfcWind * sfcWind * dt
        + 2.47417178e-7 * tas * tas * sfcWind * sfcWind * dt
        + 1.07596466e-8 * tas * tas * tas * sfcWind * sfcWind * dt
        + 8.49242932e-5 * sfcWind * sfcWind * sfcWind * dt
        + 1.35191328e-6 * tas * sfcWind * sfcWind * sfcWind * dt
        + -6.21531254e-9 * tas * tas * sfcWind * sfcWind * sfcWind * dt
        + -4.99410301e-6 * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + -1.89489258e-8 * tas * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + 8.15300114e-8 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + 7.55043090e-4 * dt * dt
        + -5.65095215e-5 * tas * dt * dt
        + -4.52166564e-7 * tas * tas * dt * dt
        + 2.46688878e-8 * tas * tas * tas * dt * dt
        + 2.42674348e-10 * tas * tas * tas * tas * dt * dt
        + 1.54547250e-4 * sfcWind * dt * dt
        + 5.24110970e-6 * tas * sfcWind * dt * dt
        + -8.75874982e-8 * tas * tas * sfcWind * dt * dt
        + -1.50743064e-9 * tas * tas * tas * sfcWind * dt * dt
        + -1.56236307e-5 * sfcWind * sfcWind * dt * dt
        + -1.33895614e-7 * tas * sfcWind * sfcWind * dt * dt
        + 2.49709824e-9 * tas * tas * sfcWind * sfcWind * dt * dt
        + 6.51711721e-7 * sfcWind * sfcWind * sfcWind * dt * dt
        + 1.94960053e-9 * tas * sfcWind * sfcWind * sfcWind * dt * dt
        + -1.00361113e-8 * sfcWind * sfcWind * sfcWind * sfcWind * dt * dt
        + -1.21206673e-5 * dt * dt * dt
        + -2.18203660e-7 * tas * dt * dt * dt
        + 7.51269482e-9 * tas * tas * dt * dt * dt
        + 9.79063848e-11 * tas * tas * tas * dt * dt * dt
        + 1.25006734e-6 * sfcWind * dt * dt * dt
        + -1.81584736e-9 * tas * sfcWind * dt * dt * dt
        + -3.52197671e-10 * tas * tas * sfcWind * dt * dt * dt
        + -3.36514630e-8 * sfcWind * sfcWind * dt * dt * dt
        + 1.35908359e-10 * tas * sfcWind * sfcWind * dt * dt * dt
        + 4.17032620e-10 * sfcWind * sfcWind * sfcWind * dt * dt * dt
        + -1.30369025e-9 * dt * dt * dt * dt
        + 4.13908461e-10 * tas * dt * dt * dt * dt
        + 9.22652254e-12 * tas * tas * dt * dt * dt * dt
        + -5.08220384e-9 * sfcWind * dt * dt * dt * dt
        + -2.24730961e-11 * tas * sfcWind * dt * dt * dt * dt
        + 1.17139133e-10 * sfcWind * sfcWind * dt * dt * dt * dt
        + 6.62154879e-10 * dt * dt * dt * dt * dt
        + 4.03863260e-13 * tas * dt * dt * dt * dt * dt
        + 1.95087203e-12 * sfcWind * dt * dt * dt * dt * dt
        + -4.73602469e-12 * dt * dt * dt * dt * dt * dt
        + 5.12733497e0 * wvp
        + -3.12788561e-1 * tas * wvp
        + -1.96701861e-2 * tas * tas * wvp
        + 9.99690870e-4 * tas * tas * tas * wvp
        + 9.51738512e-6 * tas * tas * tas * tas * wvp
        + -4.66426341e-7 * tas * tas * tas * tas * tas * wvp
        + 5.48050612e-1 * sfcWind * wvp
        + -3.30552823e-3 * tas * sfcWind * wvp
        + -1.64119440e-3 * tas * tas * sfcWind * wvp
        + -5.16670694e-6 * tas * tas * tas * sfcWind * wvp
        + 9.52692432e-7 * tas * tas * tas * tas * sfcWind * wvp
        + -4.29223622e-2 * sfcWind * sfcWind * wvp
        + 5.00845667e-3 * tas * sfcWind * sfcWind * wvp
        + 1.00601257e-6 * tas * tas * sfcWind * sfcWind * wvp
        + -1.81748644e-6 * tas * tas * tas * sfcWind * sfcWind * wvp
        + -1.25813502e-3 * sfcWind * sfcWind * sfcWind * wvp
        + -1.79330391e-4 * tas * sfcWind * sfcWind * sfcWind * wvp
        + 2.34994441e-6 * tas * tas * sfcWind * sfcWind * sfcWind * wvp
        + 1.29735808e-4 * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + 1.29064870e-6 * tas * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + -2.28558686e-6 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + -3.69476348e-2 * dt * wvp
        + 1.62325322e-3 * tas * dt * wvp
        + -3.14279680e-5 * tas * tas * dt * wvp
        + 2.59835559e-6 * tas * tas * tas * dt * wvp
        + -4.77136523e-8 * tas * tas * tas * tas * dt * wvp
        + 8.64203390e-3 * sfcWind * dt * wvp
        + -6.87405181e-4 * tas * sfcWind * dt * wvp
        + -9.13863872e-6 * tas * tas * sfcWind * dt * wvp
        + 5.15916806e-7 * tas * tas * tas * sfcWind * dt * wvp
        + -3.59217476e-5 * sfcWind * sfcWind * dt * wvp
        + 3.28696511e-5 * tas * sfcWind * sfcWind * dt * wvp
        + -7.10542454e-7 * tas * tas * sfcWind * sfcWind * dt * wvp
        + -1.24382300e-5 * sfcWind * sfcWind * sfcWind * dt * wvp
        + -7.38584400e-9 * tas * sfcWind * sfcWind * sfcWind * dt * wvp
        + 2.20609296e-7 * sfcWind * sfcWind * sfcWind * sfcWind * dt * wvp
        + -7.32469180e-4 * dt * dt * wvp
        + -1.87381964e-5 * tas * dt * dt * wvp
        + 4.80925239e-6 * tas * tas * dt * dt * wvp
        + -8.75492040e-8 * tas * tas * tas * dt * dt * wvp
        + 2.77862930e-5 * sfcWind * dt * dt * wvp
        + -5.06004592e-6 * tas * sfcWind * dt * dt * wvp
        + 1.14325367e-7 * tas * tas * sfcWind * dt * dt * wvp
        + 2.53016723e-6 * sfcWind * sfcWind * dt * dt * wvp
        + -1.72857035e-8 * tas * sfcWind * sfcWind * dt * dt * wvp
        + -3.95079398e-8 * sfcWind * sfcWind * sfcWind * dt * dt * wvp
        + -3.59413173e-7 * dt * dt * dt * wvp
        + 7.04388046e-7 * tas * dt * dt * dt * wvp
        + -1.89309167e-8 * tas * tas * dt * dt * dt * wvp
        + -4.79768731e-7 * sfcWind * dt * dt * dt * wvp
        + 7.96079978e-9 * tas * sfcWind * dt * dt * dt * wvp
        + 1.62897058e-9 * sfcWind * sfcWind * dt * dt * dt * wvp
        + 3.94367674e-8 * dt * dt * dt * dt * wvp
        + -1.18566247e-9 * tas * dt * dt * dt * dt * wvp
        + 3.34678041e-10 * sfcWind * dt * dt * dt * dt * wvp
        + -1.15606447e-10 * dt * dt * dt * dt * dt * wvp
        + -2.80626406e0 * wvp * wvp
        + 5.48712484e-1 * tas * wvp * wvp
        + -3.99428410e-3 * tas * tas * wvp * wvp
        + -9.54009191e-4 * tas * tas * tas * wvp * wvp
        + 1.93090978e-5 * tas * tas * tas * tas * wvp * wvp
        + -3.08806365e-1 * sfcWind * wvp * wvp
        + 1.16952364e-2 * tas * sfcWind * wvp * wvp
        + 4.95271903e-4 * tas * tas * sfcWind * wvp * wvp
        + -1.90710882e-5 * tas * tas * tas * sfcWind * wvp * wvp
        + 2.10787756e-3 * sfcWind * sfcWind * wvp * wvp
        + -6.98445738e-4 * tas * sfcWind * sfcWind * wvp * wvp
        + 2.30109073e-5 * tas * tas * sfcWind * sfcWind * wvp * wvp
        + 4.17856590e-4 * sfcWind * sfcWind * sfcWind * wvp * wvp
        + -1.27043871e-5 * tas * sfcWind * sfcWind * sfcWind * wvp * wvp
        + -3.04620472e-6 * sfcWind * sfcWind * sfcWind * sfcWind * wvp * wvp
        + 5.14507424e-2 * dt * wvp * wvp
        + -4.32510997e-3 * tas * dt * wvp * wvp
        + 8.99281156e-5 * tas * tas * dt * wvp * wvp
        + -7.14663943e-7 * tas * tas * tas * dt * wvp * wvp
        + -2.66016305e-4 * sfcWind * dt * wvp * wvp
        + 2.63789586e-4 * tas * sfcWind * dt * wvp * wvp
        + -7.01199003e-6 * tas * tas * sfcWind * dt * wvp * wvp
        + -1.06823306e-4 * sfcWind * sfcWind * dt * wvp * wvp
        + 3.61341136e-6 * tas * sfcWind * sfcWind * dt * wvp * wvp
        + 2.29748967e-7 * sfcWind * sfcWind * sfcWind * dt * wvp * wvp
        + 3.04788893e-4 * dt * dt * wvp * wvp
        + -6.42070836e-5 * tas * dt * dt * wvp * wvp
        + 1.16257971e-6 * tas * tas * dt * dt * wvp * wvp
        + 7.68023384e-6 * sfcWind * dt * dt * wvp * wvp
        + -5.47446896e-7 * tas * sfcWind * dt * dt * wvp * wvp
        + -3.59937910e-8 * sfcWind * sfcWind * dt * dt * wvp * wvp
        + -4.36497725e-6 * dt * dt * dt * wvp * wvp
        + 1.68737969e-7 * tas * dt * dt * dt * wvp * wvp
        + 2.67489271e-8 * sfcWind * dt * dt * dt * wvp * wvp
        + 3.23926897e-9 * dt * dt * dt * dt * wvp * wvp
        + -3.53874123e-2 * wvp * wvp * wvp
        + -2.21201190e-1 * tas * wvp * wvp * wvp
        + 1.55126038e-2 * tas * tas * wvp * wvp * wvp
        + -2.63917279e-4 * tas * tas * tas * wvp * wvp * wvp
        + 4.53433455e-2 * sfcWind * wvp * wvp * wvp
        + -4.32943862e-3 * tas * sfcWind * wvp * wvp * wvp
        + 1.45389826e-4 * tas * tas * sfcWind * wvp * wvp * wvp
        + 2.17508610e-4 * sfcWind * sfcWind * wvp * wvp * wvp
        + -6.66724702e-5 * tas * sfcWind * sfcWind * wvp * wvp * wvp
        + 3.33217140e-5 * sfcWind * sfcWind * sfcWind * wvp * wvp * wvp
        + -2.26921615e-3 * dt * wvp * wvp * wvp
        + 3.80261982e-4 * tas * dt * wvp * wvp * wvp
        + -5.45314314e-9 * tas * tas * dt * wvp * wvp * wvp
        + -7.96355448e-4 * sfcWind * dt * wvp * wvp * wvp
        + 2.53458034e-5 * tas * sfcWind * dt * wvp * wvp * wvp
        + -6.31223658e-6 * sfcWind * sfcWind * dt * wvp * wvp * wvp
        + 3.02122035e-4 * dt * dt * wvp * wvp * wvp
        + -4.77403547e-6 * tas * dt * dt * wvp * wvp * wvp
        + 1.73825715e-6 * sfcWind * dt * dt * wvp * wvp * wvp
        + -4.09087898e-7 * dt * dt * dt * wvp * wvp * wvp
        + 6.14155345e-1 * wvp * wvp * wvp * wvp
        + -6.16755931e-2 * tas * wvp * wvp * wvp * wvp
        + 1.33374846e-3 * tas * tas * wvp * wvp * wvp * wvp
        + 3.55375387e-3 * sfcWind * wvp * wvp * wvp * wvp
        + -5.13027851e-4 * tas * sfcWind * wvp * wvp * wvp * wvp
        + 1.02449757e-4 * sfcWind * sfcWind * wvp * wvp * wvp * wvp
        + -1.48526421e-3 * dt * wvp * wvp * wvp * wvp
        + -4.11469183e-5 * tas * dt * wvp * wvp * wvp * wvp
        + -6.80434415e-6 * sfcWind * dt * wvp * wvp * wvp * wvp
        + -9.77675906e-6 * dt * dt * wvp * wvp * wvp * wvp
        + 8.82773108e-2 * wvp * wvp * wvp * wvp * wvp
        + -3.01859306e-3 * tas * wvp * wvp * wvp * wvp * wvp
        + 1.04452989e-3 * sfcWind * wvp * wvp * wvp * wvp * wvp
        + 2.47090539e-4 * dt * wvp * wvp * wvp * wvp * wvp
        + 1.48348065e-3 * wvp * wvp * wvp * wvp * wvp * wvp
    )


@declare_units(
    tas="[temperature]",
    hurs="[]",
    sfcWind="[speed]",
    mrt="[temperature]",
    rsds="[radiation]",
    rsus="[radiation]",
    rlds="[radiation]",
    rlus="[radiation]",
)
def universal_thermal_climate_index(
    tas: xr.DataArray,
    hurs: xr.DataArray,
    sfcWind: xr.DataArray,
    mrt: xr.DataArray = None,
    rsds: xr.DataArray = None,
    rsus: xr.DataArray = None,
    rlds: xr.DataArray = None,
    rlus: xr.DataArray = None,
    stat: str = "average",
    mask_invalid: bool = True,
) -> xr.DataArray:
    r"""Universal thermal climate index (UTCI).

    The UTCI is the equivalent temperature for the environment derived from a
    reference environment and is used to evaluate heat stress in outdoor spaces.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean temperature
    hurs : xarray.DataArray
        Relative Humidity
    sfcWind : xarray.DataArray
        Wind velocity
    mrt: xarray.DataArray, optional
        Mean radiant temperature
    rsds : xr.DataArray, optional
        Surface Downwelling Shortwave Radiation
        This is necessary if mrt is not None.
    rsus : xr.DataArray, optional
        Surface Upwelling Shortwave Radiation
        This is necessary if mrt is not None.
    rlds : xr.DataArray, optional
        Surface Downwelling Longwave Radiation
        This is necessary if mrt is not None.
    rlus : xr.DataArray, optional
        Surface Upwelling Longwave Radiation
        This is necessary if mrt is not None.
    stat : {'average', 'instant', 'sunlit'}
        Which statistic to apply. If "average", the average of the cosine of the
        solar zenith angle is calculated. If "instant", the instantaneous cosine
        of the solar zenith angle is calculated. If "sunlit", the cosine of the
        solar zenith angle is calculated during the sunlit period of each interval.
        If "instant", the instantaneous cosine of the solar zenith angle is calculated.
        This is necessary if mrt is not None.
    mask_invalid: bool
        If True (default), UTCI values are NaN where any of the inputs are outside
        their validity ranges : -50°C < tas < 50°C,  -30°C < tas - mrt < 30°C
        and  0.5 m/s < sfcWind < 17.0 m/s.

    Returns
    -------
    xarray.DataArray
        Universal Thermal Climate Index.

    Notes
    -----
    The calculation uses water vapour partial pressure, which is derived from relative
    humidity and saturation vapour pressure computed according to the ITS-90 equation.

    This code was inspired by the `pythermalcomfort` and `thermofeel` packages.

    Notes
    -----
    See: http://www.utci.org/utcineu/utcineu.php

    References
    ----------
    :cite:cts:`brode_utci_2009,blazejczyk_introduction_2013`
    """
    e_sat = saturation_vapor_pressure(tas=tas, method="its90")
    tas = convert_units_to(tas, "degC")
    sfcWind = convert_units_to(sfcWind, "m/s")
    if mrt is None:
        mrt = mean_radiant_temperature(
            rsds=rsds, rsus=rsus, rlds=rlds, rlus=rlus, stat=stat
        )
    mrt = convert_units_to(mrt, "degC")
    delta = mrt - tas
    pa = convert_units_to(e_sat, "kPa") * convert_units_to(hurs, "1")

    utci = xr.apply_ufunc(
        _utci,
        tas,
        sfcWind,
        delta,
        pa,
        input_core_dims=[[], [], [], []],
        dask="parallelized",
        output_dtypes=[tas.dtype],
    )

    utci = utci.assign_attrs({"units": "degC"})
    if mask_invalid:
        utci = utci.where(
            (-50.0 < tas)
            & (tas < 50.0)
            & (-30 < delta)
            & (delta < 30)
            & (0.5 < sfcWind)
            & (sfcWind < 17.0)
        )
    return utci


def _fdir_ratio(
    dates: xr.DataArray,
    csza_i: xr.DataArray,
    csza_s: xr.DataArray,
    rsds: xr.DataArray,
) -> xr.DataArray:
    r"""Return ratio of direct solar radiation.

    The ratio of direct solar radiation is the fraction of the total horizontal solar irradiance
    due to the direct beam of the sun.

    Parameters
    ----------
    dates : xr.DataArray
        Series of dates and time of day
    csza_i : xr.DataArray
        Cosine of the solar zenith angle during each interval
    csza_s : xr.DataArray
        Cosine of the solar zenith angle during the sunlit period of each interval
    rsds : xr.DataArray
        Surface Downwelling Shortwave Radiation

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Ratio of direct solar radiation

    Notes
    -----
    This code was inspired by the `PyWBGT` package.

    References
    ----------
    :cite:cts:`liljegren_modeling_2008,kong_explicit_2022`
    """
    d = distance_from_sun(dates)
    s_star = rsds * ((1367 * csza_s * (d ** (-2))) ** (-1))
    s_star = xr.where(s_star > 0.85, 0.85, s_star)
    fdir_ratio = np.exp(3 - 1.34 * s_star - 1.65 * (s_star ** (-1)))
    fdir_ratio = xr.where(fdir_ratio > 0.9, 0.9, fdir_ratio)
    return xr.where(
        (fdir_ratio <= 0) | (csza_i <= np.cos(89.5 / 180 * np.pi)) | (rsds <= 0),
        0,
        fdir_ratio,
    )


@declare_units(
    rsds="[radiation]", rsus="[radiation]", rlds="[radiation]", rlus="[radiation]"
)
def mean_radiant_temperature(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
    stat: str = "average",
) -> xr.DataArray:
    r"""Mean radiant temperature.

    The mean radiant temperature is the incidence of radiation on the body from all directions.

    Parameters
    ----------
    rsds : xr.DataArray
       Surface Downwelling Shortwave Radiation
    rsus : xr.DataArray
        Surface Upwelling Shortwave Radiation
    rlds : xr.DataArray
        Surface Downwelling Longwave Radiation
    rlus : xr.DataArray
        Surface Upwelling Longwave Radiation
    stat : {'average', 'instant', 'sunlit'}
        Which statistic to apply. If "average", the average of the cosine of the
        solar zenith angle is calculated. If "instant", the instantaneous cosine
        of the solar zenith angle is calculated. If "sunlit", the cosine of the
        solar zenith angle is calculated during the sunlit period of each interval.
        If "instant", the instantaneous cosine of the solar zenith angle is calculated.

    Returns
    -------
    xarray.DataArray, [K]
        Mean radiant temperature

    Warnings
    --------
    There are some issues in the calculation of mrt in polar regions.

    Notes
    -----
    This code was inspired by the `thermofeel` package :cite:p:`brimicombe_thermofeel_2021`.

    References
    ----------
    :cite:cts:`di_napoli_mean_2020`
    """
    rsds = convert_units_to(rsds, "W m-2")
    rsus = convert_units_to(rsus, "W m-2")
    rlds = convert_units_to(rlds, "W m-2")
    rlus = convert_units_to(rlus, "W m-2")

    dates = rsds.time
    lat = _gather_lat(rsds)
    lon = _gather_lon(rsds)
    dec = solar_declination(dates)

    if stat == "sunlit":
        csza_i = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, stat="average", sunlit=False
        )
        csza_s = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, stat="average", sunlit=True
        )
    elif stat == "instant":
        tc = time_correction_for_solar_angle(dates)
        csza = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, time_correction=tc, stat="instant"
        )
        csza_i = csza.copy()
        csza_s = csza.copy()
    elif stat == "average":
        csza = cosine_of_solar_zenith_angle(
            dates, dec, lat, stat="average", sunlit=False
        )
        csza_i = csza.copy()
        csza_s = csza.copy()
    else:
        raise NotImplementedError(
            "Argument 'stat' must be one of 'average', 'instant' or 'sunlit'."
        )

    fdir_ratio = _fdir_ratio(dates, csza_i, csza_s, rsds)

    rsds_direct = fdir_ratio * rsds
    rsds_diffuse = rsds - rsds_direct

    gamma = np.arcsin(csza_i)
    fp = 0.308 * np.cos(gamma * 0.988 - (gamma**2 / 50000))
    i_star = xr.where(csza_s > 0.001, rsds_direct / csza_s, 0)

    mrt = np.power(
        (
            (1 / 5.67e-8)  # Stefan-Boltzmann constant
            * (
                0.5 * rlds
                + 0.5 * rlus
                + (0.7 / 0.97) * (0.5 * rsds_diffuse + 0.5 * rsus + fp * i_star)
            )
        ),
        0.25,
    )
    return mrt.assign_attrs({"units": "K"})
