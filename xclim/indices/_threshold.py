import datetime
from typing import Union

import numpy as np
import xarray

from xclim import run_length as rl
from xclim import utils
from xclim.utils import declare_units
from xclim.utils import units

xarray.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "cold_spell_days",
    "daily_pr_intensity",
    "maximum_consecutive_wet_days",
    "cooling_degree_days",
    "freshet_start",
    "growing_degree_days",
    "growing_season_length",
    "heat_wave_index",
    "heating_degree_days",
    "tn_days_below",
    "tx_days_above",
    "warm_day_frequency",
    "warm_night_frequency",
    "wetdays",
    "maximum_consecutive_dry_days",
    "maximum_consecutive_tx_days",
    "sea_ice_area",
    "sea_ice_extent",
    "tropical_nights",
]


@declare_units("days", tas="[temperature]", thresh="[temperature]")
def cold_spell_days(tas, thresh="-10 degC", window: int = 5, freq="AS-JUL"):
    r"""Cold spell days

    The number of days that are part of a cold spell, defined as five or more consecutive days with mean daily
    temperature below a threshold in °C.

    Parameters
    ----------
    tas : xarrray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature below which a cold spell begins [℃] or [K]. Default: '-10 degC'
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
      Resampling frequency; Defaults to "AS-JUL".

    Returns
    -------
    xarray.DataArray
      Cold spell days.

    Notes
    -----
    Let :math:`T_i` be the mean daily temperature on day :math:`i`, the number of cold spell days during
    period :math:`\phi` is given by

    .. math::

       \sum_{i \in \phi} \prod_{j=i}^{i+5} [T_j < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    """
    t = utils.convert_units_to(thresh, tas)
    over = tas < t
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim="time")


@declare_units("mm/day", pr="[precipitation]", thresh="[precipitation]")
def daily_pr_intensity(pr, thresh="1 mm/day", freq="YS"):
    r"""Average daily precipitation intensity

    Return the average precipitation over wet days.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm/d or kg/m²/s]
    thresh : str
      precipitation value over which a day is considered wet. Default : '1 mm/day'
    freq : str
      Resampling frequency defining the periods defined in
      http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The average precipitation over wet days for each period

    Notes
    -----
    Let :math:`\mathbf{p} = p_0, p_1, \ldots, p_n` be the daily precipitation and :math:`thresh` be the precipitation
    threshold defining wet days. Then the daily precipitation intensity is defined as

    .. math::

       \frac{\sum_{i=0}^n p_i [p_i \leq thresh]}{\sum_{i=0}^n [p_i \leq thresh]}

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the average
    precipitation fallen over days with precipitation >= 5 mm at seasonal
    frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> import xarray as xr
    >>> import xclim.indices
    >>> pr = xr.open_dataset("pr_day.nc").pr
    >>> daily_int = xclim.indices.daily_pr_intensity(pr, thresh='5 mm/day', freq="QS-DEC")
    """
    t = utils.convert_units_to(thresh, pr, "hydro")

    # put pr=0 for non wet-days
    pr_wd = xarray.where(pr >= t, pr, 0)
    pr_wd.attrs["units"] = pr.units

    # sum over wanted period
    s = pr_wd.resample(time=freq).sum(dim="time", keep_attrs=True)
    sd = utils.pint_multiply(s, 1 * units.day, "mm")

    # get number of wetdays over period
    wd = wetdays(pr, thresh=thresh, freq=freq)
    return sd / wd


@declare_units("days", pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_wet_days(
    pr: xarray.DataArray, thresh: str = "1 mm/day", freq: str = "YS"
):
    r"""Consecutive wet days.

    Returns the maximum number of consecutive wet days.

    Parameters
    ---------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm]
    thresh : str
      Threshold precipitation on which to base evaluation [Kg m-2 s-1] or [mm]. Default : '1 mm/day'
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive wet days.

    Notes
    -----
    Let :math:`\mathbf{x}=x_0, x_1, \ldots, x_n` be a daily precipitation series and
    :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where :math:`[p_i > thresh] \neq [p_{i+1} >
    thresh]`, that is, the days when the precipitation crosses the *wet day* threshold.
    Then the maximum number of consecutive wet days is given by

    .. math::


       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [x_{s_j} > 0^\circ C]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    thresh = utils.convert_units_to(thresh, pr, "hydro")

    group = (pr > thresh).resample(time=freq)
    return group.apply(rl.longest_run, dim="time")


@declare_units("C days", tas="[temperature]", thresh="[temperature]")
def cooling_degree_days(
    tas: xarray.DataArray, thresh: str = "18 degC", freq: str = "YS"
):
    r"""Cooling degree days

    Sum of degree days above the temperature threshold at which spaces are cooled.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Temperature threshold above which air is cooled. Default : '18 degC'
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Cooling degree days

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day :math:`i`. Then the cooling degree days above
    temperature threshold :math:`thresh` over period :math:`\phi` is given by:

    .. math::

        \sum_{i \in \phi} (x_{i}-{thresh} [x_i > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    thresh = utils.convert_units_to(thresh, tas)

    return (
        tas.pipe(lambda x: x - thresh).clip(min=0).resample(time=freq).sum(dim="time")
    )


@declare_units("", tas="[temperature]", thresh="[temperature]")
def freshet_start(
    tas: xarray.DataArray, thresh: str = "0 degC", window: int = 5, freq: str = "YS"
):
    r"""First day consistently exceeding threshold temperature.

    Returns first day of period where a temperature threshold is exceeded
    over a given number of days.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default '0 degC'
    window : int
      Minimum number of days with temperature above threshold needed for evaluation
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Day of the year when temperature exceeds threshold over a given number of days for the first time. If there are
      no such day, return np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the freshet is given by the smallest index :math:`i` for which

    .. math::

       \prod_{j=i}^{i+w} [x_j > thresh]

    is true, where :math:`w` is the number of days the temperature threshold should be exceeded,  and :math:`[P]` is
    1 if :math:`P` is true, and 0 if false.
    """
    thresh = utils.convert_units_to(thresh, tas)
    over = tas > thresh
    group = over.resample(time=freq)
    return group.apply(rl.first_run_ufunc, window=window, index="dayofyear")


@declare_units("C days", tas="[temperature]", thresh="[temperature]")
def growing_degree_days(
    tas: xarray.DataArray, thresh: str = "4.0 degC", freq: str = "YS"
):
    r"""Growing degree-days over threshold temperature value [℃].

    The sum of degree-days over the threshold temperature.

    Parameters
    ---------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '4.0 degC'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The sum of growing degree-days above 4℃

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    growing degree days are:

    .. math::

        GD4_j = \sum_{i=1}^I (TG_{ij}-{4} | TG_{ij} > {4}℃)
    """
    thresh = utils.convert_units_to(thresh, tas)
    return (
        tas.pipe(lambda x: x - thresh).clip(min=0).resample(time=freq).sum(dim="time")
    )


@declare_units("days", tas="[temperature]", thresh="[temperature]")
def growing_season_length(
    tas: xarray.DataArray,
    thresh: str = "5.0 degC",
    window: int = 6,
    mid_date: str = "07-01",
    freq: str = "YS",
):
    r"""Growing season length.

    The number of days between the first occurrence of at least
    six consecutive days with mean daily temperature over 5℃ and
    the first occurrence of at least six consecutive days with
    mean daily temperature below 5℃ after a certain date. (Usually
    July 1st in the northern hemisphere and January 1st in the southern hemisphere.)

    WARNING: The default values are only valid for the northern hemisphere.

    Parameters
    ---------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '5.0 degC'.
    window : int
      Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    mid_date : str
      Date of the year after which to look for the end of the season. Should have the format '%m-%d'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Growing season length.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then counted is
    the number of days between the first occurrence of at least 6 consecutive days with:

    .. math::

        TG_{ij} > 5 ℃

    and the first occurrence after 1 July of at least 6 consecutive days with:

    .. math::

        TG_{ij} < 5 ℃

    Examples
    --------
    If working in the Southern Hemisphere, one can use:

    >>> gsl = growing_season_length(tas, mid_date='01-01', freq='AS-Jul')
    """
    thresh = utils.convert_units_to(thresh, tas)

    mid_doy = datetime.datetime.strptime(mid_date, "%m-%d").timetuple().tm_yday

    def compute_gsl(yrdata):
        if (
            yrdata.chunks is not None
            and len(yrdata.chunks[yrdata.dims.index("time")]) > 1
        ):
            yrdata = yrdata.chunk({"time": -1})
        mid_idx = np.where(yrdata.time.dt.dayofyear == mid_doy)[0]
        if (
            mid_idx.size == 0
        ):  # The mid date is not in the group. Happens at boundaries.
            allNans = xarray.full_like(yrdata.isel(time=0), np.nan)
            allNans.attrs = {}
            return allNans
        end = rl.first_run(
            yrdata.where(yrdata.time >= yrdata.time[mid_idx][0]) < thresh,
            window,
            "time",
        )
        beg = rl.first_run(yrdata > thresh, window, "time")
        sl = end - beg
        sl = xarray.where(
            beg.isnull() & end.notnull(), 0, sl
        )  # If everything is under thresh
        sl = xarray.where(
            beg.notnull() & end.isnull(), yrdata.time.size - beg, sl
        )  # If gs is not ended at end of year
        return sl.where(sl >= 0)  # When end happens before beg.

    gsl = tas.resample(time=freq).map(compute_gsl)
    return gsl


@declare_units("days", tasmax="[temperature]", thresh="[temperature]")
def heat_wave_index(
    tasmax: xarray.DataArray,
    thresh: str = "25.0 degC",
    window: int = 5,
    freq: str = "YS",
):
    r"""Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to designate a heatwave [℃] or [K]. Default: '25.0 degC'.
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    DataArray
      Heat wave index.
    """
    thresh = utils.convert_units_to(thresh, tasmax)
    over = tasmax > thresh
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim="time")


@declare_units("C days", tas="[temperature]", thresh="[temperature]")
def heating_degree_days(
    tas: xarray.DataArray, thresh: str = "17.0 degC", freq: str = "YS"
):
    r"""Heating degree days

    Sum of degree days below the temperature threshold at which spaces are heated.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '17.0 degC'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Heating degree days index.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    heating degree days are:

    .. math::

        HD17_j = \sum_{i=1}^{I} (17℃ - TG_{ij})
    """
    thresh = utils.convert_units_to(thresh, tas)

    return tas.pipe(lambda x: thresh - x).clip(0).resample(time=freq).sum(dim="time")


@declare_units("days", tasmin="[temperature]", thresh="[temperature]")
def tn_days_below(
    tasmin: xarray.DataArray, thresh: str = "-10.0 degC", freq: str = "YS"
):
    r"""Number of days with tmin below a threshold in

    Number of days where daily minimum temperature is below a threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K] . Default: '-10 degC'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Number of days Tmin < threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmin)
    f1 = utils.threshold_count(tasmin, "<", thresh, freq)
    return f1


@declare_units("days", tasmax="[temperature]", thresh="[temperature]")
def tx_days_above(
    tasmax: xarray.DataArray, thresh: str = "25.0 degC", freq: str = "YS"
):
    r"""Number of summer days

    Number of days where daily maximum temperature exceed a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '25 degC'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Number of summer days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} > Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmax)
    f = (tasmax > thresh) * 1
    return f.resample(time=freq).sum(dim="time")


@declare_units("days", tasmax="[temperature]", thresh="[temperature]")
def warm_day_frequency(
    tasmax: xarray.DataArray, thresh: str = "30 degC", freq: str = "YS"
):
    r"""Frequency of extreme warm days

    Return the number of days with tasmax > thresh per period

    Parameters
    ----------
    tasmax : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default : '30 degC'
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Number of days exceeding threshold.

    Notes:
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]

    """
    thresh = utils.convert_units_to(thresh, tasmax)
    events = (tasmax > thresh) * 1
    return events.resample(time=freq).sum(dim="time")


@declare_units("days", tasmin="[temperature]", thresh="[temperature]")
def warm_night_frequency(
    tasmin: xarray.DataArray, thresh: str = "22 degC", freq: str = "YS"
):
    r"""Frequency of extreme warm nights

    Return the number of days with tasmin > thresh per period

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default : '22 degC'
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The number of days with tasmin > thresh per period
    """
    thresh = utils.convert_units_to(thresh, tasmin)
    events = (tasmin > thresh) * 1
    return events.resample(time=freq).sum(dim="time")


@declare_units("days", pr="[precipitation]", thresh="[precipitation]")
def wetdays(pr: xarray.DataArray, thresh: str = "1.0 mm/day", freq: str = "YS"):
    r"""Wet days

    Return the total number of days during period with precipitation over threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm]
    thresh : str
      Precipitation value over which a day is considered wet. Default: '1 mm/day'.
    freq : str
      Resampling frequency defining the periods defined in
      http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The number of wet days for each period [day]

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the number days
    with precipitation over 5 mm at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> import xarray as xr
    >>> import xclim.utils
    >>> pr = xr.open_dataset('pr.day.nc').pr
    >>> wd = xclim.indices.wetdays(pr, pr_min=5., freq="QS-DEC")
    """
    thresh = utils.convert_units_to(thresh, pr, "hydro")

    wd = (pr >= thresh) * 1
    return wd.resample(time=freq).sum(dim="time")


@declare_units("days", pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_dry_days(
    pr: xarray.DataArray, thresh: str = "1 mm/day", freq: str = "YS"
):
    r"""Maximum number of consecutive dry days

    Return the maximum number of consecutive days within the period where precipitation
    is below a certain threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [mm]
    thresh : str
      Threshold precipitation on which to base evaluation [mm]. Default : '1 mm/day'
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive dry days.

    Notes
    -----
    Let :math:`\mathbf{p}=p_0, p_1, \ldots, p_n` be a daily precipitation series and :math:`thresh` the threshold
    under which a day is considered dry. Then let :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where
    :math:`[p_i < thresh] \neq [p_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive dry days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [p_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = utils.convert_units_to(thresh, pr, "hydro")
    group = (pr < t).resample(time=freq)

    return group.apply(rl.longest_run, dim="time")


@declare_units("days", tasmax="[temperature]", thresh="[temperature]")
def maximum_consecutive_tx_days(
    tasmax: xarray.DataArray, thresh: str = "25 degC", freq: str = "YS"
):
    r"""Maximum number of consecutive summer days (Tx > 25℃)

    Return the maximum number of consecutive days within the period where temperature is above a certain threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Max daily temperature [K]
    thresh : str
      Threshold temperature [K].
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive summer days.

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily maximum temperature series and :math:`thresh` the threshold
    above which a day is considered a summer day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive dry days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = utils.convert_units_to(thresh, tasmax)
    group = (tasmax > t).resample(time=freq)

    return group.apply(rl.longest_run, dim="time")


@declare_units("[area]", sic="[]", area="[area]", thresh="[]")
def sea_ice_area(sic, area, thresh="15 pct"):
    """Return the total sea ice area.

    Sea ice area measures the total sea ice covered area where sea ice concentration is above a threshold,
    usually set to 15%.

    Parameters
    ----------
    sic : xarray.DataArray
      Sea ice concentration [0,1].
    area : xarray.DataArray
      Grid cell area [m²]
    thresh : str
      Minimum sea ice concentration for a grid cell to contribute to the sea ice extent.

    Returns
    -------
    Sea ice area [m²].

    Notes
    -----
    To compute sea ice area over a subregion, first mask or subset the input sea ice concentration data.

    References
    ----------
    `What is the difference between sea ice area and extent
    <https://nsidc.org/arcticseaicenews/faq/#area_extent>`_

    """
    t = utils.convert_units_to(thresh, sic)
    factor = utils.convert_units_to("100 pct", sic)
    out = xarray.dot(sic.where(sic >= t, 0), area) / factor
    out.attrs["units"] = area.units
    return out


@declare_units("[area]", sic="[]", area="[area]", thresh="[]")
def sea_ice_extent(sic, area, thresh="15 pct"):
    """Return the total sea ice extent.

    Sea ice extent measures the *ice-covered* area, where a region is considered ice-covered if its sea ice
    concentration is above a threshold usually set to 15%.

    Parameters
    ----------
    sic : xarray.DataArray
      Sea ice concentration [0,1].
    area : xarray.DataArray
      Grid cell area [m²]
    thresh : str
      Minimum sea ice concentration for a grid cell to contribute to the sea ice extent.

    Returns
    -------
    Sea ice extent [m²].

    Notes
    -----
    To compute sea ice area over a subregion, first mask or subset the input sea ice concentration data.

    References
    ----------
    `What is the difference between sea ice area and extent
    <https://nsidc.org/arcticseaicenews/faq/#area_extent>`_
    """
    t = utils.convert_units_to(thresh, sic)
    out = xarray.dot(sic >= t, area)
    out.attrs["units"] = area.units
    return out


@declare_units("days", tasmin="[temperature]", thresh="[temperature]")
def tropical_nights(
    tasmin: xarray.DataArray, thresh: str = "20.0 degC", freq: str = "YS"
):
    r"""Tropical nights

    The number of days with minimum daily temperature above threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '20 degC'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      Number of days with minimum daily temperature above threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmin)
    return (
        tasmin.pipe(lambda x: (tasmin > thresh) * 1).resample(time=freq).sum(dim="time")
    )
