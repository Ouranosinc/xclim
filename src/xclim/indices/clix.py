"""
Clix-meta index functions submodule
===================================

Generic indices implementing index functions from the `clix-meta <https://github.com/clix-meta/clix-meta/>`
framework for climate index definitions. Functions here are simple redefinitions of functions in
:py:mod:`xclim.indices.generic` but with signatures that match the clix-meta vocabulary.

This version of xclim implements clix-meta v0.6.1 .
"""

from typing import Literal

import xarray as xr

import xclim.indices.generic as generic
from xclim.core import DateStr, Quantified
from xclim.core.units import declare_relative_units

OPERATORS = Literal[">", "gt", "<", "lt", ">=", "ge", "<=", "le"]
REDUCERS = Literal["min", "max", "mean", "std", "var", "sum"]
TIME_RANGE = tuple[DateStr, DateStr]


# TODO: count_bivariate_percentile_occurrences, not ready in clix-meta 0.6.1


@declare_relative_units(threshold="<low_data>", high_data="<low_data>")
def count_level_crossings(
    low_data: xr.DataArray,
    high_data: xr.DataArray,
    threshold: Quantified,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the number of times low_data is below threshold while high_data is above threshold.

    This index function calculates the number of times the given threshold level is crossed during the specified time
    period, i.e. how many times the maximum data is above the threshold and the minimum data is below the threshold.
    The function takes two inputs, ``low_data`` and ``high_data``, together with one parameter, the ``threshold``.
    First, the threshold is transformed to the same standard_name and units as the input data. Then the comparison is
    done as ``low_data < threshold < high_data``, and finally the number of times when the comparison is fulfilled is
    counted.

    Parameters
    ----------
    low_data : xr.DataArray
        Variable that must be under the threshold.
    high_data : xr.DataArray
        Variable that must be above the threshold.
    threshold : Quantified
        Threshold.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [time]
        Number of times when low_data is under {threshold} and high_data is above {threshold}.
    """
    return generic.bivariate_count_occurrences(
        data_var1=low_data,
        data_var2=high_data,
        threshold_var1=threshold,
        threshold_var2=threshold,
        freq=freq,
        op_var1="<",
        op_var2=">",
        var_reducer="all",
    )


@declare_relative_units(threshold="<data>")
def count_occurrences(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the number of times data meets a condition.

    This index function calculates the number of times the given threshold is exceeded during the specified time period.
    First, the threshold is transformed to the same standard_name and units as the input data. Then the condition is
    applied, i.e. if ``condition`` is <, the comparison ``data < threshold`` has to be fulfilled. Finally, the number of
    times when the comparison is fulfilled is counted.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator. e.g. arr > thresh.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [time]
        Number of times where data is {condition} {threshold}.
    """
    return generic.count_occurrences(
        data=data, threshold=threshold, op=condition, freq=freq, constrain=OPERATORS.__args__
    )


def count_percentile_occurences(
    data: xr.DataArray,
    percentile: float,
    condition: OPERATORS,
    reference_period: TIME_RANGE,
    freq: str,
) -> xr.DataArray:
    """
    Count how many times a seasonally varying percentile-based threshold is exceeded.

    This index function calculates how many times a seasonally varying percentile-based threshold is exceeded during the
    specified time period. First, the given ``percentile`` value is used to calculate the climatology for the specified
    reference period of daily percentile levels over a 5-day window centred on each specific day. These seasonally
    varying percentile levels are used when applying the condition, i.e. if the ``condition`` is <, the comparison is
    done for each day (i) as ``data(i) < percentile_level(i)``. Finally, the number of times when the comparison is
    fulfilled is counted.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    percentile : float
        The percentile to compute on the reference period, between 0 and 100.
    condition :  {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr[i] > climatology[doy(i)]
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates are given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.

    Returns
    -------
    xr.DataArray, [time]
        Number of times data {condition} the {percentile}th percentile for the same day of year
        (computed using a 5-day window).
    """
    return generic.count_percentile_occurrences(
        data=data,
        percentile=percentile,
        op=condition,
        reference_period=reference_period,
        freq=freq,
        bootstrap=False,
        constrain=OPERATORS.__args__,
    )


# TODO: count_spell_duraction. Not ready in clix-meta v0.6.1


@declare_relative_units(threshold="<data>")
def count_thresholded_percentile_occurences(
    data: xr.DataArray,
    data_threshold: Quantified,
    data_condition: OPERATORS,
    percentile: float,
    percentile_condition: OPERATORS,
    reference_period: TIME_RANGE,
    freq: str,
) -> xr.DataArray:
    """
    Count how many times a seasonally varying percentile-based threshold is exceeded within filtered data.

    This index function calculates how many time a percentile-based threshold (fixed over the year) is exceeded during
    the specified time period. First the ``data_threshold`` is transformed to the same standard name and units as the
    input data. Then the given ``percentile`` value is used to calculate the climatological percentile-based threshold
    for the specified reference period.  This constant percentile level is then used when applying the
    ``percentile_condition``, i.e. if the percentile condition is <, the comparison is done as
    ``data < percentile_level``, and finally the number of times when the percentile condition is fulfilled is counted.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    data_threshold : Quantified
        Threshold.
    data_condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Operator for the data condition, for filtering the data prior to computing the climatology.
    percentile : float
        The percentile to compute on the reference period, between 0 and 100.
    percentile_condition :  {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Operator for the percentile_condition e.g. arr[i] > climatology[doy(i)].
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates are given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.

    Returns
    -------
    xr.DataArray, [time]
        Number of times data is {percentile_condition} the {percentile}th percentile of
        {data} {data_condition} {data_threshold}.
    """
    return generic.count_thresholded_percentile_occurrences(
        data=data,
        threshold=data_threshold,
        op_thresh=data_condition,
        percentile=percentile,
        op_perc=percentile_condition,
        reference_period=reference_period,
        freq=freq,
        constrain=OPERATORS.__args__,
        bootstrap=False,
    )


@declare_relative_units(high_data="<low_data>")
def diurnal_temperature_range(
    low_data: xr.DataArray, high_data: xr.DataArray, statistic: REDUCERS, freq: str
) -> xr.DataArray:
    """
    Calculate the diurnal temperature range and reduce according to a statistic.

    This index function calculates a statistical measure on the diurnal temperature range during the specified time
    period. It takes two inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum temperature.
    The diurnal temperature range is first calculated, and then the statistic is calculated.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    statistic : {'max', 'min', 'mean', 'sum', 'std', 'var'}
        Reducer.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {statistic} of the diurnal range between low_data and high_data.
    """
    return generic.diurnal_range(low_data=low_data, high_data=high_data, reducer=statistic, freq=freq)


@declare_relative_units(high_data="<low_data>")
def extreme_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the extreme daily temperature range.

    This index function calculates the maximum temperature difference during the specified time period. It takes two
    inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum temperature. From this it calculates the
    extreme temperature range as the maximum of daily maximum temperature minus
    the minimum of daily minimum temperature.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [difference of low_data]
        The range between the maximum of high_data and the minimum of low_data.
    """
    return generic.extreme_range(low_data=low_data, high_data=high_data, freq=freq)


@declare_relative_units(threshold="<data>")
def first_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the first time some condition is met.

    This index function calculates the first time during the specified time period when a threshold is exceeded.
    First, the threshold is transformed to the same standard_name and units as the input data. Then the condition is
    applied, i.e. if ``condition`` is <, the comparison ``data < threshold`` has to be fulfilled. Finally, the first
    occurrence when this comparison is met is located.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr > thresh.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [day of year]
        Day-of-year of the first time where data {condition} {threshold}.
    """
    return generic.day_threshold_reached(
        data=data,
        threshold=threshold,
        op=condition,
        date=None,
        which="first",
        window=1,
        freq=freq,
        constrain=OPERATORS.__args__,
    )


# TODO: first_spell. Marked as ready in clix-meta 0.6.1, but no index definitions are using it.
#       Very similar to first_run_after_date, but the "date" is given as a "dead_period" in number of timesteps (days).
#       Would be easy to implement if freq is fixed to annual, but the description doesn't mention this.


def interday_diurnal_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the average absolute day-to-day difference in diurnal range.

    This index function calculates the mean over the specified time period of the absolute day-to-day difference in
    diurnal temperature range. It takes two inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum
    temperature and calculates the diurnal temperature range. Then the day-to-day absolute difference is calculated and
    the average is formed.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest data.
    high_data : xr.DataArray
        The highest data.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [difference of low_data]
        Average day-to-day difference of the diurnal range between low_data and high_data.
    """
    return generic.interday_diurnal_range(low_data=low_data, high_data=high_data, reducer="mean", freq=freq)


@declare_relative_units(threshold="<data>")
def last_occurrence(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the last time some condition is met.

    This index function calculates the last time during the specified time period when a threshold exceeded.
    First, the threshold is transformed to the same standard_name and units as the input data, then the condition is
    applied, i.e. if ``condition`` is <, the comparison ``data < threshold`` has to be fulfilled. Finally, the last
    occurrence when this comparison is met is located.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical operator. e.g. arr > thresh.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Day-of-year of the last time where data {condition} {threshold}.
    """
    return generic.day_threshold_reached(
        data=data,
        threshold=threshold,
        op=condition,
        date=None,
        which="last",
        window=1,
        freq=freq,
        constrain=OPERATORS.__args__,
    )
