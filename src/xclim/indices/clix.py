"""
Clix-meta index functions submodule
===================================

Generic indices implementing index functions from the `clix-meta <https://github.com/clix-meta/clix-meta/>`
framework for climate index definitions. Functions here are redefinitions of functions in
:py:mod:`xclim.indices.generic` but with signatures that match the clix-meta vocabulary.

This module tries to follow the clix-meta definitions as closely as possible,
which means it can have meaningful differences with the rest of xclim.

For example, indicators where a number of occurrences (usually days) is counted will use units "1", instead of
having temporal dimensions (i.e. "days") like xclim does elsewhere.

However, indicators calculating a date will have no units in this module. "clix-meta" suggests "day", but that
already means something else.

This version of xclim implements clix-meta v0.6.1 .
"""

from __future__ import annotations

from typing import Literal

import xarray as xr

import xclim.indices.generic as generic
from xclim.core import DateStr, DayOfYearStr, Quantified
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
    Calculate the number of times the given threshold level is crossed during the specified time period.

    I.e. how many times the maximum data is above the threshold and the minimum data is below the threshold.
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
    xr.DataArray, [dimensionless]
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
    ).assign_attrs(units="1")


@declare_relative_units(threshold="<data>")
def count_occurrences(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the number of times the given threshold is exceeded during the specified time period.

    First, the threshold is transformed to the same standard_name and units as the input data. Then the condition is
    applied, i.e. if ``condition`` is <, the comparison ``data < threshold`` has to be fulfilled. Finally, the number of
    times when the comparison is fulfilled is counted.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Number of times where data is {condition} {threshold}.
    """
    return generic.count_occurrences(
        data=data,
        threshold=threshold,
        op=condition,
        freq=freq,
    ).assign_attrs(units="1")


def count_percentile_occurrences(
    data: xr.DataArray,
    percentile: float,
    condition: OPERATORS,
    reference_period: TIME_RANGE,
    freq: str,
) -> xr.DataArray:
    """
    Calculate how many times a seasonally varying percentile-based threshold is exceeded during the specified period.

    First, the given ``percentile`` value is used to calculate the climatology for the specified
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
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates are given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.

    Returns
    -------
    xr.DataArray, [dimensionless]
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
    ).assign_attrs(units="1")


# TODO: count_spell_duraction. Not ready in clix-meta v0.6.1


@declare_relative_units(data_threshold="<data>")
def count_thresholded_percentile_occurrences(
    data: xr.DataArray,
    data_threshold: Quantified,
    data_condition: OPERATORS,
    percentile: float,
    percentile_condition: OPERATORS,
    reference_period: TIME_RANGE,
    freq: str,
) -> xr.DataArray:
    """
    Calculate how many time a percentile-based threshold (fixed over the year) is exceeded during the specified period.

    First the ``data_threshold`` is transformed to the same standard name and units as the
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
        Logical comparison operator to filter the data.
    percentile : float
        The percentile to compute on the reference period, between 0 and 100.
    percentile_condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator to find percentile occurrences on filtered data.
    reference_period : tuple of two dates
        Start and end of the period used to compute the percentiles. Dates are given as YYYY-MM-DD.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        This function only makes sense with annual frequencies.

    Returns
    -------
    xr.DataArray, [dimensionless]
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
        bootstrap=False,
    ).assign_attrs(units="1")


@declare_relative_units(high_data="<low_data>")
def diurnal_temperature_range(
    low_data: xr.DataArray, high_data: xr.DataArray, statistic: REDUCERS, freq: str
) -> xr.DataArray:
    """
    Calculate a statistical measure on the diurnal temperature range during the specified time period.

    It takes two inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum temperature.
    The diurnal temperature range is first calculated, and then the statistic is calculated.

    Parameters
    ----------
    low_data : xr.DataArray
        The lowest daily temperature (tasmin).
    high_data : xr.DataArray
        The highest daily temperature (tasmax).
    statistic : {"min", "max", "mean", "std", "var", "sum"}
        Reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {statistic} of the diurnal range between low_data and high_data.
    """
    return generic.diurnal_range(low_data=low_data, high_data=high_data, reducer=statistic, freq=freq)


@declare_relative_units(high_data="<low_data>")
def extreme_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the maximum temperature difference during the specified time period.

    It takes two inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum temperature.
    From this it calculates the extreme temperature range as the maximum of daily maximum temperature minus
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
    after_date: DayOfYearStr = None,
) -> xr.DataArray:
    """
    Calculate the first time during the specified time period when a threshold is exceeded.

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
        Logical comparison operator.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    after_date : Day of year str (MM-DD)
        Earliest day the occurrence can be found. Added to definition
        to support "faf" without indexing.

    Returns
    -------
    xr.DataArray, [day of year]
        Day-of-year of the first time where data {condition} {threshold}.
    """
    return generic.day_threshold_reached(
        data=data,
        threshold=threshold,
        op=condition,
        date=after_date,
        which="first",
        window=1,
        freq=freq,
    )


# TODO: first_spell. Marked as ready in clix-meta 0.6.1, but no index definitions are using it.
#       Very similar to first_run_after_date, but the "date" is given as a "dead_period" in number of timesteps (days).
#       Would be easy to implement if freq is fixed to annual, but the description doesn't mention this.


def interday_diurnal_temperature_range(low_data: xr.DataArray, high_data: xr.DataArray, freq: str) -> xr.DataArray:
    """
    Calculate the mean over the specified period of the absolute day-to-day difference in diurnal temperature range.

    It takes two inputs, ``low_data`` and ``high_data``, i.e. daily minimum and maximum
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
    before_date: DayOfYearStr = None,
) -> xr.DataArray:
    """
    Calculate the last time during the specified time period when a threshold exceeded.

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
        Logical comparison operator.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    before_date : Day of year str (MM-DD)
        Latest day the occurrence can be found. Added to definition
        to support "lsf" without indexing.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Day-of-year of the last time where data {condition} {threshold}.
    """
    return generic.day_threshold_reached(
        data=data,
        threshold=threshold,
        op=condition,
        date=before_date,
        which="last",
        window=1,
        freq=freq,
    )


def percentile(data: xr.DataArray, percentiles: float, freq: str):
    """
    Calculate a percentile statistic over the data in the specified time period.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    percentiles : float
        A percentile (0, 100).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray, [same as data]
        {percentiles}th percentile of the data.
    """
    return generic.percentile(data, percentiles, freq)


def running_statistics(
    data: xr.DataArray, window: int, rolling_aggregator: REDUCERS, overall_statistic: REDUCERS, freq: str
):
    """
    Calculate, for the specified period, a statistic on a rolling aggregation, using the specified window size.

    Parameters
    ----------
    data : xr.DataArray
        An array.
    window : int
        The running window size.
    rolling_aggregator : {"min", "max", "mean", "std", "var", "sum"}
        The running statistic. The result is assigned to the window's center.
    overall_statistic : {"min", "max", "mean", "std", "var", "sum"}
        Reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Resampling is done after computing the running statistics.

    Returns
    -------
    xr.DataArray
        {overall_statistic} of the {window}-day {rolling_aggregator}.
    """
    return generic.select_rolling_resample_op(
        da=data, op=overall_statistic, window=window, window_center=True, window_op=rolling_aggregator, freq=freq
    )


@declare_relative_units(threshold="<data>")
def spell_length(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    statistic: REDUCERS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate the statistic over the lengths of spells during the specified time period.

    First, the ``threshold`` is transformed to the same standard_name and units as the input data.
    Then the``condition`` is applied, i.e. if ``condition`` is "<"", the comparison ``data < threshold``
    has to be fulfilled, and spell lengths are calculated from the resulting data.
    Finally the ``statistic`` over spell lengths is calculated.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    statistic : {"min", "max", "mean", "std", "var", "sum"}
        Spell lengths reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {statistic} of spells where data {condition} {threshold}.
    """
    return generic.spell_length_statistics(
        data=data,
        threshold=threshold,
        window=1,
        win_reducer=None,
        op=condition,
        spell_reducer=statistic,
        resample_before_rl=False,
        freq=freq,
    )


def statistics(data: xr.DataArray, statistic: REDUCERS, freq: str):
    """
    Calculate the statistic over the data in the specified time period.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    statistic : {"min", "max", "mean", "std", "var", "sum"}
        Reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {statistic} of data.
    """
    return generic.select_resample_op(data, op=statistic, freq=freq)


@declare_relative_units(threshold="<data>")
def temperature_sum(data: xr.DataArray, threshold: Quantified, condition: OPERATORS, freq: str) -> xr.DataArray:
    """
    Calculate the temperature sum above or below a threshold during the specified time period.

    First, the ``threshold`` is transformed to the same standard_name and units as the input data. Then the
    condition is applied, i.e. if ``condition`` is "<", the comparison ``data < threshold`` has to be fulfilled.
    Finally, for those data values that fulfil the condition the sum is calculated after
    subtraction of the threshold value. If the sum is for values below the threshold the result is multiplied by -1.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        Sum of temperature {condition} {threshold}.
    """
    return generic.cumulative_difference(data, threshold=threshold, op=condition, freq=freq)


@declare_relative_units(threshold="<data>")
def thresholded_percentile(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    percentile: float,
    freq: str,
) -> xr.DataArray:
    """
    Calculate, for the specified time period, a percentile statistic on the data that meets the specified condition.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the condition is applied, i.e. if ``condition`` is "<", the comparison ``data < threshold``
    has to be fulfilled. Finally, the percentile is calculated over the data that fulfil the condition.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    percentile : float
        A percentile (0, 100).
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {percentile}th percentile of data where it is {condition} {threshold}.
    """
    return generic.thresholded_percentile(data, threshold=threshold, op=condition, percentile=percentile, freq=freq)


@declare_relative_units(threshold="<data>")
def thresholded_running_statistics(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    rolling_aggregator: REDUCERS,
    window_size: int,
    overall_statistic: REDUCERS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate, for the specified period, a statistic on a rolling aggregation on filtered data.

    First, the threshold is transformed to the same standard_name and units as the input data. Then the condition
    is applied, i.e. if `condition` is "<", the comparison ``data < threshold`` has to be fulfilled. Then the
    ``rolling_aggregator`` is calculated on the data that fulfil the condition, and finally the ``overall_statistic``
    is calculated over the resulting data.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    rolling_aggregator : {"min", "max", "mean", "std", "var", "sum"}
        Running reducer over the window.
    window_size : int
        Size of the rolling window (centered).
    overall_statistic : {"min", "max", "mean", "std", "var", "sum"}
        Reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
        Applied after the rolling window.

    Returns
    -------
    xr.DataArray
        {overall_statistic} of {window_size}-day {rolling_aggregtor} of data where it is {condition} {threshold}.
    """
    return generic.thresholded_rolling_resample_op(
        data,
        threshold=threshold,
        op=condition,
        reducer=overall_statistic,
        window=window_size,
        window_center=True,
        window_op=rolling_aggregator,
        freq=freq,
    )


@declare_relative_units(threshold="<data>")
def thresholded_statistics(
    data: xr.DataArray,
    threshold: Quantified,
    condition: OPERATORS,
    statistic: REDUCERS,
    freq: str,
) -> xr.DataArray:
    """
    Calculate, for the specified time period, the statistic over the data for which some condition is met.

    First, the threshold is transformed to the same standard_name and units as the input data.
    Then the condition is applied, i.e. if ``condition`` is "<", the comparison ``data < threshold``
    has to be fulfilled. Finally, the statistic is calculated for those data values that fulfil the condition.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    threshold : Quantified
        Threshold.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        Logical comparison operator.
    statistic : {"min", "max", "mean", "std", "var", "sum"}
        Reducer over the resampling period.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.

    Returns
    -------
    xr.DataArray
        {statistic} of data where it is {condition} {threshold}.
    """
    return generic.thresholded_resample_op(data, threshold=threshold, op=condition, reducer=statistic, freq=freq)
