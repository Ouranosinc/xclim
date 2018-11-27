# -*- coding: utf-8 -*-
"""
Indicator calculation instances
===============================

While the `indices` module stores the computing functions, this module defines Indicator classes and instances that
include a number of functionalities, such as input validation, unit conversion, output meta-data handling,
and missing value masking.

The concept followed here is to define Indicator subclasses for each input variable, then create instances
for each indicator.

"""

from . import checks
from . import indices as _ind
from .utils import Indicator


# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


class Tas(Indicator):
    """Class for univariate indices using mean daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: mean within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


class Tasmin(Indicator):
    """Class for univariate indices using min daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: minimum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


class Tasmax(Indicator):
    """Class for univariate indices using max daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: maximum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


class TasminTasmax(Indicator):
    required_units = ('K', 'K')

    def cfprobe(self, dan, dax):
        for da in (dan, dax):
            checks.check_valid(da, 'cell_methods', 'time: maximum within days')
            checks.check_valid(da, 'standard_name', 'air_temperature')


summer_days = Tasmax(identifier='su_{thresh}',
                     units='days',
                     long_name='Number of Summer Days (Tmax > {thresh}C)',
                     cell_methods='time: maximum within days time: sum over days',
                     standard_name='number_of_days_with_air_temperature_above_threshold',
                     description="{freq} number of days where daily maximum temperature exceeds {thresh}℃",
                     compute=_ind.summer_days,
                     )

heat_wave_frequency = TasminTasmax(identifier='heat_wave_frequency',
                                   units='',
                                   long_name='Number of heat wave events',
                                   standard_name='events',
                                   description="Number of spells meeting criteria for health impacting heat wave.",
                                   keywords="health,",
                                   compute=_ind.heat_wave_frequency,
                                   )

heat_wave_index = Tasmax(identifier='hwi_{thresh}',
                         units='days',
                         description='{freq} number of days that are part of a heatwave, '
                                     'defined as five or more consecutive days over {thresh}℃',
                         long_name='Number of days that are part of a heatwave',
                         standard_name='heat_wave_index',
                         compute=_ind.heat_wave_index,
                         )

tg_mean = Tas(identifier='tg_mean',
              long_name="Mean daily mean temperature",
              units='K',
              standard_name="air_temperature",
              cell_methods='time: mean within days time: mean over days',
              description="{freq} mean of daily mean temperature.",
              keywords='',
              compute=_ind.tg_mean, )

tx_mean = Tasmax(identifier='tx_mean',
                 long_name='Mean daily maximum temperature',
                 units='K',
                 standard_name='air_temperature',
                 cell_methods='time: maximum within days time: mean over days',
                 description='{freq} mean of daily maximum temperature.',
                 keywords='',
                 compute=_ind.tx_mean,
                 )

tx_max = Tasmax(identifier='tx_max',
                long_name='Maximum daily maximum temperature',
                units='K',
                standard_name='air_temperature',
                cell_methods='time: maximum within days time: maximum over days',
                description='{freq} maximum of daily maximum temperature.',
                keywords='',
                compute=_ind.tx_max,
                )

tx_min = Tasmax(identifier='tx_min',
                long_name='Minimum daily maximum temperature',
                units='K',
                standard_name='air_temperature',
                cell_methods='time: maximum within days time: minimum over days',
                description='{freq} minimum of daily maximum temperature.',
                keywords='',
                compute=_ind.tx_min,
                )

tn_mean = Tasmin(identifier='tn_mean',
                 long_name='Mean daily minimum temperature',
                 units='K',
                 standard_name='air_temperature',
                 cell_methods='time: minimum within days time: mean over days',
                 description='{freq} mean of daily minimum temperature.',
                 keywords='',
                 compute=_ind.tn_mean,
                 )

tn_max = Tasmin(identifier='tn_max',
                long_name='Maximum daily minimum temperature',
                units='K',
                standard_name='air_temperature',
                cell_methods='time: minimum within days time: maximum over days',
                description='{freq} maximum of daily minimum temperature.',
                keywords='',
                compute=_ind.tn_max,
                )

tn_min = Tasmin(identifier='tn_min',
                long_name='Minimum daily minimum temperature',
                units='K',
                standard_name='air_temperature',
                cell_methods='time: minimum within days time: minimum over days',
                description='{freq} minimum of daily minimum temperature.',
                keywords='',
                compute=_ind.tn_min,
                )

daily_temperature_range = TasminTasmax(identifier='dtr',
                                       long_name='Mean Diurnal Temperature Range',
                                       units='K',
                                       standard_name='air_temperature',
                                       cell_methods='time range within days time: mean over days',
                                       description='{freq} mean diurnal temparature range',
                                       compute=_ind.daily_temperature_range,
                                       )

daily_temperature_range_variability = TasminTasmax(identifier='dtrvar',
                                                   long_name='Mean Diurnal Temperature Range Variability',
                                                   units='K',
                                                   standard_name='air_temperature',
                                                   cell_methods='time range within days time: difference '
                                                                'over days time: mean over days',
                                                   description='{freq} mean diurnal temparature range variability ('
                                                               'defined as the average day-to-day variation '
                                                               'in daily temperature range '
                                                               'for the given time period)',
                                                   compute=_ind.daily_temperature_range_variability,
                                                   )

extreme_temperature_range = TasminTasmax(identifier='etr',
                                         long_name='Intra-period Extreme Temperature Range',
                                         units='K',
                                         standard_name='air_temperature',
                                         description='{freq} range between the maximum of daily max temperature '
                                                     '(tx_max) and the minimum of daily min temperature (tn_min)',
                                         compute=_ind.extreme_temperature_range,
                                         )

cold_spell_duration = Tasmin(identifier='csdi_{window}',
                             standard_name='cold_spell_duration_index',
                             long_name='Cold Spell Duration Index, count of days with at '
                                       'least {window} consecutive days when Tmin < 10th percentile',
                             descrition='{freq} number of days with at least {window} consecutive days'
                                        ' where the daily minimum temperature is below the 10th '
                                        'percentile. The 10th percentile should be computed for '
                                        'a 5-day window centred on each calendar day in the  1961-1990 period',
                             units='days',
                             compute=_ind.cold_spell_duration_index,
                             )

cold_spell_index = Tas(identifier='csi_{thresh}',
                       standard_name='cold_spell_index',
                       long_name='cold spell index',
                       units='days',
                       description='{freq} number of days that are part of a cold spell, defined as {window} '
                                   'or more consecutive days with mean daily '
                                   'temperature below  {thresh}°C',
                       compute=_ind.cold_spell_index,
                       )

daily_freezethaw_cycles = TasminTasmax(identifier='dlyfrzthw',
                                       standard_name='daily_freezethaw_cycles',
                                       long_name='daily freezethaw cycles',
                                       description='{freq} number of days with a diurnal freeze-thaw cycle '
                                                   ': Tmax > 0℃ and Tmin < 0℃',
                                       compute=_ind.daily_freezethaw_cycles,
                                       units='days'
                                       )

cooling_degree_days = Tas(identifier='cddcold_{thresh}',
                          long_name='Cooling Degree Days (Tmean > {thresh}C)',
                          standard_name='integral_of_air_temperature_excess_wrt_time',
                          units='K days',
                          description='{freq} cooling degree days above {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=_ind.cooling_degree_days,
                          )

heating_degree_days = Tas(identifier='hddheat_{thresh}',
                          long_name='Heating Degree Days (Tmean < {thresh}C)',
                          standard_name='integral_of_air_temperature_deficit_wrt_time',
                          units='K days',
                          description='{freq} heating degree days below {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=_ind.heating_degree_days,
                          )

growing_degree_days = Tas(identifier='gddgrow_{thresh}',
                          standard_name='integral_of_air_temperature_excess_wrt_time',
                          long_name='growing degree days above {thresh}',
                          units='K days',
                          description='{freq} growing degree days above {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=_ind.growing_degree_days,
                          )

frost_days = Tasmin(identifier='frost_days',
                    long_name='Number of Frost Days (Tmin < 0C)',
                    standard_name='days_with_air_temperature_below_threshold',
                    units='days',
                    cell_methods='time: minimum within days time: sum over days',
                    description='{freq} number of days with minimum daily '
                                'temperature below 0°C',
                    compute=_ind.frost_days,
                    )

ice_days = Tasmax(identifier='ice_days',
                  long_name='Number of Ice Days (Tmax < 0C)',
                  standard_name='days_with_air_temperature_below_threshold',
                  units='days',
                  cell_methods='time: maximum within days time: sum over days',
                  description='{freq} number of days with maximum daily '
                              'temperature below 0°C',
                  compute=_ind.ice_days,
                  )

consecutive_frost_days = Tasmin(identifier='consecutive_frost_days',
                                standard_name='spell_length_of_days_with_air_temperature_below_threshold',
                                long_name='Maximum number of consecutive days with Tmin < 0C',
                                units='days',
                                description='{freq} maximum number of consecutive days with '
                                            'minimum daily temperature below 0°C',
                                cell_methods='time: min within days time: maximum over days',
                                compute=_ind.consecutive_frost_days,
                                )

growing_season_length = Tas(identifier='gsl_{thresh}',
                            standard_name='growing_season_length',
                            long_name='ETCCDI Growing Season Length (Tmean > {thresh}C)',
                            units='days',
                            description='{freq} number of days between the first occurrence of at least '
                                        'six consecutive days with mean daily temperature over {thresh}℃ and '
                                        'the first occurrence of at least {window} consecutive days with '
                                        'mean daily temperature below {thresh}℃ after July 1st in the northern '
                                        'hemisphere and January 1st in the southern hemisphere',
                            cell_methods='',
                            compute=_ind.growing_season_length,
                            )

tropical_nights = Tasmin(identifier='tr_{thresh}',
                         standard_name='number_of_days_with_air_temperature_above_threshold',
                         cell_methods='time: minimum within days time: sum over days',
                         long_name='Number of Tropical Nights (Tmin > {thresh}C)',
                         description='{freq} number of Tropical Nights : defined as days with minimum daily temperature'
                                     ' above {thresh}℃',
                         compute=_ind.tropical_nights,
                         )
