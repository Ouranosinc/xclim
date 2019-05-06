# -*- coding: utf-8 -*-
import abc

from xclim import indices
from xclim import checks
from xclim.utils import Indicator, Indicator2D

__all__ = ['tn_days_below', 'tx_days_above', 'tx_tn_days_above',
           'heat_wave_frequency', 'heat_wave_max_length', 'heat_wave_index', 'tg_mean', 'tg10p', 'tg90p', 'tn_min',
           'tn_max', 'tn_mean', 'tn10p', 'tn90p', 'tx_min', 'tx_max', 'tx_mean', 'tx10p', 'tx90p',
           'daily_temperature_range', 'daily_temperature_range_variability', 'extreme_temperature_range',
           'cold_spell_duration_index', 'cold_spell_days', 'daily_freezethaw_cycles', 'cooling_degree_days',
           'heating_degree_days', 'growing_degree_days', 'freshet_start', 'frost_days', 'ice_days',
           'consecutive_frost_days', 'growing_season_length', 'tropical_nights']


# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


class Tas(Indicator):
    """Class for univariate indices using mean daily temperature as the input."""

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: mean within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""


class Tasmin(Indicator):
    """Class for univariate indices using min daily temperature as the input."""

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: minimum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""


class Tasmax(Indicator):
    """Class for univariate indices using max daily temperature as the input."""

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: maximum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""


class TasminTasmax(Indicator2D):

    def cfprobe(self, dan, dax):
        for da in (dan, dax):
            checks.check_valid(da, 'cell_methods', 'time: maximum within days')
            checks.check_valid(da, 'standard_name', 'air_temperature')

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""


tn_days_below = Tasmin(identifier='tnlt_{thresh}',
                       units='days',
                       standard_name='number_of_days_with_air_temperature_below_threshold',
                       long_name='Number of days with Tmin < {thresh}C',
                       description="{freq} number of days where daily minimum temperature is below {thresh}℃",
                       cell_methods='time: minimum within days time: sum over days',
                       compute=indices.tn_days_below,
                       )

tx_days_above = Tasmax(identifier='txgt_{thresh}',
                       units='days',
                       standard_name='number_of_days_with_air_temperature_above_threshold',
                       long_name='Number of days with Tmax > {thresh}C',
                       description="{freq} number of days where daily maximum temperature exceeds {thresh}℃",
                       cell_methods='time: maximum within days time: sum over days',
                       compute=indices.tx_days_above,
                       )

tx_tn_days_above = TasminTasmax(identifier='txgt_{thresh_tasmax}_tngt_{thresh_tasmin}',
                                units='days',
                                standard_name='number_of_days_with_air_temperature_above_threshold',
                                long_name='Number of days with Tmax > {thresh_tasmax}C and Tmin > {thresh_tasmin}C',
                                description="{freq} number of days where daily maximum temperature exceeds"
                                            " {thresh_tasmax}℃ and minimum temperature exceeds {thresh_tasmin}℃",
                                cell_methods='',
                                compute=indices.tx_tn_days_above,
                                )

heat_wave_frequency = TasminTasmax(identifier='heat_wave_frequency',
                                   units='',
                                   standard_name='heat_wave_events',
                                   long_name='Number of heat wave events (Tmin > {thresh_tasmin}℃'
                                             'and Tmax > {thresh_tasmax}℃ for >= {window} days)',
                                   description="{freq} number of heat wave events over a given period. "
                                               "An event occurs when the minimum and maximum daily "
                                               "temperature both exceeds specific thresholds : "
                                               "(Tmin > {thresh_tasmin}℃ and Tmax > {thresh_tasmax}℃) "
                                               "over a minimum number of days ({window}).",
                                   cell_methods='',
                                   keywords="health,",
                                   compute=indices.heat_wave_frequency,
                                   )

heat_wave_max_length = TasminTasmax(identifier='heat_wave_max_length',
                                    units='days',
                                    standard_name='spell_length_of_days_with_air_temperature_above_threshold',
                                    long_name='Maximum length of heat wave events (Tmin > {thresh_tasmin}℃'
                                              'and Tmax > {thresh_tasmax}℃ for >= {window} days)',
                                    description="{freq} maximum length of heat wave events occuring in a given period."
                                                "An event occurs when the minimum and maximum daily "
                                                "temperature both exceeds specific thresholds "
                                                "(Tmin > {thresh_tasmin}℃ and Tmax > {thresh_tasmax}℃) over "
                                                "a minimum number of days ({window}).",
                                    cell_methods='',
                                    keywords="health,",
                                    compute=indices.heat_wave_max_length,
                                    )

heat_wave_index = Tasmax(identifier='hwi_{thresh}',
                         units='days',
                         standard_name='heat_wave_index',
                         long_name='Number of days that are part of a heatwave',
                         description='{freq} number of days that are part of a heatwave, '
                                     'defined as five or more consecutive days over {thresh}℃',
                         cell_methods='',
                         compute=indices.heat_wave_index,
                         )

tg_mean = Tas(identifier='tg_mean',
              units='K',
              standard_name="air_temperature",
              long_name="Mean daily mean temperature",
              description="{freq} mean of daily mean temperature.",
              cell_methods='time: mean within days time: mean over days',
              compute=indices.tg_mean, )

tx_mean = Tasmax(identifier='tx_mean',
                 units='K',
                 standard_name='air_temperature',
                 long_name='Mean daily maximum temperature',
                 description='{freq} mean of daily maximum temperature.',
                 cell_methods='time: maximum within days time: mean over days',
                 compute=indices.tx_mean,
                 )

tx_max = Tasmax(identifier='tx_max',
                units='K',
                standard_name='air_temperature',
                long_name='Maximum daily maximum temperature',
                description='{freq} maximum of daily maximum temperature.',
                cell_methods='time: maximum within days time: maximum over days',
                compute=indices.tx_max,
                )

tx_min = Tasmax(identifier='tx_min',
                units='K',
                standard_name='air_temperature',
                long_name='Minimum daily maximum temperature',
                description='{freq} minimum of daily maximum temperature.',
                cell_methods='time: maximum within days time: minimum over days',
                compute=indices.tx_min,
                )

tn_mean = Tasmin(identifier='tn_mean',
                 units='K',
                 standard_name='air_temperature',
                 long_name='Mean daily minimum temperature',
                 description='{freq} mean of daily minimum temperature.',
                 cell_methods='time: minimum within days time: mean over days',
                 compute=indices.tn_mean,
                 )

tn_max = Tasmin(identifier='tn_max',
                units='K',
                standard_name='air_temperature',
                long_name='Maximum daily minimum temperature',
                description='{freq} maximum of daily minimum temperature.',
                cell_methods='time: minimum within days time: maximum over days',
                compute=indices.tn_max,
                )

tn_min = Tasmin(identifier='tn_min',
                units='K',
                standard_name='air_temperature',
                long_name='Minimum daily minimum temperature',
                description='{freq} minimum of daily minimum temperature.',
                cell_methods='time: minimum within days time: minimum over days',
                compute=indices.tn_min,
                )

daily_temperature_range = TasminTasmax(identifier='dtr',
                                       units='K',
                                       standard_name='air_temperature',
                                       long_name='Mean Diurnal Temperature Range',
                                       description='{freq} mean diurnal temperature range',
                                       cell_methods='time range within days time: mean over days',
                                       compute=indices.daily_temperature_range,
                                       )

daily_temperature_range_variability = TasminTasmax(identifier='dtrvar',
                                                   units='K',
                                                   standard_name='air_temperature',
                                                   long_name='Mean Diurnal Temperature Range Variability',
                                                   description='{freq} mean diurnal temparature range variability ('
                                                               'defined as the average day-to-day variation '
                                                               'in daily temperature range '
                                                               'for the given time period)',
                                                   cell_methods='time range within days time: difference '
                                                                'over days time: mean over days',
                                                   compute=indices.daily_temperature_range_variability,
                                                   )

extreme_temperature_range = TasminTasmax(identifier='etr',
                                         units='K',
                                         standard_name='air_temperature',
                                         long_name='Intra-period Extreme Temperature Range',

                                         description='{freq} range between the maximum of daily max temperature '
                                                     '(tx_max) and the minimum of daily min temperature (tn_min)',
                                         compute=indices.extreme_temperature_range,
                                         )

cold_spell_duration_index = Tasmin(identifier='csdi_{window}',
                                   units='days',
                                   standard_name='cold_spell_duration_index',
                                   long_name='Cold Spell Duration Index, count of days with at '
                                             'least {window} consecutive days when Tmin < 10th percentile',
                                   description='{freq} number of days with at least {window} consecutive days'
                                              ' where the daily minimum temperature is below the 10th '
                                              'percentile. The 10th percentile should be computed for '
                                              'a 5-day window centred on each calendar day in the  1961-1990 period',
                                   cell_methods='',
                                   compute=indices.cold_spell_duration_index,
                                   )

cold_spell_days = Tas(identifier='csi_{thresh}',
                      units='days',
                      standard_name='cold_spell_days',
                      long_name='cold spell index',
                      description='{freq} number of days that are part of a cold spell, defined as {window} '
                                  'or more consecutive days with mean daily '
                                  'temperature below  {thresh}°C',
                      cell_methods='',
                      compute=indices.cold_spell_days,
                      )

daily_freezethaw_cycles = TasminTasmax(identifier='dlyfrzthw',
                                       units='days',
                                       standard_name='daily_freezethaw_cycles',
                                       long_name='daily freezethaw cycles',
                                       description='{freq} number of days with a diurnal freeze-thaw cycle '
                                                   ': Tmax > 0℃ and Tmin < 0℃',
                                       cell_methods='',
                                       compute=indices.daily_freezethaw_cycles,
                                       )

cooling_degree_days = Tas(identifier='cddcold_{thresh}',
                          units='K days',
                          standard_name='integral_of_air_temperature_excess_wrt_time',
                          long_name='Cooling Degree Days (Tmean > {thresh}C)',
                          description='{freq} cooling degree days above {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=indices.cooling_degree_days,
                          )

heating_degree_days = Tas(identifier='hddheat_{thresh}',
                          units='K days',
                          standard_name='integral_of_air_temperature_deficit_wrt_time',
                          long_name='Heating Degree Days (Tmean < {thresh}C)',
                          description='{freq} heating degree days below {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=indices.heating_degree_days,
                          )

growing_degree_days = Tas(identifier='gddgrow_{thresh}',
                          units='K days',
                          standard_name='integral_of_air_temperature_excess_wrt_time',
                          long_name='growing degree days above {thresh}',
                          description='{freq} growing degree days above {thresh}°C',
                          cell_methods='time: mean within days time: sum over days',
                          compute=indices.growing_degree_days,
                          )

freshet_start = Tas(identifier='freshet_start',
                    units='',
                    standard_name='day_of_year',
                    long_name="Day of year of spring freshet start",
                    description="Day of year of spring freshet start, defined as the first day a temperature "
                                "threshold of {thresh} is exceeded for at least {window} days.",
                    compute=indices.freshet_start)

frost_days = Tasmin(identifier='frost_days',
                    units='days',
                    standard_name='days_with_air_temperature_below_threshold',
                    long_name='Number of Frost Days (Tmin < 0C)',
                    description='{freq} number of days with minimum daily '
                                'temperature below 0°C',
                    cell_methods='time: minimum within days time: sum over days',
                    compute=indices.frost_days,
                    )

ice_days = Tasmax(identifier='ice_days',
                  standard_name='days_with_air_temperature_below_threshold',
                  units='days',
                  long_name='Number of Ice Days (Tmax < 0C)',
                  description='{freq} number of days with maximum daily '
                              'temperature below 0°C',
                  cell_methods='time: maximum within days time: sum over days',
                  compute=indices.ice_days,
                  )

consecutive_frost_days = Tasmin(identifier='consecutive_frost_days',
                                units='days',
                                standard_name='spell_length_of_days_with_air_temperature_below_threshold',
                                long_name='Maximum number of consecutive days with Tmin < 0C',
                                description='{freq} maximum number of consecutive days with '
                                            'minimum daily temperature below 0°C',
                                cell_methods='time: min within days time: maximum over days',
                                compute=indices.consecutive_frost_days,
                                )

growing_season_length = Tas(identifier='gsl_{thresh}',
                            units='days',
                            standard_name='growing_season_length',
                            long_name='ETCCDI Growing Season Length (Tmean > {thresh}C)',
                            description='{freq} number of days between the first occurrence of at least '
                                        'six consecutive days with mean daily temperature over {thresh}℃ and '
                                        'the first occurrence of at least {window} consecutive days with '
                                        'mean daily temperature below {thresh}℃ after July 1st in the northern '
                                        'hemisphere and January 1st in the southern hemisphere',
                            cell_methods='',
                            compute=indices.growing_season_length,
                            )

tropical_nights = Tasmin(identifier='tr_{thresh}',
                         units='days',
                         standard_name='number_of_days_with_air_temperature_above_threshold',
                         long_name='Number of Tropical Nights (Tmin > {thresh}C)',
                         description='{freq} number of Tropical Nights : defined as days with minimum daily temperature'
                                     ' above {thresh}℃',
                         cell_methods='time: minimum within days time: sum over days',
                         compute=indices.tropical_nights,
                         )

tg90p = Tas(identifier='tg90p',
            units='days',
            standard_name='days_with_air_temperature_above_threshold',
            long_name='Number of days when Tmean > 90th percentile',
            description='{freq} number of days with mean daily temperature above the 90th percentile.'
                        'The 90th percentile is to be computed for a 5 day window centered on each calendar day '
                        'for a reference period',
            cell_methods='time: mean within days time: sum over days',
            compute=indices.tg90p,
            )

tg10p = Tas(identifier='tg10p',
            units='days',
            standard_name='days_with_air_temperature_below_threshold',
            long_name='Number of days when Tmean < 10th percentile',
            description='{freq} number of days with mean daily temperature below the 10th percentile.'
                        'The 10th percentile is to be computed for a 5 day window centered on each calendar day '
                        'for a reference period',
            cell_methods='time: mean within days time: sum over days',
            compute=indices.tg10p
            )

tx90p = Tasmax(identifier='tx90p',
               units='days',
               standard_name='days_with_air_temperature_above_threshold',
               long_name='Number of days when Tmax > 90th percentile',
               description='{freq} number of days with maximum daily temperature above the 90th percentile.'
                           'The 90th percentile is to be computed for a 5 day window centered on each calendar day '
                           'for a reference period',
               cell_methods='time: maximum within days time: sum over days',
               compute=indices.tx90p,
               )

tx10p = Tasmax(identifier='tx10p',
               units='days',
               standard_name='days_with_air_temperature_below_threshold',
               long_name='Number of days when Tmax < 10th percentile',
               description='{freq} number of days with maximum daily temperature below the 10th percentile.'
                           'The 10th percentile is to be computed for a 5 day window centered on each calendar day '
                           'for a reference period',
               cell_methods='time: maximum within days time: sum over days',
               compute=indices.tx10p
               )

tn90p = Tasmin(identifier='tn90p',
               units='days',
               standard_name='days_with_air_temperature_above_threshold',
               long_name='Number of days when Tmin > 90th percentile',
               description='{freq} number of days with minimum daily temperature above the 90th percentile.'
                           'The 90th percentile is to be computed for a 5 day window centered on each calendar day '
                           'for a reference period',
               cell_methods='time: minimum within days time: sum over days',
               compute=indices.tn90p,
               )

tn10p = Tasmin(identifier='tn10p',
               units='days',
               standard_name='days_with_air_temperature_below_threshold',
               long_name='Number of days when Tmin < 10th percentile',
               description='{freq} number of days with minimum daily temperature below the 10th percentile.'
                           'The 10th percentile is to be computed for a 5 day window centered on each calendar day '
                           'for a reference period',
               cell_methods='time: minimum within days time: sum over days',
               compute=indices.tn10p
               )
