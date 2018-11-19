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


heat_wave_frequency = TasminTasmax(identifier='heat_wave_frequency',
                                   units='',
                                   long_name='Number of heat wave events',
                                   standard_name='events',
                                   description="Number of spells meeting criteria for health impacting heat wave.",
                                   keywords="health,",
                                   compute=_ind.heat_wave_frequency)

tmmean = Tas(identifier='tmmean',
             units='K',
             long_name="Mean daily mean temperature",
             standard_name="air_temperature",
             cell_methods='time: mean within days time: mean over days',
             description="{freq} mean of daily mean temperature.",
             keywords='',
             compute=_ind.tg_mean, )

txmean = Tasmax(identifier='txmean',
                required_units='K',
                long_name='Mean daily maximum temperature',
                standard_name='air_temperature',
                cell_methods='time: maximum within days time: mean over days',
                description='{freq} mean of daily maximum temperature.',
                keywords='',
                compute=_ind.tx_mean,
                )

txmax = Tasmax(identifier='txmax',
               required_units='K',
               long_name='Maximum daily maximum temperature',
               standard_name='air_temperature',
               cell_methods='time: maximum within days time: maximum over days',
               description='{freq} maximum of daily maximum temperature.',
               keywords='',
               compute=_ind.tx_max,
               )

txmin = Tasmax(identifier='txmin',
               required_units='K',
               long_name='Minimum daily maximum temperature',
               standard_name='air_temperature',
               cell_methods='time: maximum within days time: minimum over days',
               description='{freq} minimum of daily maximum temperature.',
               keywords='',
               compute=_ind.tx_min,
               )

tnmean = Tasmin(identifier='tnmean',
                required_units='K',
                long_name='Mean daily minimum temperature',
                standard_name='air_temperature',
                cell_methods='time: minimum within days time: mean over days',
                description='{freq} mean of daily minimum temperature.',
                keywords='',
                compute=_ind.tn_mean,
                )

tnmax = Tasmin(identifier='tnmax',
               required_units='K',
               long_name='Maximum daily minimum temperature',
               standard_name='air_temperature',
               cell_methods='time: minimum within days time: maximum over days',
               description='{freq} maximum of daily minimum temperature.',
               keywords='',
               compute=_ind.tn_max,
               )

tnmin = Tasmin(identifier='tnmin',
               required_units='K',
               long_name='Minimum daily minimum temperature',
               standard_name='air_temperature',
               cell_methods='time: minimum within days time: minimum over days',
               description='{freq} minimum of daily minimum temperature.',
               keywords='',
               compute=_ind.tn_min,
               )

cold_spell_duration = Tasmin(identifier='cold_spell_duration',
                             standard_name='cold_spell_duration_index',
                             units='days',
                             compute=_ind.cold_spell_duration_index,
                             )

cooling_dd = Tas(identifier='cooling_dd',
                 long_name='cooling degree days above {thresh}',
                 standard_name='cooling degree days above {thresh}',
                 units='K days',
                 compute=_ind.cooling_degree_days,
                 )

frost_days = Tasmin(identifier='frost_days',
                    long_name='number of days below 0C',
                    standard_name='number of frost days',
                    units='days',
                    compute=_ind.frost_days,
                    )

growing_degree_days = Tas(identifier='growing_degree_days',
                          units='K days',
                          compute=_ind.growing_degree_days,
                          )
