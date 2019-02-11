# -*- coding: utf-8 -*-
"""
Indicator calculation instances
===============================

While the `indices` module stores the computing functions, this module defines Indicator classes and instances that
include a number of functionalities, such as input validation, unit conversion, output meta-data handling,
and missing value masking.

The concept followed here is to define UnivariateIndicator subclasses for each input variable, then create instances
for each indicator.

"""

from . import checks
from . import indices as _ind
from .utils import UnivariateIndicator

# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


class Tas(UnivariateIndicator):
    """Class for univariate indices using mean daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: mean within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')



class Tasmin(UnivariateIndicator):
    """Class for univariate indices using min daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: minimum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


class Tasmax(UnivariateIndicator):
    """Class for univariate indices using max daily temperature as the input."""
    required_units = 'K'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: maximum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


tmmean = Tas(identifier='tmmean',
             units='K',
             long_name="Mean daily mean temperature",
             standard_name="air_temperature",
             description="{freq} mean of daily mean temperature.",
             keywords='',
             compute=_ind.tg_mean,)


tx_max = Tasmax(identifier='tx_max',
                required_units='K',
                long_name='Maximum temperature',
                standard_name='tasmax',
                description='Maximum daily maximum temperature.',
                keywords='',
                compute=_ind.tx_max,
                )


cold_spell_duration = Tasmin(identifier='cold_spell_duration',
                             standard_name='cold_spell_duration_index',
                             units='days',
                             compute=_ind.cold_spell_duration_index,
                             )

tx_min = Tasmax(identifier='tx_min',
                long_name='Minimum maximum temperature',
                standard_name='tx_min',
                description='Minimum daily maximum temperature over the period',
                cell_methods='time: minimum within {freq}',
                compute=_ind.tx_min,
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
