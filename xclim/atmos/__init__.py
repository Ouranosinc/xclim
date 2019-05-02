"""
Atmospheric indicator calculation instances
===========================================

While the `indices` module stores the computing functions, this module defines Indicator classes and instances that
include a number of functionalities, such as input validation, unit conversion, output meta-data handling,
and missing value masking.

The concept followed here is to define Indicator subclasses for each input variable, then create instances
for each indicator.

"""
from ._temperature import tn_days_below
from ._temperature import tx_days_above
from ._temperature import tx_tn_days_above
from ._temperature import heat_wave_frequency
from ._temperature import heat_wave_max_length
from ._temperature import heat_wave_index

from ._temperature import tn_mean
from ._temperature import tg_mean
from ._temperature import tx_mean
from ._temperature import tn_max
from ._temperature import tx_max
from ._temperature import tn_min
from ._temperature import tx_min
from ._temperature import tn90p
from ._temperature import tn10p
from ._temperature import tg90p
from ._temperature import tg10p
from ._temperature import tx90p
from ._temperature import tx10p

from ._temperature import daily_temperature_range
from ._temperature import daily_temperature_range_variability
from ._temperature import extreme_temperature_range
from ._temperature import cold_spell_duration_index
from ._temperature import cold_spell_days
from ._temperature import daily_freezethaw_cycles

from ._temperature import cooling_degree_days
from ._temperature import heating_degree_days
from ._temperature import growing_degree_days
from ._temperature import freshet_start
from ._temperature import frost_days
from ._temperature import ice_days
from ._temperature import consecutive_frost_days
from ._temperature import growing_season_length
from ._temperature import tropical_nights

from ._precip import precip_accumulation
from ._precip import daily_pr_intensity
from ._precip import max_1day_precipitation_amount
from ._precip import max_n_day_precipitation_amount

from ._precip import wetdays
from ._precip import maximum_consecutive_wet_days
from ._precip import maximum_consecutive_dry_days

from ._precip import rain_on_frozen_ground_days

