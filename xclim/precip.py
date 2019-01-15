# -*- coding: utf-8 -*-
from . import indices as _ind
from .utils import Indicator


class Pr(Indicator):
    required_units = 'mm/day'
    context = 'hydro'


class PrTas(Indicator):
    required_units = ('mm/day', 'K')
    context = 'hydro'


rain_on_frozen_ground_days = PrTas(identifier='rain_frzgr',
                                   units='days',
                                   standard_name='number_of_days_with_lwe_thickness_of_'
                                                 'precipitation_amount_above_threshold',
                                   long_name='Number of rain on frozen ground days',
                                   description="{freq} number of days with rain above {thresh} "
                                               "after a series of seven days "
                                               "with average daily temperature below 0℃. "
                                               "Precipitation is assumed to be rain when the"
                                               "daily average temperature is above 0℃.",
                                   cell_methods='',
                                   compute=_ind.rain_on_frozen_ground_days,
                                   )

max_1day_precipitation_amount = Pr(identifier='rx1day',
                                   units='mm/day',
                                   standard_name='lwe_thickness_of_precipitation_amount',
                                   long_name='maximum 1-day total precipitation',
                                   description="{freq} maximum 1-day total precipitation",
                                   cellmethods='time: sum within days time: maximum over days',
                                   compute=_ind.max_1day_precipitation_amount,
                                   )

max_n_day_precipitation_amount = Pr(identifier='rx{window}day',
                                    units='mm',
                                    standard_name='lwe_thickness_of_precipitation_amount',
                                    long_name='maximum {window}-day total precipitation',
                                    description="{freq} maximum {window}-day total precipitation",
                                    cellmethods='time: sum within days time: maximum over days',
                                    compute=_ind.max_n_day_precipitation_amount,
                                    )

wetdays = Pr(identifier='r{thresh}mm',
             units='days',
             standard_name='number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold',
             long_name='Number of Wet Days (precip >= {thresh} mm)',
             description='{freq} number of days with daily precipitation over {thresh} mm',
             cell_methods='time: sum within days time: sum over days',
             compute=_ind.wetdays,
             )

maximum_consecutive_wet_days = Pr(identifier='cwd',
                                  units='days',
                                  standard_name='number_of_days_with_lwe_thickness_of_'
                                                'precipitation_amount_at_or_above_threshold',
                                  long_name='Maximum consecutive wet days (Precip >= {thresh}mm)',
                                  description='{freq} maximum number of days with daily '
                                              'precipitation over {thresh} mm',
                                  cell_methods='time: sum within days time: sum over days',
                                  compute=_ind.maximum_consecutive_wet_days,
                                  )

maximum_consecutive_dry_days = Pr(identifier='cdd',
                                  units='days',
                                  standard_name='number_of_days_with_lwe_thickness_of_'
                                                'precipitation_amount_below_threshold',
                                  long_name='Maximum consecutive dry days (Precip < {thresh}mm)',
                                  description='{freq} maximum number of days with daily '
                                              'precipitation below {thresh} mm',
                                  cell_methods='time: sum within days time: sum over days',
                                  compute=_ind.maximum_consecutive_dry_days,
                                  )

daily_pr_intensity = Pr(identifier='sdii',
                        units='mm/day',
                        standard_name='lwe_thickness_of_precipitation_amount',
                        long_name='Average precipitation during Wet Days (SDII)',
                        description="{freq} Simple Daily Intensity Index (SDII) : {freq} average precipitation "
                                    "for days with daily precipitation over {thresh} mm",
                        cell_methods='',
                        compute=_ind.daily_pr_intensity,
                        )

precip_accumulation = Pr(identifier='prcptot',
                         units='mm',
                         standard_name='lwe_thickness_of_precipitation_amount',
                         long_name='Total precipitation',
                         description='{freq} total precipitation',
                         cell_methods='time: sum within days time: sum over days',
                         compute=_ind.precip_accumulation
                         )
