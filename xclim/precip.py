from . import indices as _ind
from .utils import Indicator


class Pr(Indicator):
    required_units = 'mm/day'
    context = 'hydro'


max_1day_precipitation_amount = Pr(identifier='rx1day',
                                   standard_name='lwe_thickness_of_precipitation_amount',
                                   long_name='maximum 1-day total precipitation',
                                   units='mm/day',
                                   cellmethods='time: sum within days time: maximum over days',
                                   description="{freq} maximum 1-day total precipitation",
                                   compute=_ind.max_1day_precipitation_amount,
                                   )

max_n_day_precipitation_amount = Pr(identifier='rx{window}day',
                                    standard_name='lwe_thickness_of_precipitation_amount',
                                    long_name='maximum {window}-day total precipitation',
                                    units='mm',
                                    cellmethods='time: sum within days time: maximum over days',
                                    description="{freq} maximum {window}-day total precipitation",
                                    compute=_ind.max_n_day_precipitation_amount,
                                    )

wetdays = Pr(identifier='wetdays',
             standard_name='number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold',
             units='days',
             cell_methods='time: sum within days time: sum over days',
             long_name='Number of Wet Days (precip >= {thresh} mm)',
             description='{freq} number of days with daily precipitation over {thresh} mm',
             compute=_ind.wetdays,
             )

maximum_consecutive_wet_days = Pr(identifier='cwd',
                                  standard_name='number_of_days_with_lwe_thickness_of_'
                                                'precipitation_amount_at_or_above_threshold',
                                  units='days',
                                  cell_methods='time: sum within days time: sum over days',
                                  long_name='Maximum consecutive wet days (Precip >= {thresh}mm)',
                                  description='{freq} maximum number of days with daily '
                                              'precipitation over {thresh} mm',
                                  compute=_ind.maximum_consecutive_wet_days,
                                  )

maximum_consecutive_dry_days = Pr(identifier='cdd',
                                  standard_name='number_of_days_with_lwe_thickness_of_'
                                                'precipitation_amount_below_threshold',
                                  units='days',
                                  cell_methods='time: sum within days time: sum over days',
                                  long_name='Maximum consecutive wet days (Precip < {thresh}mm)',
                                  description='{freq} maximum number of days with daily '
                                              'precipitation below {thresh} mm',
                                  compute=_ind.maximum_consecutive_dry_days,
                                  )

daily_pr_intensity = Pr(identifier='sdii',
                        standard_name='lwe_thickness_of_precipitation_amount',
                        long_name='Average precipitation during Wet Days (SDII)',
                        units='mm/day',
                        description="{freq} Simple Daily Intensity Index (SDII) : {freq} average precipitation "
                                    "for days with daily precipitation over {thresh} mm",
                        cell_methods='',
                        compute=_ind.daily_pr_intensity,
                        )

precip_accumulation = Pr(identifier='prcptot',
                         standard_name='lwe_thickness_of_precipitation_amount',
                         long_name='Total precipitation',
                         units='mm',
                         description='{freq} total precipitation',
                         cell_methods='time: sum within days time: sum over days',
                         compute=_ind.precip_accumulation
                         )
