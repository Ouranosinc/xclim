# -*- coding: utf-8 -*-
"""Top-level package for xclim."""
import sys
from functools import partial
from typing import Optional

from xclim import atmos
from xclim import indices
from xclim import land
from xclim import seaIce
from xclim import utils

# from .stats import fit, test

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.14.0"


def build_module(
    name: str,
    objs: dict,
    doc: str = "",
    source: Optional[object] = None,
    mode: str = "ignore",
):
    """Create a module from imported objects.

    Parameters
    ----------
    name : str
      New module name.
    objs : dict
      Dictionary of the objects (or their name) to import into the module,
      keyed by the name they will take in the created module.
    doc : str
      Docstring of the new module.
    source : object
      Module where objects are defined if not explicitly given.
    mode : str
      How to deal with missing objects. Acceptable parameters are 'raise', 'warn', or 'ignore'.


    Returns
    -------
    ModuleType
      A module built from a list of objects' name.

    """
    import types
    import warnings
    import logging

    logging.captureWarnings(capture=True)

    try:
        out = types.ModuleType(name, doc)
    except TypeError:
        msg = "Module '{}' is not properly formatted".format(name)
        raise TypeError(msg)

    for key, obj in objs.items():
        if isinstance(obj, str) and source is not None:
            module_mappings = getattr(source, obj, None)
        else:
            module_mappings = obj

        if module_mappings is None:
            msg = "{} has not been implemented.".format(obj)
            if mode == "ignore":
                logging.info(msg)
            elif mode == "warn":
                warnings.warn(msg)
            elif mode == "raise":
                raise NotImplementedError(msg)
            else:
                msg = "{} is not a valid missing object behaviour".format(mode)
                raise AttributeError(msg)

        else:
            out.__dict__[key] = module_mappings
            try:
                module_mappings.__module__ = name
            except AttributeError:
                msg = "{} is not a function".format(module_mappings)
                raise AttributeError(msg)

    sys.modules[name] = out
    return out


def __build_icclim(mode: str = "warn"):

    #  'SD', 'SD1', 'SD5cm', 'SD50cm',

    # TODO : Complete mappings for ICCLIM indices
    mapping = {
        "TG": indices.tg_mean,
        "TX": indices.tx_mean,
        "TN": indices.tn_mean,
        "TG90p": indices.tg90p,
        "TG10p": indices.tg10p,
        "TGx": indices.tg_max,
        "TGn": indices.tg_min,
        "TX90p": indices.tx90p,
        "TX10p": indices.tx10p,
        "TXx": indices.tx_max,
        "TXn": indices.tx_min,
        "TN90p": indices.tn90p,
        "TN10p": indices.tn10p,
        "TNx": indices.tn_max,
        "TNn": indices.tn_min,
        "CSDI": indices.cold_spell_duration_index,
        "SU": indices.tx_days_above,
        "CSU": indices.maximum_consecutive_tx_days,
        "TR": indices.tropical_nights,
        "GD4": partial(indices.growing_degree_days, thresh="4 degC"),
        "FD": indices.frost_days,
        "CFD": indices.consecutive_frost_days,
        "GSL": indices.growing_season_length,
        "ID": indices.ice_days,
        "HD17": partial(indices.heating_degree_days, thresh="17 degC"),
        "CDD": indices.maximum_consecutive_dry_days,
        "CWD": indices.maximum_consecutive_wet_days,
        "PRCPTOT": indices.precip_accumulation,
        "RR1": indices.wetdays,
        "SDII": partial(indices.daily_pr_intensity, thresh="1 mm/day"),
        "ETR": indices.extreme_temperature_range,
        "DTR": indices.daily_temperature_range,
        "vDTR": indices.daily_temperature_range_variability,
        "R10mm": partial(indices.wetdays, thresh="10 mm/day"),
        "R20mm": partial(indices.wetdays, thresh="20 mm/day"),
        "RX1day": indices.max_1day_precipitation_amount,
        "RX5day": partial(indices.max_n_day_precipitation_amount, window=5),
        "WSDI": indices.warm_spell_duration_index,
        "R75p": partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R95p": partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R99p": partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R75pTOT": partial(indices.fraction_over_precip_thresh, thresh="1 mm/day"),
        "R95pTOT": partial(indices.fraction_over_precip_thresh, thresh="1 mm/day"),
        "R99pTOT": partial(indices.fraction_over_precip_thresh, thresh="1 mm/day"),
        # 'SD': None,
        # 'SD1': None,
        # 'SD5cm': None,
        # 'SD50cm': None,
    }

    mod = build_module(
        "xclim.icclim",
        mapping,
        doc="""
            ==============
            ICCLIM indices
            ==============
            The European Climate Assessment & Dataset project (`ECAD`_) defines
            a set of 26 core climate indices. Those have been made accessible
            directly in xclim through their ECAD name for compatibility. However,
            the methods in this module are only wrappers around the corresponding
            methods of `xclim.indices`. Note that none of the checks performed by
            the `xclim.utils.Indicator` class (like with `xclim.atmos` indicators)
            are performed in this module.

            .. _ECAD: https://www.ecad.eu/
            """,
        mode=mode,
    )
    return mod


def __build_pcc(mode: str = "warn"):
    mapping = dict(
        longest_spell_of_30_degrees_celsius_days=indices.longest_hot_spell,
        # average_length_of_heat_waves=None,
        coldest_minimum_temperature=indices.tn_min,
        cooling_degree_days=indices.cooling_degree_days,
        # corn_heat_units=None,
        date_of_first_fall_frost=utils.wrapped_partial(
            indices.growing_season_end, thresh="0 degC", mid_date="07-15", window=1
        ),
        date_of_last_spring_frost=indices.date_of_last_spring_frost,
        dry_days=indices.dry_days,
        days_above_32_degrees_celsius=utils.wrapped_partial(
            indices.tx_days_above, thresh="32 degC"
        ),
        days_above_34_degrees_celsius=utils.wrapped_partial(
            indices.tx_days_above, thresh="34 degC"
        ),
        freeze_thaw_cycles=utils.wrapped_partial(
            indices.daily_freezethaw_cycles,
            thresh_tasmax="0 degC",
            thresh_tasmin="-1 degC",
        ),
        freezing_degree_days=indices.freezing_degree_days,
        frost_days=indices.frost_days,
    )

    mod = build_module(
        "xclim.pcc",
        mapping,
        doc="""
            ==============================
            Prairie Climate Centre indices
            ==============================
            The Prairie Climate Centre (`PCC`_) defines a set of 23 core climate `indices`_.
            They have been made accessible directly in xclim by their climate atlas names.
            However, the methods in this module are only wrappers around the corresponding
            methods of `xclim.indices`. Note that none of the checks performed by
            the `xclim.utils.Indicator` class (like with `xclim.atmos` indicators)
            are performed in this module.

            .. _PCC: http://prairieclimatecentre.ca/
            .. _indices: https://climateatlas.ca/variables
            """,
        mode=mode,
    )
    return mod


ICCLIM = __build_icclim("ignore")
PCC = __build_pcc("ignore")
