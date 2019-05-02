# -*- coding: utf-8 -*-

"""Top-level package for xclim."""

from functools import partial

import xclim.indices
import xclim.indices.simple
import xclim.indices.threshold
from xclim.indices import indices
import sys

# from .stats import fit, test

__author__ = """Travis Logan"""
__email__ = 'logan.travis@ouranos.ca'
__version__ = '0.8-beta'


def build_module(name, objs, doc='', source=None, mode='ignore'):
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
    source : Module object
      Module where objects are defined if not explicitly given.
    mode : {'raise', 'warn', 'ignore'}
      How to deal with missing objects.


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
            if mode == 'raise':
                raise NotImplementedError(msg)
            elif mode == 'warn':
                warnings.warn(msg)
            else:
                logging.info(msg)

        else:
            out.__dict__[key] = module_mappings
            try:
                module_mappings.__module__ = name
            except AttributeError:
                msg = "{} is not a function".format(module_mappings)
                raise AttributeError(msg)

    sys.modules[name] = out
    return out


def __build_icclim(mode='warn'):

    #  ['TG', 'TX', 'TN', 'TXx', 'TXn', 'TNx', 'TNn', 'SU', 'TR', 'CSU', 'GD4', 'FD', 'CFD', 'GSL',
    #   'ID', 'HD17', 'CDD', 'CWD', 'PRCPTOT', 'RR1', 'SDII', 'R10mm', 'R20mm', 'RX1day', 'RX5day',
    #   'SD', 'SD1', 'SD5cm', 'SD50cm', 'DTR', 'ETR', 'vDTR', 'TG10p', 'TX10p', 'TN10p', 'TG90p',
    #   'TX90p', 'TN90p', 'WSDI', 'CSDI', 'R75p', 'R75pTOT', 'R95p', 'R95pTOT', 'R99p', 'R99pTOT']

    # Use partials to specify default value ?
    # TODO : Complete mappings for ICCLIM indices
    mapping = {'TG': xclim.indices.simple.tg_mean,
               'TX': xclim.indices.simple.tx_mean,
               'TN': xclim.indices.simple.tn_mean,
               'TG90p': indices.tg90p,
               'TG10p': indices.tg10p,
               'TGx': xclim.indices.simple.tg_max,
               'TGn': xclim.indices.simple.tg_min,
               'TX90p': indices.tx90p,
               'TX10p': indices.tx10p,
               'TXx': xclim.indices.simple.tx_max,
               'TXn': xclim.indices.simple.tx_min,
               'TN90p': indices.tn90p,
               'TN10p': indices.tn10p,
               'TNx': xclim.indices.simple.tn_max,
               'TNn': xclim.indices.simple.tn_min,
               'SU': xclim.indices.threshold.tx_days_above,
               'TR': indices.tropical_nights,
               # 'CSU': None,
               'GD4': partial(xclim.indices.threshold.growing_degree_days, thresh=4),
               'FD': indices.frost_days,
               'CFD': indices.consecutive_frost_days,
               'GSL': xclim.indices.threshold.growing_season_length,
               'ID': indices.ice_days,
               'HD17': xclim.indices.threshold.heating_degree_days,
               'CDD': indices.maximum_consecutive_dry_days,
               'CWD': xclim.indices.threshold.maximum_consecutive_wet_days,
               'PRCPTOT': indices.precip_accumulation,
               'RR1': xclim.indices.threshold.wetdays,
               # 'SDII': None,
               'ETR': indices.extreme_temperature_range,
               'DTR': indices.daily_temperature_range,
               'vDTR': indices.daily_temperature_range_variability,
               # 'R10mm': None,
               # 'R20mm': None,
               # 'RX1day': None,
               # 'RX5day': None,
               # 'R75p': None,
               # 'R75pTOT':None,
               # 'R95p': None,
               # 'R95pTOT': None,
               # 'R99p': None,
               # 'R99pTOT': None,
               # 'SD': None,
               # 'SD1': None,
               # 'SD5cm': None,
               # 'SD50cm': None,
               }

    mod = build_module('xclim.icclim', mapping, doc="""ICCLIM indices""", mode=mode)
    return mod


icclim = __build_icclim('ignore')
