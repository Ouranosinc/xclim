# -*- coding: utf-8 -*-

"""Top-level package for xclim."""

__author__ = """Travis Logan"""
__email__ = 'logan.travis@ouranos.ca'
__version__ = '0.6-alpha'

from . import indices
# from .stats import fit, test
from functools import partial


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

    try:
        out = types.ModuleType(name, doc)
    except TypeError:
        msg = "Module '{}' is not properly formatted".format(name)
        raise TypeError(msg)

    for key, obj in objs.items():
        if isinstance(obj, str) and source is not None:
            f = getattr(source, obj, None)
        else:
            f = obj

        if f is None:
            msg = "{} has not been implemented.".format(obj)
            if mode == 'raise':
                raise NotImplementedError(msg)
            elif mode == 'warn':
                warnings.warn(msg)
                raise Warning(msg)

        else:
            out.__dict__[key] = f
            try:
                f.__module__ = 'xclim.'+name
            except AttributeError:
                msg = "{} is not a function".format(f)
                raise AttributeError(msg)

    return out


def __build_icclim(mode='warn'):
    import sys
    #  ['TG', 'TX', 'TN', 'TXx', 'TXn', 'TNx', 'TNn', 'SU', 'TR', 'CSU', 'GD4', 'FD', 'CFD',
    #   'ID', 'HD17', 'CDD', 'CWD', 'PRCPTOT', 'RR1', 'SDII', 'R10mm', 'R20mm', 'RX1day', 'RX5day',
    #   'SD', 'SD1', 'SD5cm', 'SD50cm', 'DTR', 'ETR', 'vDTR', 'TG10p', 'TX10p', 'TN10p', 'TG90p',
    #   'TX90p', 'TN90p', 'WSDI', 'CSDI', 'R75p', 'R75pTOT', 'R95p', 'R95pTOT', 'R99p', 'R99pTOT']

    # Use partials to specify default value ?
    # TODO : Complete mappings for ICCLIM indices
    mapping = {'TG': indices.tg_mean,
               'TX': indices.tx_mean,
               'TN': indices.tn_mean,
               'TXx': indices.tx_max,
               'TXn': indices.tx_min,
               'TNx': indices.tn_max,
               'TNn': indices.tn_min,
               'SU': indices.summer_days,
               'TR': indices.tropical_nights,
               'GD4': partial(indices.growing_degree_days, thresh=4),
               'FD': indices.frost_days,
               'CFD': indices.consecutive_frost_days,
               'GSL': indices.growing_season_length,
               'ID': indices.ice_days,
               'HD17': indices.heating_degree_days,
               # 'CDD': indices.consecutive_dry_days,
               'CWD': indices.consecutive_wet_days,
               # 'PRCPTOT': indices.prec_total,
               # 'RR1': indices.wet_days,
               'DTR': indices.daily_temperature_range,
               }

    mod = build_module('icclim', mapping, doc="""ICCLIM indices""", mode=mode)
    sys.modules['xclim.icclim'] = mod
    return mod


icclim = __build_icclim('ignore')
