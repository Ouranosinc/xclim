# -*- coding: utf-8 -*-

"""Top-level package for xclim."""

__author__ = """Travis Logan"""
__email__ = 'logan.travis@ouranos.ca'
__version__ = '0.2-alpha'


from .checks import *
from .hydro import BFI
from . import indices
from .stats import fit, test

def build_module(name, keys, source, doc='', mode='ignore'):
    """Create a module from imported objects.

    Parameters
    ----------
    name : str
      New module name.
    keys : sequence
      Iterable sequence of the names of the objects to import into the module.
    source : str
      Module where objects are defined.
    doc : str
      Docstring of the new module.
    mode : {'raise', 'warn', 'ignore'}
      How to deal with missing objects.


    Returns
    -------
    ModuleType
      A module built from a list of objects' name.

    """
    import types
    import warnings

    out = types.ModuleType(name, doc)

    for key, name in keys.items():
        f = getattr(source, name, None)
        if f is None:
            msg = "{} has not been implemented.".format(k)
            if mode == 'raise':
                raise NotImplementedError(msg)
            elif mode == 'warn':
                warnings.warn(msg)

        else:
            out.__dict__[key] = f

    return out

def __build_icclim(mode='warn'):
    keys = ['TG', 'TX', 'TN', 'TXx', 'TXn', 'TNx', 'TNn', 'SU', 'TR', 'CSU', 'GD4', 'FD', 'CFD',
            'ID', 'HD17', 'CDD', 'CWD', 'PRCPTOT', 'RR1', 'SDII', 'R10mm', 'R20mm', 'RX1day', 'RX5day',
            'SD', 'SD1', 'SD5cm', 'SD50cm', 'DTR', 'ETR', 'vDTR', 'TG10p', 'TX10p', 'TN10p', 'TG90p',
            'TX90p', 'TN90p', 'WSDI', 'CSDI', 'R75p', 'R75pTOT', 'R95p', 'R95pTOT', 'R99p', 'R99pTOT']

    mapping = {'CFD': 'consecutive_frost_days'}

    return build_module('icclim', mapping, indices, doc="""ICCLIM indices""", mode=mode)

icclim = __build_icclim('ignore')
