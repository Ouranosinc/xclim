# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicators module
=================

Indicators are the main tool xclim provides to compute climate indices. In contrast
to the function defined in `xclim.indices`, Indicators add a layer of health checks
and metadata handling. Indicator objects are split into realms : atmos, land and
seaIce.

The module also defines additional virtual modules : `icclim` and `anuclim`. For the moment, they hold indices
instead of indicators, but in the future they'll be converted to indicators.
"""
from functools import wraps
from types import ModuleType
from typing import Callable


def build_module(
    name: str,
    objs: dict,
    doc: str = "",
    source: ModuleType = None,
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
    source : Module object
      Module where objects are defined if not explicitly given.
    mode : {'raise', 'warn', 'ignore'}
      How to deal with missing objects.


    Returns
    -------
    ModuleType
      A module built from a list of objects' name.

    """
    import logging
    import sys
    import types
    import warnings

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
    from xclim import indices
    from xclim.core.utils import wrapped_partial

    #  ['SD', 'SD1', 'SD5cm', 'SD50cm',
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
        "GD4": wrapped_partial(indices.growing_degree_days, thresh="4 degC"),
        "FD": indices.frost_days,
        "CFD": wrapped_partial(indices.maximum_consecutive_frost_days, thresh="0 degC"),
        "GSL": indices.growing_season_length,
        "ID": indices.ice_days,
        "HD17": wrapped_partial(indices.heating_degree_days, thresh="17 degC"),
        "CDD": indices.maximum_consecutive_dry_days,
        "CWD": indices.maximum_consecutive_wet_days,
        "PRCPTOT": indices.precip_accumulation,
        "RR1": indices.wetdays,
        "SDII": wrapped_partial(indices.daily_pr_intensity, thresh="1 mm/day"),
        "ETR": indices.extreme_temperature_range,
        "DTR": indices.daily_temperature_range,
        "vDTR": indices.daily_temperature_range_variability,
        "R10mm": wrapped_partial(indices.wetdays, thresh="10 mm/day"),
        "R20mm": wrapped_partial(indices.wetdays, thresh="20 mm/day"),
        "RX1day": indices.max_1day_precipitation_amount,
        "RX5day": wrapped_partial(indices.max_n_day_precipitation_amount, window=5),
        "WSDI": indices.warm_spell_duration_index,
        "R75p": wrapped_partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R95p": wrapped_partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R99p": wrapped_partial(indices.days_over_precip_thresh, thresh="1 mm/day"),
        "R75pTOT": wrapped_partial(
            indices.fraction_over_precip_thresh, thresh="1 mm/day"
        ),
        "R95pTOT": wrapped_partial(
            indices.fraction_over_precip_thresh, thresh="1 mm/day"
        ),
        "R99pTOT": wrapped_partial(
            indices.fraction_over_precip_thresh, thresh="1 mm/day"
        ),
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
            methods of  `xclim.indices`. Note that none of the checks performed by
            the `xclim.utils.Indicator` class (like with `xclim.atmos` indicators)
            are performed in this module.

            .. _ECAD: https://www.ecad.eu/
            """,
        mode=mode,
    )
    return mod


def ensure_annual(func: Callable) -> Callable:
    """Ensure that supplied frequency keyword denotes annual time step."""

    @wraps(func)
    def _wrapper(*args, **kwargs):
        if "freq" not in kwargs:
            raise ValueError(
                "Frequency must be provided as a keyword argument (freq='Y[S]' or freq='A[S][-month]')"
            )
        freq = kwargs["freq"]
        if freq[0] not in ["Y", "A"]:
            raise ValueError("Frequency must be annual ('Y[S]' or 'A[S][-month]')")
        return func(*args, **kwargs)

    return _wrapper


def __build_anuclim(mode: str = "warn"):
    from xclim import indices
    from xclim.core.utils import wrapped_partial

    mapping = {
        "P1_AnnMeanTemp": ensure_annual(indices.tg_mean),
        "P2_MeanDiurnalRange": ensure_annual(indices.daily_temperature_range),
        "P3_Isothermality": ensure_annual(indices.isothermality),
        "P4_TempSeasonality": indices.temperature_seasonality,
        "P5_MaxTempWarmestPeriod": ensure_annual(indices.tx_max),
        "P6_MinTempColdestPeriod": ensure_annual(indices.tn_min),
        "P7_TempAnnualRange": ensure_annual(indices.extreme_temperature_range),
        "P8_MeanTempWettestQuarter": ensure_annual(
            wrapped_partial(indices.tg_mean_wetdry_quarter, op="wettest")
        ),
        "P9_MeanTempDriestQuarter": ensure_annual(
            wrapped_partial(indices.tg_mean_wetdry_quarter, op="driest")
        ),
        "P10_MeanTempWarmestQuarter": ensure_annual(
            wrapped_partial(indices.tg_mean_warmcold_quarter, op="warmest")
        ),
        "P11_MeanTempColdestQuarter": ensure_annual(
            wrapped_partial(indices.tg_mean_warmcold_quarter, op="coldest")
        ),
        "P12_AnnualPrecip": ensure_annual(indices.prcptot),
        "P13_PrecipWettestPeriod": ensure_annual(
            wrapped_partial(indices.prcptot_wetdry_period, op="wettest")
        ),
        "P14_PrecipDriestPeriod": ensure_annual(
            wrapped_partial(indices.prcptot_wetdry_period, op="driest")
        ),
        "P15_PrecipSeasonality": indices.precip_seasonality,
        "P16_PrecipWettestQuarter": ensure_annual(
            wrapped_partial(indices.prcptot_wetdry_quarter, op="wettest")
        ),
        "P17_PrecipDriestQuarter": ensure_annual(
            wrapped_partial(indices.prcptot_wetdry_quarter, op="driest")
        ),
        "P18_PrecipWarmestQuarter": ensure_annual(
            wrapped_partial(indices.prcptot_warmcold_quarter, op="warmest")
        ),
        "P19_PrecipColdestQuarter": ensure_annual(
            wrapped_partial(indices.prcptot_warmcold_quarter, op="coldest")
        ),
    }

    mod = build_module(
        "xclim.anuclim",
        mapping,
        doc="""
                ==============
                ANUCLIM indices
                ==============

                The ANUCLIM (v6.1) software package' BIOCLIM sub-module produces a set of Bioclimatic
                parameters derived values of temperature and precipitation. The methods in this module
                are wrappers around a subset of corresponding methods of `xclim.indices`. Note that none
                of the checks performed by the `xclim.utils.Indicator` class (like with `xclim.atmos`
                indicators) are performed in this module.

                Futhermore, according to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6),
                input values should be at a weekly (or monthly) frequency.  However, the xclim.indices
                implementation here will calculate the result with input data of any frequency.

                .. _ANUCLIM: https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6)
                """,
        mode=mode,
    )
    return mod


ICCLIM = icclim = __build_icclim("ignore")
anuclim = __build_anuclim()
