"""
Internationalization
====================

This module defines methods and object to help the internationalization of metadata for
climate indicators computed by xclim. Go to :ref:`notebooks/customize:Adding translated metadata` to see
how to use this feature.

All the methods and objects in this module use localization data given in JSON files.
These files are expected to be defined as in this example for French:

.. code-block::

    {
        "attrs_mapping": {
            "modifiers": ["", "f", "mpl", "fpl"],
            "YS": ["annuel", "annuelle", "annuels", "annuelles"],
            "YS-*": ["annuel", "annuelle", "annuels", "annuelles"],
            # ... and so on for other frequent parameters translation...
        },
        "DTRVAR": {
            "long_name": "Variabilité de l'amplitude de la température diurne",
            "description": "Variabilité {freq:f} de l'amplitude de la température diurne (définie comme la moyenne de la variation journalière de l'amplitude de température sur une période donnée)",
            "title": "Variation quotidienne absolue moyenne de l'amplitude de la température diurne",
            "comment": "",
            "abstract": "La valeur absolue de la moyenne de l'amplitude de la température diurne.",
        },
        # ... and so on for other indicators...
    }

Indicators are named by subclass identifier, the same as in the indicator registry (`xclim.core.indicators.registry`),
but which can differ from the callable name. In this case, the indicator is called through
`atmos.daily_temperature_range_variability`, but its identifier is `DTRVAR`.
Use the `ind.__class__.__name__` accessor to get its registry name.

Here, the usual parameter passed to the formatting of "description" is "freq" and is usually translated from "YS"
to "annual". However, in French and in this sentence, the feminine form should be used, so the "f" modifier is added
by the translator so that the formatting function knows which translation to use. Acceptable entries for the mappings
are limited to what is already defined in `xclim.core.indicators.utils.default_formatter`.

For user-provided internationalization dictionaries, only the "attrs_mapping" and its "modifiers" key are mandatory,
all other entries (translations of frequent parameters and all indicator entries) are optional.
For xclim-provided translations (for now only French), all indicators must have en entry and the "attrs_mapping"
entries must match exactly the default formatter.
Those default translations are found in the `xclim/locales` folder.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from xclim.core.formatting import AttrFormatter, default_formatter
from xclim.core.utils import CaseInsensitiveDict

TRANSLATABLE_ATTRS = [
    "long_name",
    "description",
    "comment",
    "title",
    "abstract",
]
"""
List of attributes to consider translatable when generating locale dictionaries.
"""

_LOCALES = {}


def list_locales() -> list:
    """
    List of loaded locales.

    Includes all loaded locales, no matter how complete the translations are.

    Returns
    -------
    list
        A list of available locales.
    """
    return list(_LOCALES.keys())


def _valid_locales(locales):
    """Check if the locales are valid."""
    if isinstance(locales, str):
        return True
    return all(
        # A locale is valid if it is a string from the list
        (isinstance(locale, str) and locale in _LOCALES)
        or (
            # Or if it is a tuple of a string and either a file or a dict.
            not isinstance(locale, str)
            and isinstance(locale[0], str)
            and (isinstance(locale[1], dict) or Path(locale[1]).is_file())
        )
        for locale in locales
    )


def get_local_dict(locale: str | Sequence[str] | tuple[str, dict]) -> tuple[str, dict]:
    """
    Return all translated metadata for a given locale.

    Parameters
    ----------
    locale : str or sequence of str
        IETF language tag or a tuple of the language tag and a translation dict, or a tuple of the language
        tag and a path to a json file defining translation of attributes.

    Returns
    -------
    str
        The best fitting locale string.
    dict
        The available translations in this locale.

    Raises
    ------
    UnavailableLocaleError
        If the given locale is not available.
    """
    _valid_locales([locale])

    if isinstance(locale, str):
        if locale not in _LOCALES:
            raise UnavailableLocaleError(locale)

        return locale, deepcopy(_LOCALES[locale])

    if isinstance(locale[1], dict):
        trans = CaseInsensitiveDict(locale[1])
    else:
        # Thus, a string pointing to a json file
        trans = read_locale_file(locale[1])

    if locale[0] in _LOCALES:
        loaded_trans = deepcopy(_LOCALES[locale[0]])
        # Passed translations have priority
        loaded_trans.update(trans)
        trans = loaded_trans
    return locale[0], trans


def get_local_attrs(
    indicator: str | Sequence[str],
    locale: str,
    var_name: str = None,
    names: Sequence[str] | None = None,
    append_locale_name: bool = True,
) -> dict:
    r"""
    Get all attributes of an indicator in the requested locale.

    Parameters
    ----------
    indicator : str or sequence of strings
        Indicator's identifier, usually the same as in `xc.core.indicator.registry`.
        If multiple names are passed, the attrs from each indicator are merged,
        with the highest priority set to the first name.
    locale : str
        IETF language tag or a tuple of the language tag and a translation dict, or a tuple of the language tag
        and a path to a json file defining translation of attributes.
    var_name : str, optional
        For multi-output indicator, this is the name of the variable for which we request attributes.
    names : sequence of str, optional
        If given, only returns translations of attributes in this list.
    append_locale_name : bool
        If True (default), append the language tag (as "{attr_name}_{locale}") to the returned attributes.

    Returns
    -------
    dict
        All attributes available for given indicator and locales.
        Warns and returns an empty dict if none were available.

    Raises
    ------
    ValueError
        If `append_locale_name` is False and multiple `locales` are requested.
    """
    if isinstance(indicator, str):
        indicator = [indicator]

    loc_name, loc_dict = get_local_dict(locale)
    loc_name = f"_{loc_name}" if append_locale_name else ""

    local_attrs = {}
    for ind in reversed(indicator):
        local_attrs = local_attrs | loc_dict.get(ind.lower(), {})
        if var_name:
            local_attrs = local_attrs | loc_dict.get(f"{ind}.{var_name}".lower(), {})

    attrs = {}
    if not local_attrs:
        warnings.warn(
            f"Attributes of indicator {', '.join(indicator)} in language {locale} were requested, but none were found."
        )
    else:
        for name in TRANSLATABLE_ATTRS:
            if (names is None or name in names) and name in local_attrs:
                attrs[f"{name}{loc_name}"] = local_attrs[name]
    return attrs


def get_local_formatter(
    locale: str | Sequence[str] | tuple[str, dict],
) -> AttrFormatter:
    """
    Return an AttrFormatter instance for the given locale.

    Parameters
    ----------
    locale : str or tuple of str
        IETF language tag or a tuple of the language tag and a translation dict, or a tuple of the language tag
        and a path to a json file defining translation of attributes.

    Returns
    -------
    AttrFormatter
        A locale-based formatter object instance.
    """
    loc_name, loc_dict = get_local_dict(locale)
    if "attrs_mapping" in loc_dict:
        attrs_mapping = loc_dict["attrs_mapping"].copy()
        mods = attrs_mapping.pop("modifiers")
        return AttrFormatter(attrs_mapping, mods)

    warnings.warn(f"No `attrs_mapping` entry found for locale {loc_name}, using default (english) formatter.")
    return default_formatter


class UnavailableLocaleError(ValueError):
    """
    Error raised when a locale is requested but doesn't exist.

    Parameters
    ----------
    locale : str
        The locale code.
    """

    def __init__(self, locale):
        super().__init__(
            f"Locale {locale} not available. Use `xclim.core.locales.list_locales()` to see available languages."
        )


def read_locale_file(filename, module: str | None = None, encoding: str = "UTF8") -> dict[str, dict]:
    """
    Read a locale file (.json) and return its dictionary.

    Parameters
    ----------
    filename : PathLike
        The file to read.
    module : str, optional
        If the module is a string, this module name is added to all identifiers translated in this file.
        Defaults to None, and no module name is added (as if the indicator was an official xclim indicator).
    encoding : str
        The encoding to use when reading the file.
        Defaults to `UTF-8`, overriding Python's default mechanism which is machine-dependent.

    Returns
    -------
    dict
        The locale dictionary. All entries are lowercase.
    """
    with open(filename, encoding=encoding) as f:
        data = json.load(f)

    modstr = f"{module}." if module is not None else ""
    locdict = CaseInsensitiveDict({(k if k == "attrs_mapping" else f"{modstr}{k}"): v for k, v in data.items()})
    return locdict


def load_locale(locdata: str | Path | dict[str, dict], locale: str) -> None:
    """
    Load translations from a json file into xclim.

    Parameters
    ----------
    locdata : str or Path or dictionary
        Either a loaded locale dictionary or a path to a json file.
    locale : str
        The locale name (IETF tag).
    """
    if isinstance(locdata, str | Path):
        filename = Path(locdata)
        locdata = read_locale_file(filename)

    if locale in _LOCALES:
        _LOCALES[locale].update(locdata)
    else:
        _LOCALES[locale] = locdata


def generate_local_dict(locale: str, init_english: bool = False) -> CaseInsensitiveDict:
    """
    Generate a dictionary with keys for each indicator and translatable attributes.

    Parameters
    ----------
    locale : str
        Locale in the IETF format.
    init_english : bool
        If True, fills the initial dictionary with the english versions of the attributes. Defaults to False.

    Returns
    -------
    dict
        Indicator translation dictionary.
    """
    from ..core.indicator import registry  # pylint: disable=import-outside-toplevel

    if locale in _LOCALES:
        _, attrs = get_local_dict(locale)
        for ind_name in attrs.copy().keys():
            if ind_name != "attrs_mapping" and ind_name not in registry:
                attrs.pop(ind_name)
    else:
        attrs = CaseInsensitiveDict()

    attrs_mapping = attrs.setdefault("attrs_mapping", {})
    attrs_mapping.setdefault("modifiers", [""])
    for key, value in default_formatter.mapping.items():
        attrs_mapping.setdefault(key, [value[0]])

    for ind_name, indicator in registry.items():
        ind_attrs = attrs.setdefault(ind_name, {})
        for prop in indicator._translatable_props:
            if init_english:
                eng = getattr(indicator, prop)
                if not isinstance(eng, str):
                    eng = ""
            else:
                eng = ""
            ind_attrs.setdefault(prop, eng)

        for atts in indicator.attrs:
            # In the case of single output, put var attrs in main dict
            if len(indicator.attrs) > 1:
                ind_attrs = attrs.setdefault(f"{ind_name}.{atts.var_name}", {})

            for attr in indicator._translatable_attrs:
                if init_english:
                    eng = atts.get(attr)
                    if not isinstance(eng, str):
                        eng = ""
                else:
                    eng = ""
                ind_attrs.setdefault(attr, eng)
    return attrs
