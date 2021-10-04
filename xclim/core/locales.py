# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Internationalization
====================

Defines methods and object to help the internationalization of metadata for the
climate indicators computed by xclim.

All the methods and objects in this module use localization data given in json files.
These files are expected to be defined as in this example for french:

.. code-block::

    {
        "attrs_mapping" : {
            "modifiers": ["", "f", "mpl", "fpl"],
            "YS" : ["annuel", "annuelle", "annuels", "annuelles"],
            "AS-*" : ["annuel", "annuelle", "annuels", "annuelles"],
            ... and so on for other frequent parameters translation...
        },
          "DTRVAR": {
            "long_name": "Variabilité de l'amplitude de la température diurne",
            "description": "Variabilité {freq:f} de l'amplitude de la température diurne (définie comme la moyenne de la variation journalière de l'amplitude de température sur une période donnée)",
            "title": "Variation quotidienne absolue moyenne de l'amplitude de la température diurne",
            "comment": "",
            "abstract": "La valeur absolue de la moyenne de l'amplitude de la température diurne."
          },
        ... and so on for other indicators...
    }

Indicators are named by subclass identifier, the same as in the indicator registry (`xclim.core.indicators.registry`),
but which can differ from the callable name. In this case, the indicator is called through
`atmos.daily_temperature_range_variability`, but its identifier is `DTRVAR`.
Use the `ind.__class__.__name__` accessor to get its registry name.

Here, the usual parameter passed to the formatting of "description" is "freq" and is usually
translated from "YS" to "annual". However, in french and in this sentence, the feminine
form should be used, so the "f" modifier is added by the translator so that the
formatting function knows which translation to use. Acceptable entries for the mappings
are limited to what is already defined in `xclim.core.indicators.utils.default_formatter`.

For user-provided internationalization dictionaries, only the "attrs_mapping" and
its "modifiers" key are mandatory, all other entries (translations of frequent parameters
and all indicator entries) are optional. For xclim-provided translations (for now only french),
all indicators must have en entry and the "attrs_mapping" entries must match exactly the default formatter.
Those default translations are found in the `xclim/locales` folder.

Attributes
----------
TRANSLATABLE_ATTRS
    List of attributes to consider translatable when generating locale dictionaries.
"""
import json
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

from .formatting import AttrFormatter, default_formatter

TRANSLATABLE_ATTRS = [
    "long_name",
    "description",
    "comment",
    "title",
    "abstract",
    "keywords",
]

_LOCALES = {}


def list_locales():
    """List of loaded locales. Includes all loaded locales, no matter how complete the translations are."""
    return list(_LOCALES.keys())


def _valid_locales(locales):
    """Check if the lcoales are valid."""
    if isinstance(locales, str):
        return True
    return all(
        [
            # A locale is valid if it is a string from the list
            (isinstance(locale, str) and locale in _LOCALES)
            or (
                # Or if it is a tuple of a string and either a file or a dict.
                not isinstance(locale, str)
                and isinstance(locale[0], str)
                and (isinstance(locale[1], dict) or Path(locale[1]).is_file())
            )
            for locale in locales
        ]
    )


def get_local_dict(locale: Union[str, Sequence[str], Tuple[str, dict]]):
    """Return all translated metadata for a given locale.

    Parameters
    ----------
    locale : str or sequence of str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes.

    Raises
    ------
    UnavailableLocaleError
        If the given locale is not available.

    Returns
    -------
    str
        The best fitting locale string
    dict
        The available translations in this locale.
    """
    _valid_locales([locale])

    if isinstance(locale, str):
        if locale not in _LOCALES:
            raise UnavailableLocaleError(locale)

        return (locale, _LOCALES[locale])

    if isinstance(locale[1], dict):
        trans = locale[1]
    else:
        # Thus a string pointing to a json file
        trans = read_locale_file(locale[1])

    if locale[0] in _LOCALES:
        loaded_trans = _LOCALES[locale[0]]
        # Passed translations have priority
        loaded_trans.update(trans)
        trans = loaded_trans
    return (locale[0], trans)


def get_local_attrs(
    indicator: Union[str, Sequence[str]],
    *locales: Union[str, Sequence[str], Tuple[str, dict]],
    names: Optional[Sequence[str]] = None,
    append_locale_name: bool = True,
) -> dict:
    """Get all attributes of an indicator in the requested locales.

    Parameters
    ----------
    indicator : str or sequence of strings
        Indicator's class name, usually the same as in `xc.core.indicator.registry`.
        If multiple names are passed, te attrs from each indicators are merged, with highest priority to the first name.
    *locales : str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes.
    names : Optional[Sequence[str]]
        If given, only returns translations of attributes in this list.
    append_locale_name : bool
        If True (default), append the language tag (as "{attr_name}_{locale}") to the
        returned attributes.

    Raises
    ------
    ValueError
        If `append_locale_name` is False and multiple `locales` are requested.

    Returns
    -------
    dict
        All CF attributes available for given indicator and locales.
        Warns and returns an empty dict if none were available.
    """
    if isinstance(indicator, str):
        indicator = [indicator]

    if not append_locale_name and len(locales) > 1:
        raise ValueError(
            "`append_locale_name` cannot be False if multiple locales are requested."
        )

    attrs = {}
    for locale in locales:
        loc_name, loc_dict = get_local_dict(locale)
        loc_name = f"_{loc_name}" if append_locale_name else ""
        local_attrs = loc_dict.get(indicator[-1], {})
        for other_ind in indicator[-2::-1]:
            local_attrs.update(loc_dict.get(other_ind, {}))
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
    locale: Union[str, Sequence[str], Tuple[str, dict]]
) -> AttrFormatter:
    """Return an AttrFormatter instance for the given locale.

    Parameters
    ----------
    locale : str or tuple of str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes.
    """
    loc_name, loc_dict = get_local_dict(locale)
    if "attrs_mapping" in loc_dict:
        attrs_mapping = loc_dict["attrs_mapping"].copy()
        mods = attrs_mapping.pop("modifiers")
        return AttrFormatter(attrs_mapping, mods)

    warnings.warn(
        "No `attrs_mapping` entry found for locale {loc_name}, using default (english) formatter."
    )
    return default_formatter


class UnavailableLocaleError(ValueError):
    """Error raised when a locale is requested but doesn't exist."""

    def __init__(self, locale):
        super().__init__(
            f"Locale {locale} not available. Use `xclim.core.locales.list_locales()` to see available languages."
        )


def read_locale_file(filename, module=None, encoding="UTF8"):
    """Read a locale file (*.json) and return its dictionary.

    Parameters
    ----------
    filename: PathLike
      The file to read.
    module: string, optional
      If module is a string, this module name is added to all identifiers translated in this file.
      Defaults to None, and no module name is added (as if the indicator was an official xclim indicator).
    encoding : string
      The encoding to use when reading the file.
      Defaults to UTF-8, overriding python's default mechanism which is machine dependent.
    """
    with open(filename, "r", encoding=encoding) as f:
        locdict = json.load(f)

    if module is not None:
        locdict = {
            (k if k == "attrs_mapping" else f"{module}.{k}"): v
            for k, v in locdict.items()
        }
    return locdict


def load_locale(locdata: Union[str, Path, Mapping[str, dict]], locale: str):
    """Load translations from a json file into xclim.

    Parameters
    ----------
    locdata : str or dictionary
      Either a loaded locale dictionary or a path to a json file.
    locale : str
      The locale name (IETF tag).
    """
    if isinstance(locdata, (str, Path)):
        filename = Path(locdata)
        locdata = read_locale_file(filename)

    if locale in _LOCALES:
        _LOCALES[locale].update(locdata)
    else:
        _LOCALES[locale] = locdata


def generate_local_dict(locale: str, init_english: bool = False):
    """Generate a dictionary with keys for each indicators and translatable attributes.

    Parameters
    ----------
    locale : str
        Locale in the IETF format
    init_english : bool
        If True, fills the initial dictionary with the english versions of the attributes.
        Defaults to False.
    """
    from xclim.core.indicator import registry

    if locale in _LOCALES:
        locname, attrs = get_local_dict(locale)
        for ind_name in attrs.copy().keys():
            if ind_name != "attrs_mapping" and ind_name not in registry:
                attrs.pop(ind_name)
    else:
        attrs = {}

    attrs_mapping = attrs.setdefault("attrs_mapping", {})
    attrs_mapping.setdefault("modifiers", [""])
    for key, value in default_formatter.mapping.items():
        attrs_mapping.setdefault(key, [value[0]])

    eng_attr = ""
    for ind_name, indicator in registry.items():
        ind_attrs = attrs.setdefault(ind_name, {})
        for translatable_attr in set(TRANSLATABLE_ATTRS).difference(
            set(indicator._cf_names)
        ):
            if init_english:
                eng_attr = getattr(indicator, translatable_attr)
                if not isinstance(eng_attr, str):
                    eng_attr = ""
            ind_attrs.setdefault(f"{translatable_attr}", eng_attr)

        for var_attrs in indicator.cf_attrs:
            # In the case of single output, put var attrs in main dict
            if len(indicator.cf_attrs) > 1:
                ind_attrs = attrs.setdefault(f"{ind_name}.{var_attrs['var_name']}", {})

            for translatable_attr in set(TRANSLATABLE_ATTRS).intersection(
                set(indicator._cf_names)
            ):
                if init_english:
                    eng_attr = var_attrs.get(translatable_attr)
                    if not isinstance(eng_attr, str):
                        eng_attr = ""
                ind_attrs.setdefault(f"{translatable_attr}", eng_attr)
    return attrs
