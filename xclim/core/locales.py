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
from typing import Optional, Sequence, Tuple, Union

import pkg_resources

from .formatting import AttrFormatter

TRANSLATABLE_ATTRS = [
    "long_name",
    "description",
    "comment",
    "title",
    "abstract",
    "keywords",
]


def list_locales():
    """Return a list of available locales in xclim."""
    locale_list = pkg_resources.resource_listdir("xclim.locales", "")
    return [locale.split(".")[0] for locale in locale_list if locale.endswith(".json")]


def _valid_locales(locales):
    return all(
        [
            (isinstance(locale, str) and get_best_locale(locale) is not None)
            or (
                not isinstance(locale, str)
                and isinstance(locale[0], str)
                and (Path(locale[1]).is_file() or isinstance(locale[1], dict))
            )
            for locale in locales
        ]
    )


def get_best_locale(locale: str):
    """Get the best fitting available locale.

    for existing locales : ['fr', 'fr-BE', 'en-US'],
    'fr-CA' returns 'fr', 'en' -> 'en-US' and 'en-GB' -> 'en-US'.

    Parameters
    ----------
    locale : str
        The requested locale, as an IETF language tag (lang or lang-territory)

    Returns
    -------
    str or None:
        The best available locale. None is none are available.
    """
    available = list_locales()
    if locale in available:
        return locale
    locale = locale.split("-")[0]
    if locale in available:
        return locale
    if locale in [av.split("-")[0] for av in available]:
        return [av for av in available if av.split("-")[0] == locale][0]
    return None


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
    if isinstance(locale, str):
        locale = get_best_locale(locale)
        if locale is None:
            raise UnavailableLocaleError(locale)

        return (
            locale,
            json.load(pkg_resources.resource_stream("xclim.locales", f"{locale}.json")),
        )
    if isinstance(locale[1], dict):
        return locale
    with open(locale[1], encoding="utf-8") as locf:
        return locale[0], json.load(locf)


def get_local_attrs(
    indicator: str,
    *locales: Union[str, Sequence[str], Tuple[str, dict]],
    names: Optional[Sequence[str]] = None,
    append_locale_name: bool = True,
):
    """Get all attributes of an indicator in the requested locales.

    Parameters
    ----------
    indicator : str
        Indicator's class name, usually the same as in `xc.core.indicator.registry`.
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
        .

    Returns
    -------
    dict
        All CF attributes available for given indicator and locales.
        Warns and returns an empty dict if none were available.
    """
    if not append_locale_name and len(locales) > 1:
        raise ValueError(
            "`append_locale_name` cannot be False if multiple locales are requested."
        )

    attrs = {}
    for locale in locales:
        loc_name, loc_dict = get_local_dict(locale)
        loc_name = f"_{loc_name}" if append_locale_name else ""
        local_attrs = loc_dict.get(indicator)
        if local_attrs is None:
            warnings.warn(
                f"Attributes of indicator {indicator} in language {locale} were requested, but none were found."
            )
        else:
            for name in TRANSLATABLE_ATTRS:
                if (names is None or name in names) and name in local_attrs:
                    attrs[f"{name}{loc_name}"] = local_attrs[name]
    return attrs


def get_local_formatter(locale: Union[str, Sequence[str], Tuple[str, dict]]):
    """Return an AttrFormatter instance for the given locale.

    Parameters
    ----------
    locale : str or tuple of str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes.
    """
    loc_name, loc_dict = get_local_dict(locale)
    attrs_mapping = loc_dict["attrs_mapping"].copy()
    mods = attrs_mapping.pop("modifiers")
    return AttrFormatter(attrs_mapping, mods)


class UnavailableLocaleError(ValueError):
    """Error raised when a locale is requested but doesn't exist."""

    def __init__(self, locale):
        super().__init__(
            f"Locale {locale} not available. Use `xclim.core.locales.list_locales()` to see available languages."
        )
