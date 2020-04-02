# -*- coding: utf-8 -*-
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
            ... and so on for other frequent parameters translation...
        },
        "atmos.dtrvar" : {
            "long_name": "Variabilité de l'intervalle de température moyen",
            "description": "Variabilité {freq:f} de l'intervalle de température moyen."
        },
        ... and so on for other indicators...
    }

Indicators are named by their module and identifier, which can differ from the callable name.
In this case, the indicator is called through `atmos.daily_temperature_range_variability`,
but its identifier is `dtrvar`.

Here, the usual parameter passed to the formatting of "description" is "freq" and is usually
translated from "YS" to "annual". However, in french and in this sentence, the feminine
form should be used, so the "f" modifier is added by the translator so that the
formatting function knows which translation to use. Acceptable entries for the mappings
are limited to what is already defined in `xclim.indicators.utils.default_formatter`.

The "attrs_mapping" and its "modifiers" key are mandatory in the locale dict, all other
entries (translations of frequent parameters and all indicator entries) are optional.

Attributes
----------
LOCALES
    List of currently set locales. Computing indicator through a xclim.indicators.indicator.Indicator
    object will add metadata in these languages as available.
TRANSLATABLE_ATTRS
    List of attributes to consider translatable when generating locale dictionaries.
"""
import json
import warnings
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import pkg_resources

from xclim.core.formatting import AttrFormatter

LOCALES = []
TRANSLATABLE_ATTRS = ["long_name", "description", "comment", "title", "abstract"]


def list_locales():
    """Return a list of available locales in xclim."""
    locale_list = pkg_resources.resource_listdir(__package__, "")
    return [locale.split(".")[0] for locale in locale_list if locale.endswith(".json")]


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
            json.load(pkg_resources.resource_stream(__package__, f"{locale}.json")),
        )
    if isinstance(locale[1], dict):
        return locale
    with open(locale[1]) as locf:
        return locale[0], json.load(locf)


def get_local_attrs(
    indicator: Any,
    *locales: Union[str, Sequence[str], Tuple[str, dict]],
    names: Optional[Sequence[str]] = None,
    fill_missing: bool = True,
    append_locale_name: bool = True,
):
    """Get all attributes of an indicator in the requested locales.

    Parameters
    ----------
    indicator : Union[utils.Indicator, utils.Indicator2D]
        Indicator object
    *locales : str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes. If none are given, defaults to the currently globally set in
        xclim.locales.LOCALES
    names : Optional[Sequence[str]]
        If given, only returns translations of attributes in this list.
    fill_missing : bool
        If True (default), fill untranslated attributes by the default (english) ones.
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
    if not locales:
        locales = LOCALES

    if not append_locale_name and len(locales) > 1:
        raise ValueError(
            "`append_locale_name` cannot be False if multiple locales are requested."
        )

    attrs = {}
    for locale in locales:
        loc_name, loc_dict = get_local_dict(locale)
        loc_name = f"_{loc_name}" if append_locale_name else ""
        ind_name = f"{indicator.__module__.split('.')[2]}.{indicator.identifier}"
        local_attrs = loc_dict.get(ind_name)
        if local_attrs is None:
            warnings.warn(
                f"Attributes of indicator {ind_name} in language {locale} were requested, but none were found."
            )
        else:
            for name in TRANSLATABLE_ATTRS:
                if (names is None or name in names) and (
                    fill_missing or name in local_attrs
                ):
                    attrs[f"{name}{loc_name}"] = local_attrs.get(
                        name, getattr(indicator, name)
                    )
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


def set_locales(*locales: Union[str, Sequence[str], Tuple[str, dict]]):
    """Set the current locales.

    All indicators computed through atmos, land or seaIce will have additionnal metadata
    in the given languages, as available and defined in xclim.locales data files or in given file.

    Parameters
    ----------
    *locales : str or tuple of str
        IETF language tag or a tuple of the language tag and a translation dict, or
        a tuple of the language tag and a path to a json file defining translation
        of attributes.

    Raises
    ------
    UnavailableLocaleError
        If a requested locale is not available.
    """
    for locale in locales:
        if (isinstance(locale, str) and get_best_locale(locale) is None) or (
            not isinstance(locale, str)
            and isinstance(locale[1], str)
            and not Path(locale[1]).is_file()
        ):
            raise UnavailableLocaleError(locale)
    LOCALES[:] = locales


class metadata_locale:
    """Set a locale for the metadata output within a context.
    """

    def __init__(self, *locales: Union[str, Sequence[str], Tuple[str, dict]]):
        """Create the context object to manage locales.

        Parameters
        ----------
        **locales : str
            IETF language tag or a tuple of the language tag and a translation dict, or
            a tuple of the language tag and a path to a json file defining translation
            of attributes.

        Raises
        ------
        UnavailableLocaleError
            If a requested locale is not defined in `xclim.locales`.

        Examples
        --------
        >>> with metadata_locale("fr", "de"):
        >>>     gdd = atmos.growing_degree_days(ds.tas)  # Will be created with english, french and german metadata.
        >>> gdd = atmos.growing_degree_days(ds.tas)  # Will be created with english metadata only.
        """

        self.locales = locales

    def __enter__(self):
        self.old_locales = LOCALES[:]
        set_locales(*self.locales)

    def __exit__(self, *args, **kwargs):
        set_locales(*self.old_locales)


class UnavailableLocaleError(ValueError):
    """Error raised when a locale is requested but doesn"t exist.
    """

    def __init__(self, locale):
        super().__init__(
            f"Locale {locale} not available. Use `xclim.locales.list_locales()` to see available languages."
        )


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
    import xclim as xc

    indicators = {}
    for module in [xc.atmos, xc.land, xc.seaIce]:
        for indicator in module.__dict__.values():
            if not isinstance(
                indicator, (xc.core.indicator.Indicator, xc.core.indicator.Indicator2D)
            ):
                continue
            ind_name = f"{indicator.__module__.split('.')[2]}.{indicator.identifier}"
            indicators[ind_name] = indicator

    best_locale = get_best_locale(locale)
    if best_locale is not None:
        locname, attrs = get_local_dict(best_locale)
        for ind_name in attrs.copy().keys():
            if ind_name != "attrs_mapping" and ind_name not in indicators:
                attrs.pop(ind_name)
    else:
        attrs = {}

    attrs_mapping = attrs.setdefault("attrs_mapping", {})
    attrs_mapping.setdefault("modifiers", [""])
    for key, value in xc.core.formatting.default_formatter.mapping.items():
        attrs_mapping.setdefault(key, [value[0]])

    eng_attr = ""
    for ind_name, indicator in indicators.items():
        ind_attrs = attrs.setdefault(ind_name, {})
        for translatable_attr in TRANSLATABLE_ATTRS:
            if init_english:
                eng_attr = getattr(indicator, translatable_attr)
                if not isinstance(eng_attr, str):
                    eng_attr = ""
            ind_attrs.setdefault(f"{translatable_attr}", eng_attr)
    return attrs
