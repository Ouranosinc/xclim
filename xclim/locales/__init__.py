# -*- coding: utf-8 -*-
"""xclim Internationalization module

Defines methods and object to help the internationalization of metadata for the
climate indicators computed by xclim.

All the methods and objects in this module use localization data given in json files.
These files are expected to be defined as in this example for french:

```json
{
    "attrs_mapping" : {
        "modifiers": ["", "_f", _mpl", "_fpl"],
        "description": {
            "YS" : ["annuel", "annuelle", "annuels", "annuelles"],
            ... and so on for other frequent parameters translation...
        }
        ... and so on for other translatable attributes...
    },
    "atmos.dtrvar" : {
        "long_name": "Variabilité de l'intervalle de température moyen",
        "description": "Variabilité {freq_f} de l'intervalle de température moyen."
    },
    ... and so on for other indicators...
}
```

Indicators are named by their module and identifier, which can differ from the callable name.
In this case, the indicator is called through `atmos.daily_temperature_range_variability`,
but its identifier is `dtrvar`.

Here, the usual parameter passed to the formatting of "description" is "freq" and is usually
translated from "YS" to "annual". However, in french and in this sentence, the feminine
form should be used, so the "_f" modifier is added by the translator so that the
formatting function knows which translation to use. Acceptable entries for the mappings
are limited to what is already defined in `xclim.utils.Indicator._attrs_mapping`.

The "attrs_mapping" and its key "modifiers" are mandatory in the locale dict, all other
entries (translations of frequent parameters and all indicator entries) are optional.

Attributes
----------
LOCALES : list
    List of currently set locales. Computing indicator through a xclim.utils.Indicator
    object will add metadata in these languages as available.
TRANSLATABLE_ATTRS : list
    List of attributes to consider translatable when generating locale dictionaries.
"""
import json
import warnings
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Union

import pkg_resources

LOCALES = []
TRANSLATABLE_ATTRS = ["long_name", "description", "comment"]


def list_locales():
    """Return a list of available locales in xclim."""
    locale_list = pkg_resources.resource_listdir(__package__, "")
    return [locale.split(".")[0] for locale in locale_list if locale.endswith(".json")]


def get_locale_dict(locale: Union[str, Sequence[str]]):
    """Return all translated metadata for a given locale.

    Parameters
    ----------
    locale : str or sequence of str
        2-char locale strings or tuple of the 2-char locale string and a path to a json
        file defining translation of attributes.

    Raises
    ------
    UnavailableLocaleError
        If the given locale is not available.

    Returns
    -------
    str
        The 2-char locale string
    dict
        The available translations in this locale.
    """
    if isinstance(locale, str):
        if locale not in list_locales():
            raise UnavailableLocaleError(locale)
        else:
            return (
                locale,
                json.load(pkg_resources.resource_stream(__package__, f"{locale}.json")),
            )
    with open(locale[1]) as locf:
        return locale[0], json.load(locf)


def get_local_format(attribute, mapping, modifiers):
    """Return a formatting function for attribute

    Allows for more control in locale-dependent string mapping.
    The returned function parses the parameters passed to format the string
    and replaces formattable-values by all available translation and their
    modified versions, as defined in this modules' doc.

    Parameters
    ----------
    attribute : str
        Attribute template string
    mapping : dict
        A mapping from a parameter value to a list of (str) translations of this
        parameter for different modifiers.
    modifiers : list of str
        A list of modifiers suffixes.

    Returns
    -------
    callable
        A formatting function for `attribute`.
        In addition to formatting the function also capitalizes its output.

    Examples
    --------
    >>> func = get_local_format('Moyenne {freq_f} de X', {'YS': ['annuel', 'annuelle']}, ['', '_f'])
    >>> func(freq='{YS}')
    "Moyenne annuelle de X"
    """

    def local_format(**kwargs):
        for key, val in kwargs.copy().items():
            if val.startswith("{") and val[1:-1] in mapping:
                kwargs.pop(key)
                kwargs.update(
                    {
                        f"{key}{modifier}": value
                        for modifier, value in zip(modifiers, mapping[val[1:-1]])
                    }
                )
        return attribute.format(**kwargs).capitalize()


def get_indicator_local_attrs(indicator: Any, *locales: Union[str, Sequence[str]]):
    """Get all attributes of an indicator in the requested locales.

    Parameters
    ----------
    indicator : Union[utils.Indicator, utils.Indicator2D]
        Indicator object
    *locales : str
        2-char locale strings or tuple of the 2-char locale string and a path to a json
        file defining translation of attributes. If none are give, defaults to the
        currently globally set in xclim.locales.LOCALES

    Returns
    -------
    dict
        All CF attributes available for given indicator and locales.
        Warns and returns an empty dict if none were available.
    """
    if not locales:
        locales = LOCALES
    attrs = {}
    for locale in locales:
        loc_name, loc_dict = get_locale_dict(locale)
        ind_name = f"{indicator.__module__.split('.')[1]}.{indicator.identifier}"
        local_attrs = loc_dict.get(ind_name)
        if local_attrs is None:
            warnings.warn(
                f"Attributes of indicator {ind_name} in language {locale} were requested, but none were found."
            )
        else:
            attrs.update(
                {
                    f"{name}_{loc_name}": get_local_format(
                        attr,
                        loc_dict["attrs_mapping"].get(name),
                        loc_dict["attrs_mapping"]["modifiers"],
                    )
                    for name, attr in local_attrs.items()
                }
            )
    return attrs


def set_locales(*locales: Union[str, Sequence[str]]):
    """Set the current locales.

    All indicators computed through atmos, land or seaIce will have additionnal metadata
    in the given languages, as available and defined in xclim.locales data files or in given file.

    Parameters
    ----------
    *locales : str or tuple of str
        2-char locale strings or tuple of the 2-char locale string and a path to a json
        file defining translation of attributes.

    Raises
    ------
    UnavailableLocaleError
        If a requested locale is not available.
    """
    for locale in locales:
        if locale not in list_locales() and not Path(locale[1]).is_file():
            raise UnavailableLocaleError(locale)
    LOCALES[:] = locales


class metadata_locale:
    """Set a locale for the metadata output within a context.
    """

    def __init__(self, **locales: str):
        """Create the context object to manage locales.

        Parameters
        ----------
        **locales : str
            Requested locales as 2-char strings.

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
        set_locales(self.locales)

    def __exit__(self, type, value, traceback):
        set_locales(self.old_locales)


class UnavailableLocaleError(ValueError):
    """Error raised when a locale is requested but doesn"t exist.
    """

    def __init__(self, locale):
        super().__init__(
            f"Locale {locale} not available. Use `metadata_locale.list_locales()` to see available languages."
        )


def generate_locale_dict(locale: str, init_english: bool = False):
    """Generate a dictionary with keys for each indicators and translatable attributes.

    Parameters
    ----------
    locale : str
        Locale 2-char string
    init_english : bool, optional
        If True, fills the initial dictionary with the english versions of the attributes.
        Defaults to False.
    """
    import xclim as xc

    indicators = {}
    for module in [xc.atmos, xc.land, xc.seaIce]:
        for indicator in module.__dict__.values():
            if not isinstance(indicator, (xc.utils.Indicator, xc.utils.Indicator2D)):
                continue
            ind_name = f"{indicator.__module__.split('.')[1]}.{indicator.identifier}"
            indicators[ind_name] = indicator

    if locale in list_locales():
        locname, attrs = get_locale_dict(locale)
        for ind_name in attrs.copy().keys():
            if ind_name not in indicators:
                attrs.pop(ind_name)
    else:
        attrs = {}

    attrs_mapping = attrs.setdefault("attrs_mapping", {})
    attrs_mapping.setdefault("modifiers", [""])
    for key, value in xc.utils.Indicator._attrs_mapping.items():
        if key in TRANSLATABLE_ATTRS:
            attrs_mapping.setdefault(
                key, {param: [val] for param, val in value.items()}
            )

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
