"""Locales and language support module."""
from xclim.core.formatting import default_formatter
from xclim.core.locales import TRANSLATABLE_ATTRS, get_best_locale, get_local_dict


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

    best_locale = get_best_locale(locale)
    if best_locale is not None:
        locname, attrs = get_local_dict(best_locale)
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
