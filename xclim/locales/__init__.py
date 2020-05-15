from xclim.core.locales import get_best_locale
from xclim.core.locales import get_local_dict
from xclim.core.locales import TRANSLATABLE_ATTRS


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
