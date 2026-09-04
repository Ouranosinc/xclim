# pylint: disable=unsubscriptable-object
# Tests for `xclim.locales`
from __future__ import annotations

import json

import numpy as np
import pytest

from xclim import atmos
from xclim.core import locales as xloc
from xclim.core.locales import generate_local_dict
from xclim.core.options import set_options

esperanto = (
    "eo",
    {
        "attrs_mapping": {"modifiers": ["adj"], "YS-*": ["jara"], "MS": ["monata"]},
        "tg_mean": {
            "long_name": "Meza ciutaga averaga temperaturo",
            "title": "Meza ciutaga averaga temperaturo",
        },
    },
)

russian = (
    "ru",
    {
        "attrs_mapping": {
            "modifiers": ["nn", "nf"],
            "YS-*": ["годовое", "годовая"],
            "MS": ["месячный", "месячная"],
        },
        "TG_MEAN": {
            "long_name": "Среднее значение среднесуточной температуры",
            "description": "Средне{freq:nf} среднесуточная температура.",
        },
    },
)


def test_local_dict(tmp_path):
    loc, dic = xloc.get_local_dict("fr")
    assert loc == "fr"
    assert dic["TG_MEAN"]["long_name"] == "Moyenne de la température moyenne quotidienne"

    loc, dic = xloc.get_local_dict(esperanto)
    assert loc == "eo"
    assert dic["TG_MEAN"]["long_name"] == "Meza ciutaga averaga temperaturo"

    with (tmp_path / "ru.json").open("w", encoding="utf-8") as f:
        json.dump(russian[1], f, ensure_ascii=False)

    loc, dic = xloc.get_local_dict(("ru", tmp_path / "ru.json"))
    assert loc == "ru"
    assert dic["TG_MEAN"]["long_name"] == "Среднее значение среднесуточной температуры"

    with pytest.raises(xloc.UnavailableLocaleError):
        xloc.get_local_dict("tlh")

    loc, dic = xloc.get_local_dict(("fr", {"TX_MAX": {"long_name": "Fait chaud."}}))
    assert loc == "fr"
    assert dic["TX_MAX"]["long_name"] == "Fait chaud."
    assert dic["TG_MEAN"]["long_name"] == "Moyenne de la température moyenne quotidienne"


def test_local_attrs():
    attrs = xloc.get_local_attrs(atmos.tg_mean.identifier, esperanto, append_locale_name=False)
    assert "description" not in attrs


def test_local_formatter():
    fmt = xloc.get_local_formatter(russian)
    assert fmt.format("{freq:nn}", freq="YS-JUL") == "годовое"
    assert fmt.format("{freq:nf}", freq="YS-DEC") == "годовая"


def test_indicator_output(tas_series):
    tas = tas_series(np.zeros(365))

    with set_options(metadata_locales="fr"):
        out = atmos.tg_mean(tas, freq="YS")

    assert "long_name_fr" in out.tg_mean.attrs
    assert out.tg_mean.attrs["description_fr"] == "Moyenne annuelle de la température quotidienne."


def test_indicator_integration():
    eo_attrs = atmos.tg_mean.translate(esperanto, fill_missing=True)
    assert "title" in eo_attrs
    assert "long_name" in eo_attrs["attrs"][0]

    eo_attrs = atmos.tg_mean.translate(esperanto, fill_missing=False)
    assert "description" not in eo_attrs["attrs"][0]


@pytest.mark.parametrize("locale", xloc.list_locales())
def test_xclim_translations(locale, official_indicators):
    _loc, dic = xloc.get_local_dict(locale)
    assert "attrs_mapping" in dic
    assert "modifiers" in dic["attrs_mapping"]
    for translatable, translations in dic["attrs_mapping"].items():
        if translatable != "modifiers":
            assert isinstance(translations, list)
            assert len(translations) <= len(dic["attrs_mapping"]["modifiers"])

    untranslated = []
    incomplete = []
    for indname, ind in official_indicators.items():
        is_complete = True
        trans = ind.translate(locale, fill_missing=False)
        if set(trans) == {"attrs"}:
            untranslated.append(indname)
            continue
        # Both global attrs are present
        is_complete = {"title", "abstract"}.issubset(set(trans))
        for _attrs, transattrs in zip(ind.attrs, trans["attrs"], strict=False):
            if {"long_name", "description"} - set(transattrs.keys()):
                is_complete = False

        if not is_complete:
            incomplete.append(indname)
    if len(untranslated) > 0 or len(incomplete) > 0:
        pytest.fail(
            f"{len(untranslated)} indicator(s) are missing translations"
            f"{': [' + ', '.join(untranslated) + ']' if len(untranslated) else ''}"
            f"{' and ' if len(incomplete) else '. '}"
            f"{', '.join(incomplete) or 'None'} have incomplete translations for official locale `{locale}`."
        )


@pytest.mark.parametrize(
    "initeng,expected",
    [
        (False, ""),
        (True, atmos.tg_mean.attrs[0]["long_name"]),
    ],
)
def test_local_dict_generation(initeng, expected):
    dic = generate_local_dict("tlh", init_english=initeng)
    assert "attrs_mapping" in dic
    assert "modifiers" in dic["attrs_mapping"]
    assert dic["tg_mean"]["long_name"] == expected
