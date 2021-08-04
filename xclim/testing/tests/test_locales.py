#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim.locales`
import json

import numpy as np
import pytest

from xclim import atmos
from xclim.core import locales as xloc
from xclim.core.formatting import default_formatter
from xclim.core.locales import generate_local_dict
from xclim.core.options import set_options

esperanto = (
    "eo",
    {
        "attrs_mapping": {"modifiers": ["adj"], "AS-*": ["jara"], "MS": ["monata"]},
        "TG_MEAN": {
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
            "AS-*": ["годовое", "годовая"],
            "MS": ["месячный", "месячная"],
        },
        "TG_MEAN": {
            "long_name": "Среднее значение среднесуточной температуры.",
            "description": "Средне{freq:nf} среднесуточная температура.",
        },
    },
)


def test_local_dict(tmp_path):
    loc, dic = xloc.get_local_dict("fr")
    assert loc == "fr"
    assert (
        dic["TG_MEAN"]["long_name"] == "Moyenne de la température journalière moyenne"
    )

    loc, dic = xloc.get_local_dict(esperanto)
    assert loc == "eo"
    assert dic["TG_MEAN"]["long_name"] == "Meza ciutaga averaga temperaturo"

    with (tmp_path / "ru.json").open("w", encoding="utf-8") as f:
        json.dump(russian[1], f, ensure_ascii=False)

    loc, dic = xloc.get_local_dict(("ru", tmp_path / "ru.json"))
    assert loc == "ru"
    assert dic["TG_MEAN"]["long_name"] == "Среднее значение среднесуточной температуры."

    with pytest.raises(xloc.UnavailableLocaleError):
        xloc.get_local_dict("tlh")


def test_local_attrs_sing():
    attrs = xloc.get_local_attrs(
        atmos.tg_mean.__class__.__name__, esperanto, append_locale_name=False
    )
    assert "description" not in attrs

    with pytest.raises(ValueError):
        attrs = xloc.get_local_attrs(
            atmos.tg_mean, "fr", esperanto, append_locale_name=False
        )


def test_local_attrs_multi(tmp_path):
    with (tmp_path / "ru.json").open("w", encoding="utf-8") as f:
        json.dump(russian[1], f, ensure_ascii=False)

    attrs = xloc.get_local_attrs(
        atmos.tg_mean.__class__.__name__,
        "fr",
        esperanto,
        ("ru", tmp_path / "ru.json"),
        append_locale_name=True,
    )
    for key in ["description_fr", "description_ru"]:
        assert key in attrs
    for key in ["description_eo"]:
        assert key not in attrs


def test_local_formatter():
    fmt = xloc.get_local_formatter(russian)
    assert fmt.format("{freq:nn}", freq="AS-JUL") == "годовое"
    assert fmt.format("{freq:nf}", freq="AS-DEC") == "годовая"


def test_indicator_output(tas_series):
    tas = tas_series(np.zeros(365))

    with set_options(metadata_locales="fr"):
        tgmean = atmos.tg_mean(tas, freq="YS")

    assert "long_name_fr" in tgmean.attrs
    assert (
        tgmean.attrs["description_fr"]
        == "Moyenne annuelle de la température journalière moyenne"
    )


def test_indicator_integration():
    eo_attrs = atmos.tg_mean.translate_attrs(esperanto, fill_missing=True)
    assert "title" in eo_attrs
    assert "long_name" in eo_attrs["outputs"][0]

    eo_attrs = atmos.tg_mean.translate_attrs(esperanto, fill_missing=False)
    assert "description" not in eo_attrs["outputs"][0]


@pytest.mark.parametrize("locale", xloc.list_locales())
def test_xclim_translations(locale, official_indicators):
    loc, dic = xloc.get_local_dict(locale)
    assert "attrs_mapping" in dic
    assert "modifiers" in dic["attrs_mapping"]
    for translatable, translations in dic["attrs_mapping"].items():
        if translatable != "modifiers":
            assert isinstance(translations, list)
            assert len(translations) <= len(dic["attrs_mapping"]["modifiers"])
    assert set(dic["attrs_mapping"].keys()).symmetric_difference(
        default_formatter.mapping.keys()
    ) == {"modifiers"}

    untranslated = []
    incomplete = []
    for indname, indcls in official_indicators.items():
        is_complete = True
        trans = indcls.translate_attrs(locale)
        if trans == {"outputs": []}:
            untranslated.append(indname)
            continue
        # Both global attrs are present
        is_complete = not ({"title", "abstract"} - set(trans.keys()))
        for attrs, transattrs in zip(indcls.cf_attrs, trans["outputs"]):
            if {"long_name", "description"} - set(transattrs.keys()):
                is_complete = False

        if not is_complete:
            incomplete.append(indname)

    if len(untranslated) > 0 or len(incomplete) > 0:
        pytest.fail(
            f"Indicators {', '.join(untranslated)} do not have translations and {', '.join(incomplete)} have incomplete ones for official locale {locale}."
        )


@pytest.mark.parametrize(
    "initeng,expected", [(False, ""), (True, atmos.tg_mean.long_name)]
)
def test_local_dict_generation(initeng, expected):
    dic = generate_local_dict("tlh", init_english=initeng)
    assert "attrs_mapping" in dic
    assert "modifiers" in dic["attrs_mapping"]
    assert dic["TG_MEAN"]["long_name"] == expected
