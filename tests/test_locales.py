#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim.locales`
import json

import numpy as np
import pytest

import xclim.locales as xloc
from xclim import atmos
from xclim.core.formatting import default_formatter


esperanto = (
    "eo",
    {
        "attrs_mapping": {"modifiers": ["adj"], "YS": ["jara"], "MS": ["monata"]},
        "atmos.tg_mean": {
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
            "YS": ["годовое", "годовая"],
            "MS": ["месячный", "месячная"],
        },
        "atmos.tg_mean": {
            "long_name": "Среднее значение среднесуточной температуры.",
            "description": "Средне{freq:nf} среднесуточная температура.",
        },
    },
)


def test_best_locale():
    assert xloc.get_best_locale("fr") == "fr"
    assert xloc.get_best_locale("fr-CA") == "fr"
    assert xloc.get_best_locale("en") is None


def test_local_dict(tmp_path):
    loc, dic = xloc.get_local_dict("fr")
    assert loc == "fr"
    assert (
        dic["atmos.tg_mean"]["long_name"]
        == "Moyenne de la température journalière moyenne"
    )

    loc, dic = xloc.get_local_dict(esperanto)
    assert loc == "eo"
    assert dic["atmos.tg_mean"]["long_name"] == "Meza ciutaga averaga temperaturo"

    with (tmp_path / "ru.json").open("w") as f:
        json.dump(russian[1], f, ensure_ascii=False)

    loc, dic = xloc.get_local_dict(("ru", tmp_path / "ru.json"))
    assert loc == "ru"
    assert (
        dic["atmos.tg_mean"]["long_name"]
        == "Среднее значение среднесуточной температуры."
    )

    with pytest.raises(xloc.UnavailableLocaleError):
        xloc.get_local_dict("tlh")


@pytest.mark.parametrize(
    "fill,isin,notin", [(True, ["description"], []), (False, [], ["description"]),]
)
def test_local_attrs_sing(fill, isin, notin):
    attrs = xloc.get_local_attrs(
        atmos.tg_mean, esperanto, fill_missing=fill, append_locale_name=False
    )
    for key in isin:
        assert key in attrs
    for key in notin:
        assert key not in attrs

    with pytest.raises(ValueError):
        attrs = xloc.get_local_attrs(
            atmos.tg_mean, "fr", esperanto, append_locale_name=False
        )


@pytest.mark.parametrize(
    "fill,isin,notin",
    [
        (True, ["description_fr", "description_eo", "description_ru"], []),
        (False, ["description_fr", "description_ru"], ["description_eo"]),
    ],
)
def test_local_attrs_multi(fill, isin, notin, tmp_path):
    with (tmp_path / "ru.json").open("w") as f:
        json.dump(russian[1], f, ensure_ascii=False)

    attrs = xloc.get_local_attrs(
        atmos.tg_mean,
        "fr",
        esperanto,
        ("ru", tmp_path / "ru.json"),
        fill_missing=fill,
        append_locale_name=True,
    )
    for key in isin:
        assert key in attrs
    for key in notin:
        assert key not in attrs


def test_local_formatter():
    fmt = xloc.get_local_formatter(russian)
    assert fmt.format("{freq:nn}", freq="YS") == "годовое"
    assert fmt.format("{freq:nf}", freq="YS") == "годовая"


def test_context():
    assert "fr" not in xloc.LOCALES
    with xloc.metadata_locale("fr"):
        assert "fr" in xloc.LOCALES
    assert "fr" not in xloc.LOCALES


@pytest.mark.parametrize("locale", ["tlh", ("tlh", "not/a/real/klingo/file.json")])
def test_set_locales_error(locale):
    with pytest.raises(xloc.UnavailableLocaleError):
        xloc.set_locales(locale)


def test_indicator_output(tas_series):
    tas = tas_series(np.zeros(365))

    with xloc.metadata_locale("fr"):
        tgmean = atmos.tg_mean(tas, freq="YS")

    assert "long_name_fr" in tgmean.attrs
    assert (
        tgmean.attrs["description_fr"]
        == "Moyenne annuelle de la température journalière moyenne"
    )


def test_indicator_integration():
    eo_attrs = atmos.tg_mean.translate_attrs(esperanto, fill_missing=True)
    assert "title" in eo_attrs
    assert "long_name" in eo_attrs

    eo_attrs = atmos.tg_mean.translate_attrs(esperanto, fill_missing=False)
    assert "description" not in eo_attrs


@pytest.mark.parametrize("locale", xloc.list_locales())
def test_xclim_translations(locale):
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

    for indicator, fields in dic.items():
        if indicator != "attrs_mapping":
            mod, name = indicator.split(".")
            assert mod in ["atmos", "land", "seaIce"]
            # no easy to test if indicator exists!
            assert set(fields.keys()).issubset(xloc.TRANSLATABLE_ATTRS)


@pytest.mark.parametrize(
    "initeng,expected", [(False, ""), (True, atmos.tg_mean.long_name)]
)
def test_local_dict_generation(initeng, expected):
    dic = xloc.generate_local_dict("tlh", init_english=initeng)
    assert "attrs_mapping" in dic
    assert "modifiers" in dic["attrs_mapping"]
    assert dic["atmos.tg_mean"]["long_name"] == expected
