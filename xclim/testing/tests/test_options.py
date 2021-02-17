#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim.core.options`
import pytest

from xclim import set_options
from xclim.core.missing import MissingBase
from xclim.core.options import OPTIONS, register_missing_method


@pytest.mark.parametrize(
    "option,value",
    [
        ("metadata_locales", ["fr"]),
        ("data_validation", "log"),
        ("data_validation", "raise"),
        ("cf_compliance", "log"),
        ("cf_compliance", "raise"),
        ("check_missing", "wmo"),
        ("check_missing", "any"),
        ("missing_options", {"wmo": {"nm": 10, "nc": 3}}),
        ("missing_options", {"pct": {"tolerance": 0.1}}),
        ("missing_options", {"wmo": {"nm": 10, "nc": 3}, "pct": {"tolerance": 0.1}}),
    ],
)
def test_set_options_valid(option, value):
    old = OPTIONS[option]
    with set_options(**{option: value}):
        if option != "missing_options":
            assert OPTIONS[option] == value
        else:
            for k, opts in value.items():
                curr_opts = OPTIONS["missing_options"][k].copy()
                assert curr_opts == opts
    assert OPTIONS[option] == old


@pytest.mark.parametrize(
    "option,value",
    [
        ("metadata_locales", ["tlh"]),
        ("metadata_locales", [("tlh", "not/a/real/klingo/file.json")]),
        ("data_validation", True),
        ("check_missing", "from_context"),
        ("cf_compliance", False),
        ("missing_options", {"pct": {"nm": 45}}),
        ("missing_options", {"wmo": {"nm": 45, "nc": 3}}),
        (
            "missing_options",
            {"wmo": {"nm": 45, "nc": 3}, "notachoice": {"tolerance": 45}},
        ),
        (
            "missing_options",
            {"wmo": {"nm": 45, "nc": 3, "_validator": lambda x: x < 1}},
        ),
    ],
)
def test_set_options_invalid(option, value):
    old = OPTIONS[option]
    with pytest.raises(ValueError):
        set_options(**{option: value})
    assert old == OPTIONS[option]


def test_register_missing_method():
    @register_missing_method("test")
    class MissingTest(MissingBase):
        def is_missing(self, null, count, a_param=2):
            return True

        @staticmethod
        def validate(a_param):
            return a_param < 3

    with pytest.raises(ValueError):
        set_options(missing_options={"test": {"a_param": 5}})

    with set_options(check_missing="test"):
        assert OPTIONS["check_missing"] == "test"
