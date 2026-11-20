#!/usr/bin/env python
# Tests for `xclim.core.options`
from __future__ import annotations

import threading
import time

import pytest

from xclim import set_options
from xclim.core.options import get_options


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
        ("missing_options", {"pct": {"subfreq": None, "tolerance": 0.1}}),
        (
            "missing_options",
            {"wmo": {"nm": 10, "nc": 3}, "pct": {"subfreq": None, "tolerance": 0.1}},
        ),
    ],
)
def test_set_options_valid(option, value):
    old = get_options()[option]
    with set_options(**{option: value}):
        if option != "missing_options":
            assert get_options()[option] == value
        else:
            for k, opts in value.items():
                curr_opts = get_options()["missing_options"][k].copy()
                assert curr_opts == opts
    assert get_options()[option] == old


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
    old = get_options()[option]
    with pytest.raises(ValueError):
        set_options(**{option: value})
    assert old == get_options()[option]


def test_setting_options_threadsafe():
    def set_check_missing():
        set_options(check_missing="wmo")
        time.sleep(2)

    def assert_option_unchanged():
        time.sleep(0.5)
        assert get_options()["check_missing"] == "any"

    t1 = threading.Thread(target=set_check_missing)
    t2 = threading.Thread(target=assert_option_unchanged)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
