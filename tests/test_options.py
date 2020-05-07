#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim.core.options`
import pytest

from xclim import set_options
from xclim.core.options import OPTIONS
from xclim.core.options import register_missing_method


@pytest.mark.parametrize(
    "option,value",
    [
        ("validate_inputs", "log"),
        ("validate_inputs", "raise"),
        ("check_missing", "wmo"),
        ("check_missing", "any"),
        ("missing_wmo", {"nm": 10, "nc": 3}),
        ("missing_pct", 0.1),
    ],
)
def test_set_options_valid(option, value):
    old = OPTIONS[option]
    with set_options(**{option: value}):
        assert OPTIONS[option] == value
    assert OPTIONS[option] == old


@pytest.mark.parametrize(
    "option,value",
    [
        ("validate_inputs", True),
        ("missing_pct", 4),
        ("missing_wmo", {"nm": 45}),
        ("missing_wmo", {"nm": 45, "nc": 3}),
    ],
)
def test_set_options_invalid(option, value):
    old = OPTIONS[option]
    with pytest.raises(ValueError):
        set_options(**{option: value})
    assert old == OPTIONS[option]


def test_register_missing_method():
    @register_missing_method("test")
    def missing_example(da, freq):
        return True

    with set_options(check_missing="test"):
        assert OPTIONS["check_missing"] == "test"
