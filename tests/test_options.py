#!/usr/bin/env python
# Tests for `xclim.core.options`
from __future__ import annotations

import threading
import time
from multiprocessing.pool import ThreadPool

import pytest
import xarray as xr

from xclim import atmos, set_options
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


def test_setting_options_threadsafe(random):
    results = [None]

    def set_check_missing():
        set_options(check_missing="wmo")
        time.sleep(2)

    def assert_option_unchanged():
        time.sleep(0.5)
        results[0] = get_options()["check_missing"] == "any"

    t1 = threading.Thread(target=set_check_missing)
    t2 = threading.Thread(target=assert_option_unchanged)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # pytest doesn't fail if a thread fails, must assert outside of it
    assert results[0]

    # Similar test, but with indicator computation and ThreadPool
    # ThreadPool will re-raise an exception in the thread
    # testing that changes to the main thread are not seen by children threads
    pr = xr.DataArray(
        random.gamma(0.8, 4, 730),
        dims="time",
        coords={"time": xr.date_range("2000-01-01", periods=730, freq="D")},
        attrs={"units": "mm/d"},
    )
    with ThreadPool(processes=1) as pool:
        # This set_options has no effect on the pool which was created from an earlier context
        with set_options(as_dataset=False):
            out = pool.apply(atmos.dry_spell_max_length, (pr,))
        assert isinstance(out, xr.Dataset)
