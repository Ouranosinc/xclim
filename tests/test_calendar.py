from __future__ import annotations

import os

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from xarray.coding.cftimeindex import CFTimeIndex

from xclim.core.calendar import (
    adjust_doy_calendar,
    climatological_mean_doy,
    common_calendar,
    compare_offsets,
    construct_offset,
    convert_calendar,
    date_range,
    datetime_to_decimal_year,
    days_in_year,
    days_since_to_doy,
    doy_to_days_since,
    ensure_cftime_array,
    get_calendar,
    interp_calendar,
    max_doy,
    parse_offset,
    percentile_doy,
    time_bnds,
)


@pytest.fixture(
    params=[dict(start="2004-01-01T12:07:01", periods=27, freq="3MS")], ids=["3MS"]
)
def time_range_kwargs(request):
    return request.param


@pytest.fixture()
def datetime_index(time_range_kwargs):
    return pd.date_range(**time_range_kwargs)


@pytest.fixture()
def cftime_index(time_range_kwargs):
    return xr.cftime_range(**time_range_kwargs)


def da(index):
    return xr.DataArray(
        np.arange(100.0, 100.0 + index.size), coords=[index], dims=["time"]
    )


@pytest.mark.parametrize("freq", ["6480H", "302431T", "23144781S"])
def test_time_bnds(freq, datetime_index, cftime_index):
    da_datetime = da(datetime_index).resample(time=freq)
    da_cftime = da(cftime_index).resample(time=freq)

    cftime_bounds = time_bnds(da_cftime, freq=freq)
    cftime_starts = cftime_bounds.isel(bnds=0)
    cftime_ends = cftime_bounds.isel(bnds=1)
    cftime_starts = CFTimeIndex(cftime_starts.values).to_datetimeindex()
    cftime_ends = CFTimeIndex(cftime_ends.values).to_datetimeindex()

    # cftime resolution goes down to microsecond only, code below corrects
    # that to allow for comparison with pandas datetime
    cftime_ends += np.timedelta64(999, "ns")
    if hasattr(da_datetime, "_full_index"):
        datetime_starts = da_datetime._full_index.to_period(freq).start_time
        datetime_ends = da_datetime._full_index.to_period(freq).end_time
    else:
        datetime_starts = (
            da_datetime.groupers[0].group_as_index.to_period(freq).start_time
        )
        datetime_ends = da_datetime.groupers[0].group_as_index.to_period(freq).end_time
    assert_array_equal(cftime_starts, datetime_starts)
    assert_array_equal(cftime_ends, datetime_ends)


@pytest.mark.parametrize("typ", ["pd", "xr"])
def test_time_bnds_irregular(typ):
    """Test time_bnds for irregular `middle of the month` time series."""
    if typ == "xr":
        start = xr.cftime_range("1990-01-01", periods=24, freq="MS")
        # Well. xarray string parsers do not support sub-second resolution, but cftime does.
        end = xr.cftime_range(
            "1990-01-01T23:59:59", periods=24, freq="M"
        ) + pd.Timedelta(0.999999, "s")
    elif typ == "pd":
        start = pd.date_range("1990-01-01", periods=24, freq="MS")
        end = pd.date_range("1990-01-01 23:59:59.999999999", periods=24, freq="M")

    time = start + (end - start) / 2

    bounds = time_bnds(time, freq="M")
    bs = bounds.isel(bnds=0)
    be = bounds.isel(bnds=1)

    assert_array_equal(bs, start)
    assert_array_equal(be, end)


@pytest.mark.parametrize("use_dask", [True, False])
def test_percentile_doy(tas_series, use_dask):
    tas = tas_series(np.arange(365), start="1/1/2001")
    if use_dask:
        tas = tas.chunk(dict(time=10))
    tas = xr.concat((tas, tas), "dim0")
    p1 = percentile_doy(tas, window=5, per=50)
    assert p1.sel(dayofyear=3, dim0=0).data == 2
    assert p1.attrs["units"] == "K"


@pytest.mark.parametrize("use_dask", [True, False])
def test_percentile_doy_nan(tas_series, use_dask):
    tas = tas_series(np.arange(365), start="1/1/2001")
    if use_dask:
        tas = tas.chunk(dict(time=10))
    tas = tas.where(tas.time.dt.dayofyear != 2)
    tas = xr.concat((tas, tas), "dim0")
    pnan = percentile_doy(tas, window=5, per=50)
    assert pnan.sel(dayofyear=3, dim0=0).data == 2.5
    assert pnan.attrs["units"] == "K"


@pytest.mark.parametrize("use_dask", [True, False])
def test_percentile_doy_no_copy(tas_series, use_dask):
    tas = tas_series(np.arange(365), start="1/1/2001")
    if use_dask:
        tas = tas.chunk(dict(time=10))
    tas = xr.concat((tas, tas), "dim0")
    original_tas = tas.copy(deep=True)
    p1 = percentile_doy(tas, window=5, per=50, copy=False)
    assert p1.sel(dayofyear=3, dim0=0).data == 2
    assert p1.attrs["units"] == "K"
    assert not np.testing.assert_array_equal(original_tas, tas)


def test_percentile_doy_invalid():
    tas = xr.DataArray(
        [0, 1],
        dims=("time",),
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="H")},
    )
    with pytest.raises(ValueError):
        percentile_doy(tas)


@pytest.mark.parametrize(
    "freqA,op,freqB,exp",
    [
        ("D", ">", "H", True),
        ("2YS", "<=", "QS-DEC", False),
        ("4W", "==", "3W", False),
        ("24H", "==", "D", True),
    ],
)
def test_compare_offsets(freqA, op, freqB, exp):
    assert compare_offsets(freqA, op, freqB) is exp


def test_adjust_doy_360_to_366():
    source = xr.DataArray(np.arange(360), coords=[np.arange(1, 361)], dims="dayofyear")
    time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")

    out = adjust_doy_calendar(source, target)

    assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
    assert out.sel(dayofyear=366) == source.sel(dayofyear=360)


def test_adjust_doy__max_93_to_max_94():
    # GIVEN
    source = xr.DataArray(np.arange(92), coords=[np.arange(152, 244)], dims="dayofyear")
    time = xr.cftime_range("2000-06-01", periods=92, freq="D", calendar="all_leap")
    target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")
    # WHEN
    out = adjust_doy_calendar(source, target)
    # THEN
    assert out[0].dayofyear == 153
    assert out[0] == source[0]
    assert out[-1].dayofyear == 244
    assert out[-1] == source[-1]


def test_adjust_doy__leap_to_noleap_djf():
    # GIVEN
    leap_source = xr.DataArray(
        np.arange(92),
        coords=[np.concatenate([np.arange(1, 61), np.arange(335, 367)])],
        dims="dayofyear",
    )
    time = xr.cftime_range("2000-12-01", periods=91, freq="D", calendar="noleap")
    no_leap_target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")
    # WHEN
    out = adjust_doy_calendar(leap_source, no_leap_target)
    # THEN
    assert out[0].dayofyear == 1
    assert out[0] == leap_source[0]
    assert 366 not in out.dayofyear
    assert out[-1].dayofyear == 365
    assert out[-1] == leap_source[-1]


def test_adjust_doy_366_to_360():
    source = xr.DataArray(np.arange(366), coords=[np.arange(1, 367)], dims="dayofyear")
    time = xr.cftime_range("2000", periods=360, freq="D", calendar="360_day")
    target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")

    out = adjust_doy_calendar(source, target)

    assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
    assert out.sel(dayofyear=360) == source.sel(dayofyear=366)


@pytest.mark.parametrize(
    "file,cal,maxdoy",
    [
        (
            (
                "CanESM2_365day",
                "pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
            ),
            "noleap",
            365,
        ),
        (
            (
                "HadGEM2-CC_360day",
                "pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
            ),
            "360_day",
            360,
        ),
        (("NRCANdaily", "nrcan_canada_daily_pr_1990.nc"), "default", 366),
    ],
)
def test_get_calendar(file, cal, maxdoy, open_dataset):
    with open_dataset(os.path.join(*file)) as ds:
        out_cal = get_calendar(ds)
        assert cal == out_cal
        assert max_doy[cal] == maxdoy


@pytest.mark.parametrize(
    "obj,cal",
    [
        ([pd.Timestamp.now()], "default"),
        (pd.Timestamp.now(), "default"),
        (cftime.DatetimeAllLeap(2000, 1, 1), "all_leap"),
        (np.array([cftime.DatetimeNoLeap(2000, 1, 1)]), "noleap"),
        (xr.cftime_range("2000-01-01", periods=4, freq="D"), "standard"),
    ],
)
def test_get_calendar_nonxr(obj, cal):
    assert get_calendar(obj) == cal


@pytest.mark.parametrize("obj", ["astring", {"a": "dict"}, lambda x: x])
def test_get_calendar_errors(obj):
    with pytest.raises(ValueError, match="Calendar could not be inferred from object"):
        get_calendar(obj)


@pytest.mark.parametrize(
    "source,target,target_as_str,freq",
    [
        ("standard", "noleap", True, "D"),
        ("noleap", "default", True, "D"),
        ("noleap", "all_leap", False, "D"),
        ("proleptic_gregorian", "noleap", False, "4H"),
        ("default", "noleap", True, "4H"),
    ],
)
def test_convert_calendar(source, target, target_as_str, freq):
    src = xr.DataArray(
        date_range("2004-01-01", "2004-12-31", freq=freq, calendar=source),
        dims=("time",),
        name="time",
    )
    da_src = xr.DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )
    tgt = xr.DataArray(
        date_range("2004-01-01", "2004-12-31", freq=freq, calendar=target),
        dims=("time",),
        name="time",
    )

    conv = convert_calendar(da_src, target if target_as_str else tgt)

    assert get_calendar(conv) == target

    if target_as_str and max_doy[source] < max_doy[target]:
        assert conv.size == src.size
    elif not target_as_str:
        assert conv.size == tgt.size

        assert conv.isnull().sum() == max(max_doy[target] - max_doy[source], 0)


@pytest.mark.parametrize(
    "source,target,freq",
    [
        ("standard", "360_day", "D"),
        ("360_day", "default", "D"),
        ("proleptic_gregorian", "360_day", "4H"),
    ],
)
@pytest.mark.parametrize("align_on", ["date", "year"])
def test_convert_calendar_360_days(source, target, freq, align_on):
    src = xr.DataArray(
        date_range("2004-01-01", "2004-12-30", freq=freq, calendar=source),
        dims=("time",),
        name="time",
    )
    da_src = xr.DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )

    conv = convert_calendar(da_src, target, align_on=align_on)

    assert get_calendar(conv) == target

    if align_on == "date":
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        )
    elif target == "360_day":
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29],
        )
    else:
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 29, 30, 30, 31, 30, 30, 31, 30, 31, 29, 31],
        )
    if source == "360_day" and align_on == "year":
        assert conv.size == 360 if freq == "D" else 360 * 4
    else:
        assert conv.size == 359 if freq == "D" else 359 * 4


def test_convert_calendar_360_days_random():
    da_std = xr.DataArray(
        np.linspace(0, 1, 366 * 2),
        dims=("time",),
        coords={
            "time": date_range(
                "2004-01-01", "2004-12-31T23:59:59", freq="12H", calendar="default"
            )
        },
    )
    da_360 = xr.DataArray(
        np.linspace(0, 1, 360 * 2),
        dims=("time",),
        coords={
            "time": date_range(
                "2004-01-01", "2004-12-30T23:59:59", freq="12H", calendar="360_day"
            )
        },
    )

    conv = convert_calendar(da_std, "360_day", align_on="random")
    assert get_calendar(conv) == "360_day"
    assert conv.size == 720
    conv2 = convert_calendar(da_std, "360_day", align_on="random")
    assert (conv != conv2).any()

    conv = convert_calendar(da_360, "default", align_on="random")
    assert get_calendar(conv) == "default"
    assert conv.size == 720
    assert np.datetime64("2004-02-29") not in conv.time
    conv2 = convert_calendar(da_360, "default", align_on="random")
    assert (conv2 != conv).any()

    conv = convert_calendar(da_360, "noleap", align_on="random", missing=np.NaN)
    conv = conv.where(conv.isnull(), drop=True)
    nandoys = conv.time.dt.dayofyear[::2]
    assert all(nandoys < np.array([74, 147, 220, 293, 366]))
    assert all(nandoys > np.array([0, 73, 146, 219, 292]))


@pytest.mark.parametrize(
    "source,target,freq",
    [
        ("standard", "noleap", "D"),
        ("noleap", "default", "4H"),
        ("noleap", "all_leap", "M"),
        ("360_day", "noleap", "D"),
        ("noleap", "360_day", "D"),
    ],
)
def test_convert_calendar_missing(source, target, freq):
    src = xr.DataArray(
        date_range(
            "2004-01-01",
            "2004-12-31" if source != "360_day" else "2004-12-30",
            freq=freq,
            calendar=source,
        ),
        dims=("time",),
        name="time",
    )
    da_src = xr.DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )
    out = convert_calendar(da_src, target, missing=np.nan, align_on="date")
    assert xr.infer_freq(out.time) == freq
    if source == "360_day":
        assert out.time[-1].dt.day == 31


@pytest.mark.parametrize(
    "source,target",
    [
        ("standard", "noleap"),
        ("noleap", "default"),
        ("standard", "360_day"),
        ("360_day", "standard"),
        ("noleap", "all_leap"),
        ("360_day", "noleap"),
    ],
)
def test_interp_calendar(source, target):
    src = xr.DataArray(
        date_range("2004-01-01", "2004-07-30", freq="D", calendar=source),
        dims=("time",),
        name="time",
    )
    tgt = xr.DataArray(
        date_range("2004-01-01", "2004-07-30", freq="D", calendar=target),
        dims=("time",),
        name="time",
    )
    da_src = xr.DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )
    conv = interp_calendar(da_src, tgt)

    assert conv.size == tgt.size
    assert get_calendar(conv) == target

    np.testing.assert_almost_equal(conv.max(), 1, 2)
    assert conv.min() == 0


@pytest.mark.parametrize(
    "inp,calout",
    [
        (
            xr.DataArray(
                date_range("2004-01-01", "2004-01-10", freq="D"),
                dims=("time",),
                name="time",
            ),
            "standard",
        ),
        (date_range("2004-01-01", "2004-01-10", freq="D"), "standard"),
        (
            xr.DataArray(date_range("2004-01-01", "2004-01-10", freq="D")).values,
            "standard",
        ),
        (date_range("2004-01-01", "2004-01-10", freq="D").values, "standard"),
        (date_range("2004-01-01", "2004-01-10", freq="D", calendar="julian"), "julian"),
    ],
)
def test_ensure_cftime_array(inp, calout):
    out = ensure_cftime_array(inp)
    assert get_calendar(out) == calout


@pytest.mark.parametrize(
    "year,calendar,exp",
    [
        (2004, "standard", 366),
        (2004, "noleap", 365),
        (2004, "all_leap", 366),
        (1500, "default", 365),
        (1500, "standard", 366),
        (1500, "proleptic_gregorian", 365),
        (2030, "360_day", 360),
    ],
)
def test_days_in_year(year, calendar, exp):
    assert days_in_year(year, calendar) == exp


@pytest.mark.parametrize(
    "source_cal, exp180",
    [
        ("standard", 0.49180328),
        ("default", 0.49180328),
        ("noleap", 0.49315068),
        ("all_leap", 0.49180328),
        ("360_day", 0.5),
        (None, 0.49180328),
    ],
)
def test_datetime_to_decimal_year(source_cal, exp180):
    times = xr.DataArray(
        date_range(
            "2004-01-01", "2004-12-30", freq="D", calendar=source_cal or "default"
        ),
        dims=("time",),
        name="time",
    )
    decy = datetime_to_decimal_year(times, calendar=source_cal)
    np.testing.assert_almost_equal(decy[180] - 2004, exp180)


def test_clim_mean_doy(tas_series):
    arr = tas_series(np.ones(365 * 10))
    mean, stddev = climatological_mean_doy(arr, window=1)

    assert "dayofyear" in mean.coords
    np.testing.assert_array_equal(mean.values, 1)
    np.testing.assert_array_equal(stddev.values, 0)

    arr = tas_series(np.arange(365 * 3), start="1/1/2001")
    mean, stddev = climatological_mean_doy(arr, window=3)

    np.testing.assert_array_equal(mean[1:-1], np.arange(365, 365 * 2)[1:-1])
    np.testing.assert_array_almost_equal(stddev[1:-1], 298.0223, 4)


def test_doy_to_days_since():
    # simple test
    time = date_range("2020-07-01", "2022-07-01", freq="AS-JUL")
    da = xr.DataArray(
        [190, 360, 3],
        dims=("time",),
        coords={"time": time},
        attrs={"is_dayofyear": 1, "calendar": "default"},
    )

    out = doy_to_days_since(da)
    np.testing.assert_array_equal(out, [7, 178, 186])

    assert out.attrs["units"] == "days after 07-01"
    assert "is_dayofyear" not in out.attrs

    da2 = days_since_to_doy(out)
    xr.testing.assert_identical(da, da2)

    out = doy_to_days_since(da, start="07-01")
    np.testing.assert_array_equal(out, [7, 178, 186])

    # other calendar
    out = doy_to_days_since(da, calendar="noleap")
    assert out.attrs["calendar"] == "noleap"
    np.testing.assert_array_equal(out, [8, 178, 186])

    da2 = days_since_to_doy(out)  # calendar read from attribute
    da2.attrs.pop("calendar")  # drop for identicality
    da.attrs.pop("calendar")  # drop for identicality
    xr.testing.assert_identical(da, da2)

    # with start
    time = date_range("2020-12-31", "2022-12-31", freq="Y")
    da = xr.DataArray(
        [190, 360, 3],
        dims=("time",),
        coords={"time": time},
        name="da",
        attrs={"is_dayofyear": 1, "calendar": "default"},
    )

    out = doy_to_days_since(da, start="01-02")
    np.testing.assert_array_equal(out, [188, 358, 1])

    da2 = days_since_to_doy(out)  # start read from attribute
    assert da2.name == da.name
    xr.testing.assert_identical(da, da2)

    # finer freq
    time = date_range("2020-01-01", "2020-03-01", freq="MS")
    da = xr.DataArray(
        [15, 33, 66],
        dims=("time",),
        coords={"time": time},
        name="da",
        attrs={"is_dayofyear": 1, "calendar": "default"},
    )

    out = doy_to_days_since(da)
    assert out.attrs["units"] == "days after time coordinate"
    np.testing.assert_array_equal(out, [14, 1, 5])

    da2 = days_since_to_doy(out)  # start read from attribute
    xr.testing.assert_identical(da, da2)


@pytest.mark.parametrize(
    "freq,em,eb,es,ea",
    [
        ("4AS-JUL", 4, "A", True, "JUL"),
        ("M", 1, "M", False, None),
        ("YS", 1, "A", True, "JAN"),
        ("3A", 3, "A", False, "DEC"),
        ("D", 1, "D", True, None),
        ("3W", 21, "D", True, None),
    ],
)
def test_parse_offset_valid(freq, em, eb, es, ea):
    m, b, s, a = parse_offset(freq)
    assert m == em
    assert b == eb
    assert s is es
    assert a == ea


def test_parse_offset_invalid():
    # This error actually comes from pandas, but why not test it anyway
    with pytest.raises(ValueError, match="Invalid frequency"):
        parse_offset("day")


@pytest.mark.parametrize(
    "m,b,s,a,exp",
    [
        (1, "A", True, None, "AS-JAN"),
        (2, "Q", False, "DEC", "2Q-DEC"),
        (1, "D", False, None, "D"),
    ],
)
def test_construct_offset(m, b, s, a, exp):
    assert construct_offset(m, b, s, a) == exp


@pytest.mark.parametrize(
    "inputs,join,expected",
    [
        (["noleap", "standard"], "outer", "standard"),
        (["noleap", "standard"], "inner", "noleap"),
        (["default", "default"], "outer", "default"),
        (["all_leap", "default"], "inner", "standard"),
    ],
)
def test_common_calendars(inputs, join, expected):
    assert expected == common_calendar(inputs, join=join)
