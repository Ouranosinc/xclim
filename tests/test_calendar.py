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
    convert_doy,
    days_since_to_doy,
    doy_to_days_since,
    ensure_cftime_array,
    get_calendar,
    max_doy,
    parse_offset,
    percentile_doy,
    stack_periods,
    time_bnds,
    unstack_periods,
)


@pytest.fixture(params=[dict(start="2004-01-01T12:07:01", periods=27, freq="3MS")], ids=["3MS"])
def time_range_kwargs(request):
    return request.param


@pytest.fixture()
def datetime_index(time_range_kwargs):
    return xr.date_range(**time_range_kwargs)


@pytest.fixture()
def cftime_index(time_range_kwargs):
    return xr.date_range(use_cftime=True, **time_range_kwargs)


def da(index):
    return xr.DataArray(np.arange(100.0, 100.0 + index.size), coords=[index], dims=["time"])


@pytest.mark.parametrize("freq", ["6480h", "302431min", "23144781s"])
def test_time_bnds(freq, datetime_index, cftime_index):
    da_datetime = da(datetime_index).resample(time=freq)
    out_time = da_datetime.mean()
    da_cftime = da(cftime_index).resample(time=freq)

    cftime_bounds = time_bnds(da_cftime, freq=freq)
    cftime_starts = cftime_bounds.isel(bnds=0)
    cftime_ends = cftime_bounds.isel(bnds=1)
    cftime_starts = CFTimeIndex(cftime_starts.values).to_datetimeindex()
    cftime_ends = CFTimeIndex(cftime_ends.values).to_datetimeindex()

    # cftime resolution goes down to microsecond only, code below corrects
    # that to allow for comparison with pandas datetime
    cftime_ends += np.timedelta64(999, "ns")
    out_periods = out_time.indexes["time"].to_period(freq)
    assert_array_equal(cftime_starts, out_periods.start_time)
    assert_array_equal(cftime_ends, out_periods.end_time)


@pytest.mark.parametrize("use_cftime", [True, False])
def test_time_bnds_irregular(use_cftime):
    """Test time_bnds for irregular `middle of the month` time series."""
    start = xr.date_range("1990-01-01", periods=24, freq="MS", use_cftime=use_cftime)
    # Well. xarray string parsers do not support sub-second resolution, but cftime does.
    end = xr.date_range("1990-01-01T23:59:59", periods=24, freq="ME", use_cftime=use_cftime) + pd.Timedelta(
        0.999999999, "s"
    )

    time = start + (end - start) / 2

    bounds = time_bnds(time, freq="ME")
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
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="h")},
    )
    with pytest.raises(ValueError):
        percentile_doy(tas)


@pytest.mark.parametrize(
    "freqA,op,freqB,exp",
    [
        ("D", ">", "h", True),
        ("2YS", "<=", "QS-DEC", False),
        ("4W", "==", "3W", False),
        ("24h", "==", "D", True),
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
    time = xr.date_range("2000-06-01", periods=92, freq="D", calendar="all_leap")
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
    time = xr.date_range("2000-12-01", periods=91, freq="D", calendar="noleap")
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
    time = xr.date_range("2000", periods=360, freq="D", calendar="360_day")
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
        (("NRCANdaily", "nrcan_canada_daily_pr_1990.nc"), "proleptic_gregorian", 366),
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
        ([pd.Timestamp.now()], "standard"),
        (pd.Timestamp.now(), "standard"),
        (cftime.DatetimeAllLeap(2000, 1, 1), "all_leap"),
        (np.array([cftime.DatetimeNoLeap(2000, 1, 1)]), "noleap"),
        (xr.date_range("2000-01-01", periods=4, freq="D"), "standard"),
    ],
)
def test_get_calendar_nonxr(obj, cal):
    assert get_calendar(obj) == cal


@pytest.mark.parametrize("obj", ["astring", {"a": "dict"}, lambda x: x])
def test_get_calendar_errors(obj):
    with pytest.raises(ValueError, match="Calendar could not be inferred from object"):
        get_calendar(obj)


def test_convert_calendar_and_doy():
    doy = xr.DataArray(
        [31, 32, 336, 364.23, 365],
        dims=("time",),
        coords={"time": xr.date_range("2000-01-01", periods=5, freq="YS", calendar="noleap")},
        attrs={"is_dayofyear": 1, "calendar": "noleap"},
    )
    out = convert_doy(doy, target_cal="360_day").convert_calendar("360_day", align_on="date")
    # out = convert_calendar(doy, "360_day", align_on="date", doy=True)
    np.testing.assert_allclose(out, [30.575342, 31.561644, 331.39726, 359.240548, 360.0])
    assert out.time.dt.calendar == "360_day"
    out = convert_doy(doy, target_cal="360_day", align_on="date").convert_calendar("360_day", align_on="date")
    np.testing.assert_array_equal(out, [np.nan, 31, 332, 360.23, np.nan])
    assert out.time.dt.calendar == "360_day"


@pytest.mark.parametrize(
    "inp,calout",
    [
        (
            xr.DataArray(
                xr.date_range("2004-01-01", "2004-01-10", freq="D"),
                dims=("time",),
                name="time",
            ),
            "standard",
        ),
        (xr.date_range("2004-01-01", "2004-01-10", freq="D"), "standard"),
        (
            xr.DataArray(xr.date_range("2004-01-01", "2004-01-10", freq="D")).values,
            "standard",
        ),
        (xr.date_range("2004-01-01", "2004-01-10", freq="D").values, "standard"),
        (
            xr.date_range("2004-01-01", "2004-01-10", freq="D", calendar="julian"),
            "julian",
        ),
    ],
)
def test_ensure_cftime_array(inp, calout):
    out = ensure_cftime_array(inp)
    assert get_calendar(out) == calout


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
    time = xr.date_range("2020-07-01", "2022-07-01", freq="YS-JUL")
    da = xr.DataArray(
        [190, 360, 3],
        dims=("time",),
        coords={"time": time},
        attrs={"is_dayofyear": 1, "calendar": "standard"},
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
    time = xr.date_range("2020-12-31", "2022-12-31", freq="YE")
    da = xr.DataArray(
        [190, 360, 3],
        dims=("time",),
        coords={"time": time},
        name="da",
        attrs={"is_dayofyear": 1, "calendar": "proleptic_gregorian"},
    )

    out = doy_to_days_since(da, start="01-02")
    np.testing.assert_array_equal(out, [188, 358, 1])

    da2 = days_since_to_doy(out)  # start read from attribute
    assert da2.name == da.name
    xr.testing.assert_identical(da, da2)

    # finer freq
    time = xr.date_range("2020-01-01", "2020-03-01", freq="MS")
    da = xr.DataArray(
        [15, 33, 66],
        dims=("time",),
        coords={"time": time},
        name="da",
        attrs={"is_dayofyear": 1, "calendar": "proleptic_gregorian"},
    )

    out = doy_to_days_since(da)
    assert out.attrs["units"] == "days after time coordinate"
    np.testing.assert_array_equal(out, [14, 1, 5])

    da2 = days_since_to_doy(out)  # start read from attribute
    xr.testing.assert_identical(da, da2)


@pytest.mark.parametrize(
    "freq,em,eb,es,ea",
    [
        ("4YS-JUL", 4, "Y", True, "JUL"),
        ("ME", 1, "M", False, None),
        ("YS", 1, "Y", True, "JAN"),
        ("3YE", 3, "Y", False, "DEC"),
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
        (1, "Y", True, None, "YS-JAN"),
        (2, "Q", False, "DEC", "2QE-DEC"),
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


def test_convert_doy():
    doy = xr.DataArray(
        [31, 32, 336, 364.23, 365],
        dims=("time",),
        coords={"time": xr.date_range("2000-01-01", periods=5, freq="YS", calendar="noleap")},
        attrs={"is_dayofyear": 1, "calendar": "noleap"},
    )

    out = convert_doy(doy, "360_day", align_on="date")
    np.testing.assert_array_equal(out, [np.nan, 31, 332, 360.23, np.nan])
    assert out.time.dt.calendar == "noleap"
    out = convert_doy(doy, "360_day", align_on="year")
    np.testing.assert_allclose(out, [30.575342, 31.561644, 331.39726, 359.240548, 360.0])

    doy = xr.DataArray(
        [31, 200.48, 190, 60, 300.54],
        dims=("time",),
        coords={"time": xr.date_range("2000-01-01", periods=5, freq="YS-JUL", calendar="standard")},
        attrs={"is_dayofyear": 1, "calendar": "standard"},
    ).expand_dims(lat=[10, 20, 30])

    out = convert_doy(doy, "noleap", align_on="date")
    np.testing.assert_array_equal(out.isel(lat=0), [31, 200.48, 190, np.nan, 299.54])
    out = convert_doy(doy, "noleap", align_on="year")
    np.testing.assert_allclose(out.isel(lat=0), [31.0, 200.48, 190.0, 59.83607, 299.71885])


@pytest.mark.parametrize("calendar", ["standard", None])
@pytest.mark.parametrize(
    "w,s,m,f,ss",
    [(30, 10, None, "YS", 0), (3, 1, None, "QS-DEC", 60), (6, None, None, "MS", 0)],
)
def test_stack_periods(tas_series, calendar, w, s, m, f, ss):
    da = tas_series(np.arange(365 * 50), start="2000-01-01", calendar=calendar)

    da_stck = stack_periods(da, window=w, stride=s, min_length=m, freq=f, align_days=False)

    assert "period_length" in da_stck.coords

    da2 = unstack_periods(da_stck)

    xr.testing.assert_identical(da2, da.isel(time=slice(ss, da2.time.size + ss)))


def test_stack_periods_special(tas_series):
    da = tas_series(np.arange(365 * 48 + 12), calendar="standard", start="2004-01-01")

    with pytest.raises(ValueError, match="unaligned day-of-year"):
        stack_periods(da)

    da = da.convert_calendar("noleap")

    da_stck = stack_periods(da, dim="horizon")
    np.testing.assert_array_equal(da_stck.horizon_length, 10950)

    with pytest.raises(ValueError, match="can't find the window"):
        unstack_periods(da_stck)

    da2 = unstack_periods(da_stck.drop_vars("horizon_length"), dim="horizon")
    xr.testing.assert_identical(da2, da.isel(time=slice(0, da2.time.size)))
