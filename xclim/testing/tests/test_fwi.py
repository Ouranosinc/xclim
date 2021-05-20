import numpy as np
import pytest
import xarray as xr

from xclim import atmos
from xclim.core.units import convert_units_to
from xclim.indices.fwi import (
    _day_length,
    _day_length_factor,
    _drought_code,
    _duff_moisture_code,
    _fine_fuel_moisture_code,
    _overwintering_drought_code,
    build_up_index,
    fire_season,
    fire_weather_index,
    fire_weather_ufunc,
    initial_spread_index,
)
from xclim.indices.run_length import run_bounds
from xclim.testing import open_dataset

fwi_url = "FWI/cffdrs_test_fwi.nc"


def test_fine_fuel_moisture_code():
    fwi_data = open_dataset(fwi_url)
    ffmc = np.full(fwi_data.time.size + 1, np.nan)
    ffmc[0] = 85
    for i, t in enumerate(fwi_data.time):
        ffmc[i + 1] = _fine_fuel_moisture_code(
            fwi_data.sel(time=t).tas.values,
            fwi_data.sel(time=t).pr.values,
            fwi_data.sel(time=t).ws.values,
            fwi_data.sel(time=t).rh.values,
            ffmc[i],
        )

    np.testing.assert_allclose(ffmc[1:], fwi_data.ffmc.isel(test=0), rtol=1e-6)


def test_duff_moisture_code():
    fwi_data = open_dataset(fwi_url)
    dmc = np.full(fwi_data.time.size + 1, np.nan)
    dmc[0] = 6
    for i, t in enumerate(fwi_data.time):
        dmc[i + 1] = _duff_moisture_code(
            fwi_data.sel(time=t).tas.values,
            fwi_data.sel(time=t).pr.values,
            fwi_data.sel(time=t).rh.values,
            fwi_data.sel(time=t).time.dt.month.values,
            fwi_data.lat.values,
            dmc[i],
        )

    np.testing.assert_allclose(dmc[1:], fwi_data.dmc.isel(test=0), rtol=1e-6)


def test_drought_code():
    fwi_data = open_dataset(fwi_url)
    dc = np.full(fwi_data.time.size + 1, np.nan)
    dc[0] = 15
    for i, t in enumerate(fwi_data.time):
        dc[i + 1] = _drought_code(
            fwi_data.sel(time=t).tas.values,
            fwi_data.sel(time=t).pr.values,
            fwi_data.sel(time=t).time.dt.month.values,
            fwi_data.lat.values,
            dc[i],
        )

    np.testing.assert_allclose(dc[1:], fwi_data.dc.isel(test=0), rtol=1e-6)


def test_initial_spread_index():
    fwi_data = open_dataset(fwi_url)
    isi = np.full(fwi_data.time.size, np.nan)
    for i, t in enumerate(fwi_data.time):
        isi[i] = initial_spread_index(
            fwi_data.sel(time=t).ws.values,
            fwi_data.sel(time=t).isel(test=0).ffmc.values,
        )
    np.testing.assert_allclose(isi, fwi_data.isi.isel(test=0), rtol=1e-6)


def test_build_up_index():
    fwi_data = open_dataset(fwi_url)
    bui = np.full(fwi_data.time.size, np.nan)
    for i, t in enumerate(fwi_data.time):
        bui[i] = build_up_index(
            fwi_data.sel(time=t).isel(test=0).dmc.values,
            fwi_data.sel(time=t).isel(test=0).dc.values,
        )
    np.testing.assert_allclose(bui, fwi_data.bui.isel(test=0), rtol=1e-6)


@pytest.mark.parametrize(
    "inputs,exp",
    [
        ([300, 110, 0.75, 0.75, 15], 109.4657),
        ([300, 110, 1.0, 0.9, 15], 16.35315),
        ([100, 50, 0.75, 0.75, 15], 105.176),
        ([1, 550, 0.75, 0.75, 10], 10),
    ],
)
def test_overwintering_drought_code(inputs, exp):
    wDC = _overwintering_drought_code(*inputs)
    np.testing.assert_allclose(wDC, exp, rtol=1e-6)


def test_fire_weather_index():
    fwi_data = open_dataset(fwi_url)
    fwi = np.full(fwi_data.time.size, np.nan)
    for i, t in enumerate(fwi_data.time):
        fwi[i] = fire_weather_index(
            fwi_data.sel(time=t).isel(test=0).isi.values,
            fwi_data.sel(time=t).isel(test=0).bui.values,
        )
    np.testing.assert_allclose(fwi, fwi_data.fwi.isel(test=0), rtol=1e-6)


def test_day_length():
    assert _day_length(44, 1) == 6.5


def test_day_lengh_factor():
    assert _day_length_factor(44, 1) == -1.6


def test_fire_weather_indicator():
    fwi_data = open_dataset(fwi_url)
    dc, dmc, ffmc, isi, bui, fwi = atmos.fire_weather_indexes(
        tas=fwi_data.tas,
        pr=fwi_data.pr,
        rh=fwi_data.rh,
        ws=fwi_data.ws,
        lat=fwi_data.lat,
    )

    dc2, dmc2, ffmc2, isi2, bui2, fwi2 = atmos.fire_weather_indexes(
        tas=fwi_data.tas,
        pr=fwi_data.pr,
        rh=fwi_data.rh,
        ws=fwi_data.ws,
        lat=fwi_data.lat,
        ffmc0=ffmc[-1],
        dmc0=dmc[-1],
        dc0=dc[-1],
    )
    xr.testing.assert_allclose(dc, fwi_data.dc.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(dmc, fwi_data.dmc.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(ffmc, fwi_data.ffmc.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(isi, fwi_data.isi.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(bui, fwi_data.bui.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(fwi, fwi_data.fwi.isel(test=0), rtol=1e-6)
    xr.testing.assert_allclose(dc2, fwi_data.dc.isel(test=1), rtol=1e-6)
    xr.testing.assert_allclose(dmc2, fwi_data.dmc.isel(test=1), rtol=1e-6)
    xr.testing.assert_allclose(ffmc2, fwi_data.ffmc.isel(test=1), rtol=1e-6)
    xr.testing.assert_allclose(isi2, fwi_data.isi.isel(test=1), rtol=1e-6)
    xr.testing.assert_allclose(bui2, fwi_data.bui.isel(test=1), rtol=1e-6)
    xr.testing.assert_allclose(fwi2, fwi_data.fwi.isel(test=1), rtol=1e-6)


def test_fire_weather_ufunc_overwintering():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    ds = ds.assign(
        rh=atmos.relative_humidity_from_dewpoint(ds=ds),
        ws=convert_units_to(atmos.wind_speed_from_vector(ds=ds)[0], "km/h"),
        tas=convert_units_to(ds.tas, "degC"),
        pr=convert_units_to(ds.pr, "mm/d"),
    )
    season_mask_all = fire_season(ds.tas, method="WF93", temp_end_thresh="4 degC")
    season_mask_all_LA08 = fire_season(ds.tas, snd=ds.swe, method="LA08")
    season_mask_yr = fire_season(ds.tas, method="WF93", freq="YS")

    # Mask is computed correctly and parameters are passed
    # season not passed so computed on the fly
    out1 = fire_weather_ufunc(
        tas=ds.tas,
        pr=ds.pr,
        lat=ds.lat,
        season_method="WF93",
        overwintering=False,
        temp_end_thresh=4,
        indexes=["DC"],
    )
    np.testing.assert_array_equal(out1["season_mask"], season_mask_all)

    out2 = fire_weather_ufunc(
        tas=ds.tas,
        pr=ds.pr,
        snd=ds.swe,
        lat=ds.lat,
        season_method="LA08",
        overwintering=True,
        indexes=["DC"],
    )
    np.testing.assert_array_equal(out2["season_mask"], season_mask_all_LA08)

    # Overwintering
    # Get last season's DC (from previous comp) and mask Saskatoon and Victoria
    dc0 = out2["DC"].ffill("time").isel(time=-1).where([True, True, True, False, False])
    winter_pr = out2["winter_pr"]

    out3 = fire_weather_ufunc(
        tas=ds.tas,
        pr=ds.pr,
        lat=ds.lat,
        winter_pr=winter_pr,
        season_mask=season_mask_yr,
        dc0=dc0,
        overwintering=True,
        indexes=["DC"],
    )
    np.testing.assert_allclose(
        out3["winter_pr"].isel(location=0), 261.27353647, rtol=1e-6
    )
    np.testing.assert_array_equal(out3["DC"].notnull(), season_mask_yr)


def test_fire_weather_ufunc_errors(tas_series, pr_series, rh_series, ws_series):
    tas = tas_series(np.ones(100), start="2017-01-01")
    pr = pr_series(np.ones(100), start="2017-01-01")
    rh = rh_series(np.ones(100), start="2017-01-01")
    ws = ws_series(np.ones(100), start="2017-01-01")

    snd = xr.full_like(tas, 0)
    lat = xr.full_like(tas.isel(time=0), 45)
    DC0 = xr.full_like(tas.isel(time=0), np.nan)  # noqa
    DMC0 = xr.full_like(tas.isel(time=0), np.nan)  # noqa
    FFMC0 = xr.full_like(tas.isel(time=0), np.nan)  # noqa

    # Test invalid combination
    with pytest.raises(TypeError):
        fire_weather_ufunc(
            tas=tas,
            pr=pr,
            rh=rh,
            lat=lat,
            dc0=DC0,
            indexes=["DC", "ISI"],
        )

    # Test missing arguments
    with pytest.raises(TypeError):
        fire_weather_ufunc(
            tas=tas,
            pr=pr,
            dc0=DC0,
            indexes=["DC"],  # lat=lat,
        )

    with pytest.raises(TypeError):
        fire_weather_ufunc(
            tas=tas,
            pr=pr,
            lat=lat,
            dc0=DC0,
            indexes=["DC"],
            season_method="LA08",
        )

    # Test output is complete + dask
    out = fire_weather_ufunc(
        tas=tas.chunk(),
        pr=pr.chunk(),
        lat=lat.chunk(),
        dc0=DC0,
        indexes=["DC"],
    )
    assert len(out.keys()) == 1
    out["DC"].load()

    out = fire_weather_ufunc(
        tas=tas,
        pr=pr,
        rh=rh,
        ws=ws,
        lat=lat,
        snd=snd,
        dc0=DC0,
        dmc0=DMC0,
        ffmc0=FFMC0,
        indexes=["DSR"],
    )

    assert len(out.keys()) == 7


@pytest.mark.parametrize(
    "key,kwargs",
    [
        ("id1_default", {}),
        ("id2_default", {}),
        ("id3_default", {}),
        (
            "id1_start10_end3",
            {"temp_start_thresh": "283.15 K", "temp_end_thresh": "3 degC"},
        ),
        (
            "id1_start10_end3_YS",
            {
                "temp_start_thresh": "283.15 K",
                "temp_end_thresh": "3 degC",
                "freq": "YS",
            },
        ),
    ],
)
def test_fire_season_R(key, kwargs):
    expected = _get_cffdrs_fire_season(key)
    in_ds = open_dataset("FWI/cffdrs_test_wDC.nc")
    nid = int(key[2])

    mask = fire_season(tas=in_ds.where(in_ds.id == nid, drop=True).tasmax, **kwargs)
    bounds = run_bounds(mask, dim="time", coord=True)
    np.testing.assert_array_equal(bounds, expected)


def _get_cffdrs_fire_season(key=None):
    def to_xr(arr):
        return xr.DataArray(
            np.array(arr, dtype=np.datetime64), dims=("bounds", "events")
        )

    if key:
        return to_xr(cffdrs_fire_season[key])
    return {key: to_xr(arr) for key, arr in cffdrs_fire_season.items()}


# The following were computed with cffdrs 1.8.18, on the test_wDC data.
cffdrs_fire_season = {
    "id1_default": [["2013-03-15", "2014-03-14"], ["2013-11-23", "2014-11-14"]],
    "id2_default": [["1980-04-20", "1981-05-15"], ["1980-10-16", "1981-10-14"]],
    "id3_default": [["1999-05-02", "2000-06-16"], ["1999-10-20", "2000-10-07"]],
    "id1_start10_end3": [
        ["2013-03-12", "2014-03-09", "2014-12-13"],
        ["2013-11-23", "2014-11-15", "2014-12-18"],
    ],
    "id1_start10_end3_YS": [["2013-03-12", "2014-03-09"], ["2013-11-23", "2014-11-15"]],
}
