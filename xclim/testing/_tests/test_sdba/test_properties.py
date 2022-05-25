from __future__ import annotations

import numpy as np
import pytest

from xclim import sdba
from xclim.testing import open_dataset


def test_mean():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1980"), location="Vancouver")
        .pr
    )

    out_year = sdba.properties.mean(sim)
    np.testing.assert_array_almost_equal(out_year.values, [3.0016028e-05])

    out_season = sdba.properties.mean(sim, group="time.season")
    np.testing.assert_array_almost_equal(
        out_season.values, [4.6115547e-05, 1.7220482e-05, 2.8805329e-05, 2.825359e-05]
    )

    assert out_season.long_name == "Mean"


def test_var():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1980"), location="Vancouver")
        .pr
    )

    out_year = sdba.properties.var(sim)
    np.testing.assert_array_almost_equal(out_year.values, [2.5884779e-09])

    out_season = sdba.properties.var(sim, group="time.season")
    np.testing.assert_array_almost_equal(
        out_season.values, [3.9270796e-09, 1.2538864e-09, 1.9057025e-09, 2.8776632e-09]
    )
    assert out_season.long_name == "Variance"
    assert out_season.units == "kg^2 m-4 s-2"


def test_skewness():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1980"), location="Vancouver")
        .pr
    )

    out_year = sdba.properties.skewness(sim)
    np.testing.assert_array_almost_equal(out_year.values, [2.8497460898513745])

    out_season = sdba.properties.skewness(sim, group="time.season")
    np.testing.assert_array_almost_equal(
        out_season.values,
        [2.036650744163691, 3.7909534745807147, 2.416590445325826, 3.3521301798559566],
    )
    assert out_season.long_name == "Skewness"
    assert out_season.units == ""


def test_quantile():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1980"), location="Vancouver")
        .pr
    )

    out_year = sdba.properties.quantile(sim, q=0.2)
    np.testing.assert_array_almost_equal(out_year.values, [2.8109431013945154e-07])

    out_season = sdba.properties.quantile(sim, group="time.season", q=0.2)
    np.testing.assert_array_almost_equal(
        out_season.values,
        [
            1.5171653330980917e-06,
            9.822543773907455e-08,
            1.8135805248675763e-07,
            4.135342521749408e-07,
        ],
    )
    assert out_season.long_name == "Quantile 0.2"


def test_spell_length_distribution():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .pr
    )
    tmean = (
        sdba.properties.spell_length_distribution(sim, op="<", group="time.month")
        .sel(month=1)
        .values
    )
    tmax = (
        sdba.properties.spell_length_distribution(
            sim, op="<", group="time.month", stat="max"
        )
        .sel(month=1)
        .values
    )
    tmin = (
        sdba.properties.spell_length_distribution(
            sim, op="<", group="time.month", stat="min"
        )
        .sel(month=1)
        .values
    )

    np.testing.assert_array_almost_equal([tmean, tmax, tmin], [2.44127, 10, 1])

    simt = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .tasmax
    )
    tmean = sdba.properties.spell_length_distribution(
        simt, op=">=", group="time.month", method="quantile", thresh=0.9
    ).sel(month=6)
    tmax = (
        sdba.properties.spell_length_distribution(
            simt, op=">=", group="time.month", stat="max", method="quantile", thresh=0.9
        )
        .sel(month=6)
        .values
    )
    tmin = (
        sdba.properties.spell_length_distribution(
            simt, op=">=", group="time.month", stat="min", method="quantile", thresh=0.9
        )
        .sel(month=6)
        .values
    )

    np.testing.assert_array_almost_equal([tmean.values, tmax, tmin], [3.0, 6, 1])

    with pytest.raises(
        ValueError,
        match="percentile is not a valid method. Choose 'amount' or 'quantile'.",
    ):
        sdba.properties.spell_length_distribution(simt, method="percentile")

    assert tmean.long_name == "mean of spell length when input variable >= quantile 0.9"


def test_acf():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .pr
    )
    out = sdba.properties.acf(sim, lag=1, group="time.month").sel(month=1)
    np.testing.assert_array_almost_equal(out.values, [0.11242357313756905])

    with pytest.raises(
        ValueError,
        match="Grouping on year is not allowed for this function.",
    ):
        sdba.properties.acf(sim, group="time")

    assert out.long_name == "lag-1 autocorrelation"
    assert out.units == ""


def test_annual_cycle():
    simt = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .tasmax
    )
    amp = sdba.properties.annual_cycle_amplitude(simt, amplitude_type="absolute")
    relamp = sdba.properties.annual_cycle_amplitude(simt, amplitude_type="relative")
    phase = sdba.properties.annual_cycle_phase(simt)

    np.testing.assert_array_almost_equal(
        [amp.values, relamp.values, phase.values],
        [34.039806, 11.793684020675501, 165.33333333333334],
    )
    with pytest.raises(
        ValueError,
        match="Grouping on season is not allowed for this function.",
    ):
        sdba.properties.annual_cycle_amplitude(simt, group="time.season")

    with pytest.raises(
        ValueError,
        match="Grouping on month is not allowed for this function.",
    ):
        sdba.properties.annual_cycle_phase(simt, group="time.month")

    assert amp.long_name == "absolute amplitude of the annual cycle"
    assert phase.long_name == "Phase of the annual cycle"
    assert amp.units == "delta_degC"
    assert relamp.units == "%"
    assert phase.units == ""


def test_corr_btw_var():
    simt = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .tasmax
    )
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .pr
    )
    pc = sdba.properties.corr_btw_var(simt, sim, corr_type="Pearson")
    pp = sdba.properties.corr_btw_var(
        simt, sim, corr_type="Pearson", output="pvalue"
    ).values
    sc = sdba.properties.corr_btw_var(simt, sim).values
    sp = sdba.properties.corr_btw_var(simt, sim, output="pvalue").values
    sc_jan = (
        sdba.properties.corr_btw_var(simt, sim, group="time.month").sel(month=1).values
    )
    sim[0] = np.nan
    pc_nan = sdba.properties.corr_btw_var(sim, simt, corr_type="Pearson").values

    np.testing.assert_array_almost_equal(
        [pc.values, pp, sc, sp, sc_jan, pc_nan],
        [
            -0.20849051347480407,
            3.2160438749049577e-12,
            -0.3449358561881698,
            5.97619379511559e-32,
            0.28329503745038936,
            np.nan,
        ],
    )
    assert pc.long_name == "Pearson correlation coefficient"
    assert pc.units == ""

    with pytest.raises(
        ValueError,
        match="pear is not a valid type. Choose 'Pearson' or 'Spearman'.",
    ):
        sdba.properties.corr_btw_var(sim, simt, group="time", corr_type="pear")


def test_relative_frequency():
    sim = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .pr
    )

    test = sdba.properties.relative_frequency(sim, thresh="25 mm d-1", op=">=")
    testjan = (
        sdba.properties.relative_frequency(
            sim, thresh="25 mm d-1", op=">=", group="time.month"
        )
        .sel(month=1)
        .values
    )
    np.testing.assert_array_almost_equal(
        [test.values, testjan], [0.0045662100456621, 0.010752688172043012]
    )
    assert (
        test.long_name == "Relative frequency of days with input variable >= 25 mm d-1"
    )
    assert test.units == ""


def test_trend():
    simt = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "1952"), location="Vancouver")
        .tasmax
    )
    slope = sdba.properties.trend(simt).values
    pvalue = sdba.properties.trend(simt, output="pvalue").values
    np.testing.assert_array_almost_equal(
        [slope, pvalue], [-1.33720000e-01, 0.154605951862107], 4
    )

    slope = sdba.properties.trend(simt, group="time.month").sel(month=1)
    pvalue = (
        sdba.properties.trend(simt, output="pvalue", group="time.month")
        .sel(month=1)
        .values
    )
    np.testing.assert_array_almost_equal(
        [slope.values, pvalue], [0.8254349999999988, 0.6085783558202086], 4
    )

    assert slope.long_name == "slope of the interannual linear trend"
    assert slope.units == "K/year"


def test_return_value():
    simt = (
        open_dataset("sdba/CanESM2_1950-2100.nc")
        .sel(time=slice("1950", "2010"), location="Vancouver")
        .tasmax
    )
    out_y = sdba.properties.return_value(simt)
    out_djf = (
        sdba.properties.return_value(simt, op="min", group="time.season")
        .sel(season="DJF")
        .values
    )

    np.testing.assert_array_almost_equal([out_y.values, out_djf], [313.154, 278.072], 3)
    assert out_y.long_name == "20-year max return level"
