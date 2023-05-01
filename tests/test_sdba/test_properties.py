from __future__ import annotations

import numpy as np
import pytest

from xclim import sdba
from xclim.core.units import convert_units_to


class TestProperties:
    def test_mean(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1980"), location="Vancouver")
            .pr
        ).load()

        out_year = sdba.properties.mean(sim)
        np.testing.assert_array_almost_equal(out_year.values, [3.0016028e-05])

        out_season = sdba.properties.mean(sim, group="time.season")
        np.testing.assert_array_almost_equal(
            out_season.values,
            [4.6115547e-05, 1.7220482e-05, 2.8805329e-05, 2.825359e-05],
        )

        assert out_season.long_name.startswith("Mean")

    def test_var(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1980"), location="Vancouver")
            .pr
        ).load()

        out_year = sdba.properties.var(sim)
        np.testing.assert_array_almost_equal(out_year.values, [2.5884779e-09])

        out_season = sdba.properties.var(sim, group="time.season")
        np.testing.assert_array_almost_equal(
            out_season.values,
            [3.9270796e-09, 1.2538864e-09, 1.9057025e-09, 2.8776632e-09],
        )
        assert out_season.long_name.startswith("Variance")
        assert out_season.units == "kg^2 m-4 s-2"

    def test_skewness(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1980"), location="Vancouver")
            .pr
        ).load()

        out_year = sdba.properties.skewness(sim)
        np.testing.assert_array_almost_equal(out_year.values, [2.8497460898513745])

        out_season = sdba.properties.skewness(sim, group="time.season")
        np.testing.assert_array_almost_equal(
            out_season.values,
            [
                2.036650744163691,
                3.7909534745807147,
                2.416590445325826,
                3.3521301798559566,
            ],
        )
        assert out_season.long_name.startswith("Skewness")
        assert out_season.units == ""

    def test_quantile(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1980"), location="Vancouver")
            .pr
        ).load()

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
        assert out_season.long_name.startswith("Quantile 0.2")

    def test_spell_length_distribution(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .pr
        ).load()

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
        ).load()

        tmean = sdba.properties.spell_length_distribution(
            simt, op=">=", group="time.month", method="quantile", thresh=0.9
        ).sel(month=6)
        tmax = (
            sdba.properties.spell_length_distribution(
                simt,
                op=">=",
                group="time.month",
                stat="max",
                method="quantile",
                thresh=0.9,
            )
            .sel(month=6)
            .values
        )
        tmin = (
            sdba.properties.spell_length_distribution(
                simt,
                op=">=",
                group="time.month",
                stat="min",
                method="quantile",
                thresh=0.9,
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

        assert (
            tmean.long_name
            == "Average of spell length distribution when the variable is >= the quantile 0.9."
        )

    def test_acf(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .pr
        ).load()

        out = sdba.properties.acf(sim, lag=1, group="time.month").sel(month=1)
        np.testing.assert_array_almost_equal(out.values, [0.11242357313756905])

        with pytest.raises(ValueError, match="Grouping period year is not allowed for"):
            sdba.properties.acf(sim, group="time")

        assert out.long_name.startswith("Lag-1 autocorrelation")
        assert out.units == ""

    def test_annual_cycle(self, open_dataset):
        simt = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .tasmax
        ).load()

        amp = sdba.properties.annual_cycle_amplitude(simt)
        relamp = sdba.properties.relative_annual_cycle_amplitude(simt)
        phase = sdba.properties.annual_cycle_phase(simt)

        np.testing.assert_allclose(
            [amp.values, relamp.values, phase.values],
            [16.74645996, 5.802083, 167],
            rtol=1e-6,
        )
        with pytest.raises(
            ValueError,
            match="Grouping period season is not allowed for property",
        ):
            sdba.properties.annual_cycle_amplitude(simt, group="time.season")

        with pytest.raises(
            ValueError,
            match="Grouping period month is not allowed for property",
        ):
            sdba.properties.annual_cycle_phase(simt, group="time.month")

        assert amp.long_name.startswith("Absolute amplitude of the annual cycle")
        assert phase.long_name.startswith("Phase of the annual cycle")
        assert amp.units == "delta_degC"
        assert relamp.units == "%"
        assert phase.units == ""

    def test_annual_range(self, open_dataset):
        simt = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .tasmax
        ).load()

        # Initial annual cycle was this with window = 1
        amp = sdba.properties.mean_annual_range(simt, window=1)
        relamp = sdba.properties.mean_annual_relative_range(simt, window=1)
        phase = sdba.properties.mean_annual_phase(simt, window=1)

        np.testing.assert_allclose(
            [amp.values, relamp.values, phase.values],
            [34.039806, 11.793684020675501, 165.33333333333334],
        )

        amp = sdba.properties.mean_annual_range(simt)
        relamp = sdba.properties.mean_annual_relative_range(simt)
        phase = sdba.properties.mean_annual_phase(simt)

        np.testing.assert_array_almost_equal(
            [amp.values, relamp.values, phase.values],
            [18.715261, 6.480101, 181.6666667],
        )
        with pytest.raises(
            ValueError,
            match="Grouping period season is not allowed for property",
        ):
            sdba.properties.mean_annual_range(simt, group="time.season")

        with pytest.raises(
            ValueError,
            match="Grouping period month is not allowed for property",
        ):
            sdba.properties.mean_annual_phase(simt, group="time.month")

        assert amp.long_name.startswith("Average annual absolute amplitude")
        assert phase.long_name.startswith("Average annual phase")
        assert amp.units == "delta_degC"
        assert relamp.units == "%"
        assert phase.units == ""

    def test_corr_btw_var(self, open_dataset):
        simt = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .tasmax
        ).load()

        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .pr
        ).load()

        pc = sdba.properties.corr_btw_var(simt, sim, corr_type="Pearson")
        pp = sdba.properties.corr_btw_var(
            simt, sim, corr_type="Pearson", output="pvalue"
        ).values
        sc = sdba.properties.corr_btw_var(simt, sim).values
        sp = sdba.properties.corr_btw_var(simt, sim, output="pvalue").values
        sc_jan = (
            sdba.properties.corr_btw_var(simt, sim, group="time.month")
            .sel(month=1)
            .values
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
                -0.2090292,
            ],
        )
        assert pc.long_name == "Pearson correlation coefficient"
        assert pc.units == ""

        with pytest.raises(
            ValueError,
            match="pear is not a valid type. Choose 'Pearson' or 'Spearman'.",
        ):
            sdba.properties.corr_btw_var(sim, simt, group="time", corr_type="pear")

    def test_relative_frequency(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .pr
        ).load()

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
        assert test.long_name == "Relative frequency of values >= 25 mm d-1."
        assert test.units == ""

    def test_transition(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .pr
        ).load()

        test = sdba.properties.transition_probability(
            da=sim, initial_op="<", final_op=">="
        )

        np.testing.assert_array_almost_equal([test.values], [0.14076782449725778])
        assert (
            test.long_name
            == "Transition probability of values < 1 mm d-1 to values >= 1 mm d-1."
        )
        assert test.units == ""

    def test_trend(self, open_dataset):
        simt = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "1952"), location="Vancouver")
            .tasmax
        ).load()

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

        assert slope.long_name.startswith("Slope of the interannual linear trend")
        assert slope.units == "K/year"

    def test_return_value(self, open_dataset):
        simt = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1950", "2010"), location="Vancouver")
            .tasmax
        ).load()

        out_y = sdba.properties.return_value(simt)

        out_djf = (
            sdba.properties.return_value(simt, op="min", group="time.season")
            .sel(season="DJF")
            .values
        )

        np.testing.assert_array_almost_equal(
            [out_y.values, out_djf], [313.154, 278.072], 3
        )
        assert out_y.long_name.startswith("20-year maximal return level")

    @pytest.mark.slow
    def test_spatial_correlogram(self, open_dataset):
        # This also tests sdba.utils._pairwise_spearman and sdba.nbutils._pairwise_haversine_and_bins
        # Test 1, does it work with 1D data?
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1981", "2010"))
            .tasmax
        ).load()

        out = sdba.properties.spatial_correlogram(sim, dims=["location"], bins=3)
        np.testing.assert_allclose(out, [-1, np.nan, 0], atol=1e-6)

        # Test 2, not very exhaustive, this is more of a detect-if-we-break-it test.
        sim = open_dataset("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc").tasmax
        out = sdba.properties.spatial_correlogram(
            sim.isel(lon=slice(0, 50)), dims=["lon", "lat"], bins=20
        )
        np.testing.assert_allclose(
            out[:5],
            [0.95099902, 0.83028772, 0.66874473, 0.48893958, 0.30915054],
        )
        np.testing.assert_allclose(
            out.distance[:5],
            [26.543199, 67.716227, 108.889254, 150.062282, 191.23531],
            rtol=5e-07,
        )

    @pytest.mark.slow
    def test_decorrelation_length(self, open_dataset):
        sim = (
            open_dataset("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
            .tasmax.isel(lon=slice(0, 5), lat=slice(0, 1))
            .load()
        )

        out = sdba.properties.decorrelation_length(
            sim, dims=["lat", "lon"], bins=10, radius=30
        )
        np.testing.assert_allclose(
            out[0],
            [4.5, 4.5, 4.5, 4.5, 10.5],
        )

    def test_get_measure(self, open_dataset):
        sim = (
            open_dataset("sdba/CanESM2_1950-2100.nc")
            .sel(time=slice("1981", "2010"), location="Vancouver")
            .pr
        ).load()

        ref = (
            open_dataset("sdba/ahccd_1950-2013.nc")
            .sel(time=slice("1981", "2010"), location="Vancouver")
            .pr
        ).load()

        sim = convert_units_to(sim, ref, context="hydro")
        sim_var = sdba.properties.var(sim)
        ref_var = sdba.properties.var(ref)

        meas = sdba.properties.var.get_measure()(sim_var, ref_var)
        np.testing.assert_allclose(meas, [0.408327], rtol=1e-3)


class TestEOF:
    def test_first_eof(self, open_dataset):
        pytest.importorskip("eofs")
        sim = (
            open_dataset("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
            .tasmax.isel(lon=slice(0, 10), lat=slice(50, 60))
            .load()
        )

        out = sdba.properties.first_eof(sim)
        np.testing.assert_allclose(
            [out.mean(), out.max()], [0.099976, 0.103867], rtol=1e-5
        )
        assert (out.isnull() == sim.isnull().any("time")).all()
