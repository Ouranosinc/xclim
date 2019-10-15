#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package.
#
# We want to tests multiple things here:
#  - that data results are correct
#  - that metadata is correct and complete
#  - that missing data are handled appropriately
#  - that various calendar formats and supported
#  - that non-valid input frequencies or holes in the time series are detected
#
#
# For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it,
# saving the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as
# a first line of defense.
import glob
import os

import cftime
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import __version__
from xclim import atmos
from xclim import ensembles
from xclim import indices
from xclim import subset
from xclim import utils
from xclim.utils import adjust_doy_calendar
from xclim.utils import daily_downsampler
from xclim.utils import format_kwargs
from xclim.utils import Indicator
from xclim.utils import infer_doy_max
from xclim.utils import parse_doc
from xclim.utils import percentile_doy
from xclim.utils import pint2cfunits
from xclim.utils import units
from xclim.utils import units2pint
from xclim.utils import walk_map

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


class TestDailyDownsampler:
    def test_std_calendar(self):

        # standard calendar
        # generate test DataArray
        time_std = pd.date_range("2000-01-01", "2000-12-31", freq="D")
        da_std = xr.DataArray(np.arange(time_std.size), coords=[time_std], dims="time")

        for freq in "YS MS QS-DEC".split():
            resampler = da_std.resample(time=freq)
            grouper = daily_downsampler(da_std, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert the values of resampler and grouper are the same
            assert np.allclose(x1.values, x2.values)

    @pytest.mark.skip
    def test_365_day(self):

        # 365_day calendar
        # generate test DataArray
        units = "days since 2000-01-01 00:00"
        time_365 = cftime.num2date(np.arange(0, 1 * 365), units, "365_day")
        da_365 = xr.DataArray(
            np.arange(time_365.size), coords=[time_365], dims="time", name="data"
        )
        units = "days since 2001-01-01 00:00"
        time_std = cftime.num2date(np.arange(0, 1 * 365), units, "standard")
        da_std = xr.DataArray(
            np.arange(time_std.size), coords=[time_std], dims="time", name="data"
        )

        for freq in "YS MS QS-DEC".split():
            resampler = da_std.resample(time=freq)
            grouper = daily_downsampler(da_365, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_365.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert the values of resampler of non leap year with standard calendar
            # is identical to grouper
            assert np.allclose(x1.values, x2.values)

    def test_360_days(self):
        #
        # 360_day calendar
        #
        units = "days since 2000-01-01 00:00"
        time_360 = cftime.num2date(np.arange(0, 360), units, "360_day")
        da_360 = xr.DataArray(
            np.arange(1, time_360.size + 1), coords=[time_360], dims="time", name="data"
        )

        for freq in "YS MS QS-DEC".split():
            grouper = daily_downsampler(da_360, freq=freq)

            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_360.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert grouper values == expected values
            target_year = 180.5
            target_month = [n * 30 + 15.5 for n in range(0, 12)]
            target_season = [30.5] + [(n - 1) * 30 + 15.5 for n in [4, 7, 10, 12]]
            target = {"YS": target_year, "MS": target_month, "QS-DEC": target_season}[
                freq
            ]
            assert np.allclose(x2.values, target)


class UniIndTemp(Indicator):
    identifier = "tmin"
    var_name = "tmin{thresh}"
    units = "K"
    long_name = "{freq} mean surface temperature"
    standard_name = "{freq} mean temperature"
    cell_methods = "time: mean within {freq}"

    @staticmethod
    def compute(da, thresh=0.0, freq="YS"):
        """Docstring"""
        out = da
        out -= thresh
        return out.resample(time=freq).mean(keep_attrs=True)


class UniIndPr(Indicator):
    identifier = "prmax"
    units = "mm/s"
    context = "hydro"

    @staticmethod
    def compute(da, freq):
        """Docstring"""
        return da.resample(time=freq).mean(keep_attrs=True)


class TestIndicator:
    def test_attrs(self, tas_series):
        import datetime as dt

        a = tas_series(np.arange(360.0))
        ind = UniIndTemp()
        txm = ind(a, thresh=5, freq="YS")
        assert txm.cell_methods == "time: mean within days time: mean within years"
        assert "{:%Y-%m-%d %H}".format(dt.datetime.now()) in txm.attrs["history"]
        assert "tmin(da, thresh=5, freq='YS')" in txm.attrs["history"]
        assert "xclim version: {}.".format(__version__) in txm.attrs["history"]
        assert txm.name == "tmin5"

    def test_temp_unit_conversion(self, tas_series):
        a = tas_series(np.arange(360.0))
        ind = UniIndTemp()
        txk = ind(a, freq="YS")

        ind.units = "degC"
        txc = ind(a, freq="YS")

        np.testing.assert_array_almost_equal(txk, txc + 273.15)

    def test_json(self, pr_series):
        ind = UniIndPr()
        meta = ind.json()

        expected = {
            "identifier",
            "var_name",
            "units",
            "long_name",
            "standard_name",
            "cell_methods",
            "keywords",
            "abstract",
            "parameters",
            "description",
            "history",
            "references",
            "comment",
            "notes",
        }

        assert set(meta.keys()).issubset(expected)

    def test_signature(self):
        from inspect import signature

        ind = UniIndTemp()
        assert signature(ind.compute) == signature(ind.__call__)

    def test_doc(self):
        ind = UniIndTemp()
        assert ind.__call__.__doc__ == ind.compute.__doc__

    def test_delayed(self):
        fn = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

        # Load dataset as a dask array
        ds = xr.open_dataset(fn, chunks={"time": 10}, cache=True)

        tx = UniIndTemp()
        txk = tx(ds.tasmax)

        # Check that the calculations are delayed
        assert isinstance(txk.data, dask.array.core.Array)

        # Same with unit conversion
        tx.required_units = ("C",)
        tx.units = "C"
        txc = tx(ds.tasmax)

        assert isinstance(txc.data, dask.array.core.Array)

    def test_identifier(self):
        with pytest.warns(UserWarning):
            UniIndPr(identifier="t_{}")

    def test_formatting(self, pr_series):
        out = atmos.wetdays(
            pr_series(np.arange(366)), thresh=1.0 * units.mm / units.day
        )
        assert out.attrs["long_name"] == "Number of wet days (precip >= 1 mm/day)"

        out = atmos.wetdays(
            pr_series(np.arange(366)), thresh=1.5 * units.mm / units.day
        )
        assert out.attrs["long_name"] == "Number of wet days (precip >= 1.5 mm/day)"


class TestKwargs:
    def test_format_kwargs(self):
        attrs = dict(
            standard_name="tx_min",
            long_name="Minimum of daily maximum temperature",
            cell_methods="time: minimum within {freq}",
        )

        format_kwargs(attrs, {"freq": "YS"})
        assert attrs["cell_methods"] == "time: minimum within years"


class TestParseDoc:
    def test_simple(self):
        parse_doc(indices.tg_mean.__doc__)


class TestPercentileDOY:
    def test_simple(self, tas_series):
        tas = tas_series(np.arange(365), start="1/1/2001")
        p1 = percentile_doy(tas, window=5, per=0.5)
        assert p1.sel(dayofyear=3).data == 2
        assert p1.attrs["units"] == "K"


class TestAdjustDoyCalendar:
    def test_360_to_366(self):
        source = xr.DataArray(
            np.arange(360), coords=[np.arange(1, 361)], dims="dayofyear"
        )
        time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        target = xr.DataArray(np.arange(len(time)), coords=[time], dims="time")

        out = adjust_doy_calendar(source, target)

        assert out.sel(dayofyear=1) == source.sel(dayofyear=1)
        assert out.sel(dayofyear=366) == source.sel(dayofyear=360)

    def test_infer_doy_max(self):
        fn = os.path.join(
            TESTS_DATA,
            "CanESM2_365day",
            "pr_day_CanESM2_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        )
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 365

        fn = os.path.join(
            TESTS_DATA,
            "HadGEM2-CC_360day",
            "pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc",
        )
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 360

        fn = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")
        with xr.open_dataset(fn) as ds:
            assert infer_doy_max(ds) == 366


class TestWalkMap:
    def test_simple(self):
        d = {"a": -1, "b": {"c": -2}}
        o = walk_map(d, lambda x: 0)
        assert o["a"] == 0
        assert o["b"]["c"] == 0


class TestUnits:
    def test_temperature(self):
        assert 4 * units.d == 4 * units.day
        Q_ = units.Quantity
        assert Q_(1, units.C) == Q_(1, units.degC)

    def test_hydro(self):
        with units.context("hydro"):
            q = 1 * units.kg / units.m ** 2 / units.s
            assert q.to("mm/day") == q.to("mm/d")

    def test_lat_lon(self):
        assert 100 * units.degreeN == 100 * units.degree

    def test_pcic(self):
        with units.context("hydro"):
            fu = units.parse_units("kilogram / d / meter ** 2")
            tu = units.parse_units("mm/day")
            np.isclose(1 * fu, 1 * tu)

    def test_dimensionality(self):
        with units.context("hydro"):
            fu = 1 * units.parse_units("kg / m**2 / s")
            tu = 1 * units.parse_units("mm / d")
            fu.to("mmday")
            tu.to("mmday")


class TestConvertUnitsTo:
    def test_deprecation(self, tas_series):
        with pytest.warns(FutureWarning):
            out = utils.convert_units_to(0, units.K)
            assert out == 273.15

            out = utils.convert_units_to(10, units.mm / units.day, context="hydro")
            assert out == 10

        with pytest.warns(FutureWarning):
            tas = tas_series(np.arange(365), start="1/1/2001")
            out = indices.tx_days_above(tas, 30)
            out1 = indices.tx_days_above(tas, "30 degC")
            out2 = indices.tx_days_above(tas, "303.15 K")
            np.testing.assert_array_equal(out, out1)
            np.testing.assert_array_equal(out, out2)


class TestUnitConversion:
    def test_pint2cfunits(self):
        u = units("mm/d")
        assert pint2cfunits(u.units) == "mm d-1"

    def test_units2pint(self, pr_series):
        u = units2pint(pr_series([1, 2]))
        assert (str(u)) == "kilogram / meter ** 2 / second"
        assert pint2cfunits(u) == "kg m-2 s-1"

        u = units2pint("m^3 s-1")
        assert str(u) == "meter ** 3 / second"
        assert pint2cfunits(u) == "m^3 s-1"

        u = units2pint("2 kg m-2 s-1")
        assert (str(u)) == "kilogram / meter ** 2 / second"

    def test_pint_multiply(self, pr_series):
        a = pr_series([1, 2, 3])
        out = utils.pint_multiply(a, 1 * units.days)
        assert out[0] == 1 * 60 * 60 * 24
        assert out.units == "kg m-2"


class TestCheckUnits:
    def test_basic(self):
        utils._check_units("mm/day", "[precipitation]")
        utils._check_units("mm/s", "[precipitation]")
        utils._check_units("kg/m2/s", "[precipitation]")
        utils._check_units("kg/m2", "[length]")
        utils._check_units("cms", "[discharge]")
        utils._check_units("m3/s", "[discharge]")

        with pytest.raises(AttributeError):
            utils._check_units("mm", "[precipitation]")
            utils._check_units("m3", "[discharge]")


class TestThresholdCount:
    def test_simple(self, tas_series):
        ts = tas_series(np.arange(365))
        out = utils.threshold_count(ts, "<", 50, "Y")
        np.testing.assert_array_equal(out, [50, 0])
