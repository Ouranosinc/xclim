from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import pint.errors
import pytest
import xarray as xr
from dask import array as dsk

from xclim import indices, set_options
from xclim.core.units import (
    amount2lwethickness,
    amount2rate,
    check_units,
    convert_units_to,
    declare_units,
    infer_context,
    lwethickness2amount,
    pint2cfunits,
    pint_multiply,
    rate2amount,
    str2pint,
    units,
    units2pint,
)
from xclim.core.utils import Quantified, ValidationError


class TestUnits:
    def test_temperature(self):
        assert 4 * units.d == 4 * units.day
        Q_ = units.Quantity  # noqa
        assert Q_(1, units.C) == Q_(1, units.degC)

    def test_hydro(self):
        with pytest.raises(pint.errors.DimensionalityError):
            convert_units_to("1 kg m-2", "m")

        with units.context("hydro"):
            q = 1 * units.kg / units.m**2 / units.s
            assert q.to("mm/day") == q.to("mm/d")
            assert q.to("mmday").magnitude == 24 * 60**2

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

    def test_fraction(self):
        q = 5 * units.percent
        assert q.to("dimensionless") == 0.05

        q = 5 * units.parse_units("pct")
        assert q.to("dimensionless") == 0.05


class TestConvertUnitsTo:
    def test_deprecation(self, tas_series):
        with pytest.warns(FutureWarning):
            out = convert_units_to(0, units.K)
            assert out == 273.15

        with pytest.warns(FutureWarning):
            out = convert_units_to(10, units.mm / units.day, context="hydro")
            assert out == 10

        with pytest.warns(FutureWarning):
            tas = tas_series(np.arange(365), start="1/1/2001")
            out = indices.tx_days_above(tas, 30)  # noqa

        out1 = indices.tx_days_above(tas, "30 degC")
        out2 = indices.tx_days_above(tas, "303.15 K")
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out, out2)
        assert out1.name == tas.name

    def test_fraction(self):
        out = convert_units_to(xr.DataArray([10], attrs={"units": "%"}), "")
        assert out == 0.1

    def test_lazy(self, pr_series):
        pr = pr_series(np.arange(365), start="1/1/2001").chunk({"time": 100})
        out = convert_units_to(pr, "mm/day", context="hydro")
        assert isinstance(out.data, dsk.Array)

    @pytest.mark.parametrize(
        "alias", [units("Celsius"), units("degC"), units("C"), units("deg_C")]
    )
    def test_temperature_aliases(self, alias):
        assert alias == units("celsius")

    def test_offset_confusion(self):
        out = convert_units_to("10 degC days", "K days")
        assert out == 10

    def test_cf_conversion_amount2lwethickness_error(self):
        # It is not thickness data because the standard name is wrong (absent)
        not_thickness_data = xr.DataArray([1, 2, 3], attrs={"units": "mm"})
        with pytest.raises(pint.errors.DimensionalityError):
            convert_units_to(not_thickness_data, "kg/m**2/day")

    def test_cf_conversion_amount2lwethickness_amount2rate(self):
        thickness_data = xr.DataArray(
            [1, 2, 3],
            coords={"time": pd.date_range("1990-01-01", periods=3, freq="D")},
            dims=["time"],
            attrs={"units": "mm", "standard_name": "thickness_of_rainfall_amount"},
        )
        out = convert_units_to(thickness_data, "kg/m**2/day")
        np.testing.assert_array_almost_equal(out, thickness_data)
        assert out.attrs["units"] == "kg d-1 m-2"  # CF equivalent unit
        assert out.attrs["standard_name"] == "rainfall_flux"


class TestUnitConversion:
    def test_pint2cfunits(self):
        u = units("mm/d")
        assert pint2cfunits(u.units) == "mm d-1"

        u = units("percent")
        assert pint2cfunits(u.units) == "%"

        u = units("pct")
        assert pint2cfunits(u.units) == "%"

    def test_units2pint(self, pr_series):
        u = units2pint(pr_series([1, 2]))
        assert (str(u)) == "kilogram / meter ** 2 / second"
        assert pint2cfunits(u) == "kg m-2 s-1"

        u = units2pint("m^3 s-1")
        assert str(u) == "meter ** 3 / second"
        assert pint2cfunits(u) == "m^3 s-1"

        u = units2pint("kg m-2 s-1")
        assert (str(u)) == "kilogram / meter ** 2 / second"

        u = units2pint("%")
        assert str(u) == "percent"

        u = units2pint("1")
        assert str(u) == "dimensionless"

        u = units2pint("mm s-1")
        assert str(u) == "millimeter / second"

        u = units2pint("degrees_north")
        assert str(u) == "degrees_north"

    def test_pint_multiply(self, pr_series):
        a = pr_series([1, 2, 3])
        out = pint_multiply(a, 1 * units.days)
        assert out[0] == 1 * 60 * 60 * 24
        assert out.units == "kg m-2"

    def test_str2pint(self):
        Q_ = units.Quantity  # noqa
        assert str2pint("-0.78 m") == Q_(-0.78, units="meter")
        assert str2pint("m kg/s") == Q_(1, units="meter kilogram/second")
        assert str2pint("11.8 degC days") == Q_(11.8, units="delta_degree_Celsius days")
        assert str2pint("nan m^2 K^-3").units == Q_(1, units="m²/K³").units


class TestCheckUnits:
    def test_basic(self):
        check_units("%", "[]")
        check_units("pct", "[]")
        check_units("mm/day", "[precipitation]")
        check_units("mm/s", "[precipitation]")
        check_units("kg/m2/s", "[precipitation]")
        check_units("cms", "[discharge]")
        check_units("m3/s", "[discharge]")
        check_units("m/s", "[speed]")
        check_units("km/h", "[speed]")

        with units.context("hydro"):
            check_units("kg/m2", "[length]")

        with set_options(data_validation="raise"):
            with pytest.raises(ValidationError):
                check_units("mm", "[precipitation]")

            with pytest.raises(ValidationError):
                check_units("m3", "[discharge]")

    def test_user_error(self):
        with pytest.raises(ValidationError):
            check_units("deg C", "[temperature]")


def test_rate2amount(pr_series):
    pr = pr_series(np.ones(365 + 366 + 365), start="2019-01-01")

    am_d = rate2amount(pr)
    np.testing.assert_array_equal(am_d, 86400)

    with xr.set_options(keep_attrs=True):
        pr_ms = pr.resample(time="MS").mean()
        pr_m = pr.resample(time="M").mean()

        am_ms = rate2amount(pr_ms)
        np.testing.assert_array_equal(am_ms[:4], 86400 * np.array([31, 28, 31, 30]))
        am_m = rate2amount(pr_m)
        np.testing.assert_array_equal(am_m[:4], 86400 * np.array([31, 28, 31, 30]))
        np.testing.assert_array_equal(am_ms, am_m)

        pr_ys = pr.resample(time="YS").mean()
        am_ys = rate2amount(pr_ys)

        np.testing.assert_array_equal(am_ys, 86400 * np.array([365, 366, 365]))


def test_amount2rate(pr_series):
    pr = pr_series(np.ones(365 + 366 + 365), start="2019-01-01")
    am = rate2amount(pr)
    assert am.attrs["standard_name"] == "precipitation_amount"

    np.testing.assert_allclose(amount2rate(am), pr)

    with xr.set_options(keep_attrs=True):
        am_ms = am.resample(time="MS").sum()
        am_m = am.resample(time="M").sum()

        pr_ms = amount2rate(am_ms)
        np.testing.assert_allclose(pr_ms, 1)
        pr_m = amount2rate(am_m)
        np.testing.assert_allclose(pr_m, 1)

        am_ys = am.resample(time="YS").sum()
        pr_ys = amount2rate(am_ys)
        np.testing.assert_allclose(pr_ys, 1)


def test_amount2lwethickness(snw_series):
    snw = snw_series(np.ones(365), start="2019-01-01")

    swe = amount2lwethickness(snw, out_units="mm")
    swe.attrs["standard_name"] == "lwe_thickness_of_snowfall_amount"
    np.testing.assert_allclose(swe, 1)

    snw = lwethickness2amount(swe)
    snw.attrs["standard_name"] == "snowfall_amount"


@pytest.mark.parametrize(
    "std_name,dim,exp",
    [
        ("precipitation_flux", None, "hydro"),
        ("snowfall_flux", None, "none"),
        ("air_temperature", "[precipitation]", "hydro"),
        (None, None, "none"),
    ],
)
def test_infer_context(std_name, dim, exp):
    assert infer_context(std_name, dim) == exp


def test_declare_units():
    """Test that an error is raised when parameters with type Quantified do not declare their dimensions.
    In this example, `wo` is a Quantified parameter, but does not declare its dimension as [length].
    """

    with pytest.raises(ValueError):

        @declare_units(pr="[precipitation]", evspsblpot="[precipitation]")
        def dryness_index(
            pr: xr.DataArray,
            evspsblpot: xr.DataArray,
            lat: xr.DataArray | str | None = None,
            wo: Quantified = "200 mm",
            freq: str = "YS",
        ) -> xr.DataArray:
            pass
