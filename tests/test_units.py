import numpy as np
import pytest
import xarray as xr

from xclim import indices
from xclim.core.units import _check_units
from xclim.core.units import convert_units_to
from xclim.core.units import pint2cfunits
from xclim.core.units import pint_multiply
from xclim.core.units import units
from xclim.core.units import units2pint


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
            out = indices.tx_days_above(tas, 30)

        out1 = indices.tx_days_above(tas, "30 degC")
        out2 = indices.tx_days_above(tas, "303.15 K")
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out, out2)
        assert out1.name == tas.name

    def test_fraction(self):
        out = convert_units_to(xr.DataArray([10], attrs={"units": "%"}), "")
        assert out == 0.1


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

        u = units2pint("2 kg m-2 s-1")
        assert (str(u)) == "kilogram / meter ** 2 / second"

        u = units2pint("%")
        assert str(u) == "percent"

    def test_pint_multiply(self, pr_series):
        a = pr_series([1, 2, 3])
        out = pint_multiply(a, 1 * units.days)
        assert out[0] == 1 * 60 * 60 * 24
        assert out.units == "kg m-2"


class TestCheckUnits:
    def test_basic(self):
        _check_units("%", "[]")
        _check_units("pct", "[]")
        _check_units("mm/day", "[precipitation]")
        _check_units("mm/s", "[precipitation]")
        _check_units("kg/m2/s", "[precipitation]")
        _check_units("kg/m2", "[length]")
        _check_units("cms", "[discharge]")
        _check_units("m3/s", "[discharge]")
        _check_units("m/s", "[speed]")
        _check_units("km/h", "[speed]")

        with pytest.raises(AttributeError):
            _check_units("mm", "[precipitation]")

        with pytest.raises(AttributeError):
            _check_units("m3", "[discharge]")
