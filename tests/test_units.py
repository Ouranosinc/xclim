from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import pint.errors
import pytest
import xarray as xr
from dask import array as dsk

from xclim import indices, set_options
from xclim.core import Quantified, ValidationError
from xclim.core.units import (
    amount2lwethickness,
    amount2rate,
    check_units,
    convert_units_to,
    declare_relative_units,
    declare_units,
    infer_context,
    infer_sampling_units,
    lwethickness2amount,
    pint2cfattrs,
    pint2cfunits,
    pint_multiply,
    rate2amount,
    str2pint,
    to_agg_units,
    units,
    units2pint,
)


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

    def test_lat_lon(self):
        assert 100 * units.degreeN == 100 * units.degree

    def test_pcic(self):
        with units.context("hydro"):
            fu = units.parse_units("kilogram / d / meter ** 2")
            tu = units.parse_units("mm/day")
            np.isclose(1 * fu, 1 * tu)

    def test_dimensionality(self):
        # Check that the hydro context allows flux to rate conversion
        with units.context("hydro"):
            fu = 1 * units.parse_units("kg / m**2 / s")
            fu.to("mm/day")

    def test_fraction(self):
        q = 5 * units.percent
        assert q.to("dimensionless") == 0.05


class TestConvertUnitsTo:
    test_data = "ERA5/daily_surface_cancities_1990-1993.nc"

    def test_deprecation(self, tas_series):
        with pytest.raises(TypeError):
            convert_units_to(0, units.K)

        with pytest.raises(TypeError):
            convert_units_to(10.0, units.mm / units.day, context="hydro")

        with pytest.raises(TypeError):
            tas = tas_series(np.arange(365), start="1/1/2001")
            out = indices.tx_days_above(tas, 30)  # noqa

    def test_fraction(self):
        out = convert_units_to(xr.DataArray([10], attrs={"units": "%"}), "")
        assert out == 0.1

    def test_lazy(self, pr_series):
        pr = pr_series(np.arange(365), start="1/1/2001").chunk({"time": 100})
        out = convert_units_to(pr, "mm/day", context="hydro")
        assert isinstance(out.data, dsk.Array)

    @pytest.mark.parametrize("alias", [units("Celsius"), units("degC"), units("C"), units("deg_C")])
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

    def test_temperature_difference(self):
        delta = xr.DataArray([2], attrs={"units": "K", "units_metadata": "temperature: difference"})
        out = convert_units_to(source=delta, target="delta_degC")
        assert out == 2
        assert out.attrs["units"] == "degC"

    def test_dataset(self, open_dataset):
        ds = open_dataset(self.test_data)

        out = convert_units_to(ds, {"tas": "degC", "pr": "mm/d"})
        assert out.tas.attrs["units"] == "°C"
        assert out.pr.attrs["units"] == "mm d-1"
        assert out.snd.attrs["units"] == "m"

    def test_dataset_missing(self, open_dataset):
        ds = open_dataset(self.test_data)

        with pytest.raises(KeyError, match="No variable named"):
            convert_units_to(ds, {"nonexistingvariable": "Å / °R"})

    def test_datatree(self, open_dataset):
        ds = open_dataset(self.test_data)

        dt = xr.DataTree.from_dict(
            {
                "": ds.sel(location="Victoria", drop=True),
                "MTL": ds.sel(location="Montréal", drop=True),
                "HAL": ds.sel(location="Halifax", drop=True),
            }
        )
        out = convert_units_to(dt, {"snd": "km", "uas": "pc / yr"})
        assert out.tas.attrs["units"] == "K"
        assert out.uas.attrs["units"] == "pc yr-1"
        assert out.snd.attrs["units"] == "km"


class TestUnitConversion:
    def test_pint2cfunits(self):
        u = units("mm/d")
        assert pint2cfunits(u.units) == "mm d-1"

        u = units("percent")
        assert pint2cfunits(u.units) == "%"

    def test_units2pint(self, pr_series):
        u = units2pint(pr_series([1, 2]))
        assert pint2cfunits(u) == "kg m-2 s-1"

        u = units2pint("m^3 s-1")
        assert pint2cfunits(u) == "m3 s-1"

        u = units2pint("%")
        assert pint2cfunits(u) == "%"

        u = units2pint("1")
        assert pint2cfunits(u) == "1"

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
        check_units("mm/day", "[precipitation]")
        check_units("mm/s", "[precipitation]")
        check_units("kg/m2/s", "[precipitation]")
        check_units("m3/s", "[discharge]")
        check_units("m/s", "[speed]")
        check_units("km/h", "[speed]")
        check_units("degC", "[temperature]")

        with units.context("hydro"):
            check_units("kg/m2", "[length]")

        with set_options(data_validation="raise"):
            with pytest.raises(ValidationError):
                check_units("mm", "[precipitation]")

            with pytest.raises(ValidationError):
                check_units("m3", "[discharge]")

    def test_comparison(self):
        """Check that both units have the same dimensions."""
        check_units("mm/day", "m/hour")

        with pytest.raises(ValidationError):
            check_units("mm/day", "m")

        check_units(
            xr.DataArray([1], attrs={"units": "degC"}),
            xr.DataArray([1], attrs={"units": "degK"}),
        )

        with pytest.raises(ValidationError):
            check_units(xr.DataArray([1], attrs={"units": "degC"}), "2 mm")

        with pytest.raises(ValidationError):
            """There is no context information to know that mm/day is a precipitation unit."""
            check_units("kg/m2/s", "mm/day")

    def test_user_error(self):
        with pytest.raises(ValidationError):
            check_units("deg C", "[temperature]")


def test_rate2amount(pr_series):
    pr = pr_series(np.ones(365 + 366 + 365), start="2019-01-01")

    am_d = rate2amount(pr)
    np.testing.assert_array_equal(am_d, 86400)

    with xr.set_options(keep_attrs=True):
        pr_ms = pr.resample(time="MS").mean()
        pr_m = pr.resample(time="ME").mean()

        am_ms = rate2amount(pr_ms)
        np.testing.assert_array_equal(am_ms[:4], 86400 * np.array([31, 28, 31, 30]))
        am_m = rate2amount(pr_m)
        np.testing.assert_array_equal(am_m[:4], 86400 * np.array([31, 28, 31, 30]))
        np.testing.assert_array_equal(am_ms, am_m)

        pr_ys = pr.resample(time="YS").mean()
        am_ys = rate2amount(pr_ys)

        np.testing.assert_array_equal(am_ys, 86400 * np.array([365, 366, 365]))


@pytest.mark.parametrize("srcfreq, exp", [("h", 3600), ("min", 60), ("s", 1), ("ns", 1e-9)])
def test_rate2amount_subdaily(srcfreq, exp):
    pr = xr.DataArray(
        np.ones(1000),
        dims=("time",),
        coords={"time": xr.date_range("2019-01-01", periods=1000, freq=srcfreq)},
        attrs={"units": "kg m-2 s-1"},
    )
    am = rate2amount(pr)
    np.testing.assert_array_equal(am, exp)


def test_amount2rate(pr_series):
    pr = pr_series(np.ones(365 + 366 + 365), start="2019-01-01")
    am = rate2amount(pr)
    assert am.attrs["standard_name"] == "precipitation_amount"

    np.testing.assert_allclose(amount2rate(am), pr)

    with xr.set_options(keep_attrs=True):
        am_ms = am.resample(time="MS").sum()
        am_m = am.resample(time="ME").sum()

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
    assert swe.attrs["standard_name"] == "lwe_thickness_of_surface_snow_amount"
    np.testing.assert_allclose(swe, 1)

    snw = lwethickness2amount(swe)
    assert snw.attrs["standard_name"] == "surface_snow_amount"


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
    """
    Test that an error is raised when parameters with type Quantified do not declare their dimensions.

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


def test_declare_relative_units():
    def index(
        data: xr.DataArray,
        thresh: Quantified,
        dthreshdt: Quantified,  # noqa: F841
    ):
        return xr.DataArray(1, attrs={"units": "rad"})

    index_relative = declare_relative_units(thresh="<data>", dthreshdt="<data>/[time]")(index)
    assert hasattr(index_relative, "relative_units")

    index_full_mm = declare_units(data="mm")(index_relative)
    assert index_full_mm.in_units == {
        "data": "mm",
        "thresh": "(mm)",
        "dthreshdt": "(mm)/[time]",
    }

    index_full_area = declare_units(data="[area]")(index_relative)
    assert index_full_area.in_units == {
        "data": "[area]",
        "thresh": "([area])",
        "dthreshdt": "([area])/[time]",
    }

    # No failures
    index_full_mm("1 mm", "2 km", "3 mm/s")

    with pytest.raises(ValidationError):
        index_full_mm("1 mm", "2 Pa", "3 mm/s")


@pytest.mark.parametrize(
    "in_u,opfunc,op,exp,exp_u",
    [
        ("m/h", "sum", "integral", 8760, "m"),
        ("m/h", "sum", "sum", 365, "m/h"),
        ("K", "mean", "mean", 1, "K"),
        ("", "sum", "count", 365, "d"),
        ("", "sum", "count", 365, "d"),
        ("kg m-2", "var", "var", 0, "kg2 m-4"),
        (
            "°C",
            "argmax",
            "doymax",
            0,
            "1",
        ),
        (
            "°C",
            "sum",
            "integral",
            365,
            ("degC d", "d degC"),
        ),  # dependent on numpy/pint version
        ("°F", "sum", "integral", 365, "d degF"),  # not sure why the order is different
    ],
)
def test_to_agg_units(in_u, opfunc, op, exp, exp_u):
    da = xr.DataArray(
        np.ones((365,)),
        dims=("time",),
        coords={"time": xr.date_range("1993-01-01", periods=365, freq="D")},
        attrs={"units": in_u},
    )
    if units(in_u).dimensionality == "[temperature]":
        da.attrs["units_metadata"] = "temperature: difference"

    # FIXME: This is emitting warnings from deprecated DataArray.argmax() usage.
    out = to_agg_units(getattr(da, opfunc)(), da, op)
    np.testing.assert_allclose(out, exp)
    if isinstance(exp_u, tuple):
        assert out.attrs["units"] in exp_u
    else:
        assert out.attrs["units"] == exp_u


def test_pint2cfattrs():
    attrs = pint2cfattrs(units.degK, is_difference=True)
    assert attrs == {"units": "K", "units_metadata": "temperature: difference"}

    attrs = pint2cfattrs(units.meter, is_difference=True)
    assert "units_metadata" not in attrs

    attrs = pint2cfattrs(units.delta_degC)
    assert attrs == {"units": "degC", "units_metadata": "temperature: difference"}


def test_temp_difference_rountrip():
    """Test roundtrip of temperature difference units."""
    attrs = {"units": "degC", "units_metadata": "temperature: difference"}
    da = xr.DataArray([1], attrs=attrs)
    pu = units2pint(da)
    # Confirm that we get delta pint units
    assert pu == units.delta_degC

    # and that converting those back to cf attrs gives the same result
    attrs = pint2cfattrs(pu)
    assert attrs == {"units": "degC", "units_metadata": "temperature: difference"}


@pytest.mark.parametrize(
    "freq,expm,expu", [("3D", 3, "d"), ("MS", 1, "month"), ("QS-DEC", 3, "month"), ("W", 1, "week"), ("min", 1, "min")]
)
def test_infer_sampling_units(freq, expm, expu):
    time = xr.date_range("14-04-2025", periods=10, freq=freq)
    da = xr.DataArray(list(range(10)), dims=("time",), coords={"time": time})
    m, u = infer_sampling_units(da)
    assert expm == m
    assert expu == u


def test_infer_sampling_units_errors():
    time = xr.date_range("14-04-2025", periods=10, freq="D")
    da = xr.DataArray(list(range(10)), dims=("time",), coords={"time": time})
    da = da[[0, 1, 5, 6]]
    with pytest.raises(ValueError, match="Unable to find"):
        infer_sampling_units(da)
