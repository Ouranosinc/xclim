# pylint: disable=unsubscriptable-object,function-redefined
# Tests for the Indicator objects
from __future__ import annotations

import json
from inspect import signature

import dask
import numpy as np
import pytest
import xarray as xr

import xclim
from xclim import atmos
from xclim.compute import tg_mean
from xclim.core import VARIABLES, InputKind, MissingVariableError, Quantified
from xclim.core.calendar import select_time
from xclim.core.indicator import Daily, IndexWrapper, Indicator, ResamplingIndicator, registry
from xclim.core.units import convert_units_to, declare_units, units
from xclim.testing import list_input_variables

try:
    from xarray import DataTree
except ImportError:
    DataTree = False


@declare_units(da="[temperature]", thresh="[temperature]")
def uniindtemp_compute(
    da: xr.DataArray,
    thresh: Quantified = "0.0 degC",
    freq: str = "YS",
    method: str = "injected",
):
    """Docstring"""
    out = da - convert_units_to(thresh, da)
    out = out.resample(time=freq).mean()
    out.attrs["units"] = da.units
    return out


uniIndTemp = Daily(
    realm="atmos",
    identifier="test.tmin",
    module="test",
    attrs=[
        dict(
            var_name="tmin{thresh}",
            units="K",
            long_name="{freq} mean surface temperature with {thresh} threshold.",
            standard_name="{freq} mean temperature",
            cell_methods="time: mean within {freq:noun}",
            another_attr="With a value.",
        )
    ],
    compute=uniindtemp_compute,
    parameters={"method": "injected"},
)


@declare_units(da="[precipitation]")
def uniindpr_compute(da: xr.DataArray, freq: str):
    """Docstring"""
    return da.resample(time=freq).mean(keep_attrs=True)


uniIndPr = Daily(
    realm="atmos",
    identifier="test.prmax",
    attrs=[dict(units="mm/s")],
    context="hydro",
    module="test",
    compute=uniindpr_compute,
)


@declare_units(da="[temperature]")
def uniclim_compute(da: xr.DataArray, freq="YS", **indexer):
    select = select_time(da, **indexer)
    return select.mean(dim="time", keep_attrs=True).expand_dims("time")


uniClim = ResamplingIndicator(
    src_freq="D",
    realm="atmos",
    identifier="test.clim",
    attrs=[dict(units="K")],
    module="test",
    compute=uniclim_compute,
)


@declare_units(tas="[temperature]")
def multitemp_compute(tas: xr.DataArray, freq: str):
    return (
        tas.resample(time=freq).min(keep_attrs=True),
        tas.resample(time=freq).max(keep_attrs=True),
    )


multiTemp = Daily(
    realm="atmos",
    identifier="test.minmaxtemp",
    attrs=[
        dict(
            var_name="tmin",
            units="K",
            standard_name="Min temp",
            description="Grouped computation of tmax and tmin",
        ),
        dict(
            var_name="tmax",
            units="K",
            description="Grouped computation of tmax and tmin",
        ),
    ],
    module="test",
    compute=multitemp_compute,
)


@declare_units(tas="[temperature]", tasmin="[temperature]", tasmax="[temperature]")
def multioptvar_compute(
    tas: xr.DataArray | None = None,
    tasmax: xr.DataArray | None = None,
    tasmin: xr.DataArray | None = None,
):
    if tas is None:
        tasmax = convert_units_to(tasmax, tasmin)
        return ((tasmin + tasmax) / 2).assign_attrs(units=tasmin.units)
    return tas


multiOptVar = Indicator(
    src_freq="D",
    realm="atmos",
    identifier="test.multiopt",
    attrs=[dict(units="K")],
    module="test",
    compute=multioptvar_compute,
)


def test_attrs(tas_series):

    a = tas_series(np.arange(360.0))
    out = uniIndTemp(a, thresh="5 degC", freq="YS")
    assert out.tmin5degC.cell_methods == "time: mean time: mean within years"
    assert uniIndTemp.standard_name == "{freq} mean temperature"
    assert uniIndTemp.attrs[0]["another_attr"] == "With a value."

    thresh = xr.DataArray(
        [1],
        dims=("adim",),
        coords={"adim": [1]},
        attrs={"long_name": "A thresh", "units": "degC"},
        name="TT",
    )
    with xclim.set_options(as_dataset=False):
        txm = uniIndTemp(a, thresh=thresh, freq="YS")
    assert txm.attrs["long_name"].endswith("with <an array> threshold.")


@pytest.mark.parametrize(
    "xropt,exp",
    [
        ("default", True),
        (False, False),
        (True, True),
    ],
)
def test_keep_attrs(tasmin_series, tasmax_series, xropt, exp):
    pytest.importorskip("xarray", minversion="2025.11")
    tx = tasmax_series(np.arange(360.0))
    tn = tasmin_series(np.arange(360.0))
    tx.attrs.update(something="blabla", bing="bang", foo="bar")
    tn.attrs.update(something="blabla", bing="bong")
    with xr.set_options(keep_attrs=xropt), xclim.set_options(as_dataset=False):
        tg = multiOptVar(tasmin=tn, tasmax=tx)

    assert (tg.attrs.get("something") == "blabla") is exp
    assert (tg.attrs.get("foo") == "bar") is exp
    assert "bing" not in tg.attrs


@pytest.mark.parametrize("xrkeep", [True, False])
def test_as_dataset(tasmax_series, tasmin_series, xrkeep):
    tx = tasmax_series(np.arange(360.0))
    tn = tasmin_series(np.arange(360.0))
    tx.attrs.update(something="blabla", bing="bang", foo="bar")
    tn.attrs.update(something="blabla", bing="bong")
    dsin = xr.Dataset({"tasmax": tx, "tasmin": tn}, attrs={"fou": "barre"})
    with xr.set_options(keep_attrs=xrkeep):
        dsout = multiOptVar(ds=dsin)
    assert isinstance(dsout, xr.Dataset)
    if xrkeep:
        assert dsout.attrs["fou"] == "barre"
        assert dsout.multiopt.attrs.get("something") == "blabla"
    else:
        assert "fou" not in dsout.attrs
        # not testing for "something" because xclim doesn't define the
        # behaviour of dataarray attributes when xarray's option is not True.


def test_as_dataset_multi(tas_series):
    tg = tas_series(np.arange(360.0))
    dsout = multiTemp(tas=tg, freq="YS")
    assert isinstance(dsout, xr.Dataset)
    assert "tmin" in dsout.data_vars
    assert "tmax" in dsout.data_vars


@pytest.mark.skipif(DataTree is False, reason="Old xarray doesn't have DataTree")
def test_datatree(tasmin_series, tasmax_series):
    ds1 = xr.Dataset({"tasmax": tasmax_series(np.arange(360)), "tasmin": tasmin_series(np.arange(360))})
    ds2 = ds1.expand_dims(lat=[45, 46])
    ds3 = xr.Dataset(
        {
            "tasmax": tasmax_series(np.arange(720), start="1950-01-01", calendar="noleap"),
            "tasmin": tasmin_series(np.arange(720), start="1950-01-01", calendar="noleap"),
        }
    )

    dt = DataTree.from_dict({"/base": ds1, "/base/withlats": ds2, "/noleap": ds3})

    dtout = multiOptVar(ds=dt)
    ds1out = multiOptVar(ds=ds1)

    xr.testing.assert_equal(dtout["base"].dataset.multiopt, ds1out.multiopt)


@pytest.mark.skipif(DataTree is False, reason="Old xarray doesn't have DataTree")
def test_datatree_error(tas_series, tasmax_series):
    ds1 = xr.Dataset({"tasmax": tasmax_series(np.arange(360)), "tas": tas_series(np.arange(360))})
    ds2 = ds1.expand_dims(lat=[45, 46])
    ds3 = xr.Dataset({"tasmax": tasmax_series(np.arange(720), start="1950-01-01", calendar="noleap")})

    dt = DataTree.from_dict({"/base": ds1, "/base/withlats": ds2, "/noleap": ds3})

    with pytest.raises(MissingVariableError):
        multiTemp(ds=dt, freq="MS")


def test_opt_vars(tasmin_series, tasmax_series):
    tn = tasmin_series(np.zeros(365))
    tx = tasmax_series(np.zeros(365))

    multiOptVar(tasmin=tn, tasmax=tx)
    assert multiOptVar.parameters["tasmin"].kind == InputKind.OPTIONAL_VARIABLE


def test_registering():
    assert "test.tmin" in registry


def test_module():
    """Translations are keyed according to the module where the indicators are defined."""
    assert atmos.tg_mean.__module__.split(".")[2] == "atmos"


def test_temp_unit_conversion(tas_series, as_da):
    a = tas_series(np.arange(365), start="2001-01-01")
    txk = uniIndTemp(a, freq="YS")

    # This is not supposed to work
    uniIndTemp.units = "degC"
    txc = uniIndTemp(a, freq="YS")
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(txk, txc + 273.15)

    uniIndTemp.attrs[0].units = "degC"
    txc = uniIndTemp(a, freq="YS")
    np.testing.assert_array_almost_equal(txk, txc + 273.15)


def test_temp_diff_unit_conversion(tasmax_series, tasmin_series, as_da):
    tx = tasmax_series(np.arange(365) + 1, start="2001-01-01")
    tn = tasmin_series(np.arange(365), start="2001-01-01")
    txC = convert_units_to(tx, "degC")
    tnC = convert_units_to(tn, "degC")

    ind = xclim.atmos.daily_temperature_range.__class__(
        identifier="test.dtr_degC", attrs=[{"units": "degC", "units_metadata": "temperature: difference"}]
    )
    out = ind(tasmax=txC, tasmin=tnC)
    assert out.attrs["units"] == "degC"
    assert out.attrs["units_metadata"] == "temperature: difference"


def test_multiindicator(tas_series):
    tas = tas_series(np.arange(366), start="2000-01-01")
    out = multiTemp(tas, freq="YS")

    assert out.tmin[0] == tas.min()
    assert out.tmax[0] == tas.max()
    assert out.tmin.attrs["standard_name"] == "Min temp"
    assert out.tmin.attrs["description"] == "Grouped computation of tmax and tmin"
    assert out.tmax.attrs["description"] == "Grouped computation of tmax and tmin"
    assert multiTemp.units == ["K", "K"]

    # Attrs passed as keywords - together
    ind = Daily(
        realm="atmos",
        identifier="test.minmaxtemp2",
        attrs=[
            dict(
                var_name="tmin",
                units="K",
                standard_name="Min temp",
                description="Grouped computation of tmax and tmin",
            ),
            dict(
                var_name="tmax",
                units="K",
                description="Grouped computation of tmax and tmin",
            ),
        ],
        compute=multitemp_compute,
    )
    out = ind(tas, freq="YS")
    assert out.tmin[0] == tas.min()
    assert out.tmax[0] == tas.max()
    assert out.tmin.attrs["standard_name"] == "Min temp"
    assert out.tmin.attrs["description"] == "Grouped computation of tmax and tmin"
    assert out.tmax.attrs["description"] == "Grouped computation of tmax and tmin"

    with pytest.raises(ValueError, match="Output #2 of minmaxtemp2 is missing a var_name"):
        ind = Daily(
            realm="atmos",
            identifier="minmaxtemp2",
            attrs=[
                dict(
                    var_name="tmin",
                    units="K",
                ),
                dict(
                    units="K",
                ),
            ],
            compute=multitemp_compute,
        )

    # Attrs passed as keywords - individually
    ind = Daily(
        realm="atmos",
        identifier="test.minmaxtemp3",
        var_name=["tmin", "tmax"],
        units="K",
        standard_name=["Min temp", ""],
        description="Grouped computation of tmax and tmin",
        compute=multitemp_compute,
    )
    out = ind(tas, freq="YS")
    assert out.tmin[0] == tas.min()
    assert out.tmax[0] == tas.max()
    assert out.tmin.attrs["standard_name"] == "Min temp"
    assert out.tmin.attrs["description"] == "Grouped computation of tmax and tmin"
    assert out.tmax.attrs["description"] == "Grouped computation of tmax and tmin"
    assert ind.units == ["K", "K"]

    # All must be the same length
    with pytest.raises(ValueError, match="Attribute standard_name has 1 elements"):
        ind = Daily(
            realm="atmos",
            identifier="minmaxtemp3",
            var_name=["tmin", "tmax"],
            units="K",
            standard_name=["Min temp"],
            description="Grouped computation of tmax and tmin",
            compute=uniindpr_compute,
        )

    ind = Daily(
        realm="atmos",
        identifier="test.minmaxtemp4",
        var_name=["tmin", "tmax"],
        units="K",
        standard_name=["Min temp", ""],
        description="Grouped computation of tmax and tmin",
        compute=uniindtemp_compute,
    )
    with pytest.raises(ValueError, match="Indicator test.minmaxtemp4 was wrongly defined"):
        ind(tas, freq="YS")


def test_missing(tas_series, as_da):
    a = tas_series(np.ones(365, float), start="1/1/2000")

    # By default, missing is set to "from_context", and the default missing option is "any"
    # Cannot set missing_options with "from_context"
    with pytest.raises(ValueError, match="Cannot set `missing_options`"):
        uniClim.__class__(missing_options={"tolerance": 0.01})

    # Null value
    a[5] = np.nan

    m = uniIndTemp(a, freq="MS")
    assert m[0].isnull()

    with xclim.set_options(as_dataset=True, check_missing="pct", missing_options={"pct": {"tolerance": 0.05}}):
        dm = uniIndTemp(a, freq="MS")
        assert not dm["tmin0.0degC"].isnull().all()
        assert "check_missing=pct, missing_options={'tolerance': 0.05}" in dm.history
    # With freq=None
    c = uniClim(a)
    assert c.isnull()

    # With indexer
    ci = uniClim(a, month=[2])
    assert not ci.isnull()

    out = uniClim(a, month=[1])
    assert out.isnull()


def test_missing_from_context(tas_series, as_da):
    a = tas_series(np.ones(365, float), start="1/1/2000")
    # Null value
    a[5] = np.nan

    ind = uniIndTemp.__class__(identifier="test.uniIndTemp2", missing="from_context")

    m = ind(a, freq="MS")
    assert m[0].isnull()


def test_json(pr_series):
    meta = uniIndPr.json()

    expected = {
        "identifier",
        "title",
        "keywords",
        "abstract",
        "parameters",
        "history",
        "references",
        "notes",
        "outputs",
    }

    output_exp = {
        "var_name",
        "dimensionality",
        "units",
        "units_metadata",
        "long_name",
        "standard_name",
        "cell_methods",
        "description",
        "comment",
    }

    assert set(meta.keys()).issubset(expected)
    for output in meta["outputs"]:
        assert set(output.keys()).issubset(output_exp)


def test_all_jsonable(official_indicators):
    problems = []
    err = None
    for identifier, ind in official_indicators.items():
        try:
            json.dumps(ind.json())
        except (KeyError, TypeError) as e:
            problems.append(identifier)
            err = e
    if problems:
        raise ValueError(f"Indicators {problems} provide problematic json serialization.: {err}")


def test_no_constrain(official_indicators):
    # constrain is meant to constrain the condition, it makes no sense without it.
    problems = set()
    for identifier, ind in official_indicators.items():
        for name in ind.parameters:
            if "generic" in ind.__module__:
                if name.startswith("constrain") and name.replace("constrain", "condition") not in ind.parameters:
                    problems.add(identifier)
            else:
                if name.startswith("constrain"):
                    problems.add(identifier)
    if problems:
        raise ValueError(
            "The following indicators have a 'constrain' argument, but shouldn't "
            f"(either not generic or without a condition argument):  {problems}"
        )


def test_no_out_units(official_indicators):
    # indicators set the units, it makes no sense to have this argument
    problems = set()
    for identifier, ind in official_indicators.items():
        if "out_units" in ind.parameters:
            problems.add(identifier)
    if problems:
        raise ValueError(f"The following indicators have a 'out_units' argument : {problems}")


def test_all_parameters_understood(official_indicators):
    problems = set()
    for identifier, ind in official_indicators.items():
        for name, param in ind.parameters.items():
            if param.kind == InputKind.OTHER_PARAMETER:
                problems.add((identifier, name))
    # lat is present in many indicators, but is exceptionally allowed.
    if problems - {("INDICE", "test_param")}:
        raise ValueError(f"The following indicator/parameter couple(s) {problems} use types not listed in InputKind.")


def test_signature():
    sig = signature(xclim.atmos.solid_precip_accumulation)
    assert list(sig.parameters.keys()) == [
        "pr",
        "tas",
        "thresh",
        "freq",
        "ds",
        "indexer",
    ]
    assert sig.parameters["pr"].annotation == xr.DataArray | str
    assert sig.parameters["tas"].default == "tas"
    assert sig.parameters["tas"].kind == sig.parameters["tas"].POSITIONAL_OR_KEYWORD
    assert sig.parameters["thresh"].kind == sig.parameters["thresh"].KEYWORD_ONLY
    assert sig.return_annotation == xr.DataArray

    for indicator in ["wind_speed_from_vector", "wind_vector_from_speed"]:
        sig = signature(getattr(xclim.convert, indicator))
        assert sig.return_annotation == tuple[xr.DataArray, xr.DataArray]


def test_doc():
    doc = xclim.atmos.tx_max.__doc__
    assert doc.startswith("Maximum temperature")
    assert "This indicator will check for missing values according to the method" in doc
    assert "Based on function :py:func:`~xclim.compute.generic.statistics" in doc
    assert "tasmax : str or DataArray" in doc
    assert "Returns\n-------" in doc

    doc = xclim.atmos.cffwis_indices.__doc__
    assert "snd : str or DataArray, optional" in doc
    assert "See :cite:t:`code-natural_resources_canada_data_nodate`, " in doc
    assert "the :py:mod:`xclim.compute.fire` module documentation," in doc
    assert "and the docstring of :py:func:`fire_weather_ufunc` for more information." in doc


def test_delayed(tasmax_series, as_da):
    tasmax = tasmax_series(np.arange(360.0)).chunk({"time": 5})
    out = uniIndTemp(tasmax)
    assert isinstance(out.data, dask.array.Array)


def test_formatting(pr_series):
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.0 * units.mm / units.day)
    # pint 0.10 now pretty print day as d.
    assert out.wetdays.attrs["long_name"] == "Number of days with daily precipitation at or above 1 mm d-1"
    assert out.wetdays.attrs["description"] in ["Annual number of days with daily precipitation at or above 1 mm d-1."]
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.5 * units.mm / units.day)
    assert out.wetdays.attrs["long_name"] == "Number of days with daily precipitation at or above 1.5 mm d-1"
    assert out.wetdays.attrs["description"] in [
        "Annual number of days with daily precipitation at or above 1.5 mm d-1."
    ]


def test_IndexWrapper():
    doc = IndexWrapper(compute=tg_mean)
    assert doc.title == "Mean of daily average temperature."
    assert doc.abstract == "Resample the original daily mean temperature series by taking the mean over each period."
    assert doc.parameters["tas"].description == "Mean daily temperature."
    assert doc.parameters["freq"].description == "Resampling frequency."
    assert doc.notes.startswith("Let")
    assert "math::" in doc.notes
    assert doc.references == ""
    assert doc.attrs[0]["long_name"] == "The mean daily temperature at the given time frequency."

    doc = IndexWrapper(compute=xclim.compute.converters.saturation_vapor_pressure)
    assert doc.parameters["ice_thresh"].description == (
        "Threshold temperature under which to switch to equations in reference to ice instead of water. "
        "If None (default) everything is computed with reference to water. "
        "If given, see `interp_power` for more options."
    )
    assert "goff_low-pressure_1946" in doc.references


def test_parsed_doc():
    assert "tas" in xclim.atmos.liquid_precip_accumulation.parameters

    params = xclim.atmos.drought_code.parameters
    assert params["tas"].description == "Noon temperature."
    assert params["tas"].units == "[temperature]"
    assert params["tas"].kind is InputKind.VARIABLE
    assert params["tas"].default == "tas"
    assert params["snd"].default is None
    assert params["snd"].kind is InputKind.OPTIONAL_VARIABLE
    assert params["snd"].units == "[length]"
    assert params["season_method"].kind is InputKind.STRING
    assert params["season_method"].choices == {"GFWED", None, "WF93", "LA08"}

    params = xclim.atmos.standardized_precipitation_evapotranspiration_index.parameters
    assert params["fitkwargs"].kind is InputKind.DICT


def test_indicator_docstring():
    doc = xclim.atmos.heat_wave_frequency.__doc__.split("\n")
    assert doc[0] == "Heat wave frequency"
    assert doc[5] == "Based on function :py:func:`~xclim.compute.generic.bivariate_spell_length_statistics`."
    assert (
        doc[6] == "With injected parameters: window_statistic=min, statistic=count, min_gap=1, constrain=('>', '>=')."
    )
    assert doc[11] == " Minimum surface temperature. Default: 'tasmin'. [Required units : [temperature]]"
    assert doc[34] == (
        "  Total number of series of at least {window} consecutive days with daily minimum temperature above "
        "{thresh_tasmin} and daily maximum temperature above {thresh_tasmax}. With additional attributes: "
        "**description**: ``{freq} number of heat wave events within a given period. A heat wave occurs when daily "
        "minimum and maximum temperatures exceed {thresh_tasmin} and {thresh_tasmax}, respectively, over at least "
        "{window} days.``"
    )


def test_input_dataset(open_dataset):
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    # Use defaults
    _ = xclim.atmos.daily_temperature_range(freq="YS", ds=ds)

    # Use non-defaults (inverted on purpose)
    with xclim.set_options(cf_compliance="log"):
        _ = xclim.atmos.daily_temperature_range("tasmax", "tasmin", freq="YS", ds=ds)

    # Use a mix
    _ = xclim.atmos.daily_temperature_range(tasmax=ds.tasmax, freq="YS", ds=ds)

    # Inexistent variable:
    dsx = ds.drop_vars("tasmin")
    with pytest.raises(MissingVariableError):
        out = xclim.atmos.daily_temperature_range(freq="YS", ds=dsx)  # noqa

    # dataset is not given
    with pytest.raises(ValueError):
        xclim.atmos.daily_temperature_range(tasmax="tmax")


def test_indicator_errors():
    def func(data: xr.DataArray, thresh: str = "0 degC", freq: str = "YS"):  # noqa
        return data

    doc = [
        "    The title",
        "",
        "    The abstract",
        "",
        "    Parameters",
        "    ----------",
        "    data : xr.DataArray",
        "      A variable.",
        "    thresh : str",
        "      A threshold",
        "    freq : str",
        "      The resampling frequency.",
        "",
        "    Returns",
        "    -------",
        "    xr.DataArray, [K]",
        "      An output",
    ]
    func.__doc__ = "\n".join(doc)

    d = dict(
        realm="atmos",
        attrs=dict(
            var_name="tmean{threshold}",
            units="K",
            long_name="{freq} mean surface temperature",
            standard_name="{freq} mean temperature",
            cell_methods=[{"time": "mean within days"}],
        ),
        compute=func,
        input={"data": "tas"},
    )
    ind = Daily(identifier="indi", module="test", **d)

    with pytest.raises(TypeError, match="Missing argument 'identifier'"):
        Daily(**d)

    d["identifier"] = "bad_indi"
    d["module"] = "test"

    func.__doc__ = "\n".join(doc)
    d["parameters"] = {}
    d["parameters"]["thresh"] = "1 degK"
    d["parameters"]["extra"] = "woopsi again"
    with pytest.raises(ValueError, match="Parameter extra was passed but it does"):
        Daily(**d)

    del d["parameters"]["extra"]
    d["input"]["data"] = "3nsd6sk72"
    with pytest.raises(ValueError, match="Compute argument data was mapped to"):
        Daily(**d)

    d2 = dict(input={"tas": "sfcWind"})
    with pytest.raises(ValueError, match="When changing the name of a variable by"):
        ind.__class__(**d2)

    del d["input"]
    # with pytest.raises(ValueError, match="variable data is missing expected units"):
    #     Daily(**d)

    def func(data: xr.DataArray, thresh: str = "0 degC"):
        return data

    func.__doc__ = "\n".join(doc[:10] + doc[12:])
    d = dict(
        realm="atmos",
        attrs=dict(
            var_name="tmean{threshold}",
            units="K",
            long_name="{freq} mean surface temperature",
            standard_name="{freq} mean temperature",
            cell_methods=[{"time": "mean within days"}],
        ),
        compute=func,
        input={"data": "tas"},
    )
    with pytest.raises(ValueError, match="ResamplingIndicator require a 'freq'"):
        Daily(identifier="indi", module="test", **d)


def test_indicator_call_errors(tas_series):
    tas = tas_series(np.arange(730), start="2001-01-01")
    uniIndTemp(da=tas, thresh="3 K")

    with pytest.raises(TypeError, match="too many positional arguments"):
        uniIndTemp(tas, tas)

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'oups'"):
        uniIndTemp(tas, oups=3)


def test_resamplingIndicator_new_error():
    with pytest.raises(ValueError, match="ResamplingIndicator require a 'freq'"):
        Daily(
            realm="atmos",
            identifier="multiopt",
            attrs=[dict(units="K")],
            module="test",
            compute=multioptvar_compute,
        )


def test_resampling_indicator_with_indexing(tas_series):
    tas = tas_series(np.ones(731), start="2003-01-01", units="°C")

    out = xclim.atmos.tx_days_above(tas, thresh="0 degC", freq="YS")
    np.testing.assert_allclose(out.tx_days_above, [365, 366])

    out = xclim.atmos.tx_days_above(tas, thresh="0 degC", freq="YS", month=2)
    np.testing.assert_allclose(out.tx_days_above, [28, 29])

    out = xclim.atmos.tx_days_above(tas, thresh="0 degC", freq="YS-JUL", doy_bounds=(1, 50))
    np.testing.assert_allclose(out.tx_days_above, [50, 50, np.nan])

    out = xclim.atmos.tx_days_above(tas, thresh="0 degC", freq="YS", doy_bounds=(50, None))
    np.testing.assert_allclose(out.tx_days_above, [316, 317])

    out = xclim.atmos.tx_days_above(tas, thresh="0 degC", freq="YS", date_bounds=("02-29", "04-01"))
    np.testing.assert_allclose(out.tx_days_above, [32, 33])

    out = xclim.atmos.tx_days_above(
        tas, thresh="0 degC", freq="YS-JUL", date_bounds=(None, "04-01"), bounds_freq="YS-JUL"
    )
    np.testing.assert_allclose(out.tx_days_above, [np.nan, 275, np.nan])


def test_indicator_indexing_doy_bounds_spatial(tasmin_series):
    da = tasmin_series(np.ones(730), start="2005-01-01", units="°C").expand_dims(lat=[0, 10, 15, 20, 25])

    start = xr.DataArray([50, 340, 100, np.nan, np.nan], dims=("lat",), coords={"lat": da.lat})
    end = xr.DataArray([200, 20, np.nan, 200, np.nan], dims=("lat",), coords={"lat": da.lat})
    out = atmos.tn_days_above(da, thresh="0 °C", doy_bounds=(start, end))

    # 340, 20 is an invalid indexer for default freq (YS)
    # such cases return an entirely masked array
    # No values are missing as there are no values to count
    np.testing.assert_array_equal(
        out.tn_days_above,
        [[151.0, 151.0], [0, 0], [266.0, 266.0], [200.0, 200.0], [365.0, 365.0]],
    )

    out = atmos.tn_days_above(da, thresh="0 °C", freq="YS-JUL", doy_bounds=(start, end), bounds_freq="YS-JUL")
    # with freq YS-JUL
    # 50, 200 is an invalid indexer
    # 340, 20 is valid but incomplete for first and last year
    # 100, NaN is valid but incomplete for last year
    # NaN, 200 is valid but incomplete for first year
    # NaN, NaN is valid (include NaN by default) but incomplete for first and last year
    np.testing.assert_array_equal(
        out.tn_days_above, [[0, 0, 0], [np.nan, 46, np.nan], [82, 82, np.nan], [np.nan, 19, 19], [np.nan, 365, np.nan]]
    )


def test_indicator_indexing_doy_bounds_temporal(tasmin_series):
    da = tasmin_series(np.ones(365 * 5 + 1), start="2005-01-01", units="°C")

    time = xr.date_range("2005-01-01", freq="YS", periods=5)
    start = xr.DataArray([50, 340, 100, np.nan, np.nan], dims=("time",), coords={"time": time})
    end = xr.DataArray([200, 20, np.nan, 200, np.nan], dims=("time",), coords={"time": time})
    out = atmos.tn_days_above(da, thresh="0 °C", doy_bounds=(start, end))

    # 340, 20 is an invalid indexer for freq YS.
    # such cases return an entirely masked array
    # No values are missing as there are no values to count
    np.testing.assert_array_equal(out.tn_days_above, [151, 0, 266, 200, 365])

    out = atmos.tn_days_above(da, thresh="0 °C", freq="YS", doy_bounds=(None, end))
    # start as None = 1 for freq YS (default) and includes doy NaNs (default)
    # None, 200: 1 to 200
    # None, 20: 1 to 20
    # None, NaN: 1 to 365
    # None, 200: 1 to 200
    # None, NaN: 1 to 365
    np.testing.assert_array_equal(out.tn_days_above, [200, 20, 365, 200, 365])

    out = atmos.tn_days_above(da, thresh="0 °C", freq="YS", doy_bounds=(start, None), include_doy_bounds_nans=False)
    # end as None = 365 for freq YS (default) and excludes doy NaNs
    # 50, None: 50 to 365
    # 340, None: 340 to 365
    # 100, None: 100 to 365
    # NaN, None: masked, no values to count (0)
    # NaN, None: masked, no values to count (0)
    np.testing.assert_array_equal(out.tn_days_above, [316, 26, 266, 0, 0])


def test_all_inputs_known():
    var_and_inds = list_input_variables()
    known_vars = (
        set(var_and_inds.keys())
        - {
            "dc0",
            "season_mask",
            "ffmc0",
            "dmc0",
            "kbdi0",
            "drought_factor",
        }  # FWI optional inputs
        - {var for var in var_and_inds.keys() if var.endswith("_per")}  # percentiles
        - {"pr_annual", "pr_cal", "wb_cal"}  # other optional or uncommon
        - {"q", "da"}  # Generic inputs
        - {"mrt", "wb"}  # TODO: add Mean Radiant Temperature and water budget
    )
    if not set(VARIABLES.keys()).issuperset(known_vars):
        raise AssertionError(
            "All input variables of xclim indicators must be registered in "
            "data/variables.yml, or skipped explicitly in this test. "
            f"The yaml file is missing: {known_vars - VARIABLES.keys()}."
        )


def test_freq_doc():
    from xclim import atmos

    doc = atmos.latitude_temperature_index.__doc__
    allowed_periods = ["Y"]
    exp = f"Requested resampling periods are restricted to {', '.join(allowed_periods)}"
    assert exp in doc


def test_no_rewrapping():
    with pytest.raises(TypeError, match="Can't change the compute"):
        uniIndTemp.__class__(
            compute=uniindtemp_compute,
            parameters={"thresh": "0 °C"},
        )
