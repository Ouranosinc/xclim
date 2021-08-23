#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for the Indicator objects
import gc
import json
from typing import Optional, Union

import dask
import numpy as np
import pytest
import xarray as xr

import xclim
from xclim import __version__, atmos
from xclim.core.formatting import (
    AttrFormatter,
    default_formatter,
    merge_attributes,
    parse_doc,
    update_history,
)
from xclim.core.indicator import Daily, Indicator, registry
from xclim.core.units import units
from xclim.core.utils import InputKind, MissingVariableError
from xclim.indices import tg_mean
from xclim.indices.generic import select_time
from xclim.testing import open_dataset


class UniIndTemp(Daily):
    realm = "atmos"
    identifier = "tmin"
    var_name = "tmin{thresh}"
    units = "K"
    long_name = "{freq} mean surface temperature"
    standard_name = "{freq} mean temperature"
    cell_methods = "time: mean within {freq:noun}"

    @staticmethod
    def compute(da: xr.DataArray, thresh: int = 0.0, freq: str = "YS"):
        """Docstring"""
        out = da
        out -= thresh
        return out.resample(time=freq).mean(keep_attrs=True)


class UniIndPr(Daily):
    realm = "atmos"
    identifier = "prmax"
    units = "mm/s"
    context = "hydro"

    @staticmethod
    def compute(da: xr.DataArray, freq):
        """Docstring"""
        return da.resample(time=freq).mean(keep_attrs=True)


class UniClim(Daily):
    realm = "atmos"
    identifier = "clim"
    units = "K"

    @staticmethod
    def compute(da: xr.DataArray, freq="YS", **indexer):
        select = select_time(da, **indexer)
        return select.mean(dim="time", keep_attrs=True)


class MultiTemp(Daily):
    realm = "atmos"
    identifier = "minmaxtemp"
    var_name = ["tmin", "tmax"]
    units = "K"
    standard_name = ["Min temp", ""]
    description = "Grouped computation of tmax and tmin"

    @staticmethod
    def compute(tas: xr.DataArray, freq):
        return (
            tas.resample(time=freq).min(keep_attrs=True),
            tas.resample(time=freq).max(keep_attrs=True),
        )


class MultiOptVar(Daily):
    realm = "atmos"
    identifier = "multiopt"
    units = "K"

    @staticmethod
    def compute(
        tas: Optional[xr.DataArray] = None,
        tasmax: Optional[xr.DataArray] = None,
        tasmin: Optional[xr.DataArray] = None,
    ):
        if tas is None:
            with xr.set_options(keep_attrs=True):
                return (tasmin + tasmax) / 2
        return tas


def test_attrs(tas_series):
    import datetime as dt

    a = tas_series(np.arange(360.0))
    ind = UniIndTemp()
    txm = ind(a, thresh=5, freq="YS")
    assert txm.cell_methods == "time: mean within days time: mean within years"
    assert f"{dt.datetime.now():%Y-%m-%d %H}" in txm.attrs["history"]
    assert "TMIN(da=tas, thresh=5, freq='YS')" in txm.attrs["history"]
    assert f"xclim version: {__version__}." in txm.attrs["history"]
    assert txm.name == "tmin5"


def test_opt_vars(tasmin_series, tasmax_series):
    tn = tasmin_series(np.zeros(365))
    tx = tasmax_series(np.zeros(365))

    ind = MultiOptVar()

    ind(tasmin=tn, tasmax=tx)


def test_registering():
    UniIndTemp(module="test")
    assert "test.TMIN" in registry

    # Because this has not been instantiated, it's not in any registry.
    class Test123(registry["test.TMIN"]):
        identifier = "test123"

    assert "test.TEST123" not in registry
    Test123(module="test")
    assert "test.TEST123" in registry

    # Confirm registries live in subclasses.
    class IndicatorNew(Indicator):
        pass

    # Identifier must be given
    with pytest.raises(AttributeError, match="has not been set."):
        IndicatorNew()

    # Realm must be given
    with pytest.raises(AttributeError, match="realm must be given"):
        IndicatorNew(identifier="i2d")

    indnew = IndicatorNew(identifier="i2d", realm="atmos", module="test")
    assert "test.I2D" in registry
    assert registry["test.I2D"].get_instance() is indnew

    del indnew
    gc.collect()
    with pytest.raises(ValueError, match="There is no existing instance"):
        registry["test.I2D"].get_instance()


def test_module():
    """Translations are keyed according to the module where the indicators are defined."""
    assert atmos.tg_mean.__module__.split(".")[2] == "atmos"
    # Virtual module also are stored under xclim.indicators
    assert xclim.indicators.cf.fg.__module__ == "xclim.indicators.cf"
    assert xclim.indicators.icclim.GD4.__module__ == "xclim.indicators.icclim"


def test_temp_unit_conversion(tas_series):
    a = tas_series(np.arange(360.0))
    ind = UniIndTemp()
    txk = ind(a, freq="YS")

    ind.units = "degC"
    txc = ind(a, freq="YS")

    np.testing.assert_array_almost_equal(txk, txc + 273.15)


def test_multiindicator(tas_series):
    tas = tas_series(np.arange(366), start="2000-01-01")
    ind = MultiTemp()

    tmin, tmax = ind(tas, freq="YS")
    assert tmin[0] == tas.min()
    assert tmax[0] == tas.max()
    assert tmin.attrs["standard_name"] == "Min temp"
    assert tmin.attrs["description"] == "Grouped computation of tmax and tmin"
    assert tmax.attrs["description"] == "Grouped computation of tmax and tmin"


def test_missing(tas_series):
    a = tas_series(np.ones(360, float), start="1/1/2000")

    # By default, missing is set to "from_context", and the default missing option is "any"
    ind = UniIndTemp()

    # Cannot set missing_options with "from_context"
    with pytest.raises(ValueError, match="Cannot set `missing_options`"):
        UniClim(missing_options={"tolerance": 0.01})

    clim = UniClim()

    # Null value
    a[5] = np.nan

    m = ind(a, freq="MS")
    assert m[0].isnull()

    with xclim.set_options(
        check_missing="pct", missing_options={"pct": {"tolerance": 0.05}}
    ):
        m = ind(a, freq="MS")
        assert not m[0].isnull()

    with xclim.set_options(check_missing="wmo"):
        m = ind(a, freq="YS")
        assert m[0].isnull()

    # With freq=None
    c = clim(a)
    assert c.isnull()

    # With indexer
    ci = clim(a, month=[2])
    assert not ci.isnull()

    out = clim(a, month=[1])
    assert out.isnull()


def test_missing_from_context(tas_series):
    a = tas_series(np.ones(360, float), start="1/1/2000")
    # Null value
    a[5] = np.nan

    ind = UniIndTemp(missing="from_context")

    m = ind(a, freq="MS")
    assert m[0].isnull()


def test_json(pr_series):
    ind = UniIndPr()
    meta = ind.json()

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
        "units",
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
    for identifier, ind in official_indicators.items():
        indinst = ind.get_instance()
        try:
            json.dumps(indinst.json())
        except (TypeError, KeyError):
            problems.append(identifier)
    if problems:
        raise ValueError(
            f"Indicators {problems} provide problematic json serialization."
        )


def test_all_parameters_understood(official_indicators):
    problems = []
    for identifier, ind in official_indicators.items():
        indinst = ind.get_instance()
        for name, param in indinst.parameters.items():
            if param["kind"] == InputKind.OTHER_PARAMETER:
                problems.append((identifier, name))
    if problems:
        raise ValueError(
            f"The following indicator/parameter couple {problems} use types not listed in InputKind."
        )


def test_signature():
    from inspect import signature

    ind = UniIndTemp()
    assert ind._sig == signature(ind.__call__)
    assert ind._sig.parameters["da"].annotation is Union[str, xr.DataArray]

    compsig = signature(ind.compute)
    assert compsig.parameters["da"].annotation is xr.DataArray
    assert "ds" not in compsig.parameters
    assert "ds" in ind._sig.parameters


def test_doc():
    ind = UniIndTemp()
    assert ind.__call__.__doc__.startswith("Docstring (realm: atmos)")


def test_delayed(tasmax_series):
    tasmax = tasmax_series(np.arange(360.0)).chunk({"time": 5})

    tx = UniIndTemp()
    txk = tx(tasmax)

    # Check that the calculations are delayed
    assert isinstance(txk.data, dask.array.core.Array)

    # Same with unit conversion
    tx.required_units = ("C",)
    tx.units = "C"
    txc = tx(tasmax)

    assert isinstance(txc.data, dask.array.core.Array)


def test_identifier():
    with pytest.warns(UserWarning):
        UniIndPr(identifier="t_{}")


def test_formatting(pr_series):
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.0 * units.mm / units.day)
    # pint 0.10 now pretty print day as d.
    assert out.attrs["long_name"] in [
        "Number of wet days (precip >= 1 mm/day)",
        "Number of wet days (precip >= 1 mm/d)",
    ]
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.5 * units.mm / units.day)
    assert out.attrs["long_name"] in [
        "Number of wet days (precip >= 1.5 mm/day)",
        "Number of wet days (precip >= 1.5 mm/d)",
    ]


def test_parse_doc():
    doc = parse_doc(tg_mean.__doc__)
    assert doc["title"] == "Mean of daily average temperature."
    assert (
        doc["abstract"]
        == "Resample the original daily mean temperature series by taking the mean over each period."
    )
    assert doc["parameters"]["tas"]["description"] == "Mean daily temperature."
    assert doc["parameters"]["freq"]["description"] == "Resampling frequency."
    assert doc["notes"].startswith("Let")
    assert "math::" in doc["notes"]
    assert "references" not in doc
    assert doc["long_name"] == "The mean daily temperature at the given time frequency"

    doc = parse_doc(xclim.indices.saturation_vapor_pressure.__doc__)
    assert (
        doc["parameters"]["ice_thresh"]["description"]
        == "Threshold temperature under which to switch to equations in reference to ice instead of water. If None (default) everything is computed with reference to water."
    )
    assert "Goff, J. A., and S. Gratch (1946)" in doc["references"]


def test_parsed_doc():
    assert "tas" in xclim.atmos.liquid_precip_accumulation.parameters
    assert "tas" not in xclim.atmos.precip_accumulation.parameters

    params = xclim.atmos.drought_code.parameters
    assert params["tas"]["description"] == "Noon temperature."
    assert params["tas"]["units"] == "[temperature]"
    assert params["tas"]["kind"] is InputKind.VARIABLE
    assert params["tas"]["default"] == "tas"
    assert params["snd"]["default"] is None
    assert params["snd"]["kind"] is InputKind.OPTIONAL_VARIABLE
    assert params["snd"]["units"] == "[length]"
    assert params["season_method"]["kind"] is InputKind.STRING
    assert params["season_method"]["choices"] == {"GFWED", None, "WF93", "LA08"}


def test_default_formatter():
    assert default_formatter.format("{freq}", freq="YS") == "annual"
    assert default_formatter.format("{freq:noun}", freq="MS") == "months"
    assert default_formatter.format("{month}", month="m3") == "march"


def test_AttrFormatter():
    fmt = AttrFormatter(
        mapping={"evil": ["méchant", "méchante"], "nice": ["beau", "belle"]},
        modifiers=["m", "f"],
    )
    # Normal cases
    assert fmt.format("{adj:m}", adj="evil") == "méchant"
    assert fmt.format("{adj:f}", adj="nice") == "belle"
    # Missing mod:
    assert fmt.format("{adj}", adj="evil") == "méchant"
    # Mod with unknown value
    with pytest.raises(ValueError):
        fmt.format("{adj:m}", adj="funny")


@pytest.mark.parametrize("new_line", ["<>", "\n"])
@pytest.mark.parametrize("missing_str", ["<Missing>", None])
def test_merge_attributes(missing_str, new_line):
    a = xr.DataArray([0], attrs={"text": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={})
    c = xr.Dataset(attrs={"text": "Text3"})

    merged = merge_attributes(
        "text", a, missing_str=missing_str, new_line=new_line, b=b, c=c
    )

    assert merged.startswith("a: Text1")

    if missing_str is not None:
        assert merged.count(new_line) == 2
        assert f"b: {missing_str}" in merged
    else:
        assert merged.count(new_line) == 1
        assert "b:" not in merged


def test_update_history():
    a = xr.DataArray([0], attrs={"history": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={"history": "Text2"})
    c = xr.Dataset(attrs={"history": "Text3"})

    merged = update_history("text", a, new_name="d", b=b, c=c)

    assert "d: text" in merged.split("\n")[-1]
    assert merged.startswith("a: Text1")


def test_input_dataset():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    # Use defaults
    out = xclim.atmos.daily_temperature_range(freq="YS", ds=ds)

    # Use non-defaults (inverted on purpose)
    with xclim.set_options(cf_compliance="log"):
        out = xclim.atmos.daily_temperature_range("tasmax", "tasmin", freq="YS", ds=ds)

    # Use a mix
    out = xclim.atmos.daily_temperature_range(tasmax=ds.tasmax, freq="YS", ds=ds)

    # Inexistent variable:
    dsx = ds.drop_vars("tasmin")
    with pytest.raises(MissingVariableError):
        out = xclim.atmos.daily_temperature_range(freq="YS", ds=dsx)  # noqa


def test_indicator_from_dict():
    d = dict(
        realm="atmos",
        output=dict(
            var_name="tmean{thresh}",
            units="K",
            long_name="{freq} mean surface temperature",
            standard_name="{freq} mean temperature",
            cell_methods=[{"time": "mean within days"}],
        ),
        index_function=dict(
            name="thresholded_statistics",
            parameters=dict(
                threshold={"data": {"thresh": None}, "description": "A threshold temp"},
                condition={"data": "`<"},
                reducer={"data": "mean"},
            ),
        ),
        input={"data": "tas"},
    )

    ind = Daily.from_dict(d, identifier="tmean", module="test")

    assert ind.realm == "atmos"
    # Parameters metadata modification
    assert ind.parameters["threshold"]["description"] == "A threshold temp"
    # Injection of paramters
    assert "condition" in ind.compute._injected
    # Placeholders were translated to name in signature
    assert ind.cf_attrs[0]["var_name"] == "tmean{threshold}"
    # Default value for input variable injected and meta injected
    assert ind._sig.parameters["data"].default == "tas"
    assert ind.parameters["data"]["units"] == "K"
