#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package, command line interface
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

import xclim as xc
from xclim.cli import _isusable
from xclim.cli import cli


@pytest.mark.parametrize(
    "indicators,indnames",
    [
        ([xc.atmos.tg_mean], ["atmos.tg_mean"]),
        (
            [xc.atmos.tn_mean, xc.atmos.daily_freezethaw_cycles],
            ["atmos.tn_mean", "atmos.daily_freezethaw_cycles"],
        ),
    ],
)
def test_info(indicators, indnames):
    runner = CliRunner()
    results = runner.invoke(cli, ["info"] + indnames)

    for ind in indicators:
        assert ind.title in results.output
        assert ind.identifier in results.output


@pytest.mark.parametrize(
    "modules,modname", [([xc.atmos, xc.land, xc.seaIce], []), ([xc.atmos], ["atmos"])]
)
def test_indices(modules, modname):
    runner = CliRunner()
    results = runner.invoke(cli, ["indices"] + modname)

    for module in modules:
        for name, ind in module.__dict__.items():
            if _isusable(ind):
                assert name in results.output


@pytest.mark.parametrize(
    "indicator,indname",
    [
        (xc.atmos.heating_degree_days, "atmos.heating_degree_days"),
        (xc.land.base_flow_index, "land.base_flow_index"),
    ],
)
def test_indicator_help(indicator, indname):
    runner = CliRunner()
    results = runner.invoke(cli, [indname, "--help"])

    for name in indicator._sig.parameters.keys():
        assert name in results.output


@pytest.mark.parametrize(
    "indicator,expected",
    [
        ("atmos.tg_mean", 272.15),
        ("atmos.daily_temperature_range_variability", 0.0),
        ("atmos.heating_degree_days", 6588.0),
        ("atmos.solid_precip_accumulation", 31622400.0),
    ],
)
def test_normal_computation(
    tasmin_series, tasmax_series, pr_series, tmp_path, indicator, expected
):
    tasmin = tasmin_series(np.ones(366,) + 270.15, start="1/1/2000")
    tasmax = tasmax_series(np.ones(366,) + 272.15, start="1/1/2000")
    pr = pr_series(np.ones(366,), start="1/1/2000")
    ds = xr.Dataset(data_vars={"tasmin": tasmin, "tasmax": tasmax, "pr": pr})
    input_file = tmp_path / "in.nc"
    output_file = tmp_path / "out.nc"

    ds.to_netcdf(input_file)

    args = ["-i", str(input_file), "-o", str(output_file), indicator]
    # if indicator == "atmos.solid_precip_accumulation":
    #     args.extend(["--tas", "tas"])
    runner = CliRunner()
    results = runner.invoke(cli, args)
    assert "Processing :" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    outvar = list(out.data_vars.values())[0]
    np.testing.assert_allclose(outvar[0], expected)


@pytest.mark.parametrize("name_suffix", ["", "the_"])
def test_auto_tas_calc(tasmin_series, tasmax_series, tmp_path, name_suffix):
    tasmin = tasmin_series(np.zeros(366,), start="1/1/2000")
    tasmax = tasmax_series(np.ones(366,), start="1/1/2000")
    input_file = tmp_path / "temp.nc"
    output_file = tmp_path / "out.nc"

    xr.Dataset(
        {name_suffix + "tasmin": tasmin, name_suffix + "tasmax": tasmax}
    ).to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        (
            ["--tas-from", name_suffix + "tasmin", name_suffix + "tasmax"]
            if name_suffix
            else []
        )
        + ["-i", str(input_file), "-o", str(output_file), "atmos.tg_mean"],
    )

    assert "Processing : Mean daily mean temperature" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tg_mean[0] == 0.5


def test_renaming_variable(tas_series, tmp_path):
    tas = tas_series(np.ones(366,), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "atmos.tn_mean",
            "--tasmin",
            "tas",
        ],
    )
    assert "Processing : Mean daily minimum temperature" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tn_mean[0] == 1.0


def test_indicator_chain(tas_series, tmp_path):
    tas = tas_series(np.ones(366,), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "atmos.tg_mean",
            "atmos.growing_degree_days",
        ],
    )

    assert "Processing : Mean daily mean temperature" in results.output
    assert "Processing : Growing degree days" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tg_mean[0] == 1.0
    assert out.growing_degree_days[0] == 0


def test_missing_variable(tas_series, tmp_path):
    tas = tas_series(np.ones(366,), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli, ["-i", str(input_file), "-o", str(output_file), "atmos.tn_mean"]
    )
    assert results.exit_code == 2
    assert "tasmin absent from input dataset." in results.output


@pytest.mark.parametrize(
    "options,output",
    [
        (["--dask-nthreads", "2"], "Error: '--dask-maxmem' must be given"),
        (["--chunks", "time:90"], "Writing everything to file"),
        (["--chunks", "time:90,lat:5"], "Writing everything to file"),
        (["--version"], xc.__version__),
    ],
)
def test_global_options(tas_series, tmp_path, options, output):
    tas = tas_series(np.ones(366,), start="1/1/2000")
    tas = xr.concat([tas] * 10, dim="lat")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        ["-i", str(input_file), "-o", str(output_file)] + options + ["atmos.tg_mean"],
    )

    assert output in results.output
