#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package, command line interface
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

import xclim as xc
from xclim.cli import cli

xc.set_options(cf_compliance="warn")


@pytest.mark.parametrize(
    "indicators,indnames",
    [
        ([xc.atmos.tg_mean], ["tg_mean"]),
        (
            [xc.atmos.tn_mean, xc.atmos.daily_freezethaw_cycles],
            ["tn_mean", "dlyfrzthw"],
        ),
    ],
)
def test_info(indicators, indnames):
    runner = CliRunner()
    results = runner.invoke(cli, ["info"] + indnames)

    for ind in indicators:
        assert ind.title in results.output
        assert ind.identifier in results.output


def test_indices():
    runner = CliRunner()
    results = runner.invoke(cli, ["indices"])

    for name, ind in xc.core.indicator.registry.items():
        assert name.lower() in results.output


@pytest.mark.parametrize(
    "indicator,indname",
    [
        (xc.atmos.heating_degree_days, "heating_degree_days"),
        (xc.land.base_flow_index, "base_flow_index"),
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
        ("tg_mean", 272.15),
        ("dtrvar", 0.0),
        ("heating_degree_days", 6588.0),
        ("solidprcptot", 31622400.0),
    ],
)
def test_normal_computation(
    tasmin_series, tasmax_series, pr_series, tmp_path, indicator, expected
):
    tasmin = tasmin_series(np.ones(366) + 270.15, start="1/1/2000")
    tasmax = tasmax_series(np.ones(366) + 272.15, start="1/1/2000")
    pr = pr_series(np.ones(366), start="1/1/2000")
    ds = xr.Dataset(
        data_vars={
            "tasmin": tasmin,
            "tasmax": tasmax,
            "tas": xc.atmos.tg(tasmin, tasmax),
            "pr": pr,
        }
    )
    input_file = tmp_path / "in.nc"
    output_file = tmp_path / "out.nc"

    ds.to_netcdf(input_file)

    args = ["-i", str(input_file), "-o", str(output_file), indicator]
    runner = CliRunner()
    results = runner.invoke(cli, args)
    assert "Processing :" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    outvar = list(out.data_vars.values())[0]
    np.testing.assert_allclose(outvar[0], expected)


def test_renaming_variable(tas_series, tmp_path):
    tas = tas_series(np.ones(366), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"
    tas.name = "tas"
    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        ["-i", str(input_file), "-o", str(output_file), "tn_mean", "--tasmin", "tas"],
    )
    assert "Processing : tn_mean" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tn_mean[0] == 1.0


def test_indicator_chain(tas_series, tmp_path):
    tas = tas_series(np.ones(366), start="1/1/2000")
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
            "tg_mean",
            "growing_degree_days",
        ],
    )

    assert "Processing : tg_mean" in results.output
    assert "Processing : growing_degree_days" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tg_mean[0] == 1.0
    assert out.growing_degree_days[0] == 0


def test_missing_variable(tas_series, tmp_path):
    tas = tas_series(np.ones(366), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli, ["-i", str(input_file), "-o", str(output_file), "tn_mean"]
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
    tas = tas_series(np.ones(366), start="1/1/2000")
    tas = xr.concat([tas] * 10, dim="lat")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        ["-i", str(input_file), "-o", str(output_file)] + options + ["tg_mean"],
    )

    assert output in results.output
