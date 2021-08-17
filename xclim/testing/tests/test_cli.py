#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package, command line interface
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

import xclim
from xclim.cli import cli
from xclim.testing import open_dataset

try:
    from dask.distributed import Client
except ImportError:
    Client = None


@pytest.mark.parametrize(
    "indicators,indnames",
    [
        ([xclim.atmos.tg_mean], ["tg_mean"]),
        (
            # Note: This test is dependent on indicator name length and terminal dimensions.
            [xclim.atmos.tn_mean, xclim.atmos.ice_days],
            ["tn_mean", "ice_days"],
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

    for name, ind in xclim.core.indicator.registry.items():
        assert name.lower() in results.output


@pytest.mark.parametrize(
    "indicator,indname",
    [
        (xclim.atmos.heating_degree_days, "heating_degree_days"),
        (xclim.land.base_flow_index, "base_flow_index"),
    ],
)
def test_indicator_help(indicator, indname):
    runner = CliRunner()
    results = runner.invoke(cli, [indname, "--help"])

    for name in indicator.parameters.keys():
        if name != "ds":
            assert name in results.output


@pytest.mark.parametrize(
    "indicator,expected,varnames",
    [
        ("tg_mean", 272.15, ["tas"]),
        ("dtrvar", 0.0, ["tasmin", "tasmax"]),
        ("heating_degree_days", 6588.0, ["tas"]),
        ("solidprcptot", 31622400.0, ["tas", "pr"]),
    ],
)
def test_normal_computation(
    tasmin_series, tasmax_series, pr_series, tmp_path, indicator, expected, varnames
):
    tasmin = tasmin_series(np.ones(366) + 270.15, start="1/1/2000")
    tasmax = tasmax_series(np.ones(366) + 272.15, start="1/1/2000")
    pr = pr_series(np.ones(366), start="1/1/2000")
    ds = xr.Dataset(
        data_vars={
            "tasmin": tasmin,
            "tasmax": tasmax,
            "tas": xclim.atmos.tg(tasmin, tasmax),
            "pr": pr,
        }
    )
    input_file = tmp_path / "in.nc"
    output_file = tmp_path / "out.nc"

    ds.to_netcdf(input_file)

    args = ["-i", str(input_file), "-o", str(output_file), "-v", indicator]
    runner = CliRunner()
    results = runner.invoke(cli, args)
    for varname in varnames:
        assert f"Parsed {varname} = {varname}" in results.output
    assert "Processing :" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    outvar = list(out.data_vars.values())[0]
    np.testing.assert_allclose(outvar[0], expected)


def test_multi_input(tas_series, pr_series, tmp_path):
    tas = tas_series(np.ones(366) + 273.15, start="1/1/2000")
    pr = pr_series(np.ones(366), start="1/1/2000")
    tas_file = tmp_path / "multi_tas_in.nc"
    pr_file = tmp_path / "multi_pr_in.nc"
    output_file = tmp_path / "out.nc"

    tas.to_dataset().to_netcdf(tas_file)
    pr.to_dataset().to_netcdf(pr_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        [
            "-i",
            str(tmp_path / "multi_*_in.nc"),
            "-o",
            str(output_file),
            "-v",
            "solidprcptot",
        ],
    )
    assert "Processing : solidprcptot" in results.output

    out = xr.open_dataset(output_file)
    assert out.solidprcptot.sum() == 0


def test_multi_output(tmp_path):
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    input_file = tmp_path / "ws_in.nc"
    output_file = tmp_path / "out.nc"
    ds.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli,
        [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "-v",
            "wind_speed_from_vector",
        ],
    )
    assert "Processing : wind_speed_from_vector" in results.output


def test_renaming_variable(tas_series, tmp_path):
    tas = tas_series(np.ones(366), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"
    tas.name = "tas"
    tas.to_netcdf(input_file)
    with xclim.set_options(cf_compliance="warn"):
        runner = CliRunner()
        results = runner.invoke(
            cli,
            [
                "-i",
                str(input_file),
                "-o",
                str(output_file),
                "-v",
                "tn_mean",
                "--tasmin",
                "tas",
            ],
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
            "-v",
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
    assert "'tasmin' was not found in the input dataset." in results.output


@pytest.mark.parametrize(
    "options,output",
    [
        (["--dask-nthreads", "2"], "Error: '--dask-maxmem' must be given"),
        (["--chunks", "time:90"], "100% Complete"),
        (["--chunks", "time:90,lat:5"], "100% Completed"),
        (["--version"], xclim.__version__),
    ],
)
def test_global_options(tas_series, tmp_path, options, output):
    if "dask" in options[0]:
        pytest.importorskip("dask.distributed")
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


def test_bad_usage(tas_series, tmp_path):
    tas = tas_series(np.ones(366), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()

    # No command
    results = runner.invoke(cli, ["-i", str(input_file)])
    assert "Missing command" in results.output

    # Indicator not found:
    results = runner.invoke(cli, ["info", "mean_ether_velocity"])
    assert "Indicator 'mean_ether_velocity' not found in xclim" in results.output

    # No input file given
    results = runner.invoke(cli, ["-o", str(output_file), "base_flow_index"])
    assert "No input file name given" in results.output

    # No output file given
    results = runner.invoke(cli, ["-i", str(input_file), "tg_mean"])
    assert "No output file name given" in results.output

    results = runner.invoke(
        cli,
        [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "--dask-nthreads",
            "2",
            "tg_mean",
        ],
    )
    if Client is None:  # dask.distributed not installed
        assert "distributed scheduler is not installed" in results.output
    else:
        assert "'--dask-maxmem' must be given" in results.output
