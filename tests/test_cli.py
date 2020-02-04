#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package, command line interface
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
import xarray as xr
from click import BadOptionUsage
from click import UsageError
from click.testing import CliRunner

import xclim as xc
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
            if isinstance(ind, xc.utils.Indicator):
                assert name in results.output


@pytest.mark.parametrize(
    "indicator,indname",
    [
        (xc.atmos.heating_degree_days, "atmos.heating_degree_days"),
        (xc.seaIce.sea_ice_area, "seaIce.sea_ice_area"),
    ],
)
def test_indicator_help(indicator, indname):
    runner = CliRunner()
    results = runner.invoke(cli, [indname, "--help"])

    for name in indicator._sig.parameters.keys():
        assert name in results.output


def test_normal_computation(tas_series, tmp_path):
    tas = tas_series(np.ones(366,), start="1/1/2000")
    input_file = tmp_path / "tas.nc"
    output_file = tmp_path / "out.nc"

    tas.to_netcdf(input_file)

    runner = CliRunner()
    results = runner.invoke(
        cli, ["-i", str(input_file), "-o", str(output_file), "atmos.tg_mean"]
    )

    assert "Processing : Mean daily mean temperature" in results.output
    assert "100% Completed" in results.output

    out = xr.open_dataset(output_file)
    assert out.tg_mean[0] == 1.0


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
    assert "Processing : growing degree days" in results.output
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
        (["--dask-nthreads", "2", "--dask-maxmem", "1GB"], "Dask client started"),
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
