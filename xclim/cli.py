# -*- coding: utf-8 -*-
"""
xclim command line interface module
"""
import inspect

import click
import xarray as xr
from dask.diagnostics import ProgressBar

import xclim as xc

xcmodules = {"atmos": xc.atmos, "land": xc.land, "seaIce": xc.seaIce}


def _get_indicator(indname):
    if "." in indname:
        modname, indname = indname.split(".")
    else:
        raise click.BadArgumentUsage(
            f"Indicator name must include the module name (ex: atmos.tg_mean) (got {indname})"
        )

    if modname not in xcmodules or indname not in xcmodules[modname].__dict__:
        raise click.BadArgumentUsage(
            f"Indicator '{indname}' or module '{modname}' not found in xclim."
        )

    return xcmodules[modname].__dict__[indname]


def _get_input(ctx):
    arg = ctx.obj["input"]
    if arg is None:
        raise click.BadOptionUsage("input", "No input file name given.", ctx.parent)
    if isinstance(arg, xr.Dataset):
        return arg
    if isinstance(arg, tuple) or "*" in arg:
        ctx.obj["xr_kwargs"].setdefault("combine", "by_coords")
        ds = xr.open_mfdataset(arg, **ctx.obj["xr_kwargs"])
    else:
        ctx.obj["xr_kwargs"].pop("combine", None)
        ds = xr.open_dataset(arg, **ctx.obj["xr_kwargs"])
    ctx.obj["input"] = ds
    return ds


def _get_output(ctx):
    if "ds_out" not in ctx.obj:
        dsin = _get_input(ctx)
        ctx.obj["ds_out"] = xr.Dataset(attrs=dsin.attrs)
        if ctx.obj["output"] is None:
            raise click.BadOptionUsage(
                "output", "No output file name given.", ctx.parent
            )
    return ctx.obj["ds_out"]


def _process_indicator(indicator, ctx, **params):
    click.echo(
        f"Processing : {indicator.format({'long_name': indicator.long_name}, params)['long_name']}"
    )
    dsin = _get_input(ctx)
    dsout = _get_output(ctx)

    for key, val in params.items():
        # a Dataset is expected
        if indicator._sig.parameters[key].default is inspect._empty:
            if key == "tas" and val is None and key not in dsin.data_vars:
                # Special case for tas.
                try:
                    params["tas"] = xc.atmos.tg(
                        dsin[ctx.obj["tas_from"][0]], dsin[ctx.obj["tas_from"][1]]
                    )
                except KeyError:
                    raise click.UsageError(
                        f"Dataset neither provides needed variable {key} nor the pair {ctx.obj['tas_from']} that could be used to construct it. Set the '--tas-from' global option or directly give a name with the '--tas' indicator option.",
                        ctx,
                    )
            else:
                # Either a variable name was given or the key is the name
                try:
                    params[key] = dsin[val or key]
                except KeyError:
                    raise click.BadArgumentUsage(
                        f"Variable {val or key} absent from input dataset. You can provide an alternative name with --{key}",
                        ctx,
                    )

    var = indicator(**params)
    dsout = dsout.assign({var.name: var})
    ctx.obj["ds_out"] = dsout


def _create_command(indname):
    indicator = _get_indicator(indname)
    params = []
    for name, param in indicator._sig.parameters.items():
        params.append(
            click.Option(
                param_decls=[f"--{name}"],
                default=None if param.default is inspect._empty else param.default,
                show_default=True,
                help=indicator._parameters_doc.get(name),
            )
        )

    @click.pass_context
    def _process(ctx, **kwargs):
        return _process_indicator(indicator, ctx, **kwargs)

    return click.Command(
        indname,
        callback=_process,
        params=params,
        help=indicator.abstract,
        short_help=indicator.long_name,
    )


@click.command(short_help="List indicators.")
@click.argument("module", required=False, nargs=-1)
@click.option(
    "-i", "--info", is_flag=True, help="Prints more details for each indicator."
)
def indices(module, info):
    """List all indicators in MODULE

    If MODULE is ommitted, lists everything.
    """
    if len(module) == 0:
        module = "all"
    formatter = click.HelpFormatter()
    formatter.write_heading("Listing all available indicators for computation.")
    for xcmod in [xc.atmos, xc.land, xc.seaIce]:
        modname = xcmod.__name__.split(".")[-1]
        if module == "all" or modname in module:
            with formatter.section(
                click.style("Indicators in module ", fg="blue")
                + click.style(f"{modname}", fg="yellow")
            ):
                rows = []
                for name, ind in xcmod.__dict__.items():
                    if isinstance(ind, xc.utils.Indicator):
                        left = click.style(name, fg="yellow")
                        if ind.var_name != name:
                            left += f" ({ind.var_name})"
                        right = ind.long_name
                        if info:
                            right += "\n" + ind.abstract
                        rows.append((left, right))
                formatter.write_dl(rows)
    click.echo(formatter.getvalue())


@click.command()
@click.argument("indicator", nargs=-1)
@click.pass_context
def info(ctx, indicator):
    """Gives information about INDICATOR.

    Same as "xclim INDICATOR --help"

    INDICATOR must include its module (ex: atmos.tg_mean)
    """
    for indname in indicator:
        ind = _get_indicator(indname)
        command = _create_command(indname)

        formatter = click.HelpFormatter()
        with formatter.section(
            click.style("Indicator", fg="blue")
            + click.style(f" {indname}", fg="yellow")
        ):
            for attrs in [
                "identifier",
                "var_name",
                "title",
                "long_name",
                "units",
                "cell_methods",
                "abstract",
                "description",
            ]:
                formatter.write_text(
                    click.style(f"{attrs}: ", fg="blue") + f"{getattr(ind, attrs)}"
                )

        command.format_options(ctx, formatter)

        click.echo(formatter.getvalue())


class XclimCli(click.MultiCommand):
    def list_commands(self, ctx):
        return "indices", "info"

    def get_command(self, ctx, name):
        command = {"indices": indices, "info": info}.get(name)
        if command is None:
            command = _create_command(name)
        return command


@click.command(
    cls=XclimCli,
    chain=True,
    help="Command line tool to compute indices on netCDF datasets",
    invoke_without_command=True,
    subcommand_metavar="INDICATOR1 [OPTIONS] ... [INDICATOR2 [OPTIONS] ... ] ...",
)
@click.option(
    "-i",
    "--input",
    help="Input files. Can be a netCDF path or a glob pattern.",
    multiple=True,
)
@click.option("-o", "--output", help="Output filepath. A new file will be created")
@click.option("-v", "--verbose", help="Make it more verbose", count=True)
@click.option(
    "--tas-from",
    nargs=2,
    help="Variable names in the input dataset for 'tasmin' and 'tasmax', used when 'tas' is needed but absent from the dataset",
    default=("tasmax", "tasmin"),
)
@click.option("--version", is_flag=True, help="Prints xclim's version number and exits")
@click.pass_context
def cli(ctx, **kwargs):
    if kwargs["version"]:
        click.echo(f"xclim {xc.__version__}")
    elif ctx.invoked_subcommand is None:
        raise click.UsageError("Missing command.", ctx)
    if len(kwargs["input"]) == 0:
        kwargs["input"] = None
    elif len(kwargs["input"]) == 1:
        kwargs["input"] = kwargs["input"][0]
    kwargs["xr_kwargs"] = {"chunks": {}}
    ctx.obj = kwargs


@cli.resultcallback()
@click.pass_context
def write_file(ctx, *args, **kwargs):
    if ctx.obj["output"] is not None:
        click.echo(f"Writing everything to file {ctx.obj['output']}")
        with ProgressBar():
            ctx.obj["ds_out"].to_netcdf(ctx.obj["output"])


if __name__ == "__main__":
    cli()
