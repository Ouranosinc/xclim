# -*- coding: utf-8 -*-
"""xclim command line interface module."""
import sys
import warnings

import click
import xarray as xr
from dask.diagnostics import ProgressBar

import xclim as xc
from xclim.core.dataflags import DataQualityException, data_flags, ecad_compliant
from xclim.core.utils import InputKind

try:
    from dask.distributed import Client, progress
except ImportError:
    # Distributed is not a dependency of xclim
    Client = None


def _get_indicator(indname):
    try:
        return xc.core.indicator.registry[indname.upper()].get_instance()  # noqa
    except KeyError:
        raise click.BadArgumentUsage(f"Indicator '{indname}' not found in xclim.")


def _get_input(ctx):
    """Return the input dataset stored in the given context.

    If the dataset is not open, opens it with open_dataset if a single path was given,
    or with open_mfdataset if a tuple or glob path was given.
    """
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
    """Return the output dataset stored in the given context.

    If the output dataset doesn't exist, create it.
    """
    if "ds_out" not in ctx.obj:
        dsin = _get_input(ctx)
        ctx.obj["ds_out"] = xr.Dataset(attrs=dsin.attrs)
        if ctx.obj["output"] is None:
            raise click.BadOptionUsage(
                "output", "No output file name given.", ctx.parent
            )
    return ctx.obj["ds_out"]


def _process_indicator(indicator, ctx, **params):
    """Add given climate indicator to the output dataset from variables in the input dataset.

    Computation is not triggered here if dask is enabled.
    """
    if ctx.obj["verbose"]:
        click.echo(f"Processing : {indicator.identifier}")
    dsin = _get_input(ctx)
    dsout = _get_output(ctx)

    for key, val in params.items():
        if val == "None" or val is None:
            params[key] = None
        elif ctx.obj["verbose"]:
            click.echo(f"Parsed {key} = {val}")
    params["ds"] = dsin

    try:
        out = indicator(**params)
    except xc.core.utils.MissingVariableError as err:
        raise click.BadArgumentUsage(err.args[0])

    if isinstance(out, tuple):
        dsout = dsout.assign(**{var.name: var for var in out})
    else:
        dsout = dsout.assign({out.name: out})
    ctx.obj["ds_out"] = dsout


def _create_command(indname):
    """Generate a Click.Command from an xclim Indicator."""
    indicator = _get_indicator(indname)
    params = []
    for name, param in indicator.parameters.items():
        if name in ["ds"] or param["kind"] == InputKind.KWARGS:
            continue
        choices = "" if "choices" not in param else f" Choices: {param['choices']}"
        params.append(
            click.Option(
                param_decls=[f"--{name}"],
                default=param["default"],
                show_default=True,
                help=param["description"] + choices,
                metavar=(
                    "VAR_NAME"
                    if param["kind"]
                    in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]
                    else "TEXT"
                ),
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
        short_help=indicator.title,
    )


@click.command(short_help="Run data flag checks for input variables.")
@click.argument("variables", required=False, nargs=-1)
@click.option(
    "-r",
    "--raise-flags",
    is_flag=True,
    help="Print an exception in the event that a variable is found to have quality control issues.",
)
@click.option(
    "-a",
    "--append",
    is_flag=True,
    help="Return the netCDF dataset with the `ecad_qc_flag` array appended as a data_var.",
)
@click.option(
    "-d",
    "--dims",
    default="all",
    help='Dimensions upon which aggregation should be performed. Default: "all". Ignored if no variable provided.',
)
@click.option(
    "-f",
    "--freq",
    default=None,
    help="Resampling periods frequency used for aggregation. Default: None. Ignored if no variable provided.",
)
@click.pass_context
def dataflags(ctx, variables, raise_flags, append, dims, freq):
    """Run quality control checks on input data variables and flag for quality control issues or suspicious values."""
    ds = _get_input(ctx)
    flagged = xr.Dataset()
    output = ctx.obj["output"]
    if dims == "none":
        dims = None

    if output and raise_flags:
        ctx.fail(
            click.BadOptionUsage(
                "raise_flags",
                "Cannot use 'raise_flags' with output netCDF.",
                ctx.parent,
            )
        )
    if not output and not raise_flags:
        ctx.fail(
            click.BadOptionUsage(
                "raise_flags",
                "Must specify output or call with 'raise_flags'.",
                ctx.parent,
            )
        )

    if variables:
        exit_code = 0
        for v in variables:
            try:
                flagged_var = data_flags(
                    ds[v], ds, dims=dims, freq=freq, raise_flags=raise_flags
                )
                if output:
                    flagged = xr.merge([flagged, flagged_var])
            except DataQualityException as e:
                exit_code = 1
                tb = sys.exc_info()
                click.echo(e.with_traceback(tb[2]))
        if raise_flags:
            ctx.exit(exit_code)
    else:
        try:
            flagged = ecad_compliant(
                ds, dims=dims, raise_flags=raise_flags, append=append
            )
            if raise_flags:
                click.echo("Dataset passes quality control checks!")
                ctx.exit()
        except DataQualityException as e:
            tb = sys.exc_info()
            click.echo(e.with_traceback(tb[2]))
            ctx.exit(1)

    if output:
        ctx.obj["ds_out"] = flagged


@click.command(short_help="List indicators.")
@click.option(
    "-i", "--info", is_flag=True, help="Prints more details for each indicator."
)
def indices(info):
    """List all indicators."""
    formatter = click.HelpFormatter()
    formatter.write_heading("Listing all available indicators for computation.")
    rows = list()
    for name, indcls in xc.core.indicator.registry.items():  # noqa
        left = click.style(name.lower(), fg="yellow")
        right = ", ".join(
            [var.get("long_name", var["var_name"]) for var in indcls.cf_attrs]
        )
        if indcls.cf_attrs[0]["var_name"] != name.lower():
            right += (
                " (" + ", ".join([var["var_name"] for var in indcls.cf_attrs]) + ")"
            )
        if info:
            right += "\n" + indcls.abstract
        rows.append((left, right))
    rows.sort(key=lambda row: row[0])
    formatter.write_dl(rows)
    click.echo(formatter.getvalue())


@click.command()
@click.argument("indicator", nargs=-1)
@click.pass_context
def info(ctx, indicator):
    """Give information about INDICATOR."""
    for indname in indicator:
        ind = _get_indicator(indname)
        command = _create_command(indname)
        formatter = click.HelpFormatter()
        with formatter.section(
            click.style("Indicator", fg="blue")
            + click.style(f" {indname}", fg="yellow")
        ):
            data = ind.json()
            data.pop("parameters")
            _format_dict(data, formatter, key_fg="blue", spaces=2)

        command.format_options(ctx, formatter)

        click.echo(formatter.getvalue())


def _format_dict(data, formatter, key_fg="blue", spaces=2):
    for attr, val in data.items():
        if isinstance(val, list):
            for isub, sub in enumerate(val):
                formatter.write_text(
                    click.style(" " * spaces + f"{attr} (#{isub + 1})", fg=key_fg)
                )
                _format_dict(sub, formatter, key_fg=key_fg, spaces=spaces + 2)
        elif isinstance(val, dict):
            formatter.write_text(click.style(" " * spaces + f"{attr}:", fg=key_fg))
            _format_dict(val, formatter, key_fg=key_fg, spaces=spaces + 2)
        else:
            formatter.write_text(
                click.style(" " * spaces + attr + " :", fg=key_fg) + " " + str(val)
            )


class XclimCli(click.MultiCommand):
    """Main cli class."""

    def list_commands(self, ctx):
        """Return the available commands (other than the indicators)."""
        return "indices", "info", "dataflags"

    def get_command(self, ctx, name):
        """Return the requested command."""
        command = {"indices": indices, "info": info, "dataflags": dataflags}.get(name)
        if command is None:
            command = _create_command(name)
        return command


@click.command(
    cls=XclimCli,
    chain=True,
    help="Command line tool to compute indices on netCDF datasets. Indicators are referred to by their "
    "(case-insensitive) identifier, as in xclim.core.indicator.registry.",
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
@click.option(
    "-v", "--verbose", help="Print details about context and progress.", count=True
)
@click.option(
    "-V", "--version", is_flag=True, help="Prints xclim's version number and exits"
)
@click.option(
    "--dask-nthreads",
    type=int,
    help="Start a dask.distributed Client with this many threads and 1 worker. "
    "If not specified, the local scheduler is used. If specified, '--dask-maxmem' must also be given",
)
@click.option(
    "--dask-maxmem",
    help="Memory limit for the dask.distributed Client as a human readable string (ex: 4GB). "
    "If specified, '--dask-nthreads' must also be specified.",
)
@click.option(
    "--chunks",
    help="Chunks to use when opening the input dataset(s). "
    "Given as <dim1>:num,<dim2:num>. Ex: time:365,lat:168,lon:150.",
)
@click.pass_context
def cli(ctx, **kwargs):
    """Entry point for the command line interface.

    Manages the global options.
    """
    if not kwargs["verbose"]:
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

    if kwargs["version"]:
        click.echo(f"xclim {xc.__version__}")
    elif ctx.invoked_subcommand is None:
        raise click.UsageError("Missing command.", ctx)

    if len(kwargs["input"]) == 0:
        kwargs["input"] = None
    elif len(kwargs["input"]) == 1:
        kwargs["input"] = kwargs["input"][0]

    if kwargs["dask_nthreads"] is not None:
        if Client is None:
            raise click.BadOptionUsage(
                "dask_nthreads",
                "Dask's distributed scheduler is not installed, only the "
                "local scheduler (non-customizable) can be used.",
                ctx,
            )
        if kwargs["dask_maxmem"] is None:
            raise click.BadOptionUsage(
                "dask_nthreads",
                "'--dask-maxmem' must be given if '--dask-nthreads' is given.",
                ctx,
            )

        client = Client(
            n_workers=1,
            threads_per_worker=kwargs["dask_nthreads"],
            memory_limit=kwargs["dask_maxmem"],
        )
        click.echo(
            "Dask client started. The dashboard is available at http://127.0.0.1:"
            f"{client.scheduler_info()['services']['dashboard']}/status"
        )
    if kwargs["chunks"] is not None:
        kwargs["chunks"] = {
            dim: int(num)
            for dim, num in map(lambda x: x.split(":"), kwargs["chunks"].split(","))
        }

    kwargs["xr_kwargs"] = {"chunks": kwargs["chunks"] or {}}
    ctx.obj = kwargs


@cli.resultcallback()
@click.pass_context
def write_file(ctx, *args, **kwargs):
    """Write the output dataset to file."""
    if ctx.obj["output"] is not None:
        if ctx.obj["verbose"]:
            click.echo(f"Writing to file {ctx.obj['output']}")
        with ProgressBar():
            r = ctx.obj["ds_out"].to_netcdf(ctx.obj["output"], compute=False)
            if ctx.obj["dask_nthreads"] is not None:
                progress(r.data)
            r.compute()
        if ctx.obj["dask_nthreads"] is not None:
            click.echo("")  # Distributed's progress doesn't print a final \n.
