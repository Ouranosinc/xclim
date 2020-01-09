# -*- coding: utf-8 -*-
"""
xclim command line interface module
"""
import inspect
import sys

import click

import xclim as xc
from xclim import atmos
from xclim import land
from xclim import seaIce

xcmodules = {"atmos": atmos, "land": land, "seaIce": seaIce}
CONTEXT = {}


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


def _process_indicator(indicator):
    print(f"processing {indicator.identifier}")


def _create_command(name):
    indicator = _get_indicator(name)
    params = []
    for name, param in indicator._sig.parameters.items():
        params.append(
            click.Option(
                param_decls=["--" + name],
                default=None if param.default is inspect._empty else param.default,
                show_default=True,
                help=indicator._parameters_doc.get(name),
            )
        )

    def _process():
        return _process_indicator(indicator)

    return click.Command(
        name,
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
def list(module, info):
    """List all indicators in MODULE

    If MODULE is ommitted, lists everything.
    """
    if len(module) == 0:
        module == "all"
    print("Listing all available indicators for computation.", end="\n\n")
    for xcmod in [atmos, land, seaIce]:
        modname = xcmod.__name__.split(".")[-1]
        if module == "all" or modname in module:
            print(f"Indicators in module : {modname}")
            for name, ind in module.__dict__.items():
                if isinstance(ind, xc.utils.Indicator):
                    print(
                        f"\t{name} "
                        + (f"{ind.identifier}" if ind.identifier != name else "")
                    )
                    print(f"\t\t{ind.long_name}")
                    if info:
                        print(ind.abstract)
                    print()


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
        return ["list", "info"]

    def get_command(self, ctx, name):
        command = {"list": list, "info": info}.get(name)
        if command is None:
            command = _create_command(name)
        return command


cli = XclimCli(help="Command line tool to compute indices on netCDF datasets")


if __name__ == "__main__":
    cli()
