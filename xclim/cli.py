# -*- coding: utf-8 -*-
"""
xclim command line interface module
"""
import inspect

import click

from xclim import atmos
from xclim import land
from xclim import seaIce
from xclim.utils import available_indicators


def _process_indicator(indicator):
    print(f"processing {indicator.identifier}")


def _create_command(indicator):
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
        indicator.identifier,
        callback=_process,
        params=params,
        help=indicator.abstract,
        short_help=indicator.long_name,
    )


indicator_commands = {
    name: _create_command(indicator) for name, indicator in available_indicators.items()
}


class XclimCli(click.MultiCommand):
    def list_commands(self, ctx):
        return sorted(list(indicator_commands.keys()))

    def get_command(self, ctx, name):
        return indicator_commands[name]


cli = XclimCli(help="Command line tool to compute indices on netCDF datasets")


if __name__ == "__main__":
    cli()
