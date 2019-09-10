import xclim.utils as xcu


def _get_indicators(modules):
    """For all modules or classes listed, return the children that are instances of xclim.utils.Indicator.

    modules : sequence
      Sequence of modules to inspect.
    """
    out = []
    for obj in modules:
        for key, val in obj.__dict__.items():
            if isinstance(val, xcu.Indicator):
                out.append(val)

    return out


def _indicator_table():
    """Return a sequence of dicts storing metadata about all available indices."""
    from xclim import atmos
    import inspect

    inds = _get_indicators([atmos])
    table = []
    for ind in inds:
        # Apply default values
        args = {
            name: p.default
            for (name, p) in ind._sig.parameters.items()
            if p.default != inspect._empty
        }
        table.append(ind.json(args))
    return table


indicators = _indicator_table()
