# -*- coding: utf-8 -*-
"""
Miscellaneous indices utilities
===============================

Helper functions for the indices computation, things that do not belong in neither
`xclim.indices.calendar`, `xclim.indices.fwi`, `xclim.indices.generic` or `xclim.indices.run_length`.
"""
from collections import defaultdict
from functools import partial
from types import FunctionType
from boltons.funcutils import update_wrapper


def wrapped_partial(func: FunctionType, suggested: dict = None, **fixed):
    """Wrap a function, updating its signature but keeping its docstring.

    Parameters
    ----------
    func : FunctionType
        The function to be wrapped
    suggested : dict
        Keyword arguments that should have new default values
        but still appear in the signature.
    fixed : dict
        Keyword arguments that should be fixed by the wrapped
        and removed from the signature.

    Examples
    --------

    >>> from inspect import signature
    >>> def func(a, b=1, c=1):
            print(a, b, c)
    >>> newf = wrapped_partial(func, b=2)
    >>> signature(newf)
    (a, *, c=1)
    >>> newf(1)
    1, 2, 1
    >>> newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    >>> signature(newf)
    (a, *, c=2)
    >>> newf(1)
    1, 2, 2
    """
    suggested = suggested or {}
    partial_func = partial(func, **suggested, **fixed)

    fully_wrapped = update_wrapper(
        partial_func, func, injected=list(fixed.keys()), hide_wrapped=True
    )
    return fully_wrapped


# TODO Reconsider the utility of this
def walk_map(d: dict, func: FunctionType):
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
      Input dictionary, possibly nested.
    func : FunctionType
      Function to apply to dictionary values.

    Returns
    -------
    dict
      Dictionary whose values are the output of the given function.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, (dict, defaultdict)):
            out[k] = walk_map(v, func)
        else:
            out[k] = func(v)
    return out
