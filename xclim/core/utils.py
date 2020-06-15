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

import dask.array as dsk
import numpy as np
import xarray as xr
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
    fixed : kwargs
        Keyword arguments that should be fixed by the wrapped
        and removed from the signature.

    Examples
    --------

    >>> from inspect import signature
    >>> def func(a, b=1, c=1):
    ...     print(a, b, c)
    >>> newf = wrapped_partial(func, b=2)
    >>> signature(newf)
    <Signature (a, *, c=1)>
    >>> newf(1)
    1 2 1
    >>> newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    >>> signature(newf)
    <Signature (a, *, c=2)>
    >>> newf(1)
    1 2 2
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


class ValidationError(ValueError):
    @property
    def msg(self):
        return self.args[0]


def ensure_chunk_size(da: xr.DataArray, **minchunks):
    if not isinstance(da.data, dsk.Array):
        return da

    chunks = dict(zip(da.dims, da.chunks))
    chunking = {}
    for dim, minchunk in minchunks.items():
        if minchunk == -1 and len(chunks[dim]) != 1:
            chunking[dim] = -1
        toosmall = (np.array(chunks[dim]) < minchunk).sum()
        if toosmall == 1:
            ind = np.where(toosmall)[0]
            new_chunks = list(chunks[dim])
            sml = new_chunks.pop(ind)
            new_chunks[max(ind - 1, 0)] += sml
            chunking[dim] = new_chunks
        elif toosmall > 1:
            fac = np.ceil(minchunk / min(chunks[dim]))
            chunking[dim] = [
                chunks[dim][i : i + fac] for i in range(len(chunks[dim]), fac)
            ]
    if chunking:
        return da.chunk(chunks=chunking)
    return da
