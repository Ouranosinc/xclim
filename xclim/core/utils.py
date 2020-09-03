# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Miscellaneous indices utilities
===============================

Helper functions for the indices computation, things that do not belong in neither
`xclim.indices.calendar`, `xclim.indices.fwi`, `xclim.indices.generic` or `xclim.indices.run_length`.
"""
from collections import defaultdict
from functools import partial
from types import FunctionType

import numpy as np
import xarray as xr
from boltons.funcutils import update_wrapper
from dask import array as dsk


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
    """Xclim ValidationError class."""

    @property
    def msg(self):  # noqa
        return self.args[0]


def ensure_chunk_size(da: xr.DataArray, max_iter=10, **minchunks):
    """Ensure that the input dataarray has chunks of at least the given size.

    If only one chunk is too small, it is merged with an adjacent chunk.
    If many chunks are too small, they are grouped together by merging adjacent chunks.

    Parameters
    ----------
    da : xr.DataArray
      The input dataarray, with or without the dask backend. Does nothing when passed a non-dask array.
    **minchunks : int
      A kwarg mapping from dimension name to minimum chunk size.
      Pass -1 to force a single chunk along that dimension.
    """
    if not isinstance(da.data, dsk.Array):
        return da

    all_chunks = dict(zip(da.dims, da.chunks))
    chunking = {}
    for dim, minchunk in minchunks.items():
        chunks = all_chunks[dim]
        if minchunk == -1 and len(chunks) > 1:
            # Rechunk to single chunk only if it's not already one
            chunking[dim] = -1

        toosmall = np.array(chunks) < minchunk  # Chunks that are too small
        if toosmall.sum() > 1:
            # Many chunks are too small, merge them by groups
            fac = np.ceil(minchunk / min(chunks)).astype(int)
            chunking[dim] = tuple(
                sum(chunks[i : i + fac]) for i in range(0, len(chunks), fac)
            )
            # Reset counter is case the last chunks are still too small
            chunks = chunking[dim]
            toosmall = np.array(chunks) < minchunk
        if toosmall.sum() == 1:
            # Only one, merge it with adjacent chunk
            ind = np.where(toosmall)[0][0]
            new_chunks = list(chunks)
            sml = new_chunks.pop(ind)
            new_chunks[max(ind - 1, 0)] += sml
            chunking[dim] = tuple(new_chunks)

    if chunking:
        return da.chunk(chunks=chunking)
    return da
