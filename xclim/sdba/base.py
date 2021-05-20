"""Base classes."""
from inspect import signature
from types import FunctionType
from typing import Callable, Mapping, Optional, Sequence, Set, Union

import dask.array as dsk
import jsonpickle
import numpy as np
import xarray as xr
from boltons.funcutils import wraps

from xclim.core.calendar import days_in_year, get_calendar
from xclim.core.utils import uses_dask


# ## Base class for the sdba module
class Parametrizable(dict):
    """Helper base class resembling a dictionary.

    This object is _completely_ defined by the content of its internal dictionary, accessible through item access
    (`self['attr']`) or in `self.parameters`. When serializing and restoring this object, only members of that internal
    dict are preserved. All other attributes set directly with `self.attr = value` will not be preserved upon serialization
    and restoration of the object with `[json]pickle`.
    dictionary. Other variables set with `self.var = data` will be lost in the serialization process.
    This class is best serialized and restored with `jsonpickle`.
    """

    _repr_hide_params = []

    def __getstate__(self):
        """For (json)pickle, a Parametrizable should be defined by its internal dict only."""
        return self.parameters

    def __setstate__(self, state):
        """For (json)pickle, a Parametrizable in only defined by its internal dict."""
        self.update(state)

    def __getattr__(self, attr):
        """Get attributes."""
        try:
            return self.__getitem__(attr)
        except KeyError as err:
            # Raise the proper error type for getattr
            raise AttributeError(*err.args)

    @property
    def parameters(self):
        """All parameters as a dictionary. Read-only."""
        return dict(**self)

    def __repr__(self):
        """Return a string representation that allows eval to recreate it."""
        params = ", ".join(
            [
                f"{k}={repr(v)}"
                for k, v in self.items()
                if k not in self._repr_hide_params
            ]
        )
        return f"{self.__class__.__name__}({params})"


class ParametrizableWithDataset(Parametrizable):
    """Parametrizeable class that also has a `ds` attribute storing a dataset."""

    _attribute = "_xclim_parameters"

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """Create an instance from a dataset.

        The dataset must have a global attribute with a name corresponding to `cls._attribute`,
        and that attribute must be the result of `jsonpickle.encode(object)` where object is
        of the same type as this object.
        """
        obj = jsonpickle.decode(ds.attrs[cls._attribute])
        obj.set_dataset(ds)
        return obj

    def set_dataset(self, ds: xr.Dataset):
        """Stores an xarray dataset in the `ds` attribute.

        Useful with custom object initialization or if some external processing was performed.
        """
        self.ds = ds
        self.ds.attrs[self._attribute] = jsonpickle.encode(self)


class Grouper(Parametrizable):
    """Helper object to perform grouping actions on DataArrays and Datasets."""

    _repr_hide_params = ["dim", "prop"]  # For a concise repr
    # Two constants for use of `map_blocks` and `map_groups`.
    # They provide better code readability, nothing more
    PROP = "<PROP>"
    DIM = "<DIM>"

    def __init__(
        self,
        group: str,
        window: int = 1,
        add_dims: Optional[Union[Sequence[str], Set[str]]] = None,
        interp: Union[bool, str] = False,
    ):
        """Create the Grouper object.

        Parameters
        ----------
        group : str
          The usual grouping name as xarray understands it. Ex: "time.month" or "time".
          The dimension name before the dot is the "main dimension" stored in `Grouper.dim` and
          the property name after is stored in `Grouper.prop`.
        window : int
          If larger than 1, a centered rolling window along the main dimension is created when grouping data.
          Units are the sampling frequency of the data along the main dimension.
        add_dims : Optional[Sequence[str]]
          Additionnal dimensions that should be reduced in grouping operations. This behaviour is also controlled
          by the `main_only` parameter of the `apply` method. If any of these dimensions are absent from the dataarrays,
          they will be omitted.
        interp : Union[bool, str]
          Whether to return an interpolatable index in the `get_index` method. Only effective for `month` grouping.
          Interpolation method names are accepted for convenience, "nearest" is translated to False, all other names are translated to True.
          This modifies the default, but `get_index` also accepts an `interp` argument overriding the one defined here..
        """
        if "." in group:
            dim, prop = group.split(".")
        else:
            dim, prop = group, "group"

        if isinstance(interp, str):
            interp = interp != "nearest"

        add_dims = add_dims or []
        if window > 1:
            add_dims.insert(1, "window")
        super().__init__(
            dim=dim,
            add_dims=add_dims,
            prop=prop,
            name=group,
            window=window,
            interp=interp,
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        kwargs["group"] = cls(
            group=kwargs.pop("group"),
            window=kwargs.pop("window", 1),
            add_dims=kwargs.pop("add_dims", []),
            interp=kwargs.get("interp", False),
        )
        return kwargs

    def get_coordinate(self, ds=None):
        """Return the coordinate as in the output of group.apply.

        Currently only implemented for groupings with prop == month or dayofyear.
        For prop == dayfofyear a ds (dataset or dataarray) can be passed to infer
        the max doy from the available years and calendar.
        """
        if self.prop == "month":
            return xr.DataArray(np.arange(1, 13), dims=("month",), name="month")
        if self.prop == "dayofyear":
            if ds is not None:
                cal = get_calendar(ds, dim=self.dim)
                mdoy = max(
                    days_in_year(yr, cal) for yr in np.unique(ds[self.dim].dt.year)
                )
            else:
                mdoy = 365
            return xr.DataArray(
                np.arange(1, mdoy + 1), dims=("dayofyear"), name="dayofyear"
            )
        if self.prop == "group":
            return xr.DataArray([1], dims=("group",), name="group")
        # TODO woups what happens when there is no group? (prop is None)
        raise NotImplementedError()

    def group(
        self,
        da: Union[xr.DataArray, xr.Dataset] = None,
        main_only=False,
        **das: xr.DataArray,
    ):
        """Return a xr.core.groupby.GroupBy object.

        More than one array can be combined to a dataset before grouping using the `das`  kwargs.
        A new `window` dimension is added if `self.window` is larger than 1.
        If `Grouper.dim` is 'time', but 'prop' is None, the whole array is grouped together.

        When multiple arrays are passed, some of them can be grouped along the same group as self.
        They are boadcasted, merged to the grouping dataset and regrouped in the output.
        """
        if das:
            from .utils import broadcast  # pylint: disable=cyclic-import

            if da is not None:
                das[da.name] = da

            da = xr.Dataset(
                data_vars={
                    name: das.pop(name)
                    for name in list(das.keys())
                    if self.dim in das[name].dims
                }
            )

            # "Ungroup" the grouped arrays
            da = da.assign(
                {
                    name: broadcast(var, da[self.dim], group=self, interp="nearest")
                    for name, var in das.items()
                }
            )

        if not main_only and self.window > 1:
            da = da.rolling(center=True, **{self.dim: self.window}).construct(
                window_dim="window"
            )
            if uses_dask(da):
                # Rechunk. There might be padding chunks.
                da = da.chunk({self.dim: -1})

        if self.prop == "group":
            group = self.get_index(da)
        else:
            group = self.name

        return da.groupby(group)

    def get_index(
        self,
        da: Union[xr.DataArray, xr.Dataset],
        interp: Optional[Union[bool, str]] = None,
    ):
        """Return the group index of each element along the main dimension.

        Parameters
        ----------
        da : Union[xr.DataArray, xr.Dataset]
          The input array/dataset for which the group index is returned.
          It must have Grouper.dim as a coordinate.
        interp : Union[bool, str]
          Argument `interp` defaults to `self.interp`. If True, the returned index can be
          used for interpolation. For month grouping, integer values represent the middle of the month, all other
          days are linearly interpolated in between.

        Returns
        -------
        xr.DataArray
          The index of each element along `Grouper.dim`.
          If `Grouper.dim` is `time` and `Grouper.prop` is None, an uniform array of True is returned.
          If `Grouper.prop` is a time accessor (month, dayofyear, etc), an numerical array is returned,
            with a special case of `month` and `interp=True`.
          If `Grouper.dim` is not `time`, the dim is simply returned.
        """
        if self.prop == "group":
            if self.dim == "time":
                return xr.full_like(da[self.dim], 1, dtype=int).rename("group")
            return da[self.dim].rename("group")

        ind = da.indexes[self.dim]
        i = getattr(ind, self.prop)

        if not np.issubdtype(i.dtype, np.integer):
            raise ValueError(
                f"Index {self.name} is not of type int (rather {i.dtype}), but {self.__class__.__name__} requires integer indexes."
            )

        interp = (
            (interp or self.interp)
            if not isinstance(interp, str)
            else interp != "nearest"
        )
        if interp:
            if self.dim == "time":
                if self.prop == "month":
                    i = ind.month - 0.5 + ind.day / ind.days_in_month
                elif self.prop == "dayofyear":
                    i = ind.dayofyear
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        xi = xr.DataArray(
            i,
            dims=self.dim,
            coords={self.dim: da.coords[self.dim]},
            name=self.dim + " group index",
        )

        # Expand dimensions of index to match the dimensions of da
        # We want vectorized indexing with no broadcasting
        # xi = xi.broadcast_like(da)
        xi.name = self.prop
        return xi

    def apply(
        self,
        func: Union[FunctionType, str],
        da: Union[xr.DataArray, Mapping[str, xr.DataArray], xr.Dataset],
        main_only: bool = False,
        **kwargs,
    ):
        """Apply a function group-wise on DataArrays.

        Parameters
        ----------
        func : Union[FunctionType, str]
          The function to apply to the groups, either a callable or a `xr.core.groupby.GroupBy` method name as a string.
          The function will be called as `func(group, dim=dims, **kwargs)`. See `main_only` for the behaviour of `dims`.
        da : Union[xr.DataArray, Mapping[str, xr.DataArray], xr.Dataset]
          The DataArray on which to apply the function. Multiple arrays can be passed through a dictionary. A dataset will be created before grouping.
        main_only : bool
          Whether to call the function with the main dimension only (if True)
          or with all grouping dims (if False, default) (including the window and dimensions given through `add_dims`).
          The dimensions used are also written in the "group_compute_dims" attribute.
          If all the input arrays are missing one of the 'add_dims', it is silently omitted.
        **kwargs :
          Other keyword arguments to pass to the function.

        Returns
        -------
        DataArray or Dataset
          Attributes "group", "group_window" and "group_compute_dims" are added.
          If the function did not reduce the array:
            - The output is sorted along the main dimension.
            - The output is rechunked to match the chunks on the input
                If multiple inputs with differing chunking were given as inputs, the chunking with the smallest number of chunks is used.
          If the function reduces the array:
            - If there is only one group, the singleton dimension is squeezed out of the output
            - The output is rechunked as to have only 1 chunk along the new dimension.


        Notes
        -----
        For the special case where a Dataset is returned, but only some of its variable where reduced by the grouping, xarray's `GroupBy.map` will
        broadcast everything back to the ungrouped dimensions. To overcome this issue, function may add a "_group_apply_reshape" attribute set to
        True on the variables that should be reduced and these will be re-grouped by calling `da.groupby(self.name).first()`.
        """
        if isinstance(da, (dict, xr.Dataset)):
            grpd = self.group(main_only=main_only, **da)
            dim_chunks = min(  # Get smallest chunking to rechunk if the operation is non-grouping
                [
                    d.chunks[d.get_axis_num(self.dim)]
                    for d in da.values()
                    if uses_dask(d) and self.dim in d.dims
                ]
                or [[]],  # pass [[]] if no dataarrays have chunks so min doesnt fail
                key=len,
            )
        else:
            grpd = self.group(da, main_only=main_only)
            # Get chunking to rechunk is the operation is non-grouping
            # To match the behaviour of the case above, an empty list signifies that dask is not used for the input.
            dim_chunks = (
                [] if not uses_dask(da) else da.chunks[da.get_axis_num(self.dim)]
            )

        dims = self.dim
        if not main_only:
            dims = [dims] + [dim for dim in self.add_dims if dim in grpd.dims]

        if isinstance(func, str):
            out = getattr(grpd, func)(dim=dims, **kwargs)
        else:
            out = grpd.map(func, dim=dims, **kwargs)

        # Case where the function wants to return more than one variables
        # and that some have grouped dims and other have the same dimensions as the input.
        # In that specific case, groupby broadcasts everything back to the input's dim, copying the grouped data.
        if isinstance(out, xr.Dataset):
            for name, outvar in out.data_vars.items():
                if "_group_apply_reshape" in outvar.attrs:
                    out[name] = self.group(outvar, main_only=True).first(
                        skipna=False, keep_attrs=True
                    )
                    del out[name].attrs["_group_apply_reshape"]

        # Save input parameters as attributes of output DataArray.
        out.attrs["group"] = self.name
        out.attrs["group_compute_dims"] = dims
        out.attrs["group_window"] = self.window

        # On non reducing ops, drop the constructed window
        if self.window > 1 and "window" in out.dims:
            out = out.isel(window=self.window // 2, drop=True)

        # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
        if self.dim in out.dims:
            out = out.sortby(self.dim)
            # The expected behavior for downstream methods would be to conserve chunking along dim
            if uses_dask(out):
                # or -1 in case dim_chunks is [], when no input is chunked (only happens if the operation is chunking the output)
                out = out.chunk({self.dim: dim_chunks or -1})
        if self.prop in out.dims and uses_dask(out):
            # Same as above : downstream methods expect only one chunk along the group
            out = out.chunk({self.prop: -1})

        return out


def parse_group(func: Callable) -> Callable:
    """Parse the "group" argument of a function and return a Grouper object.

    Adds the possiblity to pass a window argument and a list of dimensions in group.
    """
    sig = signature(func)
    if "group" in sig.parameters:
        default_group = sig.parameters["group"].default
    else:
        default_group = None

    @wraps(func)
    def _parse_group(*args, **kwargs):
        if default_group:
            kwargs.setdefault("group", default_group)
        elif "group" not in kwargs:
            raise ValueError("'group' argument not given.")
        if not isinstance(kwargs["group"], Grouper):
            kwargs = Grouper.from_kwargs(**kwargs)
        return func(*args, **kwargs)

    return _parse_group


def duck_empty(dims, sizes, chunks=None):
    """Return an empty DataArray based on a numpy or dask backend, depending on the chunks argument."""
    shape = [sizes[dim] for dim in dims]
    if chunks:
        chnks = [chunks.get(dim, (sizes[dim],)) for dim in dims]
        content = dsk.empty(shape, chunks=chnks)
    else:
        content = np.empty(shape)
    return xr.DataArray(content, dims=dims)


def map_blocks(reduces=None, **outvars):
    """
    Decorator for declaring functions and wrapping them into a map_blocks. It takes care of constructing
    the template dataset.

    If `group` is in the kwargs, it is assumed that `group.dim` is the only dimension reduced or modified,
    and that some other dimensions might be added, but no other existing dimension will be modified.

    Arguments to the decorator are mappings from variable name in the output to its *new* dimensions.
    Dimension order is not preserved.
    The placeholders "<PROP>" and "<DIM>" can be used to signify `group.prop` and `group.dim` respectively.

    The decorated function must always have the signature: func(ds, **kwargs), where ds is a DataArray or a Dataset.
    It must always output a dataset matching the mapping passed to the decorator.
    """

    def merge_dimensions(*seqs):
        """Merge several dimensions lists while preserving order."""
        out = seqs[0].copy()
        for seq in seqs[1:]:
            last_index = 0
            for i, e in enumerate(seq):
                if e in out:
                    indx = out.index(e)
                    if indx < last_index:
                        raise ValueError(
                            "Dimensions order mismatch, lists are not mergeable."
                        )
                    last_index = indx
                else:
                    out.insert(last_index + 1, e)
        return out

    # Ordered list of all added dimensions
    out_dims = merge_dimensions(*outvars.values())
    # List of dimensions reduced by the function.
    red_dims = reduces or []

    def _decorator(func):

        # @wraps(func, hide_wrapped=True)
        @parse_group
        def _map_blocks(ds, **kwargs):
            if isinstance(ds, xr.Dataset):
                ds = ds.unify_chunks()

            # Get group if present
            group = kwargs.get("group")

            # Ensure group is given as it might not be in the signature of the wrapped func
            if {Grouper.PROP, Grouper.DIM}.intersection(
                out_dims + red_dims
            ) and group is None:
                raise ValueError("Missing required `group` argument.")

            if uses_dask(ds):
                # Use dask if any of the input is dask-backed.
                chunks = (
                    dict(ds.chunks)
                    if isinstance(ds, xr.Dataset)
                    else dict(zip(ds.dims, ds.chunks))
                )
                if (
                    group is not None
                    and group.dim in chunks
                    and len(chunks[group.dim]) > 1
                ):
                    raise ValueError(
                        f"The dimension over which we group cannot be chunked ({group.dim} has chunks {chunks[group.dim]})."
                    )
            else:
                chunks = None

            # Make translation dict
            if group is not None:
                placeholders = {Grouper.PROP: group.prop, Grouper.DIM: group.dim}
            else:
                placeholders = {}

            # Get new dimensions (in order), translating placeholders to real names.
            new_dims = [placeholders.get(dim, dim) for dim in out_dims]
            reduced_dims = [placeholders.get(dim, dim) for dim in red_dims]

            for dim in new_dims:
                if dim in ds.dims and dim not in reduced_dims:
                    raise ValueError(
                        f"Dimension {dim} is meant to be added by the computation but it is already on one of the inputs."
                    )

            # Dimensions untouched by the function.
            base_dims = list(set(ds.dims) - set(new_dims) - set(reduced_dims))

            # All dimensions of the output data, new_dims are added at the end on purpose.
            all_dims = base_dims + new_dims
            # The coordinates of the output data.
            coords = {}
            for dim in all_dims:
                if dim == group.prop:
                    coords[group.prop] = group.get_coordinate(ds=ds)
                elif dim == group.dim:
                    coords[group.dim] = ds[group.dim]
                elif dim in ds.coords:
                    coords[dim] = ds[dim]
                elif dim in kwargs:
                    coords[dim] = xr.DataArray(kwargs[dim], dims=(dim,), name=dim)
                else:
                    raise ValueError(
                        f"This function adds the {dim} dimension, its coordinate must be provided as a keyword argument."
                    )
            sizes = {name: crd.size for name, crd in coords.items()}

            # Create the output dataset, but empty
            tmpl = xr.Dataset(coords=coords)
            for var, dims in outvars.items():
                # Out variables must have the base dims + new_dims
                dims = base_dims + [placeholders.get(dim, dim) for dim in dims]
                # duck empty calls dask if chunks is not None
                tmpl[var] = duck_empty(dims, sizes, chunks)

            def _call_and_transpose_on_exit(dsblock, **kwargs):
                """Call the decorated func and transpose to ensure the same dim order as on the templace."""
                out = func(dsblock, **kwargs).transpose(*all_dims)
                for name, crd in dsblock.coords.items():
                    if name not in out.coords and set(crd.dims).issubset(out.dims):
                        out = out.assign_coords({name: dsblock[name]})
                return out

            # Fancy patching for explicit dask task names
            _call_and_transpose_on_exit.__name__ = f"block_{func.__name__}"

            out = ds.map_blocks(
                _call_and_transpose_on_exit, template=tmpl, kwargs=kwargs
            )

            return out

        return _map_blocks

    return _decorator


def map_groups(reduces=[Grouper.DIM], main_only=False, **outvars):
    """
    Decorator for declaring functions acting only on groups and wrapping them into a map_blocks.
    See :py:func:`map_blocks`.

    This is the same as `map_blocks` but adds a call to `group.apply()` in the mapped func.

    It also adds an additional "main_only" argument which is the same as for group.apply.

    Finally, the decorated function must have the signature: func(ds, dim, **kwargs).
    Where ds is a DataAray or Dataset, dim is the group.dim (and add_dims). The `group` argument
    is stripped from the kwargs, but must evidently be provided in the call.
    """

    def _decorator(func):
        decorator = map_blocks(reduces=reduces, **outvars)

        def _apply_on_group(dsblock, **kwargs):
            group = kwargs.pop("group")
            return group.apply(func, dsblock, main_only=main_only, **kwargs)

        # Fancy patching for explicit dask task names
        _apply_on_group.__name__ = f"group_{func.__name__}"

        # wraps(func, injected=['dim'], hide_wrapped=True)(
        return decorator(_apply_on_group)

    return _decorator
