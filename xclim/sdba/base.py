"""Base classes."""
from inspect import signature
from types import FunctionType
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr
from boltons.funcutils import wraps


# ## Base class for the sdba module
class Parametrizable(dict):
    """Helper base class resembling a dictionary.

    Only parameters passed in the init or set using item access "[ ]" are considered as such and returned in the
    :py:meth:`Parametrizable.parameters` dictionary, the copy method and the class representation.
    """

    def __getattr__(self, attr):
        """Get attributes."""
        try:
            return self.__getitem__(attr)
        except KeyError as err:
            # Raise the proper error type for getattr
            raise AttributeError(*err.args)

    @property
    def parameters(self):
        """All parameters as a dictionary."""
        return dict(**self)

    def copy(self):
        """Return a copy of this instance."""
        return self.__class__(**self.parameters)

    def __repr__(self):
        """Return a string representation that allows eval to recreate it."""
        params = ", ".join([f"{k}={repr(v)}" for k, v in self.items()])
        return f"{self.__class__.__name__}({params})"


class Grouper(Parametrizable):
    """Helper object to perform grouping actions on DataArrays and Datasets."""

    def __init__(
        self,
        group: str,
        window: int = 1,
        add_dims: Optional[Sequence[str]] = None,
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
            dim, prop = group, None

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

    def group(self, da: xr.DataArray = None, **das: xr.DataArray):
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

        if self.window > 1:
            da = da.rolling(center=True, **{self.dim: self.window}).construct(
                window_dim="window"
            )

        if self.prop is None and self.dim == "time":
            group = self.get_index(da)
            group.name = self.dim
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
        if self.prop is None:
            if self.dim == "time":
                return xr.full_like(da[self.dim], True, dtype=bool)
            return da[self.dim]

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
        da: Union[xr.DataArray, Mapping[str, xr.DataArray]],
        main_only: bool = False,
        **kwargs,
    ):
        """Apply a function group-wise on DataArrays.

        Parameters
        ----------
        func : Union[FunctionType, str]
          The function to apply to the groups, either a callable or a `xr.core.groupby.GroupBy` method name as a string.
          The function will be called as `func(group, dim=dims, **kwargs)`. See `main_only` for the behaviour of `dims`.
        da : Union[xr.DataArray, Mapping[str, xr.DataArray]]
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
        if isinstance(da, dict):
            grpd = self.group(**da)
            dim_chunks = min(  # Get smallest chunking to rechunk if the operation is non-grouping
                [
                    d.chunks[d.get_axis_num(self.dim)]
                    for d in da.values()
                    if d.chunks and self.dim in d.dims
                ]
                or [[]],  # pass [[]] if no dataarrays have chunks so min doesnt fail
                key=len,
            )
        else:
            grpd = self.group(da)
            # Get chunking to rechunk is the operation is non-grouping
            # To match the behaviour of the case above, an empty list signifies that dask is not used for the input.
            dim_chunks = (
                [] if da.chunks is None else da.chunks[da.get_axis_num(self.dim)]
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
                    if self.prop is not None:
                        out[name] = outvar.groupby(self.name).first(
                            skipna=False, keep_attrs=True
                        )
                    else:
                        out[name] = out[name].isel({self.dim: 0})
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
            if out[self.dim].size == 1:
                out = out.squeeze(self.dim, drop=True)  # .drop_vars(self.dim)
            else:
                out = out.sortby(self.dim)
                # The expected behavior for downstream methods would be to conserve chunking along dim
                if out.chunks is not None:
                    # or -1 in case dim_chunks is [], when no input is chunked (only happens if the operation is chunking the output)
                    out = out.chunk({self.dim: dim_chunks or -1})
        if self.prop in out.dims and bool(out.chunks):
            # Same as above : downstream methods expect only one chunk along the group
            out = out.chunk({self.prop: -1})

        return out


def parse_group(func: Callable) -> Callable:
    """Parse the "group" argument of a function and return a Grouper object.

    Adds the possiblity to pass a window argument and a list of dimensions in group.
    """
    default_group = signature(func).parameters["group"].default

    @wraps(func)
    def _parse_group(*args, **kwargs):
        group = kwargs.setdefault("group", default_group)
        if not isinstance(group, Grouper):
            if not isinstance(group, str):
                dim, *add_dims = group
            else:
                dim = group
                add_dims = []
            kwargs["group"] = Grouper(
                group=dim,
                window=kwargs.pop("window", 1),
                add_dims=add_dims,
                interp=kwargs.get("interp", False),
            )
        return func(*args, **kwargs)

    return _parse_group
