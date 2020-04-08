"""Detrending classes"""
from inspect import signature
from types import FunctionType
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import xarray as xr
from boltons.funcutils import wraps


# ## Base class for the downscaling module
class ParametrizableClass(object):
    """Helper base class that sets as attribute every kwarg it receives in __init__.

    Only parameters passed in the init are considered as such and returned in the
    :py:meth:`ParametrizableClass.parameters` dictionary and the :py:meth:`ParametrizableCalss.parameters_to_json` method.
    """

    def __init__(self, **kwargs):
        self._parameter_names = []
        for key, val in kwargs.items():
            setattr(self, key, val)
            self._parameter_names.append(key)

    @property
    def parameters(self):
        """All parameters as a dictionary."""
        return {key: getattr(self, key) for key in self._parameter_names}

    def parameters_to_json(self):
        """A json-compliant dictionary of the parameters.

        Parameter values that are of another type than str, int, float, bool or None are casted as strings.
        """
        return {
            key: val
            if isinstance(val, (str, float, int, bool, type(None)))
            else str(val)
            for key, val in self.parameters.items()
        }

    def __str__(self):
        params_str = ", ".join(
            [f"{key}: {val}" for key, val in self.parameters_to_json().items()]
        )
        return f"<{self.__class__.__name__}: {params_str}>"


class Grouper(ParametrizableClass):
    """Helper object to perform grouping actions on dataarrays and datasets."""

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
          by the `main_only` parameter of the `apply` method.
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
        dims = [dim] + (add_dims or [])

        if window > 1:
            dims.insert(1, "window")
        super().__init__(
            dim=dim, dims=dims, prop=prop, name=group, window=window, interp=interp
        )

    def group(self, da: xr.DataArray = None, **das: xr.DataArray):
        """Return a xr.core.groupby.GroupBy object.

        More than one arrays can be combined to a dataset before grouping using the `das`  kwargs.
        A new `window` dimension is added if `self.window` is larger than 1.
        """
        if das:
            if da is not None:
                das[da.name] = da
            da = xr.Dataset(data_vars=das)

        if self.window > 1:
            da = da.rolling(center=True, **{self.dim: self.window}).construct(
                window_dim="window"
            )

        if self.prop is None:
            group = xr.full_like(da[self.dim], True, dtype=bool)
            group.name = self.dim
        else:
            group = self.name

        return da.groupby(group)

    def get_index(self, da: xr.DataArray, interp: Optional[Union[bool, str]] = None):
        """Return the group index of each element along the main dimension.

        Argument `interp` defaults to `self.interp`. Ifs True, the returned index can be
        used for interpolation. For month grouping, integer values represent the middle of
        the month, all other  days are linearly interpolated in between.
        """
        if self.prop is None:
            da[self.dim]

        ind = da.indexes[self.dim]
        i = getattr(ind, self.prop)
        interp = (
            (interp or self.interp)
            if not isinstance(interp, str)
            else interp != "nearest"
        )
        if interp:
            if self.dim == "time":
                if self.prop == "month":
                    i = ind.month - 0.5 + ind.day / ind.daysinmonth
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
          Whether to call the function with the main dimension only (if True) or with all grouping dims (if False, default)
          (including the window and dimensions given through `add_dims`). The dimensions used are also written in the "group_compute_dims" attribute.
        **kwargs :
          Other keyword arguments to pass to the function.

        Returns
        -------
        DataArray or Dataset
          Attributes "group", "group_window" and "group_compute_dims" are added.
          If the function did not reduce the array, its is sorted along the main dimension.

        Notes
        -----
        For the special case where a Dataset is returned, but only some of its variable where reduced by the grouping, xarray's `GroupBy.map` will
        broadcast everything back to the ungrouped dimensions. To overcome this issue, function may add a "_group_apply_reshape" attribute set to
        True on the variables that should be reduced and these will be re-grouped by calling `da.groupby(self.name).first()`.
        """
        if isinstance(da, dict):
            grpd = self.group(**da)
        else:
            grpd = self.group(da)

        dims = self.dim if main_only else self.dims
        if isinstance(func, str):
            out = getattr(grpd, func)(dim=dims, **kwargs)
        else:
            out = grpd.map(func, dim=dims, **kwargs)

        # Case where the function wants to return more than one variables
        # and that some have grouped dims and other have the same dimensions as the input.
        # In that specific case, groupby broadcasts everything back to the input's dim, copying the grouped data.
        if isinstance(out, xr.Dataset):
            for name, da in out.data_vars.items():
                if "_group_apply_reshape" in da.attrs:
                    if da.attrs["_group_apply_reshape"] and self.prop is not None:
                        out[name] = da.groupby(self.name).first(
                            skipna=False, keep_attrs=True
                        )
                    del out[name].attrs["_group_apply_reshape"]

        # Save input parameters as attributes of output DataArray.
        out.attrs["group"] = self.name
        out.attrs["group_compute_dims"] = dims
        out.attrs["group_window"] = self.window

        # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
        if self.dim in out.dims:
            if self.prop is None and out[self.dim].size == 1:
                out = out.squeeze(self.dim, drop=True)  # .drop_vars(self.dim)
            else:
                out = out.sortby(self.dim)

        return out


def parse_group(func):
    """Decorator to parse the "group" argument of a function and return a Grouper object.

    Adds the possiblity to pass a window argument and a list of dimensions in group.
    """
    default_group = signature(func).parameters["group"].default

    @wraps(func)
    def _parse_group(*args, **kwargs):
        group = kwargs.get("group", default_group)
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
