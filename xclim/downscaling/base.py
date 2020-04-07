"""Detrending classes"""
from types import FunctionType
from typing import Mapping
from typing import Union

import xarray as xr


# ## Base class for the downscaling module
class ParametrizableClass(object):
    """Helper base class that sets as attribute every kwarg it receives in __init__.

    Parameters are all public attributes. Subclasses should use private attributes (starting with _).
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def parameters(self):
        """Return all parameters as a dictionary"""
        return {
            key: val for key, val in self.__dict__.items() if not key.startswith("_")
        }

    def parameters_to_json(self):
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


# ## Grouping object
class Grouper(ParametrizableClass):
    """Applies `groupby` method to one or more data arrays.

    Parameters
    ----------
    """

    def __init__(
        self,
        group: str,
        window: int = 1,
        add_dims=None,
        interp: Union[bool, str] = False,
    ):
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
        if das:
            if da is not None:
                das[da.name] = da
            da = xr.Dataset(data_vars={da.name: da for da in das})

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

    def get_index(self, da: xr.DataArray):
        if self.prop is None:
            da[self.dim]

        ind = da.indexes[self.dim]
        i = getattr(ind, self.prop)

        if self.interp:
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
        xi = xi.expand_dims(**{k: v for (k, v) in da.coords.items() if k != self.dim})
        xi.name = self.prop
        return xi

    def apply(
        self,
        func: Union[FunctionType, str],
        da: Union[xr.DataArray, Mapping[str, xr.DataArray]],
        **kwargs,
    ):
        if isinstance(da, dict):
            grpd = self.group(**da)
        else:
            grpd = self.group(da)

        if isinstance(func, str):
            out = getattr(grpd, func)(dim=self.dims, **kwargs)
        else:
            out = grpd.map(func, dim=self.dims, **kwargs)

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
        out.attrs["group_compute_dims"] = self.dims
        out.attrs["group_window"] = self.window

        # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
        if self.dim in out.dims:
            if self.prop is None and out[self.dim].size == 1:
                out = out.squeeze(self.dim).drop_vars(self.dim)
            else:
                out = out.sortby(self.dim)
        return out
