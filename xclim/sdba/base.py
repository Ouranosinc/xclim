"""Base classes."""
import warnings
from inspect import signature
from types import FunctionType
from typing import Callable, Mapping, Optional, Sequence, Set, Union

import dask.array as dsk
import jsonpickle
import numpy as np
import xarray as xr
from boltons.funcutils import wraps

from xclim.core.calendar import days_in_year, get_calendar, max_doy, parse_offset
from xclim.core.options import OPTIONS, SDBA_ENCODE_CF
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
        """Return a string representation."""
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
    ADD_DIMS = "<ADD_DIMS>"

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
        add_dims : Optional[Union[Sequence[str], str]]
          Additional dimensions that should be reduced in grouping operations. This behaviour is also controlled
          by the `main_only` parameter of the `apply` method. If any of these dimensions are absent from the dataarrays,
          they will be omitted.
        interp : Union[bool, str]
          Whether to return an interpolatable index in the `get_index` method. Only effective for `month` grouping.
          Interpolation method names are accepted for convenience, "nearest" is translated to False, all other names
          are translated to True.
          This modifies the default, but `get_index` also accepts an `interp` argument overriding the one defined here..
        """
        if "." in group:
            dim, prop = group.split(".")
        else:
            dim, prop = group, "group"

        if isinstance(interp, str):
            interp = interp != "nearest"

        if isinstance(add_dims, str):
            add_dims = [add_dims]

        add_dims = add_dims or []
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

        if main_only:
            dims = self.dim
        else:
            dims = [self.dim] + self.add_dims
            if self.window > 1:
                dims += ["window"]

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


def parse_group(func: Callable, kwargs=None) -> Callable:
    """Parse the kwargs given to a function to set the `group` arg with a Grouper object.

    This function can be used as a decorator, in which case the parsing and updating of the kwargs is done at call time.
    It can also be called with a function from which extract the default group and kwargs to update, in which case it returns the updated kwargs.
    """
    sig = signature(func)
    if "group" in sig.parameters:
        default_group = sig.parameters["group"].default
    else:
        default_group = None

    def _update_kwargs(kwargs):
        if default_group or "group" in kwargs:
            kwargs.setdefault("group", default_group)
            if not isinstance(kwargs["group"], Grouper):
                kwargs = Grouper.from_kwargs(**kwargs)
        return kwargs

    if kwargs is not None:  # Not used as a decorator
        return _update_kwargs(kwargs)

    # else (then it's a decorator)
    @wraps(func)
    def _parse_group(*args, **kwargs):
        kwargs = _update_kwargs(kwargs)
        return func(*args, **kwargs)

    return _parse_group


def duck_empty(dims, sizes, dtype="float64", chunks=None):
    """Return an empty DataArray based on a numpy or dask backend, depending on the chunks argument."""
    shape = [sizes[dim] for dim in dims]
    if chunks:
        chnks = [chunks.get(dim, (sizes[dim],)) for dim in dims]
        content = dsk.empty(shape, chunks=chnks, dtype=dtype)
    else:
        content = np.empty(shape, dtype=dtype)
    return xr.DataArray(content, dims=dims)


def _decode_cf_coords(ds):
    """Decodes coords in-place."""
    crds = xr.decode_cf(ds.coords.to_dataset())
    for crdname in ds.coords.keys():
        ds[crdname] = crds[crdname]


def map_blocks(reduces=None, **outvars):
    """
    Decorator for declaring functions and wrapping them into a map_blocks. It takes care of constructing
    the template dataset.

    Dimension order is not preserved.

    The decorated function must always have the signature: func(ds, **kwargs), where ds is a DataArray or a Dataset.
    It must always output a dataset matching the mapping passed to the decorator.

    Parameters
    ----------
    reduces : sequence of strings
      Name of the dimensions that are removed by the function.
    **outvars
      Mapping from variable names in the output to their *new* dimensions.
      The placeholders `Grouper.PROP`, `Grouper.DIM` and `Grouper.ADD_DIMS` can be used to signify
      `group.prop`,`group.dim` and `group.add_dims` respectively.
      If an output keeps a dimension that another loses, that dimension name must be given in `reduces` and in
      the list of new dimensions of the first output.
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
            if {Grouper.PROP, Grouper.DIM, Grouper.ADD_DIMS}.intersection(
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
                if group is not None:
                    badchunks = {
                        dim: chunks.get(dim)
                        for dim in group.add_dims + [group.dim]
                        if len(chunks.get(dim, [])) > 1
                    }
                    if badchunks:
                        raise ValueError(
                            f"The dimension(s) over which we group cannot be chunked ({badchunks})."
                        )
            else:
                chunks = None

            # Make translation dict
            if group is not None:
                placeholders = {
                    Grouper.PROP: [group.prop],
                    Grouper.DIM: [group.dim],
                    Grouper.ADD_DIMS: group.add_dims,
                }
            else:
                placeholders = {}

            # Get new dimensions (in order), translating placeholders to real names.
            new_dims = []
            for dim in out_dims:
                new_dims.extend(placeholders.get(dim, [dim]))

            reduced_dims = []
            for dim in red_dims:
                reduced_dims.extend(placeholders.get(dim, [dim]))

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
            added_coords = []
            coords = {}
            sizes = {}
            for dim in all_dims:
                if dim == group.prop:
                    coords[group.prop] = group.get_coordinate(ds=ds)
                elif dim == group.dim:
                    coords[group.dim] = ds[group.dim]
                elif dim in kwargs:
                    coords[dim] = xr.DataArray(kwargs[dim], dims=(dim,), name=dim)
                elif dim in ds.dims:
                    # If a dim has no coords : some sdba function will add them, so to be safe we add them right now
                    # and note them to remove them afterwards.
                    if dim not in ds.coords:
                        added_coords.append(dim)
                    ds[dim] = ds[dim]
                    coords[dim] = ds[dim]
                else:
                    raise ValueError(
                        f"This function adds the {dim} dimension, its coordinate must be provided as a keyword argument."
                    )
            sizes.update({name: crd.size for name, crd in coords.items()})

            # Create the output dataset, but empty
            tmpl = xr.Dataset(coords=coords)
            if isinstance(ds, xr.Dataset):
                # Get largest dtype of the inputs, assign it to the output.
                dtype = max(
                    (da.dtype for da in ds.data_vars.values()), key=lambda d: d.itemsize
                )
            else:
                dtype = ds.dtype

            for var, dims in outvars.items():
                var_new_dims = []
                for dim in dims:
                    var_new_dims.extend(placeholders.get(dim, [dim]))
                # Out variables must have the base dims + new_dims
                dims = base_dims + var_new_dims
                # duck empty calls dask if chunks is not None
                tmpl[var] = duck_empty(dims, sizes, dtype=dtype, chunks=chunks)

            if OPTIONS[SDBA_ENCODE_CF]:
                ds = ds.copy()
                # Optimization to circumvent the slow pickle.dumps(cftime_array)
                for name, crd in ds.coords.items():
                    if xr.core.common._contains_cftime_datetimes(crd.values):
                        ds[name] = xr.conventions.encode_cf_variable(crd)

            def _call_and_transpose_on_exit(dsblock, **kwargs):
                """Call the decorated func and transpose to ensure the same dim order as on the templace."""
                try:
                    _decode_cf_coords(dsblock)
                    out = func(dsblock, **kwargs).transpose(*all_dims)
                except Exception as err:
                    raise ValueError(
                        f"{func.__name__} failed on block with coords : {dsblock.coords}."
                    ) from err
                return out

            # Fancy patching for explicit dask task names
            _call_and_transpose_on_exit.__name__ = f"block_{func.__name__}"

            # Remove all auxiliary coords on both tmpl and ds
            extra_coords = {
                nam: crd for nam, crd in ds.coords.items() if nam not in crd.dims
            }
            ds = ds.drop_vars(extra_coords.keys())
            # Coords not sharing dims with `all_dims` (like scalar aux coord on reduced 1D input) are absent from tmpl
            tmpl = tmpl.drop_vars(extra_coords.keys(), errors="ignore")

            # Call
            out = ds.map_blocks(
                _call_and_transpose_on_exit, template=tmpl, kwargs=kwargs
            )

            # Add back the extra coords, but only those which have compatible dimensions (like xarray would have done)
            out = out.assign_coords(
                {
                    nam: crd
                    for nam, crd in extra_coords.items()
                    if set(crd.dims).issubset(out.dims)
                }
            )

            # Finally remove coords we added... 'ignore' in case they were already removed.
            out = out.drop_vars(added_coords, errors="ignore")
            return out

        _map_blocks.__dict__["func"] = func
        return _map_blocks

    return _decorator


def map_groups(reduces=None, main_only=False, **outvars):
    """
    Decorator for declaring functions acting only on groups and wrapping them into a map_blocks.
    See :py:func:`map_blocks`.

    This is the same as `map_blocks` but adds a call to `group.apply()` in the mapped func and the default
    value of `reduces` is changed.

    The decorated function must have the signature: func(ds, dim, **kwargs).
    Where ds is a DataAray or Dataset, dim is the group.dim (and add_dims). The `group` argument
    is stripped from the kwargs, but must evidently be provided in the call.

    Parameters
    ----------
    reduces: sequence of str
      Dimensions that are removed from the inputs by the function. Defaults to [Grouper.DIM, Grouper.ADD_DIMS] if main_only is False,
      and [Grouper.DIM] if main_only is True. See :py:func:`map_blocks`.
    main_only: bool
        Same as for :py:meth:`Grouper.apply`.
    """
    defreduces = [Grouper.DIM]
    if not main_only:
        defreduces.append(Grouper.ADD_DIMS)
    reduces = reduces or defreduces

    def _decorator(func):
        decorator = map_blocks(reduces=reduces, **outvars)

        def _apply_on_group(dsblock, **kwargs):
            group = kwargs.pop("group")
            return group.apply(func, dsblock, main_only=main_only, **kwargs)

        # Fancy patching for explicit dask task names
        _apply_on_group.__name__ = f"group_{func.__name__}"

        # wraps(func, injected=['dim'], hide_wrapped=True)(
        wrapper = decorator(_apply_on_group)
        wrapper.__dict__["func"] = func
        return wrapper

    return _decorator


def _get_number_of_elements_by_year(time):
    """Get the number of elements in time in a year by inferring its sampling frequency.

    Only calendar with uniform year lengths are supported : 360_day, noleap, all_leap.
    """
    cal = get_calendar(time)

    # Calendar check
    if cal in ["standard", "gregorian", "default", "proleptic_gregorian"]:
        raise ValueError(
            "For moving window computations, the data must have a uniform calendar (360_day, no_leap or all_leap)"
        )

    mult, freq, _, _ = parse_offset(xr.infer_freq(time))
    days_in_year = max_doy[cal]
    elements_in_year = {"Q": 4, "M": 12, "D": days_in_year, "H": days_in_year * 24}
    N_in_year = elements_in_year.get(freq, 1) / int(mult or 1)
    if N_in_year % 1 != 0:
        raise ValueError(
            f"Sampling frequency of the data must be Q, M, D or H and evenly divide a year (got {mult}{freq})."
        )

    return int(N_in_year)


def construct_moving_yearly_window(
    da: xr.Dataset, window: int = 21, step: int = 1, dim: str = "movingwin"
):
    """Construct a moving window DataArray.

    Stacks windows of `da` in a new 'movingwin' dimension.
    Windows are always made of full years, so calendar with non uniform year lengths are not supported.

    Windows are constructed starting at the beginning of `da`, if number of given years is not
    a multiple of `step`, then the last year(s) will be missing as a supplementary window would be incomplete.

    Parameters
    ----------
    da : xr.DataArray
      A DataArray with a `time` dimension.
    window : int
      The length of the moving window as a number of years.
    step : int
      The step between each window as a number of years.
    dim : str
      The new dimension name. If given, must also be given to `unpack_moving_yearly_window`.

    Return
    ------
    xr.DataArray
      A DataArray with a new `movingwin` dimension and a `time` dimension with a length of 1 window.
      This assumes downstream algorithms do not make use of the _absolute_ year of the data.
      The correct timeseries can be reconstructed with :py:func:`unpack_moving_yearly_window`.
      The coordinates of `movingwin` are the first date of the windows.
    """
    # Get number of samples per year (and perform checks)
    N_in_year = _get_number_of_elements_by_year(da.time)

    # Number of samples in a window
    N = window * N_in_year

    first_slice = da.isel(time=slice(0, N))
    first_slice = first_slice.expand_dims({dim: np.atleast_1d(first_slice.time[0])})
    daw = [first_slice]

    i_start = N_in_year * step
    # This is the first time I use `while` in real python code. What an event.
    while i_start + N <= da.time.size:
        # Cut and add _full_ slices only, partial window are thrown out
        # Use isel so that we don't need to deal with a starting date.
        slc = da.isel(time=slice(i_start, i_start + N))
        slc = slc.expand_dims({dim: np.atleast_1d(slc.time[0])})
        slc["time"] = first_slice.time
        daw.append(slc)
        i_start += N_in_year * step

    daw = xr.concat(daw, dim)
    return daw


def unpack_moving_yearly_window(da: xr.DataArray, dim: str = "movingwin"):
    """Unpack a constructed moving window dataset to a normal timeseries, only keeping the central data.

    Unpack DataArrays created with :py:func:`construct_moving_yearly_window` and recreate a timeseries data.
    Only keeps the central non-overlapping years. The final timeseries will be (window - step) years shorter than
    the initial one.

    The window length and window step are inferred from the coordinates.

    Parameters
    ----------
    da: xr.DataArray
      As constructed by :py:func:`construct_moving_yearly_window`.
    dim : str
      The window dimension name as given to the construction function.
    """
    # Get number of samples by year (and perform checks)
    N_in_year = _get_number_of_elements_by_year(da.time)

    # Might be smaller than the original moving window, doesn't matter
    window = da.time.size / N_in_year

    if window % 1 != 0:
        warnings.warn(
            f"Incomplete data received as number of years covered is not an integer ({window})"
        )

    # Get step in number of years
    days_in_year = max_doy[get_calendar(da)]
    step = np.unique(da[dim].diff(dim).dt.days / days_in_year)
    if len(step) > 1:
        raise ValueError("The spacing between the windows is not equal.")
    step = int(step[0])

    # Which years to keep: length step, in the middle of window
    left = int((window - step) // 2)  # first year to keep

    # Keep only the middle years
    da = da.isel(time=slice(left * N_in_year, (left + step) * N_in_year))

    out = []
    for win_start in da[dim]:
        slc = da.sel({dim: win_start}).drop_vars(dim)
        dt = win_start.values - da[dim][0].values
        slc["time"] = slc.time + dt
        out.append(slc)

    return xr.concat(out, "time")


def stack_variables(ds, rechunk=True, dim="variables"):
    """Stack different variables of a dataset into a single DataArray with a new "variables" dimension.

    Variable attributes are all added as lists of attributes to the new coordinate, prefixed with "_".

    Parameters
    ----------
    ds : xr.Dataset
      Input dataset.
    rechunk : bool
      If True (default), dask arrays are rechunked with `variables : -1`.
    dim : str
      Name of dimension along which variables are indexed.

    Returns
    -------
    xr.DataArray
      Array with variables stacked along `dim` dimension. Units are set to "".
    """
    # Store original arrays' attributes
    attrs = {}
    nvar = len(ds.data_vars)
    for i, var in enumerate(ds.data_vars.values()):
        for name, attr in var.attrs.items():
            attrs.setdefault("_" + name, [None] * nvar)[i] = attr

    # Special key used for later `unstacking`
    attrs["is_variables"] = True
    var_crd = xr.DataArray(
        list(ds.data_vars.keys()), dims=(dim,), name=dim, attrs=attrs
    )

    da = xr.concat(ds.data_vars.values(), var_crd, combine_attrs="drop")

    if uses_dask(da) and rechunk:
        da = da.chunk({dim: -1})

    da.attrs.update(ds.attrs)
    da.attrs["units"] = ""
    return da.rename("multivariate")


def unstack_variables(da, dim=None):
    """Unstack a DataArray created by `stack_variables` to a dataset.

    Parameters
    ----------
    da : xr.DataArray
      Array holding different variables along `dim` dimension.
    dim : str
      Name of dimension along which the variables are stacked. If not specified (default),
      `dim` is inferred from attributes of the coordinate.

    Returns
    -------
    xr.Dataset
      Dataset holding each variable in an individual DataArray.
    """
    if dim is None:
        for dim, crd in da.coords.items():
            if crd.attrs.get("is_variables"):
                break
        else:
            raise ValueError("No variable coordinate found, were attributes removed?")

    ds = xr.Dataset(
        {name.item(): da.sel({dim: name.item()}, drop=True) for name in da[dim]},
        attrs=da.attrs,
    )
    del ds.attrs["units"]

    # Reset attributes
    for name, attr_list in da.variables.attrs.items():
        if not name.startswith("_"):
            continue
        for attr, var in zip(attr_list, da.variables):
            if attr is not None:
                ds[var.item()].attrs[name[1:]] = attr

    return ds
