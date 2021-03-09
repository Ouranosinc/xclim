"""
Quantile Mapping algorithms.

This file defines the different QM steps, to be wrapped into the Adjustment objects.
"""
import dask.array as da
import numpy as np
import xarray as xr
from boltons.funcutils import wraps

import xclim.sdba.nbutils as nbu
import xclim.sdba.utils as u


def duck_empty(dims, sizes, chunks=None):
    """Return an empty DataArray based on a numpy or dask backend, depending on the chunks argument."""
    shape = [sizes[dim] for dim in dims]
    if chunks:
        chnks = [chunks.get(dim, (sizes[dim],)) for dim in dims]
        content = da.empty(shape, chunks=chnks)
    else:
        content = np.empty(shape)
    return xr.DataArray(content, dims=dims)


def groupize(**outvars):
    """Decorator for declaring functions acting on single groups and wrapping them into a map_blocks.

    A "group" kwarg is injected in the function.
    It is assumed that `group.dim` is the only dimension touched, and some other dimensions might be added.

    Arguments to the decorator are mappings from variable name in the output to its *new* dimensions.
    Dimension order is not preserved.
    The placeholders "<PROP>" and "<DIM>" can be used to signify `group.prop` and `group.dim` respectively.
    """

    def _decorator(func):
        def _map_blocks_and_groupby(ds, group, **kwargs):
            # TODO : What if they are not alignable? (i.e. different calendars)
            ds = ds.unify_chunks()

            if any(isinstance(var.data, da.Array) for var in ds.data_vars.values()):
                # Use dask if any of the input is dask-backed.
                if len(ds.chunks[group.dim]) > 1:
                    raise ValueError(
                        f"The dimension over which we group cannot be chunked ({group.dim} as chunks {ds.chunks[group.dim]})."
                    )
                chunks = dict(ds.chunks)
            else:
                chunks = None

            # Make template

            # TODO : What if variables do not have the same dimensions?
            # Base dims are untouched by func, we also keep the order
            # "reference" object for dimension handling
            ref = list(ds.data_vars.values())[0]
            base_dims = [d for d in ref.dims if d != group.dim]
            alldims = set()
            alldims.update(*[set(dims) for dims in outvars.values()])
            alldims = base_dims + list(
                alldims
            )  # Ensure the untouched dimensions are first.

            coords = {}
            for i in range(len(alldims)):
                dim = alldims[i]
                if dim == "<PROP>":
                    alldims[i] = group.prop
                    coords[group.prop] = group.get_coordinate(ds=ds)
                elif dim == "<DIM>":
                    alldims[i] = group.dim
                    coords[group.dim] = ds[group.dim]
                elif dim in ds:
                    coords[dim] = ds[dim]
                elif dim in kwargs:
                    coords[dim] = xr.DataArray(kwargs[dim], dims=(dim,), name=dim)
                else:
                    raise ValueError(
                        f"This function adds the {dim} dimension, its coordinate must be provided as a keyword argument."
                    )

            placeholders = {"<PROP>": group.prop, "<DIM>": group.dim}
            sizes = {name: crd.size for name, crd in coords.items()}

            tmpl = xr.Dataset(coords=coords)
            for var, dims in outvars.items():
                dims = base_dims + [placeholders.get(dim, dim) for dim in dims]
                tmpl[var] = duck_empty(dims, sizes, chunks)
            tmpl = tmpl.transpose(*alldims)  # To be sure.

            def _apply_on_group(dsblock, **kwargs):
                return group.apply(func, dsblock, **kwargs).transpose(*alldims)

            return ds.map_blocks(_apply_on_group, template=tmpl, kwargs=kwargs)

        return _map_blocks_and_groupby

    return _decorator


@groupize(
    af=["<PROP>", "quantiles"], hist_q=["<PROP>", "quantiles"], scaling=["<PROP>"]
)
def dqm_train(ds, *, dim="time", kind="+", quantiles=None):
    """DQM: Train step: Element on one group of a 1D timeseries"""
    refn = u.apply_correction(ds.ref, u.invert(ds.ref.mean(dim), kind), kind)
    histn = u.apply_correction(ds.hist, u.invert(ds.hist.mean(dim), kind), kind)

    ref_q = nbu.quantile(refn, quantiles, dim)
    hist_q = nbu.quantile(histn, quantiles, dim)

    af = u.get_correction(hist_q, ref_q, kind)
    mu_ref = ds.ref.mean(dim)
    mu_hist = ds.hist.mean(dim)
    scaling = u.get_correction(mu_hist, mu_ref, kind=kind)

    return xr.Dataset(data_vars=dict(af=af, hist_q=hist_q, scaling=scaling))


def dqm_train_main(ref, hist, group, nquantiles=15, kind="+"):
    """DQM: Train step: Entry point."""
    quantiles = np.array(u.equally_spaced_nodes(nquantiles, eps=1e-6), dtype="float32")
    ds = xr.merge((ref, hist))
    return dqm_train(ds, group, quantiles=quantiles, kind=kind)


# def dqm_adjust_1d(
#     af,
#     hist_q,
#     sim,
#     group,
#     time,
#     coords,
#     dims,
#     interp="nearest",
#     extrapolation="constant",
#     kind="+",
# ):
#     """DQM: Adjust step: Atomic on a 1D timeseries."""
#     prop = group.prop
#     dim = group.dim  # noqa
#     for d, s in zip(dims, af.shape):
#         coords[d] = np.arange(s)
#     af = xr.DataArray(af, dims=(*dims, prop, "quantiles"), coords=coords)
#     hist_q = xr.DataArray(hist_q, dims=(*dims, prop, "quantiles"), coords=coords)
#     sim = xr.DataArray(
#         sim,
#         dims=(*dims, "time"),
#         coords={crd: val for crd, val in coords.items() if crd in dims},
#     )
#     sim["time"] = time

#     af, hist_q = u.extrapolate_qm(af, hist_q, method=extrapolation)
#     af = u.interp_on_quantiles(sim, hist_q, af, group=group, method="linear")

#     scen = u.apply_correction(sim, af, kind)
#     return scen.values


# def dqm_prepare_sim(
#     sim,
#     scaling,
#     *,
#     group,
#     other_dims,
#     core_dims,
#     coords,
#     out_dims,
#     interp="linear",
#     kind="+",
# ):
#     """DQM: Sim preprocessing: Atomic a 1D timeseries."""
#     sim, scaling = reconstruct_xr(
#         sim, scaling, other_dims=other_dims, core_dims=core_dims, coords=coords
#     )
#     out = u.apply_correction(
#         sim,
#         u.broadcast(
#             scaling,
#             sim,
#             group=group,
#             interp=interp if group.prop != "dayofyear" else "nearest",
#         ),
#         kind,
#     )
#     return deconstruct_xr(out, other_dims=other_dims, out_dims=out_dims)


# def polydetrend_get_trend_group(da, dim, deg):
#     """Polydetrend, atomic func on 1 group of a 1D timeseries."""
#     pfc = da.polyfit(dim=dim, deg=deg)
#     return xr.polyval(coord=da[dim], coeffs=pfc.polyfit_coefficients)


# def polydetrend_get_trend_1d(da, deg, group, coord, dims):
#     """Polydetrend, wrapper for a 1D timeseries."""
#     da = xr.DataArray(da, dims=(*dims, group.dim), coords={group.dim: coord})
#     out = group.apply(polydetrend_get_trend_group, da, main_only=True, deg=deg)
#     out.transpose(..., group.dim)
#     return out.values


# def polydetrend_get_trend(da, group, deg=1):
#     """Polydetrend, main."""
#     dims = list(da.dims)
#     dims.remove(group.dim)
#     trend = xr.apply_ufunc(
#         polydetrend_get_trend_1d,
#         da,
#         kwargs={"group": group, "deg": deg, "coord": da[group.dim], "dims": dims},
#         input_core_dims=[[group.dim]],
#         output_core_dims=[[group.dim]],
#         output_dtypes=[da.dtype],
#         dask_gufunc_kwargs=dict(
#             meta=(np.ndarray((), dtype=da.dtype),),
#         ),
#         dask="parallelized",
#     )
#     return trend


# def dqm_adjust(ds, sim, group, **kwargs):
#     """DQM: Adjust step: Main."""
#     dim = group.dim
#     prop = group.prop
#     dims = list(sim.dims)
#     dims.remove(dim)

#     sim = xr.apply_ufunc(
#         dqm_prepare_sim,
#         sim,
#         ds.scaling,
#         kwargs={
#             "coords": {prop: ds[prop].values, dim: sim[dim].values},
#             "group": group,
#             "other_dims": dims,
#             "core_dims": [[dim], [prop]],
#             "out_dims": OrderedDict([("sim", [dim])]),
#             **kwargs,
#         },
#         input_core_dims=[[dim], [prop]],
#         output_core_dim=[[dim]],
#         output_dtypes=[sim.dtype],
#         dask_gufunc_kwargs=dict(
#             meta=(np.ndarray((), dtype=sim.dtype),),
#         ),
#         dask="parallelized",
#     )

#     kind = kwargs.get("kind", "+")
#     trend = polydetrend_get_trend(sim, group, deg=kwargs.pop("detrend", 1))
#     sim_detrended = u.apply_correction(sim, u.invert(trend, kind), kind)

#     scen = xr.apply_ufunc(
#         dqm_adjust_1d,
#         ds.af,
#         ds.hist_q,
#         sim_detrended,
#         kwargs={
#             "coords": {prop: ds[prop].values, "quantiles": ds.quantiles.values},
#             "time": sim[dim],
#             "group": group,
#             "dims": dims,
#             **kwargs,
#         },
#         input_core_dims=[[prop, "quantiles"], [prop, "quantiles"], [dim]],
#         output_core_dims=[[dim]],
#         output_dtypes=[sim.dtype],
#         dask_gufunc_kwargs=dict(
#             meta=(np.ndarray((), dtype=sim.dtype),),
#         ),
#         dask="parallelized",
#     )

#     return u.apply_correction(scen, trend, kind)
