"""
Measures Submodule
==================
SDBA diagnostic tests are made up of properties and measures. Measures compare adjusted simulations to a reference,
through statistical properties or directly. This framework for the diagnostic tests was inspired by the
`VALUE <http://www.value-cost.eu/>`_ project.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from xclim.core.indicator import Indicator, base_registry
from xclim.core.units import convert_units_to, ensure_delta, units2pint
from xclim.core.utils import InputKind

from .base import Grouper
from .utils import _pairwise_spearman


class StatisticalMeasure(Indicator):
    """Base indicator class for statistical measures used when validating bias-adjusted outputs.

    Statistical measures use input data where the time dimension was reduced, usually by the computation
    of a :py:class:`xclim.sdba.properties.StatisticalProperty` instance.  They usually take two arrays
    as input: "sim" and "ref", "sim" being measured against "ref". The two arrays must have identical
    coordinates on their common dimensions.

    Statistical measures are generally unit-generic. If the inputs have different units, "sim" is converted
    to match "ref".
    """

    realm = "generic"

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        inputs = {k for k, p in parameters.items() if p.kind == InputKind.VARIABLE}
        if not inputs.issuperset({"sim", "ref"}):
            raise ValueError(
                f"{cls.__name__} requires 'sim' and 'ref' as inputs. Got {inputs}."
            )
        return super()._ensure_correct_parameters(parameters)

    def _preprocess_and_checks(self, das, params):
        """Perform parent's checks and also check convert units so that sim matches ref."""
        das, params = super()._preprocess_and_checks(das, params)

        # Convert grouping and check if allowed:
        sim = das["sim"]
        ref = das["ref"]

        # Check if common coordinates are identical.
        newsim, newref = xr.broadcast(sim, ref)
        for dim in set(sim.dims).union(ref.dims):
            if [sim[dim].size, ref[dim].size] != [newsim[dim].size, newref[dim].size]:
                raise ValueError(
                    f"Common dimension {dim} has different coordinates between ref and sim."
                )

        units_sim = units2pint(sim)
        units_ref = units2pint(ref)

        if units_sim != units_ref:
            das["sim"] = convert_units_to(sim, ref)

        return das, params


class StatisticalPropertyMeasure(Indicator):
    """Base indicator class for statistical properties that include the comparison measure, used when validating bias-adjusted outputs.

    StatisticalPropertyMeasure objects combine the functionalities of
    :py:class:`xclim.sdba.properties.StatisticalProperty` and
    :py:class:`xclim.sdba.properties.StatisticalMeasure`.

    Statistical properties usually reduce the time dimension and sometimes more dimensions
    (for example in spatial properties), sometimes adding a grouping dimension according to
    the passed value of `group` (e.g.: group='time.month' means the loss of the time dimension
    and the addition of a month one).

    Statistical measures usually take two arrays as input: "sim" and "ref", "sim" being measured against "ref".

    Statistical property-measures are generally unit-generic. If the inputs have different units,
    "sim" is converted to match "ref".
    """

    aspect = None
    """The aspect the statistical property studies: marginal, temporal, multivariate or spatial."""

    allowed_groups = None
    """A list of allowed groupings. A subset of dayofyear, week, month, season or group.
    The latter stands for no temporal grouping."""

    realm = "generic"

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        inputs = {k for k, p in parameters.items() if p.kind == InputKind.VARIABLE}
        if not inputs.issuperset({"sim", "ref"}):
            raise ValueError(
                f"{cls.__name__} requires 'sim' and 'ref' as inputs. Got {inputs}."
            )

        if "group" not in parameters:
            raise ValueError(
                f"{cls.__name__} require a 'group' argument, use the base Indicator"
                " class if your computation doesn't perform any regrouping."
            )

        return super()._ensure_correct_parameters(parameters)

    def _preprocess_and_checks(self, das, params):
        """Perform parent's checks and also check convert units so that sim matches ref."""
        das, params = super()._preprocess_and_checks(das, params)

        # Convert grouping and check if allowed:
        if isinstance(params["group"], str):
            params["group"] = Grouper(params["group"])

        if (
            self.allowed_groups is not None
            and params["group"].prop not in self.allowed_groups
        ):
            raise ValueError(
                f"Grouping period {params['group'].prop_name} is not allowed for property "
                f"{self.identifier} (needs something in "
                f"{list(map(lambda g: '<dim>.' + g.replace('group', ''), self.allowed_groups))})."
            )

        # Convert grouping and check if allowed:
        sim = das["sim"]
        ref = das["ref"]
        units_sim = units2pint(sim)
        units_ref = units2pint(ref)

        if units_sim != units_ref:
            das["sim"] = convert_units_to(sim, ref)

        return das, params

    def _postprocess(self, outs, das, params):
        """Squeeze `group` dim if needed."""
        outs = super()._postprocess(outs, das, params)

        for i in range(len(outs)):
            if "group" in outs[i].dims:
                outs[i] = outs[i].squeeze("group", drop=True)

        return outs


base_registry["StatisticalMeasure"] = StatisticalMeasure
base_registry["StatisticalPropertyMeasure"] = StatisticalPropertyMeasure


def _bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Bias.

    The bias is the simulation minus the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray, [same as ref]
      Absolute bias
    """
    out = sim - ref
    out.attrs["units"] = ensure_delta(ref.attrs["units"])
    return out


bias = StatisticalMeasure(identifier="bias", compute=_bias)


def _relative_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Relative Bias.

    The relative bias is the simulation minus reference, divided by the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray, [dimensionless]
      Relative bias
    """
    out = (sim - ref) / ref
    return out.assign_attrs(units="")


relative_bias = StatisticalMeasure(
    identifier="relative_bias", compute=_relative_bias, units=""
)


def _circular_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Circular bias.

    Bias considering circular time series.
    E.g. The bias between doy 365 and doy 1 is 364, but the circular bias is -1.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray, [days]
      Circular bias
    """
    out = (sim - ref) % 365
    out = out.where(
        out <= 365 / 2, 365 - out
    )  # when condition false, replace by 2nd arg
    out = out.where(ref >= sim, out * -1)  # when condition false, replace by 2nd arg
    return out.assign_attrs(units="days")


circular_bias = StatisticalMeasure(
    identifier="circular_bias", compute=_circular_bias, units="days"
)


def _ratio(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Ratio.

    The ratio is the quotient of the simulation over the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray, [dimensionless]
      Ratio
    """
    out = sim / ref
    out.attrs["units"] = ""
    return out


ratio = StatisticalMeasure(identifier="ratio", compute=_ratio, units="")


def _rmse(
    sim: xr.DataArray, ref: xr.DataArray, group: str | Grouper = "time"
) -> xr.DataArray:
    """Root mean square error.

    The root mean square error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xr.DataArray
        Data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
        Data from the reference (observations) (a time-series for each grid-point)
    group: str
        Compute the property and measure for each temporal groups individually.
        Currently not implemented.

    Returns
    -------
    xr.DataArray, [same as ref]
      Root mean square error
    """

    def _rmse(sim, ref):
        return np.sqrt(np.mean((sim - ref) ** 2, axis=-1))

    out = xr.apply_ufunc(
        _rmse,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        dask="parallelized",
    )
    return out.assign_attrs(units=ensure_delta(ref.units))


rmse = StatisticalPropertyMeasure(
    identifier="rmse",
    aspect="temporal",
    compute=_rmse,
    allowed_groups=["group"],
    cell_methods="time: mean",
)


def _mae(
    sim: xr.DataArray, ref: xr.DataArray, group: str | Grouper = "time"
) -> xr.DataArray:
    """Mean absolute error.

    The mean absolute error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xr.DataArray
        data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
        data from the reference (observations) (a time-series for each grid-point)
    group : str
        Compute the property and measure for each temporal groups individually.
        Currently not implemented.

    Returns
    -------
    xr.DataArray, [same as ref]
      Mean absolute error
    """

    def _mae(sim, ref):
        return np.mean(np.abs(sim - ref), axis=-1)

    out = xr.apply_ufunc(
        _mae,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        dask="parallelized",
    )
    return out.assign_attrs(units=ensure_delta(ref.units))


mae = StatisticalPropertyMeasure(
    identifier="mae",
    aspect="temporal",
    compute=_mae,
    allowed_groups=["group"],
    cell_methods="time: mean",
)


def _annual_cycle_correlation(
    sim: xr.DataArray,
    ref: xr.DataArray,
    window: int = 15,
    group: str | Grouper = "time",
) -> xr.DataArray:
    """Annual cycle correlation.

    Pearson correlation coefficient between the smooth day-of-year averaged annual cycles of the simulation and
    the reference. In the smooth day-of-year averaged annual cycles, each day-of-year is averaged over all years
    and over a window of days around that day.

    Parameters
    ----------
    sim : xr.DataArray
        data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
        data from the reference (observations) (a time-series for each grid-point)
    window : int
        Size of window around each day of year around which to take the mean.
        E.g. If window=31, Jan 1st is averaged over from December 17th to January 16th.
    group : str
        Compute the property and measure for each temporal groups individually.
        Currently not implemented.

    Returns
    -------
    xr.DataArray, [dimensionless]
      Annual cycle correlation
    """
    # group by day-of-year and window around each doy
    grouper_test = Grouper("time.dayofyear", window=window)
    # for each day, mean over X day window and over all years to create a smooth avg annual cycle
    sim_annual_cycle = grouper_test.apply("mean", sim)
    ref_annual_cycle = grouper_test.apply("mean", ref)
    out = xr.corr(ref_annual_cycle, sim_annual_cycle, dim="dayofyear")
    return out.assign_attrs(units="")


annual_cycle_correlation = StatisticalPropertyMeasure(
    identifier="annual_cycle_correlation",
    aspect="temporal",
    compute=_annual_cycle_correlation,
    allowed_groups=["group"],
)


def _scorr(
    sim: xr.DataArray,
    ref: xr.DataArray,
    *,
    dims: Sequence | None = None,
    group: str | Grouper = "time",
):
    """Spatial correllogram.

    Compute the inter-site correlations of each array, compute the difference in correlations and sum.
    Taken from Vrac (2018). The spatial and temporal dimensions are reduced.

    Parameters
    ----------
    sim : xr.DataArray
        data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
        data from the reference (observations) (a time-series for each grid-point)
    dims : sequence of strings, optional
        Name of the spatial dimensions. If None (default), all dimensions except 'time' are used.
    group : str
        Compute the property and measure for each temporal groups individually.
        Currently not implemented.

    Returns
    -------
    xr.DataArray, [dimensionless]
      Sum of the inter-site correlation differences.
    """
    if dims is None:
        dims = [d for d in sim.dims if d != "time"]

    refcorr = _pairwise_spearman(ref, dims)
    simcorr = _pairwise_spearman(sim, dims)
    S_corr = (simcorr - refcorr).sum(["_spatial", "_spatial2"])
    return S_corr.assign_attrs(units="")


scorr = StatisticalPropertyMeasure(
    identifier="Scorr", aspect="spatial", compute=_scorr, allowed_groups=["group"]
)


def _taylordiagram(
    sim: xr.DataArray,
    ref: xr.DataArray,
    dim: str = "time",
    group: str | Grouper = "time",
) -> xr.DataArray:
    """Taylor diagram.

    Compute the respective standard deviations of a simulation and a reference array, as well as the Pearson
    correlation coefficient between both, all necessary parameters to plot points on a Taylor diagram.

    Parameters
    ----------
    sim : xr.DataArray
        data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
        data from the reference (observations) (a time-series for each grid-point)
    dim : str
        Dimension across which the correlation and standard deviation should be computed.
    group : str
        Compute the property and measure for each temporal groups individually.
        Currently not implemented.


    Returns
    -------
    xr.DataArray, [same as ref]
        Standard deviations of sim, ref and correlation coefficient between both.
    """
    corr = xr.corr(sim, ref, dim=dim)

    ref_std = ref.std(dim=dim, skipna=True, keep_attrs=True)
    sim_std = sim.std(dim=dim, skipna=True, keep_attrs=True)

    new_dim = xr.DataArray(
        ["ref_std", "sim_std", "corr"], dims=("taylor_param",), name="taylor_param"
    )

    out = xr.concat(
        [ref_std, sim_std, corr],
        new_dim,
        coords="minimal",
        compat="override",  # Take common coords from `ref_std`.
    ).assign_attrs(
        {
            "correlation_type": "Pearson correlation coefficient",
            "units": ref.units,
        }
    )

    return out


taylordiagram = StatisticalPropertyMeasure(
    identifier="taylordiagram",
    aspect="temporal",
    compute=_taylordiagram,
    allowed_groups=["group"],
)
