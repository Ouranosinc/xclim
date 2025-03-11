"""
Ensemble filters for data processing
====================================
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _concat_hist(da: xr.DataArray, **hist) -> xr.DataArray:
    r"""
    Concatenate historical scenario with future scenarios along the time dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data where the historical scenario is stored alongside other, future, scenarios.
    **hist : dict
        Mapping of the scenario dimension name to the historical scenario coordinate, e.g. `scenario="historical"`.

    Returns
    -------
    xr.DataArray
        Data with the historical scenario is stacked in time before each one of the other scenarios.

    Notes
    -----
    Data goes from:

        +------------+----------------------------------+
        | scenario   | time                             |
        +============+==================================+
        | historical | ``hhhhhhhhhhhhhhhh------------`` |
        +------------+----------------------------------+
        | ssp245     | ``----------------111111111111`` |
        +------------+----------------------------------+
        | ssp370     | ``----------------222222222222`` |
        +------------+----------------------------------+

    to:

        +----------+----------------------------------+
        | scenario | time                             |
        +==========+==================================+
        | ssp245   | ``hhhhhhhhhhhhhhhh111111111111`` |
        +----------+----------------------------------+
        | ssp370   | ``hhhhhhhhhhhhhhhh222222222222`` |
        +----------+----------------------------------+
    """
    if len(hist) > 1:
        raise ValueError("Too many values in hist scenario.")

    # Scenario dimension, and name of the historical scenario
    ((dim, _),) = hist.items()  # pylint: disable=unbalanced-dict-unpacking

    # Select historical scenario and drop it from the data
    h = da.sel(drop=True, **hist).dropna("time", how="all")
    ens = da.drop_sel(**hist)

    index = ens[dim]
    bare = ens.drop_vars(dim).dropna("time", how="all")

    return xr.concat([h, bare], dim="time").assign_coords({dim: index})


def _model_in_all_scens(da: xr.DataArray, dimensions: dict | None = None) -> xr.DataArray:
    """
    Return data with only simulations that have at least one member in each scenario.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimensions for time, member, model and scenario.
    dimensions : dict, optional
        Mapping from original dimension names to standard dimension names: scenario, model, member.

    Returns
    -------
    xr.DataArray
      Data for models that have values for all scenarios.

    Notes
    -----
    In the following example, model `C` would be filtered out from the data because it has no member for `ssp370`.

    +-------+--------+--------+
    | model | members         |
    +-------+-----------------+
    |       | ssp245 | ssp370 |
    +=======+========+========+
    | A     | 1,2,3  | 1,2,3  |
    +-------+--------+--------+
    | B     | 1      | 2,3    |
    +-------+--------+--------+
    | C     | 1,2,3  |        |
    +-------+--------+--------+
    """
    if dimensions is None:
        dimensions = {}

    da = da.rename(reverse_dict(dimensions))

    ok = da.notnull().any("time").any("member").all("scenario")

    return da.sel(model=ok).rename(dimensions)


def _single_member(da: xr.DataArray, dimensions: dict | None = None) -> xr.DataArray:
    """
    Return data for a single member per model.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimensions for time, member, model and scenario.
    dimensions : dict
        Mapping from original dimension names to standard dimension names: scenario, model, member.

    Returns
    -------
    xr.DataArray
        Data with only one member per model.

    Notes
    -----
    In the following example, the original members would be filtered to return only the first member found for each
    scenario.

    +-------+--------+--------+----+--------+--------+
    | model | member          |    | Selected        |
    +-------+-----------------+----+-----------------+
    |       | ssp245 | ssp370 |    | ssp245 | ssp370 |
    +=======+========+========+====+========+========+
    | A     | 1,2,3  | 1,2,3  |    | 1      | 1      |
    +-------+--------+--------+----+--------+--------+
    | B     | 1,2    | 2,3    |    | 1      | 2      |
    +-------+--------+--------+----+--------+--------+
    """
    if dimensions is None:
        dimensions = {}

    da = da.rename(reverse_dict(dimensions))

    # Stack by simulation specifications - drop simulations with missing values
    full = da.stack(i=("scenario", "model", "member")).dropna("i", how="any")

    # Pick first run with data
    s = full.i.to_series()
    s[:] = np.arange(len(s))
    i = s.unstack().T.min().to_list()

    out = full.isel(i=i).unstack().squeeze()
    return out.rename(dimensions)


def reverse_dict(d: dict) -> dict:
    """
    Reverse dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to reverse.

    Returns
    -------
    dict
        Reversed dictionary.
    """
    return {v: k for (k, v) in d.items()}
