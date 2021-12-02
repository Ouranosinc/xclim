import xarray

def bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Bias.

    The bias is the simulation minus reference.

    Parameters
    ----------
    sim : xarray.DataArray
      diagnostic from the simulation
    ref : xarray.DataArray
      diagnostic from the reference

    Returns
    -------
    xarray.DataArray,
      Bias between the simulation and the reference

    """
    out = sim - ref
    out.attrs["long_name"] = f"Bias of the {sim.attrs['standard_name']}"
    out.attrs["units"] = sim.attrs["units"]
    return out
