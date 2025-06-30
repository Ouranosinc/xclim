"""Agroclimatic classification module."""

import xarray as xr

# TODO: What is the scope of the MCA module?

__all__ = ["geoviticulture_mcc"]


def geoviticulture_mcc(hi: xr.DataArray, cni: xr.DataArray, di: xr.DataArray) -> xr.DataArray:
    """
    Return the geoviticulture MCC classification.

    Parameters
    ----------
    hi : xr.DataArray
        Heliothermal index (HI) array.
    cni : xr.DataArray
        Cool nights index (CNI) array.
    di : xr.DataArray
        Dryness index (DI) array.

    Returns
    -------
    xr.DataArray
        Geoviticulture MCC classification based on the provided indices.
    """
    hi_intervals = {  # noqa: F841
        "HI +3": {"Very warm": (3000, 4000)},
        "HI +2": {"Warm": (2400, 3000)},
        "HI +1": {"Temperate warm": (2100, 2400)},
        "HI -1": {"Temperate": (1800, 2100)},
        "HI -2": {"Cool": (1500, 1800)},
        "HI -3": {"Very cool": (0, 1500)},
    }

    di_intervals = {  # noqa: F841
        "DI +2": {"Very dry": (-300, -100)},
        "DI +1": {"Moderately dry": (-100, 50)},
        "DI -1": {"Sub-humid": (50, 150)},
        "DI -2": {"Humid": (150, 300)},
    }

    cni_intervals = {  # noqa: F841
        "CI +2": {"Very cool nights": (0, 5)},
        "CI +1": {"Cool nights": (5, 10)},
        "CI -1": {"Temperate nights": (10, 15)},
        "CI -2": {"Warm nights": (15, 20)},
    }

    # TODO: Are we calculation the indices here or are they arriving already calculated?
    pass
