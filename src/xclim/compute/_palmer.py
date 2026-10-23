"""
Palmer drought indices.

Calculation of Palmer drought indices from precipitation,
potential evapotranspiration, and available water-holding capacity.

Implements:

* Z-index
* PDSI
* PHDI
* PMDI
* SC-PDSI

References
----------
Palmer (1965)
Wells et al. (2004)
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = [
    "compute_pdsi_from_zindex",
    "compute_scpdsi_from_zindex",
    "palmer_drought_indices",
]


_PDSI_MIN = -4.0
_PDSI_MAX = 4.0


@dataclass
class PalmerWaterBalance:
    et: xr.DataArray
    pr: xr.DataArray
    r: xr.DataArray
    ro: xr.DataArray
    pro: xr.DataArray
    l: xr.DataArray
    pl: xr.DataArray
    soil_water: xr.DataArray
    surface_storage: xr.DataArray
    under_storage: xr.DataArray


@dataclass
class PalmerWaterBalanceState:
    surface_storage: xr.DataArray
    under_storage: xr.DataArray


def palmer_drought_indices(
    pr: xr.DataArray,
    pet: xr.DataArray,
    awhc: xr.DataArray,
    *,
    soil_mask: xr.DataArray | None = None,
    calibration_start_year: int | None = None,
    calibration_end_year: int | None = None,
    monthly_inputs: bool = False,
) -> xr.Dataset:
    """
    Compute Palmer drought indices and supporting water-balance variables.

    Parameters
    ----------
    pr : xr.DataArray
        Precipitation. Daily values are aggregated to monthly totals
        unless ``monthly_inputs=True``.
    pet : xr.DataArray
        Potential evapotranspiration. Daily values are aggregated to
        monthly totals unless ``monthly_inputs=True``.
    awhc : xr.DataArray
        Available water-holding capacity of the soil profile.
    soil_mask : xr.DataArray, optional
        Boolean mask used to exclude non-soil grid cells from drought
        calculations.
    calibration_start_year : int, optional
        First year of the Palmer calibration period.
    calibration_end_year : int, optional
        Last year of the Palmer calibration period.
    monthly_inputs : bool, default: False
        Indicates that ``pr`` and ``pet`` are already monthly values and
        should not be resampled.

    Returns
    -------
    xr.Dataset
        Dataset containing Palmer drought indices, self-calibrated
        Palmer drought indices, Palmer water-balance components, CAFEC
        coefficients, and supporting diagnostic variables.

    Notes
    -----
    The Palmer workflow consists of:

    precipitation, PET
        → Palmer water balance
        → CAFEC coefficients
        → climatic characteristic (K)
        → Z-index
        → PDSI, PHDI, PMDI
        → SC-PDSI

    References
    ----------
    Palmer, W. C. (1965)
    Wells, N., Goddard, S., and Hayes, M. J. (2004)
    """
    if monthly_inputs:
        monthly = xr.Dataset(
            {
                "P": pr,
                "PET": pet,
            }
        )
    else:
        monthly = monthly_water_balance(
            pr,
            pet,
        )

    wb = palmer_water_balance(
        awhc=awhc,
        monthly_pr=monthly["P"],
        monthly_pet=monthly["PET"],
    )

    monthly["AET"] = wb.et
    monthly["PR"] = wb.pr
    monthly["R"] = wb.r
    monthly["RO"] = wb.ro
    monthly["PRO"] = wb.pro
    monthly["L"] = wb.l
    monthly["PL"] = wb.pl

    # Restrict Palmer calculations to cells with valid
    # soil water-holding capacity.
    valid_mask = np.isfinite(awhc)
    for var in [
        "P",
        "PET",
        "AET",
        "PR",
        "R",
        "RO",
        "PRO",
        "L",
        "PL",
    ]:
        monthly[var] = monthly[var].where(valid_mask)

    # Subset data used to estimate CAFEC coefficients and
    # Palmer climatic characteristic (K).
    calibration_monthly = subset_calibration_period(
        monthly,
        calibration_start_year,
        calibration_end_year,
    )

    if calibration_monthly.sizes["time"] == 0:
        raise ValueError(
            "Calibration period contains no data. "
            f"Requested {calibration_start_year}-{calibration_end_year}, "
            f"but loaded data spans "
            f"{int(monthly.time.dt.year.min())}-"
            f"{int(monthly.time.dt.year.max())}."
        )

    cafec = compute_cafec_coefficients(
        calibration_monthly,
    )
    calibration_departure = compute_monthly_departure(
        calibration_monthly,
        cafec,
    )

    departure = compute_monthly_departure(
        monthly,
        cafec,
    )

    k = compute_k_factor(
        calibration_monthly,
        calibration_departure["departure"],
    )

    Z = compute_z_index(
        departure,
        k,
    )
    if soil_mask is not None:
        Z = Z.where(soil_mask)

    # Compute Palmer and self-calibrated Palmer indices
    # from the monthly Z-index.
    pdsi_ds = compute_pdsi_from_zindex(
        Z,
    )

    # SC-PDSI uses the calibration period to derive
    # self-calibrated wet and dry duration factors.
    scpdsi_ds = compute_scpdsi_from_zindex(
        Z,
        calibration_start_year,
        calibration_end_year,
    )

    return xr.Dataset(
        {
            # Palmer indices
            "Z": Z,
            "pdsi": pdsi_ds["pdsi"],
            "phdi": pdsi_ds["phdi"],
            "pmdi": pdsi_ds["pmdi"],
            "scz": scpdsi_ds["scz"],
            "scpdsi": scpdsi_ds["scpdsi"],
            "X1": scpdsi_ds["X1"],
            "X2": scpdsi_ds["X2"],
            "X3": scpdsi_ds["X3"],
            # Water balance
            "precip": monthly["P"],
            "pet": monthly["PET"],
            "aet": wb.et,
            "pr": wb.pr,
            "r": wb.r,
            "ro": wb.ro,
            "pro": wb.pro,
            "l": wb.l,
            "pl": wb.pl,
            "soil_water": wb.soil_water,
            "surface_storage": wb.surface_storage,
            "under_storage": wb.under_storage,
            # Diagnostics
            "departure": departure["departure"],
            "k": k,
        }
    )


def monthly_water_balance(
    pr: xr.DataArray,
    pet: xr.DataArray,
    *,
    aet: xr.DataArray | None = None,
    recharge: xr.DataArray | None = None,
    runoff: xr.DataArray | None = None,
    loss: xr.DataArray | None = None,
    potential_recharge: xr.DataArray | None = None,
    potential_runoff: xr.DataArray | None = None,
    potential_loss: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Aggregate Palmer water-balance variables to monthly totals.
    """
    monthly = {
        "P": pr.resample(time="MS").sum(),
        "PET": pet.resample(time="MS").sum(),
    }

    if aet is not None:
        monthly["AET"] = aet.resample(time="MS").sum()
    if recharge is not None:
        monthly["R"] = recharge.resample(time="MS").sum()
    if runoff is not None:
        monthly["RO"] = runoff.resample(time="MS").sum()
    if loss is not None:
        monthly["L"] = loss.resample(time="MS").sum()
    if potential_recharge is not None:
        monthly["PR"] = potential_recharge.resample(time="MS").sum()
    if potential_runoff is not None:
        monthly["PRO"] = potential_runoff.resample(time="MS").sum()
    if potential_loss is not None:
        monthly["PL"] = potential_loss.resample(time="MS").sum()

    return xr.Dataset(monthly)


def step_palmer_water_balance(
    state: PalmerWaterBalanceState,
    awhc: xr.DataArray,
    precipitation: xr.DataArray,
    pet: xr.DataArray,
) -> tuple[
    PalmerWaterBalanceState,
    xr.DataArray,  # ET
    xr.DataArray,  # PR
    xr.DataArray,  # R
    xr.DataArray,  # RO
    xr.DataArray,  # PRO
    xr.DataArray,  # L
    xr.DataArray,  # PL
]:
    """
    Advance the Palmer two-layer soil-moisture accounting model by one month.

    Returns the updated water-balance state together with actual and
    potential Palmer water-balance components.
    """
    surface_capacity = xr.where(
        awhc < 25.4,
        awhc,
        25.4,
    )

    under_capacity = awhc - surface_capacity

    surface_storage = state.surface_storage
    under_storage = state.under_storage

    # --------------------------------------------------
    # Potential quantities
    # --------------------------------------------------
    pr_surface = surface_capacity - surface_storage
    pr_under = under_capacity - under_storage

    pr = pr_surface + pr_under
    pro = surface_storage + under_storage

    a = precipitation - pet

    potential_loss_surface = xr.where(
        a >= 0,
        pet,
        surface_storage,
    )

    potential_loss_under = xr.where(
        a >= 0,
        0.0,
        ((pet - potential_loss_surface) * under_storage / awhc),
    )

    potential_loss_under = xr.apply_ufunc(
        np.minimum,
        potential_loss_under,
        under_storage,
    )

    pl = potential_loss_surface + potential_loss_under

    # --------------------------------------------------
    # Palmer moisture accounting
    # --------------------------------------------------

    excess_precip = precipitation - pet

    wet_month = excess_precip >= 0

    # Wet-month branch
    recharge_surface = xr.where(
        wet_month,
        xr.apply_ufunc(
            np.minimum,
            excess_precip,
            surface_capacity - surface_storage,
        ),
        0,
    )

    remaining = xr.where(
        wet_month,
        excess_precip - recharge_surface,
        0,
    )

    recharge_under = xr.where(
        wet_month,
        xr.apply_ufunc(
            np.minimum,
            remaining,
            under_capacity - under_storage,
        ),
        0,
    )

    recharge = recharge_surface + recharge_under

    runoff = xr.where(
        wet_month,
        remaining - recharge_under,
        0,
    )

    surface_loss = xr.zeros_like(surface_storage)
    under_loss = xr.zeros_like(under_storage)

    # Dry-month branch
    deficit = xr.where(
        wet_month,
        0,
        pet - precipitation,
    )

    surface_can_supply = surface_storage >= deficit

    dry_surface_loss = xr.where(
        surface_can_supply,
        deficit,
        surface_storage,
    )

    dry_under_loss = xr.where(
        surface_can_supply,
        0,
        ((deficit - dry_surface_loss) * under_storage / awhc),
    )

    dry_under_loss = xr.apply_ufunc(
        np.minimum,
        dry_under_loss,
        under_storage,
    )

    surface_loss = xr.where(
        wet_month,
        0,
        dry_surface_loss,
    )

    under_loss = xr.where(
        wet_month,
        0,
        dry_under_loss,
    )

    loss = surface_loss + under_loss

    # Palmer ET equation
    et = xr.where(
        wet_month,
        pet,
        precipitation + loss,
    )

    surface_storage = surface_storage + recharge_surface - surface_loss

    under_storage = under_storage + recharge_under - under_loss
    surface_storage = xr.where(
        surface_storage < 0,
        0,
        surface_storage,
    )

    surface_storage = xr.apply_ufunc(
        np.minimum,
        surface_storage,
        surface_capacity,
    )

    under_storage = xr.where(
        under_storage < 0,
        0,
        under_storage,
    )

    under_storage = xr.apply_ufunc(
        np.minimum,
        under_storage,
        under_capacity,
    )

    state_next = PalmerWaterBalanceState(
        surface_storage=surface_storage,
        under_storage=under_storage,
    )

    return (
        state_next,
        et,
        pr,
        recharge,
        runoff,
        pro,
        loss,
        pl,
    )


def palmer_water_balance(
    awhc: xr.DataArray,
    monthly_pr: xr.DataArray,
    monthly_pet: xr.DataArray,
) -> PalmerWaterBalance:
    """
    Compute the Palmer monthly water balance.

    Applies the Palmer two-layer soil-moisture accounting model to a
    monthly precipitation and potential evapotranspiration series and
    returns the resulting water-balance components and soil-moisture
    states.
    """
    surface_capacity = xr.where(
        awhc < 25.4,
        awhc,
        25.4,
    )

    under_capacity = awhc - surface_capacity

    state = PalmerWaterBalanceState(
        surface_storage=surface_capacity.copy(),
        under_storage=under_capacity.copy(),
    )

    et_list = []
    pr_list = []
    r_list = []

    ro_list = []
    pro_list = []

    l_list = []
    pl_list = []

    surface_storage_list = []
    under_storage_list = []

    for i in range(monthly_pr.sizes["time"]):
        p = monthly_pr.isel(time=i)
        pet = monthly_pet.isel(time=i)

        (
            state,
            et,
            pr,
            r,
            ro,
            pro,
            l,
            pl,
        ) = step_palmer_water_balance(
            state,
            awhc,
            p,
            pet,
        )

        et_list.append(et.expand_dims(time=[monthly_pr.time.values[i]]))

        pr_list.append(pr.expand_dims(time=[monthly_pr.time.values[i]]))

        r_list.append(r.expand_dims(time=[monthly_pr.time.values[i]]))

        ro_list.append(ro.expand_dims(time=[monthly_pr.time.values[i]]))

        pro_list.append(pro.expand_dims(time=[monthly_pr.time.values[i]]))

        l_list.append(l.expand_dims(time=[monthly_pr.time.values[i]]))

        pl_list.append(pl.expand_dims(time=[monthly_pr.time.values[i]]))

        surface_storage_list.append(state.surface_storage.expand_dims(time=[monthly_pr.time.values[i]]))

        under_storage_list.append(state.under_storage.expand_dims(time=[monthly_pr.time.values[i]]))

    surface_storage = xr.concat(
        surface_storage_list,
        dim="time",
    )

    under_storage = xr.concat(
        under_storage_list,
        dim="time",
    )

    return PalmerWaterBalance(
        et=xr.concat(et_list, dim="time"),
        pr=xr.concat(pr_list, dim="time"),
        r=xr.concat(r_list, dim="time"),
        ro=xr.concat(ro_list, dim="time"),
        pro=xr.concat(pro_list, dim="time"),
        l=xr.concat(l_list, dim="time"),
        pl=xr.concat(pl_list, dim="time"),
        soil_water=(surface_storage + under_storage),
        surface_storage=surface_storage,
        under_storage=under_storage,
    )


def subset_calibration_period(
    monthly: xr.Dataset,
    calibration_start_year: int | None,
    calibration_end_year: int | None,
) -> xr.Dataset:
    """
    Subset a dataset to the requested Palmer calibration period.

    If no calibration years are provided, the full record is returned.
    """
    if calibration_start_year is None and calibration_end_year is None:
        return monthly

    start_year = calibration_start_year if calibration_start_year is not None else int(monthly.time.dt.year.min())

    end_year = calibration_end_year if calibration_end_year is not None else int(monthly.time.dt.year.max())

    return monthly.where(
        (monthly.time.dt.year >= start_year) & (monthly.time.dt.year <= end_year),
        drop=True,
    )


def compute_cafec_coefficients(
    monthly: xr.Dataset,
) -> xr.Dataset:
    """
    Compute Palmer CAFEC coefficients.

    Parameters
    ----------
    monthly : xr.Dataset
        Monthly Palmer water-balance variables.

    Returns
    -------
    xr.Dataset
        Monthly CAFEC coefficients (alpha, beta, gamma, delta).
    """
    pet_bar = monthly["PET"].groupby("time.month").mean("time")
    aet_bar = monthly["AET"].groupby("time.month").mean("time")

    pr_bar = monthly["PR"].groupby("time.month").mean("time")
    r_bar = monthly["R"].groupby("time.month").mean("time")

    pro_bar = monthly["PRO"].groupby("time.month").mean("time")
    ro_bar = monthly["RO"].groupby("time.month").mean("time")

    pl_bar = monthly["PL"].groupby("time.month").mean("time")
    l_bar = monthly["L"].groupby("time.month").mean("time")

    alpha = xr.where(pet_bar == 0, 1.0, aet_bar / pet_bar)
    beta = xr.where(pr_bar == 0, 1.0, r_bar / pr_bar)
    gamma = xr.where(pro_bar == 0, 1.0, ro_bar / pro_bar)
    delta = xr.where(pl_bar == 0, 1.0, l_bar / pl_bar)

    return xr.Dataset(
        {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
        }
    )


def compute_monthly_departure(
    monthly: xr.Dataset,
    cafec: xr.Dataset,
) -> xr.Dataset:
    """
    Compute CAFEC precipitation and monthly moisture departure.

    Parameters
    ----------
    monthly : xr.Dataset
        Monthly Palmer water-balance variables.
    cafec : xr.Dataset
        Monthly CAFEC coefficients.

    Returns
    -------
    xr.Dataset
        CAFEC precipitation and moisture departure.
    """
    month = monthly.time.dt.month

    alpha = cafec.alpha.sel(month=month)
    beta = cafec.beta.sel(month=month)
    gamma = cafec.gamma.sel(month=month)
    delta = cafec.delta.sel(month=month)

    cafec_precip = alpha * monthly["PET"] + beta * monthly["PR"] + gamma * monthly["PRO"] - delta * monthly["PL"]

    departure = monthly["P"] - cafec_precip

    cafec_precip.name = "CAFEC"
    departure.name = "departure"

    return xr.Dataset(
        {
            "CAFEC": cafec_precip,
            "departure": departure,
        }
    )


def compute_k_factor(
    monthly: xr.Dataset,
    departure: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the Palmer climatic characteristic (K).

    Parameters
    ----------
    monthly : xr.Dataset
        Monthly Palmer water-balance variables.
    departure : xr.DataArray
        Monthly moisture departure.

    Returns
    -------
    xr.DataArray
        Monthly climatic characteristic (K).
    """
    p_bar = monthly["P"].groupby("time.month").mean("time")
    pet_bar = monthly["PET"].groupby("time.month").mean("time")
    r_bar = monthly["R"].groupby("time.month").mean("time")
    ro_bar = monthly["RO"].groupby("time.month").mean("time")
    l_bar = monthly["L"].groupby("time.month").mean("time")

    d_hat = np.abs(departure).groupby("time.month").mean("time")
    t_hat = (pet_bar + r_bar + ro_bar) / (p_bar + l_bar)
    k_hat = 1.5 * np.log10((t_hat + 2.8) / d_hat) + 0.5
    z_hat_month = d_hat * k_hat
    z_hat_total = z_hat_month.sum("month")

    # Monthly departure weighting used to normalize Z-index values.
    k = 17.67 * k_hat / z_hat_total

    k.name = "K"

    return k


def compute_z_index(
    departure: xr.Dataset,
    k_factor: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the Palmer moisture anomaly (Z-index).

    Parameters
    ----------
    departure : xr.Dataset
        Monthly moisture departure.
    k_factor : xr.DataArray
        Monthly climatic characteristic (K).

    Returns
    -------
    xr.DataArray
        Monthly Palmer moisture anomaly (Z-index).
    """
    month = departure.time.dt.month
    k = k_factor.sel(month=month)
    Z = k * departure["departure"]
    Z.name = "Z"

    return Z


def compute_pdsi_from_zindex(
    Z: xr.DataArray,
) -> xr.Dataset:
    """
    Compute Palmer drought indices from monthly Z-index values.

    Parameters
    ----------
    Z : xr.DataArray
        Monthly Palmer moisture anomaly (Z-index).

    Returns
    -------
    xr.Dataset
        Dataset containing PDSI, PHDI, and PMDI.
    """
    pdsi, phdi, pmdi = xr.apply_ufunc(
        _pdsi_from_zindex,
        Z,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            np.float64,
            np.float64,
            np.float64,
        ],
    )

    pdsi.name = "pdsi"
    phdi.name = "phdi"
    pmdi.name = "pmdi"

    return xr.Dataset(
        {
            "pdsi": pdsi,
            "phdi": phdi,
            "pmdi": pmdi,
        }
    )


def _pdsi_from_zindex(
    Z: np.ndarray,
    *,
    persistence: float = 0.897,
    z_weight: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Palmer drought indices from a monthly Z-index series.

    Parameters
    ----------
    Z : np.ndarray
        Monthly Palmer moisture anomaly index.
    persistence : float, default: 0.897
        Persistence coefficient used in the Palmer recurrence.
    z_weight : float, default: 1/3
        Monthly Z-index weighting coefficient.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        PDSI, PHDI, and PMDI time series.
    """
    if np.all(np.isnan(Z)):
        return (
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
        )
    if np.all(~np.isfinite(Z)):
        return (
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
        )
    if np.nanmax(np.abs(Z)) == 0:
        return (
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
        )

    V = 0.0
    Pe = 0.0
    X1 = 0.0
    X2 = 0.0
    X3 = 0.0

    number_of_months = Z.shape[0]

    BT = np.zeros((number_of_months,), dtype=np.int8)

    PX1 = np.zeros((number_of_months,))
    PX2 = np.zeros((number_of_months,))
    PX3 = np.zeros((number_of_months,))
    PPe = np.zeros((number_of_months,))
    X = np.zeros((number_of_months,))
    PMDI = np.zeros((number_of_months,))

    # The Palmer spell-state logic is retained while allowing
    # alternative persistence and Z-index weighting coefficients.
    for k in range(number_of_months):
        if (Pe == 100) or (Pe == 0):
            if abs(X3) <= 0.5:
                PV = 0
                PPe[k] = 0
                PX3[k] = 0

                PX1, PX2, PX3, X, BT = _compute_X(Z, k, PPe, X1, X2, PX1, PX2, PX3, X, BT, persistence, z_weight)

            elif X3 > 0.5:
                if Z[k] >= 0.15:
                    PV, PPe, PX1, PX2, PX3, X, BT = _between_0s(
                        k, Z, X3, PX1, PX2, PX3, PPe, BT, X, persistence, z_weight
                    )

                else:
                    PV, PPe, PX1, PX2, PX3, X, BT = _wet_spell_abatement(
                        k,
                        Z,
                        V,
                        Pe,
                        PPe,
                        PX1,
                        PX2,
                        PX3,
                        X1,
                        X2,
                        X3,
                        X,
                        BT,
                        persistence,
                        z_weight,
                    )

            elif X3 < -0.5:
                if Z[k] <= -0.15:
                    PV, PPe, PX1, PX2, PX3, X, BT = _between_0s(
                        k, Z, X3, PX1, PX2, PX3, PPe, BT, X, persistence, z_weight
                    )

                else:
                    PV, PPe, PX1, PX2, PX3, X, BT = _dry_spell_abatement(
                        k,
                        Z,
                        V,
                        Pe,
                        PPe,
                        PX1,
                        PX2,
                        PX3,
                        X1,
                        X2,
                        X3,
                        X,
                        BT,
                        persistence,
                        z_weight,
                    )

        else:
            if X3 > 0:
                PV, PPe, PX1, PX2, PX3, X, BT = _wet_spell_abatement(
                    k,
                    Z,
                    V,
                    Pe,
                    PPe,
                    PX1,
                    PX2,
                    PX3,
                    X1,
                    X2,
                    X3,
                    X,
                    BT,
                    persistence,
                    z_weight,
                )

            else:
                PV, PPe, PX1, PX2, PX3, X, BT = _dry_spell_abatement(
                    k,
                    Z,
                    V,
                    Pe,
                    PPe,
                    PX1,
                    PX2,
                    PX3,
                    X1,
                    X2,
                    X3,
                    X,
                    BT,
                    persistence,
                    z_weight,
                )

        V = PV
        Pe = PPe[k]
        X1 = PX1[k]
        X2 = PX2[k]
        X3 = PX3[k]

        PMDI[k] = _pmdi(Pe, X1, X2, X3)

        _assign_X(
            k,
            number_of_months,
            BT,
            PX1,
            PX2,
            PX3,
            X,
        )

    PDSI = X
    PHDI = _phdi_select(PX3, X)

    return (
        PDSI,
        PHDI,
        PMDI,
    )


def _pmdi(probability, X1, X2, X3):
    """
    Compute the Palmer Modified Drought Index (PMDI).
    """
    if X3 == 0:
        return X2 if abs(X2) > abs(X1) else X1

    if 0 < probability < 100:
        pro = probability / 100.0
        if X3 <= 0:
            return ((1.0 - pro) * X3) + (pro * X1)
        return ((1.0 - pro) * X3) + (pro * X2)

    return X3


def _phdi_select(px3, x):
    """
    Select the Palmer Hydrological Drought Index (PHDI) series.
    """
    return np.where(px3 == 0, x, px3)


def _find_previous_nonzero(backtrack, k_index):
    """
    Locate the previous nonzero backtracking marker.
    """
    index = 0
    for c in range(k_index - 1, 0, -1):
        if backtrack[c] != 0:
            index = c + 1
            break
    return index


def _assign_X_backtracking(
    X,
    backtrack,
    preliminary_X1,
    preliminary_X2,
    current_month_index,
    previous_nonzero_index,
):
    """
    Propagate backtracked Palmer index assignments.
    """
    # Follow the stored backtracking path and assign the
    # appropriate preliminary wet or dry index.
    for i in range(current_month_index - 1, previous_nonzero_index - 1, -1):
        backtrack[i] = backtrack[i + 1]

        if backtrack[i] == 2:
            if preliminary_X2[i] == 0:
                X[i] = preliminary_X1[i]
                backtrack[i] = 1
            else:
                X[i] = preliminary_X2[i]

        elif backtrack[i] == 1:
            if preliminary_X1[i] == 0:
                X[i] = preliminary_X2[i]
                backtrack[i] = 2
            else:
                X[i] = preliminary_X1[i]


def _assign_X(
    k,
    number_of_months,
    BT,
    PX1,
    PX2,
    PX3,
    X,
):
    """
    Finalize Palmer index assignments after backtracking.
    """
    # Apply deferred backtracking when a spell assignment
    # could not be resolved in the previous timestep.
    if k > 0:
        if (PX3[k - 1] == 0) and (BT[k - 1] == 0):
            previous_nonzero_index = _find_previous_nonzero(BT, k)
            _assign_X_backtracking(
                X,
                BT,
                PX1,
                PX2,
                k,
                previous_nonzero_index,
            )

    if (k == number_of_months - 1) and (PX3[k] == 0) and (X[k] == 0):
        X[k] = PX1[k] if abs(PX1[k]) > abs(PX2[k]) else PX2[k]


def _backtrack(k, PPe, PX1, PX2, PX3, X, BT):
    """
    Perform Palmer spell backtracking.
    """
    r = 0

    # Find the most recent timestep where spell-ending
    # probability was zero.
    for c in range(k, 0, -1):
        if PPe[c] == 0:
            r = c
            break

    # Walk backward through the candidate spell history and
    # resolve the final Palmer index assignments.
    for count in range(k, r, -1):
        if (PPe[count] == 100) and (abs(PX3[count]) > 1):
            X[count] = PX3[count]
            BT[count - 1] = 2 if PX3[count] < 0 else 1

        elif BT[count] == 2:
            if PX2[count] == 0:
                X[count] = PX1[count]
                BT[count] = 1
                BT[count - 1] = 1
            else:
                X[count] = PX2[count]
                BT[count - 1] = 2

        elif BT[count] == 1:
            if PX1[count] == 0:
                X[count] = PX2[count]
                BT[count] = 2
                BT[count - 1] = 2
            else:
                X[count] = PX1[count]
                BT[count - 1] = 1

    return X, BT


def _compute_X(Z, k, PPe, X1, X2, PX1, PX2, PX3, X, BT, persistence, z_weight):
    """
    Update preliminary and established Palmer drought indices.
    """
    # Apply the Palmer recurrence using configurable
    # persistence and Z-index weighting factors.
    PX1[k] = max(0.0, persistence * X1 + z_weight * Z[k])
    PX2[k] = min(0.0, persistence * X2 + z_weight * Z[k])

    if (PX1[k] >= 1) and (PX3[k] == 0):
        X[k] = PX1[k]
        PX3[k] = PX1[k]
        PX1[k] = 0
        BT[k] = 1
        X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)

    elif (PX2[k] <= -1) and (PX3[k] == 0):
        X[k] = PX2[k]
        PX3[k] = PX2[k]
        PX2[k] = 0
        BT[k] = 2
        X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)

    elif PX3[k] == 0:
        if PX1[k] == 0:
            X[k] = PX2[k]
            BT[k] = 2
            X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)

        elif PX2[k] == 0:
            X[k] = PX1[k]
            BT[k] = 1
            X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)

    else:
        X[k] = PX3[k]

    return PX1, PX2, PX3, X, BT


def _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X, persistence, z_weight):
    """
    Handle transitions between established Palmer spells.
    """
    PV = 0
    # Continue the established spell while resetting
    # competing preliminary spell indices.
    PX1[k] = 0
    PX2[k] = 0
    PPe[k] = 0

    PX3[k] = persistence * X3 + z_weight * Z[k]

    X[k] = PX3[k]
    BT[k] = 3

    r = 0
    for count1 in range(k, 0, -1):
        if PPe[count1] == 0:
            r = count1
            break

    for count in range(k, r - 1, -1):
        if BT[count] == 3:
            X[count] = PX3[count]
            if count != r:
                BT[count - 1] = 3

    return PV, PPe, PX1, PX2, PX3, X, BT


def _wet_spell_abatement(
    k,
    Z,
    V,
    Pe,
    PPe,
    PX1,
    PX2,
    PX3,
    X1,
    X2,
    X3,
    X,
    BT,
    persistence,
    z_weight,
):
    """
    Update an established wet spell following Palmer's procedure.
    """
    Ud = Z[k] - 0.15
    # Accumulate evidence that the established wet spell
    # should terminate.
    PV = Ud + min(V, 0)
    PV = Ud + min(V, 0)

    if PV >= 0:
        return _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X, persistence, z_weight)

    Ze = -2.691 * X3 + 1.5
    Q = Ze if Pe == 100 else Ze + V

    # Compute the probability that the established wet
    # spell has ended.
    PPe[k] = (PV / Q) * 100

    if PPe[k] >= 100:
        PPe[k] = 100
        PX3[k] = 0
    else:
        PX3[k] = persistence * X3 + z_weight * Z[k]

    PX1, PX2, PX3, X, BT = _compute_X(Z, k, PPe, X1, X2, PX1, PX2, PX3, X, BT, persistence, z_weight)

    return PV, PPe, PX1, PX2, PX3, X, BT


def _dry_spell_abatement(
    k,
    Z,
    V,
    Pe,
    PPe,
    PX1,
    PX2,
    PX3,
    X1,
    X2,
    X3,
    X,
    BT,
    persistence,
    z_weight,
):
    """
    Update an established drought following Palmer's procedure.
    """
    Uw = Z[k] + 0.15
    # Accumulate evidence that the established drought
    # should terminate.
    PV = Uw + max(V, 0)
    PV = Uw + max(V, 0)

    if PV <= 0:
        return _between_0s(
            k,
            Z,
            X3,
            PX1,
            PX2,
            PX3,
            PPe,
            BT,
            X,
            persistence,
            z_weight,
        )

    Ze = (-2.691 * X3) - 1.5
    Q = Ze if Pe == 100 else Ze + V

    # Compute the probability that the established drought
    # has ended.
    PPe[k] = (PV / Q) * 100

    if PPe[k] >= 100:
        PPe[k] = 100
        PX3[k] = 0
    else:
        PX3[k] = persistence * X3 + z_weight * Z[k]

    PX1, PX2, PX3, X, BT = _compute_X(
        Z,
        k,
        PPe,
        X1,
        X2,
        PX1,
        PX2,
        PX3,
        X,
        BT,
        persistence,
        z_weight,
    )

    return (
        PV,
        PPe,
        PX1,
        PX2,
        PX3,
        X,
        BT,
    )


def compute_scpdsi_from_zindex(
    Z: xr.DataArray,
    calibration_start_year: int | None,
    calibration_end_year: int | None,
) -> xr.Dataset:
    """
    Compute self-calibrated Palmer drought indices from monthly Z-index values.

    Parameters
    ----------
    Z : xr.DataArray
        Monthly Palmer moisture anomaly (Z-index).
    calibration_start_year : int, optional
        First year of the calibration period.
    calibration_end_year : int, optional
        Last year of the calibration period.

    Returns
    -------
    xr.Dataset
        Dataset containing self-calibrated Z-index, SC-PDSI, and
        intermediate Palmer spell-state indices.
    """
    input_start_year = int(Z.time.dt.year.values[0])
    input_end_year = int(Z.time.dt.year.values[-1])

    # Use the full record if no calibration period is specified.
    if calibration_start_year is None:
        calibration_start_year = input_start_year
    if calibration_end_year is None:
        calibration_end_year = input_end_year

    scz, scpdsi, X1, X2, X3 = xr.apply_ufunc(
        _scpdsi_from_zindex,
        Z,
        calibration_start_year,
        calibration_end_year,
        input_start_year,
        input_core_dims=[["time"], [], [], []],
        output_core_dims=[
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            ["time"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
        ],
    )
    scz.name = "scz"
    scpdsi.name = "scpdsi"
    X1.name = "X1"
    X2.name = "X2"
    X3.name = "X3"

    return xr.Dataset(
        {
            "scz": scz,
            "scpdsi": scpdsi,
            "X1": X1,
            "X2": X2,
            "X3": X3,
        }
    )


def _scpdsi_from_zindex(
    Z: np.ndarray,
    calibration_start_year: int,
    calibration_end_year: int,
    input_start_year: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply self-calibration to a monthly Z-index series.
    """
    if np.all(np.isnan(Z)):
        return (
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
        )
    if np.all(~np.isfinite(Z)):
        return (
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
            np.full_like(Z, np.nan),
        )
    if np.nanmax(np.abs(Z)) == 0:
        return (
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
            np.full_like(Z, 0.0),
        )

    # Compute baseline Palmer indices prior to self-calibration.
    pdsi, _, _ = _pdsi_from_zindex(Z)

    (
        scz,
        pdsi,
        scpdsi,
        X1,
        X2,
        X3,
    ) = _self_calibrate(
        pdsi_values=pdsi.copy(),
        sczindex_values=Z.copy(),
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        input_start_year=input_start_year,
    )

    return (
        scz,
        scpdsi,
        X1,
        X2,
        X3,
    )


def _pdsi_at_percentile(
    pdsi_values: np.ndarray,
    percentile: float,
) -> float:
    """
    Return a percentile of the calibration-period PDSI distribution.

    Missing values are ignored.
    """
    valid = pdsi_values[np.isfinite(pdsi_values)]

    if valid.size == 0:
        return np.nan

    return np.percentile(
        valid,
        percentile * 100.0,
    )


def _z_sum(
    interval: int,
    wet_or_dry: str,
    sczindex_values: np.ndarray,
    periods_per_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    input_start_year: int,
) -> float:
    """
    Return the most extreme rolling accumulated Z value for the
    specified interval length within the calibration period.
    """
    cal_slice = _calibration_slice(
        calibration_start_year,
        calibration_end_year,
        input_start_year,
        periods_per_year,
    )

    # Restrict analysis to the calibration period and remove
    # missing values before computing rolling accumulation.
    Z = sczindex_values[cal_slice]
    Z = Z[np.isfinite(Z)]

    if len(Z) < interval:
        return np.nan

    # Rolling accumulated Z anomalies for the requested duration.
    rolling = np.convolve(
        Z,
        np.ones(interval),
        mode="valid",
    )

    if wet_or_dry == "WET":
        return float(np.max(rolling))
    if wet_or_dry == "DRY":
        return float(np.min(rolling))

    raise ValueError(f"Invalid wet_or_dry value: {wet_or_dry}")


def _duration_factors(
    zindex_values: np.ndarray,
    calibration_start_year: int,
    calibration_end_year: int,
    data_start_year: int,
    wet_or_dry: str,
) -> tuple[float, float]:
    """
    Estimate self-calibration duration-factor coefficients.

    Fits a linear relationship between accumulation interval length
    and the most extreme rolling accumulated Z anomalies within the
    calibration period.
    """
    month_scales = [3, 6, 9, 12, 18, 24, 30, 36, 42, 48]

    z_sums = np.zeros((len(month_scales),))

    for i, scale_months in enumerate(month_scales):
        z_sums[i] = _z_sum(
            scale_months,
            wet_or_dry,
            zindex_values,
            12,
            calibration_start_year,
            calibration_end_year,
            data_start_year,
        )

    # Ordinary least-squares fitting is retained to remain
    # consistent with the Palmer/Wells duration-factor derivation.
    # Alternative regression methods were evaluated but are not
    # used in the current methodology.
    slope, intercept = np.polyfit(
        month_scales,
        z_sums,
        1,
    )

    # Scale regression coefficients to the Palmer
    # severity-class limits (+4 wet, -4 dry).
    pdsi_limit = _PDSI_MAX

    if wet_or_dry == "DRY":
        pdsi_limit = _PDSI_MIN

    slope = slope / pdsi_limit
    intercept = intercept / pdsi_limit

    if slope == 0 and intercept == 0:
        return np.nan, np.nan

    return slope, intercept


def _calibration_slice(
    calibration_start_year: int,
    calibration_end_year: int,
    input_start_year: int,
    periods_per_year: int = 12,
) -> slice:
    """
    Return the calibration-period slice within a monthly series.
    """
    start = (calibration_start_year - input_start_year) * periods_per_year
    n_periods = (calibration_end_year - calibration_start_year + 1) * periods_per_year

    return slice(
        start,
        start + n_periods,
    )


def _self_calibrate(
    pdsi_values: np.ndarray,
    sczindex_values: np.ndarray,
    calibration_start_year: int,
    calibration_end_year: int,
    input_start_year: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply Wells-style self-calibration to Palmer drought indices.

    Derives local wet and dry scaling factors and duration-factor
    coefficients from the calibration period, then computes
    self-calibrated Z values and SC-PDSI.

    Returns
    -------
    tuple
        Self-calibrated Z values, PDSI values, SC-PDSI values,
        wet spell indices, dry spell indices, and established
        spell indices.
    """
    cal_slice = _calibration_slice(
        calibration_start_year,
        calibration_end_year,
        input_start_year,
    )

    calibration_pdsi = pdsi_values[cal_slice]

    # Scale positive and negative extremes independently so that
    # local wet and dry conditions map to the Palmer severity scale.
    dry_extreme = _pdsi_at_percentile(
        calibration_pdsi,
        0.02,
    )

    if dry_extreme == 0.0:
        dry_ratio = 1.0
    else:
        dry_ratio = _PDSI_MIN / dry_extreme

    wet_extreme = _pdsi_at_percentile(
        calibration_pdsi,
        0.98,
    )

    if wet_extreme == 0.0:
        wet_ratio = 1.0
    else:
        wet_ratio = _PDSI_MAX / wet_extreme

    for time_step, sczindex in enumerate(sczindex_values):
        if not np.isnan(sczindex):
            adjustment_factor = wet_ratio if sczindex >= 0 else dry_ratio
            sczindex_values[time_step] = sczindex * adjustment_factor

    established_index_values = np.full(
        pdsi_values.shape,
        np.nan,
    )
    scpdsi_values = np.full(
        pdsi_values.shape,
        np.nan,
    )
    wet_index_values = np.full(
        pdsi_values.shape,
        np.nan,
    )
    dry_index_values = np.full(
        pdsi_values.shape,
        np.nan,
    )

    # Estimate separate wet and dry duration relationships from the
    # self-calibrated Z-index series.
    wet_m, wet_b = _duration_factors(
        sczindex_values,
        calibration_start_year,
        calibration_end_year,
        input_start_year,
        "WET",
    )

    dry_m, dry_b = _duration_factors(
        sczindex_values,
        calibration_start_year,
        calibration_end_year,
        input_start_year,
        "DRY",
    )

    if any(np.isnan(value) for value in (wet_m, wet_b, dry_m, dry_b)):
        nan_arr = np.full(
            pdsi_values.shape,
            np.nan,
        )

        return (
            sczindex_values,
            pdsi_values,
            nan_arr,  # scpdsi
            nan_arr.copy(),  # X1
            nan_arr.copy(),  # X2
            nan_arr.copy(),  # X3
        )

    (
        pdsi_values,
        scpdsi_values,
        wet_index_values,
        dry_index_values,
        established_index_values,
    ) = _compute_scpdsi(
        established_index_values,
        sczindex_values,
        scpdsi_values,
        pdsi_values,
        wet_index_values,
        dry_index_values,
        wet_m,
        wet_b,
        dry_m,
        dry_b,
        True,
    )

    return (
        sczindex_values,
        pdsi_values,
        scpdsi_values,
        wet_index_values,
        dry_index_values,
        established_index_values,
    )


def _compute_scpdsi(
    established_index_values: np.ndarray,
    sczindex_values: np.ndarray,
    scpdsi_values: np.ndarray,
    pdsi_values: np.ndarray,
    wet_index_values: np.ndarray,
    dry_index_values: np.ndarray,
    wet_M: float,
    wet_B: float,
    dry_M: float,
    dry_B: float,
    calibration_complete: bool,
    tolerance: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute the self-calibrated Palmer Drought Severity Index (SC-PDSI).

    Applies separate wet and dry duration-factor relationships to the
    self-calibrated Z-index series and updates Palmer spell-state indices
    using the Wells et al. (2004) self-calibration procedure.
    """
    # Candidate wet and dry spell histories used for
    # SC-PDSI backtracking.
    wet_index_deque = collections.deque([])
    dry_index_deque = collections.deque([])

    # Accumulated spell-abatement evidence.
    V = 0.0

    previous_key = -1

    for period in range(established_index_values.size):
        previous_established_index_X3 = 0.0

        if previous_key >= 0 and not np.isnan(established_index_values[previous_key]):
            previous_established_index_X3 = established_index_values[previous_key]

        # Select wet- or dry-spell duration factors based on the
        # sign of the established Palmer spell.
        if previous_established_index_X3 >= 0:
            m = wet_M
            b = wet_B
        else:
            m = dry_M
            b = dry_B

        if np.isnan(m) or np.isnan(b) or (m + b) == 0:
            return (
                pdsi_values,
                scpdsi_values,
                wet_index_values,
                dry_index_values,
                established_index_values,
            )

        if not np.isnan(sczindex_values[period]) and ((m + b) != 0):
            # Convert duration-factor coefficients to the Palmer
            # persistence coefficient used in the recurrence relation.
            c = 1 - (m / (m + b))

            # Spell direction: wet (+1) or dry (-1).
            if previous_established_index_X3 >= 0:
                wd = 1
            else:
                wd = -1

            # No established spell exists. Skip spell-abatement
            # calculations and initialize a new candidate spell.
            if previous_established_index_X3 == 0:
                new_X3 = 0
                new_V = 0

                new_X, new_X1, new_X2, new_X3 = _choose_X(
                    pdsi_values,
                    established_index_values,
                    wet_index_values,
                    dry_index_values,
                    sczindex_values,
                    wet_index_deque,
                    dry_index_deque,
                    wet_M,
                    wet_B,
                    dry_M,
                    dry_B,
                    new_X3,
                    period,
                    previous_key,
                )

            else:
                new_X3 = c * previous_established_index_X3 + sczindex_values[period] / (m + b)

                # ZE is the Z anomaly required to terminate an established spell.
                ZE = (m + b) * (wd * 0.5 - c * previous_established_index_X3)

                Q = ZE + V

                if wd > 0:
                    # Accumulate evidence that an established wet spell
                    # should terminate.
                    new_V = sczindex_values[period] - 0.5 + min(V, 0)
                else:
                    # Accumulate evidence that an established drought
                    # should terminate.
                    new_V = sczindex_values[period] + 0.5 + max(V, 0)

                if (wd * new_V) > 0:
                    new_X1 = 0
                    new_X2 = 0
                    new_X = new_X3

                    wet_index_deque.clear()
                    dry_index_deque.clear()

                else:
                    # Probability that the established spell has ended.
                    new_probability = (new_V / Q) * 100

                    if new_probability >= (100 - tolerance):
                        new_X3 = 0
                        new_V = 0

                    new_X, new_X1, new_X2, new_X3 = _choose_X(
                        pdsi_values,
                        established_index_values,
                        wet_index_values,
                        dry_index_values,
                        sczindex_values,
                        wet_index_deque,
                        dry_index_deque,
                        wet_M,
                        wet_B,
                        dry_M,
                        dry_B,
                        new_X3,
                        period,
                        previous_key,
                    )

            wet_index_values[period] = new_X1
            dry_index_values[period] = new_X2
            established_index_values[period] = new_X3

            if calibration_complete:
                scpdsi_values[period] = new_X
            else:
                pdsi_values[period] = new_X

            V = new_V

        else:
            pdsi_values[period] = np.nan
            wet_index_values[period] = np.nan
            dry_index_values[period] = np.nan
            established_index_values[period] = np.nan

            if calibration_complete:
                scpdsi_values[period] = np.nan
            else:
                pdsi_values[period] = np.nan

        previous_key = period

    return (
        pdsi_values,
        scpdsi_values,
        wet_index_values,
        dry_index_values,
        established_index_values,
    )


def _choose_X(
    pdsi_values: np.ndarray,
    established_index_values: np.ndarray,
    wet_index_values: np.ndarray,
    dry_index_values: np.ndarray,
    sczindex_values: np.ndarray,
    wet_index_deque,
    dry_index_deque,
    wet_M: float,
    wet_B: float,
    dry_M: float,
    dry_B: float,
    new_X3: float,
    month_index: int,
    previous_key: int,
    tolerance=0.0,
) -> tuple[float, float, float, float]:
    """
    Update candidate and established self-calibrated Palmer indices.

    Computes the wet (X1) and dry (X2) candidate spell indices and
    determines whether either candidate becomes the established spell
    (X3). Candidate spell histories are retained for possible
    backtracking during spell transitions.
    """
    previous_wet_index_X1 = 0
    previous_dry_index_X2 = 0

    if (previous_key >= 0) and not np.isnan(established_index_values[previous_key]):
        previous_wet_index_X1 = wet_index_values[previous_key]
        previous_dry_index_X2 = dry_index_values[previous_key]

    # Convert wet and dry duration-factor coefficients to
    # Palmer persistence coefficients.
    wetc = 1 - (wet_M / (wet_M + wet_B))
    dryc = 1 - (dry_M / (dry_M + dry_B))

    z_index = sczindex_values[month_index]

    # Update candidate wet and dry spell indices.
    new_X1 = wetc * previous_wet_index_X1 + z_index / (wet_M + wet_B)
    if new_X1 < 0:
        new_X1 = 0.0

    new_X2 = dryc * previous_dry_index_X2 + z_index / (dry_M + dry_B)
    if new_X2 > 0:
        new_X2 = 0.0

    if (new_X1 >= 0.5) and (new_X3 == 0):
        _backtrack_self_calibrated(
            pdsi_values,
            wet_index_deque,
            dry_index_deque,
            tolerance,
            new_X1,
            month_index,
        )
        new_X = new_X1
        new_X3 = new_X1
        new_X1 = 0.0

    else:
        if (new_X2 <= -0.5) and (new_X3 == 0):
            _backtrack_self_calibrated(
                pdsi_values,
                wet_index_deque,
                dry_index_deque,
                tolerance,
                new_X2,
                month_index,
            )
            new_X = new_X2
            new_X3 = new_X2
            new_X2 = 0.0

        elif new_X3 == 0:
            if new_X1 == 0:
                _backtrack_self_calibrated(
                    pdsi_values,
                    wet_index_deque,
                    dry_index_deque,
                    tolerance,
                    new_X2,
                    month_index,
                )
                new_X = new_X2

            elif new_X2 == 0:
                _backtrack_self_calibrated(
                    pdsi_values,
                    wet_index_deque,
                    dry_index_deque,
                    tolerance,
                    new_X1,
                    month_index,
                )
                new_X = new_X1

            else:
                wet_index_deque.appendleft(new_X1)
                dry_index_deque.appendleft(new_X2)
                new_X = new_X3

        else:
            # Retain candidate spell histories for possible
            # backtracking during future transitions.
            wet_index_deque.appendleft(new_X1)
            dry_index_deque.appendleft(new_X2)
            new_X = new_X3

    return new_X, new_X1, new_X2, new_X3


def _backtrack_self_calibrated(
    pdsi_values: np.ndarray,
    wet_index_deque,
    dry_index_deque,
    tolerance: float,
    new_X: float,
    month_index: int,
):
    """
    Backtrack self-calibrated candidate spell histories.

    Resolves stored wet and dry spell candidates after a spell
    transition and updates the Palmer index assignment history.
    """
    num1 = new_X

    # Walk backward through the stored candidate spell
    # histories following the sign of the current spell.
    while wet_index_deque and dry_index_deque:
        if num1 > 0:
            num1 = wet_index_deque.popleft()
            num2 = dry_index_deque.popleft()
        else:
            num1 = dry_index_deque.popleft()
            num2 = wet_index_deque.popleft()

        # Switch to the competing candidate when the current
        # value is effectively zero.
        if ((-1.0 * tolerance) <= num1) and (num1 <= tolerance):
            num1 = num2

        pdsi_values[month_index] = num1


def _is_monthly(data: xr.DataArray) -> bool:
    """
    Determine whether a time series is monthly frequency.
    """
    if data.time.size < 2:
        return False

    delta = data.time.values[1] - data.time.values[0]

    return delta.astype("timedelta64[D]").astype(int) > 27
