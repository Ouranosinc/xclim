# noqa: D205,D400
"""
Uncertainty Partitioning
========================

This module implements methods and tools meant to partition climate projection uncertainties into different components.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

"""
Implemented partitioning algorithms:

 - `hawkins_sutton`
 - `lafferty_sriver`

# References for other more recent algorithms that could be added here.

Yip, S., Ferro, C. A. T., Stephenson, D. B., and Hawkins, E. (2011). A Simple, Coherent Framework for Partitioning
Uncertainty in Climate Predictions. Journal of Climate 24, 17, 4634-4643, doi:10.1175/2011JCLI4085.1

Northrop, P. J., & Chandler, R. E. (2014). Quantifying sources of uncertainty in projections of future climate.
Journal of Climate, 27(23), 8793–8808, doi:10.1175/JCLI-D-14-00265.1

Goldenson, N., Mauger, G., Leung, L. R., Bitz, C. M., & Rhines, A. (2018). Effects of ensemble configuration on
estimates of regional climate uncertainties. Geophysical Research Letters, 45, 926– 934.
https://doi.org/10.1002/2017GL076297

Lehner, F., Deser, C., Maher, N., Marotzke, J., Fischer, E. M., Brunner, L., Knutti, R., and Hawkins,
E. (2020). Partitioning climate projection uncertainty with multiple large ensembles and CMIP5/6, Earth Syst. Dynam.,
11, 491–508, https://doi.org/10.5194/esd-11-491-2020.

Evin, G., Hingray, B., Blanchet, J., Eckert, N., Morin, S., & Verfaillie, D. (2019). Partitioning Uncertainty
Components of an Incomplete Ensemble of Climate Projections Using Data Augmentation, Journal of Climate, 32(8),
2423-2440, https://doi.org/10.1175/JCLI-D-18-0606.1

Beigi E, Tsai FT-C, Singh VP, Kao S-C. Bayesian Hierarchical Model Uncertainty Quantification for Future Hydroclimate
Projections in Southern Hills-Gulf Region, USA. Water. 2019; 11(2):268. https://doi.org/10.3390/w11020268

Related bixtex entries:
 - yip_2011
 - northrop_2014
 - goldenson_2018
 - lehner_2020
 - evin_2019
"""

# TODO: Add ref for Brekke and Barsugli (2013)


def hawkins_sutton(
    da: xr.DataArray,
    sm: xr.DataArray | None = None,
    weights: xr.DataArray | None = None,
    baseline: tuple[str, str] = ("1971", "2000"),
    kind: str = "+",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the mean and partitioned variance of an ensemble based on method from Hawkins & Sutton (2009).

    Parameters
    ----------
    da : xr.DataArray
        Time series with dimensions 'time', 'scenario' and 'model'.
    sm : xr.DataArray, optional
        Smoothed time series over time, with the same dimensions as `da`. By default, this is estimated using a
        4th-order polynomial. Results are sensitive to the choice of smoothing function, use this to set another
        polynomial order, or a LOESS curve.
    weights : xr.DataArray, optional
        Weights to be applied to individual models. Should have `model` dimension.
    baseline : (str, str)
        Start and end year of the reference period.
    kind : {'+', '*'}
        Whether the mean over the reference period should be subtracted (+) or divided by (*).

    Returns
    -------
    xr.DataArray, xr.DataArray
        The mean relative to the baseline, and the components of variance of the ensemble. These components are
        coordinates along the `uncertainty` dimension: `variability`, `model`, `scenario`, and `total`.

    Notes
    -----
    To prepare input data, make sure `da` has dimensions `time`, `scenario` and `model`,
    e.g. `da.rename({"scen": "scenario"})`.

    To reproduce results from :cite:t:`hawkins_2009`, input data should meet the following requirements:
      - annual time series starting in 1950 and ending in 2100;
      - the same models are available for all scenarios.

    To get the fraction of the total variance instead of the variance itself, call `fractional_uncertainty` on the
    output.

    References
    ----------
    :cite:cts:`hawkins_2009,hawkins_2011`
    """
    if xr.infer_freq(da.time)[0] not in ["A", "Y"]:
        raise ValueError("This algorithm expects annual time series.")

    if not {"time", "scenario", "model"}.issubset(da.dims):
        raise ValueError(
            "DataArray dimensions should include 'time', 'scenario' and 'model'."
        )

    # Confirm the same models have data for all scenarios
    check = da.notnull().any("time").all("scenario")
    if not check.all():
        raise ValueError(f"Some models are missing data for some scenarios: \n {check}")

    if weights is None:
        weights = xr.ones_like(da.model, float)

    if sm is None:
        # Fit 4th order polynomial to smooth natural fluctuations
        # Note that the order of the polynomial has a substantial influence on the results.
        fit = da.polyfit(dim="time", deg=4, skipna=True)
        sm = xr.polyval(coord=da.time, coeffs=fit.polyfit_coefficients).where(
            da.notnull()
        )

    # Decadal mean residuals
    res = (da - sm).rolling(time=10, center=True).mean()

    # Individual model variance after 2000: V
    # Note that the historical data is the same for all scenarios.
    nv_u = (
        res.sel(time=slice("2000", None))
        .var(dim=("scenario", "time"))
        .weighted(weights)
        .mean("model")
    )

    # Compute baseline average
    ref = sm.sel(time=slice(*baseline)).mean(dim="time")

    # Remove baseline average from smoothed time series
    if kind == "+":
        sm -= ref
    elif kind == "*":
        sm /= ref
    else:
        raise ValueError(kind)

    # Model uncertainty: M(t)
    model_u = sm.weighted(weights).var(dim="model").mean(dim="scenario")

    # Scenario uncertainty: S(t)
    scenario_u = sm.weighted(weights).mean(dim="model").var(dim="scenario")

    # Total uncertainty: T(t)
    total = nv_u + scenario_u + model_u

    # Create output array with the uncertainty components
    u = pd.Index(["variability", "model", "scenario", "total"], name="uncertainty")
    uncertainty = xr.concat([nv_u, model_u, scenario_u, total], dim=u)

    # Keep a trace of the elements for each uncertainty component
    for d in ["model", "scenario"]:
        uncertainty.attrs[d] = da[d].values

    # Mean projection: G(t)
    g = sm.weighted(weights).mean(dim="model").mean(dim="scenario")

    return g, uncertainty


def hawkins_sutton_09_weighting(da, obs, baseline=("1971", "2000")):
    """Return weights according to the ability of models to simulate observed climate change.

    Weights are computed by comparing the 2000 value to the baseline mean: w_m = 1 / (x_{obs} + | x_{m,
    2000} - x_obs | )

    Parameters
    ----------
    da : xr.DataArray
        Input data over the historical period. Should have a time and model dimension.
    obs : float
        Observed change.
    baseline : (str, str)
        Baseline start and end year.

    Returns
    -------
    xr.DataArray
      Weights over the model dimension.
    """
    mm = da.sel(time=slice(*baseline)).mean("time")
    xm = da.sel(time=baseline[1]) - mm
    xm = xm.drop_vars("time").squeeze()
    return 1 / (obs + np.abs(xm - obs))


def lafferty_sriver(
    da: xr.DataArray,
    sm: xr.DataArray = None,
    bb13: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the mean and partitioned variance of an ensemble based on method from Lafferty and Sriver (2023).

    Parameters
    ----------
    da : xr.DataArray
        Time series with dimensions 'time', 'scenario', 'downscaling' and 'model'.
    sm : xr.DataArray
        Smoothed time series over time, with the same dimensions as `da`. By default, this is estimated using a
        4th-order polynomial. Results are sensitive to the choice of smoothing function, use this to set another
        polynomial order, or a LOESS curve.
    bb13 : bool
        Whether to apply the Brekke and Barsugli (2013) method to estimate scenario uncertainty, where the variance
        over scenarios is computed before taking the mean over models and downscaling methods.

    Returns
    -------
    xr.DataArray, xr.DataArray
        The mean relative to the baseline, and the components of variance of the ensemble. These components are
        coordinates along the `uncertainty` dimension: `variability`, `model`, `scenario`, `downscaling` and `total`.

    Notes
    -----
    To prepare input data, make sure `da` has dimensions `time`, `scenario`, `downscaling` and `model`,
    e.g. `da.rename({"experiment": "scenario"})`.

    To get the fraction of the total variance instead of the variance itself, call `fractional_uncertainty` on the
    output.

    References
    ----------
    :cite:cts:`Lafferty2023`
    """
    if xr.infer_freq(da.time)[0] not in ["A", "Y"]:
        raise ValueError("This algorithm expects annual time series.")

    if not {"time", "scenario", "model", "downscaling"}.issubset(da.dims):
        raise ValueError(
            "DataArray dimensions should include 'time', 'scenario', 'downscaling' and 'model'."
        )

    if sm is None:
        # Fit a 4th order polynomial
        fit = da.polyfit(dim="time", deg=4, skipna=True)
        sm = xr.polyval(coord=da.time, coeffs=fit.polyfit_coefficients).where(
            da.notnull()
        )

    # "Interannual variability is then estimated as the centered rolling 11-year variance of the difference
    # between the extracted forced response and the raw outputs, averaged over all outputs."
    nv_u = (
        (da - sm)
        .rolling(time=11, center=True)
        .var()
        .mean(dim=["scenario", "model", "downscaling"])
    )

    # Scenario uncertainty: U_s(t)
    if bb13:
        scenario_u = sm.var(dim="scenario").mean(dim=["model", "downscaling"])
    else:
        scenario_u = sm.mean(dim=["model", "downscaling"]).var(dim="scenario")

    # Model uncertainty: U_m(t)

    # Count the number of parent models that have been downscaled using method $d$ for scenario $s$.
    # In the paper, weights are constant, here they may vary across time if there are missing values.
    mw = sm.count("model")
    # In https://github.com/david0811/lafferty-sriver_2023_npjCliAtm/blob/main/unit_test/lafferty_sriver.py
    # weights are set to zero when there is only one model, but the var for a single element is 0 anyway.
    model_u = sm.var(dim="model").weighted(mw).mean(dim=["scenario", "downscaling"])

    # Downscaling uncertainty: U_d(t)
    dw = sm.count("downscaling")
    downscaling_u = (
        sm.var(dim="downscaling").weighted(dw).mean(dim=["scenario", "model"])
    )

    # Total uncertainty: T(t)
    total = nv_u + scenario_u + model_u + downscaling_u

    # Create output array with the uncertainty components
    u = pd.Index(
        ["model", "scenario", "downscaling", "variability", "total"], name="uncertainty"
    )
    uncertainty = xr.concat([model_u, scenario_u, downscaling_u, nv_u, total], dim=u)

    # Keep a trace of the elements for each uncertainty component
    for d in ["model", "scenario", "downscaling"]:
        uncertainty.attrs[d] = da[d].values

    # Mean projection:
    # This is not part of the original algorithm, but we want all partition algos to have similar outputs.
    g = sm.mean(dim="model").mean(dim="scenario").mean(dim="downscaling")

    return g, uncertainty


def fractional_uncertainty(u: xr.DataArray) -> xr.DataArray:
    """Return the fractional uncertainty.

    Parameters
    ----------
    u : xr.DataArray
        Array with uncertainty components along the `uncertainty` dimension.

    Returns
    -------
    xr.DataArray
        Fractional, or relative uncertainty with respect to the total uncertainty.
    """
    uncertainty = u / u.sel(uncertainty="total") * 100
    uncertainty.attrs.update(u.attrs)
    uncertainty.attrs["long_name"] = "Fraction of total variance"
    uncertainty.attrs["units"] = "%"
    return uncertainty
