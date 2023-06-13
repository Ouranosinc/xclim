"""
Pre- and Post-Processing Submodule
==================================
"""
from __future__ import annotations

import warnings
from typing import Sequence

import dask.array as dsk
import numpy as np
import xarray as xr
from xarray.core.utils import get_temp_dimname

from xclim.core.calendar import get_calendar, max_doy, parse_offset
from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to, infer_context, units
from xclim.core.utils import uses_dask

from ._processing import _adapt_freq, _normalize, _reordering
from .base import Grouper
from .nbutils import _escore
from .utils import ADDITIVE, copy_all_attrs


@update_xclim_history
def adapt_freq(
    ref: xr.DataArray,
    sim: xr.DataArray,
    *,
    group: Grouper | str,
    thresh: str = "0 mm d-1",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on :cite:t:`sdba-themesl_empirical-statistical_2012`.

    Parameters
    ----------
    ds : xr.Dataset
        With variables : "ref", Target/reference data, usually observed data, and  "sim", Simulated data.
    dim : str
        Dimension name.
    group : str or Grouper
        Grouping information, see base.Grouper
    thresh : str
        Threshold below which values are considered zero, a quantity with units.

    Returns
    -------
    sim_adj : xr.DataArray
        Simulated data with the same frequency of values under threshold than ref.
        Adjustment is made group-wise.
    pth : xr.DataArray
        For each group, the smallest value of sim that was not frequency-adjusted.
        All values smaller were either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
    dP0 : xr.DataArray
        For each group, the percentage of values that were corrected in sim.

    Notes
    -----
    With :math:`P_0^r` the frequency of values under threshold :math:`T_0` in the reference (ref) and
    :math:`P_0^s` the same for the simulated values, :math:`\\Delta P_0 = \\frac{P_0^s - P_0^r}{P_0^s}`,
    when positive, represents the proportion of values under :math:`T_0` that need to be corrected.

    The correction replaces a proportion :math:`\\Delta P_0` of the values under :math:`T_0` in sim by a uniform random
    number between :math:`T_0` and :math:`P_{th}`, where :math:`P_{th} = F_{ref}^{-1}( F_{sim}( T_0 ) )` and
    `F(x)` is the empirical cumulative distribution function (CDF).

    References
    ----------
    :cite:cts:`sdba-themesl_empirical-statistical_2012`

    """
    with units.context(infer_context(ref.attrs.get("standard_name"))):
        sim = convert_units_to(sim, ref)
        thresh = convert_units_to(thresh, ref)

    out = _adapt_freq(xr.Dataset(dict(sim=sim, ref=ref)), group=group, thresh=thresh)

    # Set some metadata
    copy_all_attrs(out, sim)
    out.sim_ad.attrs.update(sim.attrs)
    out.sim_ad.attrs.update(
        references="Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate "
        "models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4."
    )
    out.pth.attrs.update(
        long_name="Smallest value of the timeseries not corrected by frequency adaptation.",
        units=sim.units,
    )
    out.dP0.attrs.update(
        long_name=f"Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation",
    )

    return out.sim_ad, out.pth, out.dP0


def jitter_under_thresh(x: xr.DataArray, thresh: str) -> xr.DataArray:
    """Replace values smaller than threshold by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold under which to add uniform random noise to values, a quantity with units.

    Returns
    -------
    xr.DataArray

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    return jitter(x, lower=thresh, upper=None, minimum=None, maximum=None)


def jitter_over_thresh(x: xr.DataArray, thresh: str, upper_bnd: str) -> xr.DataArray:
    """Replace values greater than threshold by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold over which to add uniform random noise to values, a quantity with units.
    upper_bnd : str
        Maximum possible value for the random noise, a quantity with units.

    Returns
    -------
    xr.DataArray

    Notes
    -----
    If thresh is low, this will change the mean value of x.

    """
    return jitter(x, lower=None, upper=thresh, minimum=None, maximum=upper_bnd)


@update_xclim_history
def jitter(
    x: xr.DataArray,
    lower: str | None = None,
    upper: str | None = None,
    minimum: str | None = None,
    maximum: str | None = None,
) -> xr.DataArray:
    """Replace values under a threshold and values above another by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's `jitter`, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    lower : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the lower end.
    upper : str, optional
        Threshold over which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the upper end.
    minimum : str, optional
        Lower limit (excluded) for the lower end random noise, a quantity with units.
        If None but `lower` is not None, 0 is used.
    maximum : str, optional
        Upper limit (excluded) for the upper end random noise, a quantity with units.
        If `upper` is not None, it must be given.

    Returns
    -------
    xr.DataArray
        Same as  `x` but values < lower are replaced by a uniform noise in range (minimum, lower)
        and values >= upper are replaced by a uniform noise in range [upper, maximum).
        The two noise distributions are independent.
    """
    with units.context(infer_context(x.attrs.get("standard_name"))):
        out = x
        notnull = x.notnull()
        if lower is not None:
            lower = convert_units_to(lower, x)
            minimum = convert_units_to(minimum, x) if minimum is not None else 0
            minimum = minimum + np.finfo(x.dtype).eps
            if uses_dask(x):
                jitter = dsk.random.uniform(
                    low=minimum, high=lower, size=x.shape, chunks=x.chunks
                )
            else:
                jitter = np.random.uniform(low=minimum, high=lower, size=x.shape)
            out = out.where(~((x < lower) & notnull), jitter.astype(x.dtype))
        if upper is not None:
            if maximum is None:
                raise ValueError("If 'upper' is given, so must 'maximum'.")
            upper = convert_units_to(upper, x)
            maximum = convert_units_to(maximum, x)
            if uses_dask(x):
                jitter = dsk.random.uniform(
                    low=upper, high=maximum, size=x.shape, chunks=x.chunks
                )
            else:
                jitter = np.random.uniform(low=upper, high=maximum, size=x.shape)
            out = out.where(~((x >= upper) & notnull), jitter.astype(x.dtype))

        copy_all_attrs(out, x)  # copy attrs and same units
        return out


@update_xclim_history
def normalize(
    data: xr.DataArray,
    norm: xr.DataArray | None = None,
    *,
    group: Grouper | str,
    kind: str = ADDITIVE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Normalize an array by removing its mean.

    Normalization if performed group-wise and according to `kind`.

    Parameters
    ----------
    data : xr.DataArray
        The variable to normalize.
    norm : xr.DataArray, optional
        If present, it is used instead of computing the norm again.
    group : str or Grouper
        Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details..
    kind : {'+', '*'}
        If `kind` is "+", the mean is subtracted from the mean and if it is '*', it is divided from the data.

    Returns
    -------
    xr.DataArray
        Groupwise anomaly.
    norm : xr.DataArray
        Mean over each group.
    """
    ds = xr.Dataset(dict(data=data))

    if norm is not None:
        norm = convert_units_to(
            norm, data, context=infer_context(data.attrs.get("standard_name"))
        )
        ds = ds.assign(norm=norm)

    out = _normalize(ds, group=group, kind=kind)
    copy_all_attrs(out, ds)
    out.data.attrs.update(data.attrs)
    out.norm.attrs["units"] = data.attrs["units"]
    return out.data.rename(data.name), out.norm


def uniform_noise_like(
    da: xr.DataArray, low: float = 1e-6, high: float = 1e-3
) -> xr.DataArray:
    """Return a uniform noise array of the same shape as da.

    Noise is uniformly distributed between low and high.
    Alternative method to `jitter_under_thresh` for avoiding zeroes.
    """
    if uses_dask(da):
        mod = dsk
        kw = {"chunks": da.chunks}
    else:
        mod = np
        kw = {}

    return da.copy(
        data=(high - low) * mod.random.random_sample(size=da.shape, **kw) + low
    )


@update_xclim_history
def standardize(
    da: xr.DataArray,
    mean: xr.DataArray | None = None,
    std: xr.DataArray | None = None,
    dim: str = "time",
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray, xr.DataArray]:
    """Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns
    -------
    out : xr.DataArray or xr.Dataset
        Standardized data.
    mean : xr.DataArray
        Mean.
    std : xr.DataArray
        Standard Deviation.
    """
    if mean is None:
        mean = da.mean(dim, keep_attrs=True)
    if std is None:
        std = da.std(dim, keep_attrs=True)
    out = (da - mean) / std
    copy_all_attrs(out, da)
    return out, mean, std


@update_xclim_history
def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    out = (std * da) + mean
    copy_all_attrs(out, da)
    return out


@update_xclim_history
def reordering(ref: xr.DataArray, sim: xr.DataArray, group: str = "time") -> xr.Dataset:
    """Reorders data in `sim` following the order of ref.

    The rank structure of `ref` is used to reorder the elements of `sim` along dimension "time", optionally doing the
    operation group-wise.

    Parameters
    ----------
    sim : xr.DataArray
        Array to reorder.
    ref : xr.DataArray
        Array whose rank order sim should replicate.
    group : str
        Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.

    Returns
    -------
    xr.Dataset
        sim reordered according to ref's rank order.

    References
    ----------
    :cite:cts:`sdba-cannon_multivariate_2018`

    """
    ds = xr.Dataset({"sim": sim, "ref": ref})
    out = _reordering(ds, group=group).reordered
    copy_all_attrs(out, sim)
    return out


@update_xclim_history
def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,  # noqa
    scale: bool = False,
) -> xr.DataArray:
    r"""Energy score, or energy dissimilarity metric, based on :cite:t:`sdba-szekely_testing_2004` and :cite:t:`sdba-cannon_multivariate_2018`.

    Parameters
    ----------
    tgt: xr.DataArray
        Target observations.
    sim: xr.DataArray
        Candidate observations. Must have the same dimensions as `tgt`.
    dims: sequence of 2 strings
        The name of the dimensions along which the variables and observation points are listed.
        `tgt` and `sim` can have different length along the second one, but must be equal along the first one.
        The result will keep all other dimensions.
    N : int
        If larger than 0, the number of observations to use in the score computation. The points are taken
        evenly distributed along `obs_dim`.
    scale : bool
        Whether to scale the data before computing the score. If True, both arrays as scaled according
        to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
        statistics excluding NaN values).

    Returns
    -------
    xr.DataArray
        e-score with dimensions not in `dims`.

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,n_j`
    proposed by :cite:t:`sdba-szekely_testing_2004` is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum_{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of
    :cite:t:`sdba-cannon_multivariate_2018` to the metric. With that factor, the test becomes identical to the one
    defined by :cite:t:`sdba-baringhaus_new_2004`.
    This version is tested against values taken from Alex Cannon's MBC R package :cite:p:`sdba-cannon_mbc_2020`.

    References
    ----------
    :cite:cts:`sdba-baringhaus_new_2004,sdba-cannon_multivariate_2018,sdba-cannon_mbc_2020,sdba-szekely_testing_2004`

    """
    pts_dim, obs_dim = dims

    if N > 0:
        # If N non-zero we only take around N points, evenly distributed
        sim_step = int(np.ceil(sim[obs_dim].size / N))
        sim = sim.isel({obs_dim: slice(None, None, sim_step)})
        tgt_step = int(np.ceil(tgt[obs_dim].size / N))
        tgt = tgt.isel({obs_dim: slice(None, None, tgt_step)})

    if scale:
        tgt, avg, std = standardize(tgt)
        sim, _, _ = standardize(sim, avg, std)

    # The dimension renaming is to allow different coordinates.
    # Otherwise, apply_ufunc tries to align both obs_dim together.
    new_dim = get_temp_dimname(tgt.dims, obs_dim)
    sim = sim.rename({obs_dim: new_dim})
    out = xr.apply_ufunc(
        _escore,
        tgt,
        sim,
        input_core_dims=[[pts_dim, obs_dim], [pts_dim, new_dim]],
        output_dtypes=[sim.dtype],
        dask="parallelized",
    )

    out.name = "escores"
    out.attrs.update(
        long_name="Energy dissimilarity metric",
        description=f"Escores computed from {N or 'all'} points.",
        references="Székely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)",
    )
    return out


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
    N_in_year = elements_in_year.get(freq, 1) / mult
    if N_in_year % 1 != 0:
        raise ValueError(
            f"Sampling frequency of the data must be Q, M, D or H and evenly divide a year (got {mult}{freq})."
        )

    return int(N_in_year)


def construct_moving_yearly_window(
    da: xr.Dataset, window: int = 21, step: int = 1, dim: str = "movingwin"
):
    """Construct a moving window DataArray.

    Stack windows of `da` in a new 'movingwin' dimension.
    Windows are always made of full years, so calendar with non-uniform year lengths are not supported.

    Windows are constructed starting at the beginning of `da`, if number of given years is not
    a multiple of `step`, then the last year(s) will be missing as a supplementary window would be incomplete.

    Parameters
    ----------
    da : xr.Dataset
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


def unpack_moving_yearly_window(
    da: xr.DataArray, dim: str = "movingwin", append_ends: bool = True
):
    """Unpack a constructed moving window dataset to a normal timeseries, only keeping the central data.

    Unpack DataArrays created with :py:func:`construct_moving_yearly_window` and recreate a timeseries data.
    If `append_ends` is False, only keeps the central non-overlapping years. The final timeseries will be
    (window - step) years shorter than the initial one. If `append_ends` is True, the time points from first and last
    windows will be included in the final timeseries.

    The time points that are not in a window will never be included in the final timeseries.
    The window length and window step are inferred from the coordinates.

    Parameters
    ----------
    da : xr.DataArray
        As constructed by :py:func:`construct_moving_yearly_window`.
    dim : str
        The window dimension name as given to the construction function.
    append_ends : bool
        Whether to append the ends of the timeseries
        If False, the final timeseries will be (window - step) years shorter than the initial one,
        but all windows will contribute equally.
        If True, the year before the middle years of the first window and the years after the middle years of the last
        window are appended to the middle years. The final timeseries will be the same length as the initial timeseries
        if the windows span the whole timeseries.
        The time steps that are not in a window will be left out of the final timeseries.

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
    da_mid = da.isel(time=slice(left * N_in_year, (left + step) * N_in_year))

    out = []
    for win_start in da_mid[dim]:
        slc = da_mid.sel({dim: win_start}).drop_vars(dim)
        dt = win_start.values - da_mid[dim][0].values
        slc["time"] = slc.time + dt
        out.append(slc)

    if append_ends:
        # add front end at the front
        out.insert(
            0, da.isel({dim: 0, "time": slice(None, left * N_in_year)}).drop_vars(dim)
        )
        # add back end at the back
        back_end = da.isel(
            {dim: -1, "time": slice((left + step) * N_in_year, None)}
        ).drop_vars(dim)
        dt = da.isel({dim: -1})[dim].values - da.isel({dim: 0})[dim].values
        back_end["time"] = back_end.time + dt
        out.append(back_end)

    return xr.concat(out, "time")


@update_xclim_history
def to_additive_space(
    data: xr.DataArray,
    lower_bound: str,
    upper_bound: str = None,
    trans: str = "log",
):
    r"""Transform a non-additive variable into an additive space by the means of a log or logit transformation.

    Based on :cite:t:`sdba-alavoine_distinct_2022`.

    Parameters
    ----------
    data : xr.DataArray
        A variable that can't usually be bias-adjusted by additive methods.
    lower_bound : str
        The smallest physical value of the variable, excluded, as a Quantity string.
        The data should only have values strictly larger than this bound.
    upper_bound : str, optional
        The largest physical value of the variable, excluded, as a Quantity string.
        Only relevant for the logit transformation.
        The data should only have values strictly smaller than this bound.
    trans : {'log', 'logit'}
        The transformation to use. See notes.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, this applies a transformation to a space where
    additive methods are sensible. Given :math:`X` the variable, :math:`b_-` the lower physical bound of that variable
    and :math:`b_+` the upper physical bound, two transformations are currently implemented to get :math:`Y`,
    the additive-ready variable. :math:`\ln` is the natural logarithm.

    - `log`

        .. math::

            Y = \ln\left( X - b_- \right)

        Usually used for variables with only a lower bound, like precipitation (`pr`,  `prsn`, etc)
        and daily temperature range (`dtr`). Both have a lower bound of 0.

    - `logit`

        .. math::

            X' = (X - b_-) / (b_+ - b_-)
            Y = \ln\left(\frac{X'}{1 - X'} \right)

        Usually used for variables with both a lower and a upper bound, like relative and specific humidity,
        cloud cover fraction, etc.

    This will thus produce `Infinity` and `NaN` values where :math:`X == b_-` or :math:`X == b_+`.
    We recommend using :py:func:`jitter_under_thresh` and :py:func:`jitter_over_thresh` to remove those issues.

    See Also
    --------
    from_additive_space : for the inverse transformation.
    jitter_under_thresh : Remove values exactly equal to the lower bound.
    jitter_over_thresh : Remove values exactly equal to the upper bound.

    References
    ----------
    :cite:cts:`sdba-alavoine_distinct_2022`

    """
    with units.context(infer_context(data.attrs.get("standard_name"))):
        lower_bound = convert_units_to(lower_bound, data)
        if upper_bound is not None:
            upper_bound = convert_units_to(upper_bound, data)

    with xr.set_options(keep_attrs=True):
        if trans == "log":
            out = np.log(data - lower_bound)
        elif trans == "logit":
            data_prime = (data - lower_bound) / (upper_bound - lower_bound)
            out = np.log(data_prime / (1 - data_prime))
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Attributes to remember all this.
    out.attrs["sdba_transform"] = trans
    out.attrs["sdba_transform_lower"] = lower_bound
    if upper_bound is not None:
        out.attrs["sdba_transform_upper"] = upper_bound
    if "units" in out.attrs:
        out.attrs["sdba_transform_units"] = out.attrs.pop("units")
        out.attrs["units"] = ""
    return out


@update_xclim_history
def from_additive_space(
    data: xr.DataArray,
    lower_bound: str = None,
    upper_bound: str = None,
    trans: str = None,
    units: str = None,
):
    r"""Transform back to the physical space a variable that was transformed with `to_additive_space`.

    Based on :cite:t:`sdba-alavoine_distinct_2022`.
    If parameters are not present on the attributes of the data, they must be all given are arguments.

    Parameters
    ----------
    data : xr.DataArray
        A variable that was transformed by :py:func:`to_additive_space`.
    lower_bound : str, optional
        The smallest physical value of the variable, as a Quantity string.
        The final data will have no value smaller or equal to this bound.
        If None (default), the `sdba_transform_lower` attribute is looked up on `data`.
    upper_bound : str, optional
        The largest physical value of the variable, as a Quantity string.
        Only relevant for the logit transformation.
        The final data will have no value larger or equal to this bound.
        If None (default), the `sdba_transform_upper` attribute is looked up on `data`.
    trans : {'log', 'logit'}, optional
        The transformation to use. See notes.
        If None (the default), the `sdba_transform` attribute is looked up on `data`.
    units : str, optional
        The units of the data before transformation to the additive space.
        If None (the default), the `sdba_transform_units` attribute is looked up on `data`.

    Returns
    -------
    xr.DataArray
        The physical variable. Attributes are conserved, even if some might be incorrect.
        Except units which are taken from `sdba_transform_units` if available.
        All `sdba_transform*` attributes are deleted.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, :py:func:`to_additive_space` applied a transformation
    to a space where additive methods are sensible. Given :math:`Y` the transformed variable, :math:`b_-` the
    lower physical bound of that variable and :math:`b_+` the upper physical bound, two back-transformations are
    currently implemented to get :math:`X`, the physical variable.

    - `log`

        .. math::

            X = e^{Y} + b_-

    - `logit`

        .. math::

            X' = \frac{1}{1 + e^{-Y}}
            X = X * (b_+ - b_-) + b_-

    See Also
    --------
    to_additive_space : for the original transformation.

    References
    ----------
    :cite:cts:`sdba-alavoine_distinct_2022`

    """
    if trans is None and lower_bound is None and units is None:
        try:
            trans = data.attrs["sdba_transform"]
            units = data.attrs["sdba_transform_units"]
            lower_bound = data.attrs["sdba_transform_lower"]
            if trans == "logit":
                upper_bound = data.attrs["sdba_transform_upper"]
        except KeyError as err:
            raise ValueError(
                f"Attribute {err!s} must be present on the input data "
                "or all parameters must be given as arguments."
            ) from err
    elif (
        trans is not None
        and lower_bound is not None
        and units is not None
        and (upper_bound is not None or trans == "log")
    ):
        lower_bound = convert_units_to(lower_bound, units)
        if trans == "logit":
            upper_bound = convert_units_to(upper_bound, units)
    else:
        raise ValueError(
            "Parameters missing. Either all parameters are given as attributes of data, "
            "or all of them are given as input arguments."
        )

    with xr.set_options(keep_attrs=True):
        if trans == "log":
            out = np.exp(data) + lower_bound
        elif trans == "logit":
            out_prime = 1 / (1 + np.exp(-data))
            out = out_prime * (upper_bound - lower_bound) + lower_bound
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Remove unneeded attributes, put correct units back.
    out.attrs.pop("sdba_transform", None)
    out.attrs.pop("sdba_transform_lower", None)
    out.attrs.pop("sdba_transform_upper", None)
    out.attrs.pop("sdba_transform_units", None)
    out.attrs["units"] = units
    return out


def stack_variables(ds: xr.Dataset, rechunk: bool = True, dim: str = "multivar"):
    """Stack different variables of a dataset into a single DataArray with a new "variables" dimension.

    Variable attributes are all added as lists of attributes to the new coordinate, prefixed with "_".
    Variables are concatenated in the new dimension in alphabetical order, to ensure
    coherent behaviour with different datasets.

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
        The transformed variable. Attributes are conserved, even if some might be incorrect, except for units,
        which are replaced with `""`. Old units are stored in `sdba_transformation_units`.
        A `sdba_transform` attribute is added, set to the transformation method. `sdba_transform_lower` and
        `sdba_transform_upper` are also set if the requested bounds are different from the defaults.

        Array with variables stacked along `dim` dimension. Units are set to "".

    """
    # Store original arrays' attributes
    attrs = {}
    # sort to have coherent order with different datasets
    datavars = sorted(ds.data_vars.items(), key=lambda e: e[0])
    nvar = len(datavars)
    for i, (nm, var) in enumerate(datavars):
        for name, attr in var.attrs.items():
            attrs.setdefault("_" + name, [None] * nvar)[i] = attr

    # Special key used for later `unstacking`
    attrs["is_variables"] = True
    var_crd = xr.DataArray([nm for nm, vr in datavars], dims=(dim,), name=dim)

    da = xr.concat([vr for nm, vr in datavars], var_crd, combine_attrs="drop")

    if uses_dask(da) and rechunk:
        da = da.chunk({dim: -1})

    da.attrs.update(ds.attrs)
    da.attrs["units"] = ""
    da[dim].attrs.update(attrs)
    return da.rename("multivariate")


def unstack_variables(da: xr.DataArray, dim: str = None):
    """Unstack a DataArray created by `stack_variables` to a dataset.

    Parameters
    ----------
    da : xr.DataArray
        Array holding different variables along `dim` dimension.
    dim : str
        Name of dimension along which the variables are stacked.
        If not specified (default), `dim` is inferred from attributes of the coordinate.

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
    for name, attr_list in da[dim].attrs.items():
        if not name.startswith("_"):
            continue
        for attr, var in zip(attr_list, da[dim]):
            if attr is not None:
                ds[var.item()].attrs[name[1:]] = attr

    return ds
