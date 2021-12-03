"""
Pre and post processing
-----------------------
"""
import warnings
from typing import Optional, Sequence, Tuple, Union

import dask.array as dsk
import numpy as np
import xarray as xr
from xarray.core.utils import get_temp_dimname

from xclim.core.calendar import get_calendar, max_doy, parse_offset
from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to
from xclim.core.utils import uses_dask

from ._processing import _adapt_freq, _normalize, _reordering
from .base import Grouper
from .nbutils import _escore
from .utils import ADDITIVE


@update_xclim_history
def adapt_freq(
    ref: xr.DataArray,
    sim: xr.DataArray,
    *,
    group: Union[Grouper, str],
    thresh: str = "0 mm d-1",
) -> xr.Dataset:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on [Themessl2012]_.

    Parameters
    ----------
    ds : xr.Dataset
      With variables :  "ref", Target/reference data, usually observed data.
      and  "sim", Simulated data.
    dim : str
      Dimension name.
    group : Union[str, Grouper]
      Grouping information, see base.Grouper
    thresh : str
      Threshold below which values are considered zero, a quantity with units.

    Returns
    -------
    sim_adj : xr.DataArray
      Simulated data with the same frequency of values under threshold than ref.
      Adjustment is made group-wise.
    pth : xr.DataArray
      For each group, the smallest value of sim that was not frequency-adjusted. All values smaller were
      either left as zero values or given a random value between thresh and pth.
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
    .. [Themessl2012] Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4.
    """
    sim = convert_units_to(sim, ref)
    thresh = convert_units_to(thresh, ref)

    out = _adapt_freq(xr.Dataset(dict(sim=sim, ref=ref)), group=group, thresh=thresh)

    # Set some metadata
    out.sim_ad.attrs.update(sim.attrs)
    out.sim_ad.attrs.update(
        references="Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4."
    )
    out.pth.attrs.update(
        long_name="Smallest value of the timeseries not corrected by frequency adaptation.",
        units=sim.units,
    )
    out.dP0.attrs.update(
        long_name=f"Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation",
    )

    return out.sim_ad, out.pth, out.dP0


@update_xclim_history
def jitter_under_thresh(x: xr.DataArray, thresh: str):
    """Replace values smaller than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
      Values.
    thresh : str
      Threshold under which to add uniform random noise to values, a quantity with units.

    Returns
    -------
    array

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    thresh = convert_units_to(thresh, x)
    epsilon = np.finfo(x.dtype).eps
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=epsilon, high=thresh, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=epsilon, high=thresh, size=x.shape)
    out = x.where(~((x < thresh) & (x.notnull())), jitter.astype(x.dtype))
    out.attrs.update(x.attrs)  # copy attrs and same units
    return out


@update_xclim_history
def jitter_over_thresh(x: xr.DataArray, thresh: str, upper_bnd: str) -> xr.Dataset:
    """Replace values greater than threshold by a uniform random noise.

    Do not confuse with R's jitter, which adds uniform noise instead of replacing values.

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
    xr.Dataset

    Notes
    -----
    If thresh is low, this will change the mean value of x.
    """
    thresh = convert_units_to(thresh, x)
    upper_bnd = convert_units_to(upper_bnd, x)
    if uses_dask(x):
        jitter = dsk.random.uniform(
            low=thresh, high=upper_bnd, size=x.shape, chunks=x.chunks
        )
    else:
        jitter = np.random.uniform(low=thresh, high=upper_bnd, size=x.shape)
    out = x.where(~((x > thresh) & (x.notnull())), jitter.astype(x.dtype))
    out.attrs.update(x.attrs)  # copy attrs and same units
    return out


@update_xclim_history
def normalize(
    data: xr.DataArray,
    norm: Optional[xr.DataArray] = None,
    *,
    group: Union[Grouper, str],
    kind: str = ADDITIVE,
) -> xr.Dataset:
    """Normalize an array by removing its mean.

    Normalization if performed group-wise and according to `kind`.

    Parameters
    ----------
    data: xr.DataArray
      The variable to normalize.
    norm : xr.DataArray, optional
      If present, it is used instead of computing the norm again.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details..
    kind : {'+', '*'}
      If `kind` is "+", the mean is subtracted from the mean and if it is '*', it is divided from the data.

    Returns
    -------
    xr.DataArray
      Groupwise anomaly
    """
    ds = xr.Dataset(dict(data=data))

    if norm is not None:
        norm = convert_units_to(norm, data)
        ds = ds.assign(norm=norm)

    out = _normalize(ds, group=group, kind=kind)
    out.attrs.update(data.attrs)
    return out.data.rename(data.name)


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
    mean: Optional[xr.DataArray] = None,
    std: Optional[xr.DataArray] = None,
    dim: str = "time",
) -> Tuple[Union[xr.DataArray, xr.Dataset], xr.DataArray, xr.DataArray]:
    """Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns the standardized data, the mean and the standard deviation.
    """
    if mean is None:
        mean = da.mean(dim, keep_attrs=True)
    if std is None:
        std = da.std(dim, keep_attrs=True)
    with xr.set_options(keep_attrs=True):
        return (da - mean) / std, mean, std


@update_xclim_history
def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    with xr.set_options(keep_attrs=True):
        return (std * da) + mean


@update_xclim_history
def reordering(ref: xr.DataArray, sim: xr.DataArray, group: str = "time") -> xr.Dataset:
    """Reorders data in `sim` following the order of ref.

    The rank structure of `ref` is used to reorder the elements of `sim` along dimension "time",
    optionally doing the operation group-wise.

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

    Reference
    ---------
    Cannon, A. J. (2018). Multivariate quantile mapping bias correction: An N-dimensional probability density function
    transform for climate model simulations of multiple variables. Climate Dynamics, 50(1), 31–49.
    https://doi.org/10.1007/s00382-017-3580-6
    """
    ds = xr.Dataset({"sim": sim, "ref": ref})
    out = _reordering(ds, group=group).reordered
    out.attrs.update(sim.attrs)
    return out


@update_xclim_history
def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,
    scale: bool = False,
) -> xr.DataArray:
    r"""Energy score, or energy dissimilarity metric, based on [SkezelyRizzo]_ and [Cannon18]_.

    Parameters
    ----------
    tgt: DataArray
      Target observations.
    sim: DataArray
      Candidate observations. Must have the same dimensions as `tgt`.
    dims: sequence of 2 strings
      The name of the dimensions along which the variables and observation points are listed.
      `tgt` and `sim` can have different length along the second one, but must be equal along the first one.
      The result will keep all other dimensions.
    N : int
      If larger than 0, the number of observations to use in the score computation. The points are taken
      evenly distributed along `obs_dim`.
    scale: bool
      Whether to scale the data before computing the score. If True, both arrays as scaled according
      to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
      statistics excluding NaN values.

    Returns
    -------
    xr.DataArray
        e-score with dimensions not in `dims`.

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,,n_j`
    proposed by Szekely and Rizzo (2005) is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of [Cannon18]_ to
    the metric. With that factor, the test becomes identical to the one defined by [BaringhausFranz]_.

    References
    ----------
    .. Skezely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
    .. [BaringhausFranz] Baringhaus, L. and Franz, C. (2004) On a new multivariate two-sample test, Journal of Multivariate Analysis, 88(1), 190–206. https://doi.org/10.1016/s0047-259x(03)00079-4
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
        references="Skezely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)",
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


@update_xclim_history
def to_additive_space(
    data, trans: str = "log", lower_bound: float = 0, upper_bound: float = 1
):
    r"""Transform a non-additive variable into an addtitive space by the means of a log or logit transformation.

    Based on [AlavoineGrenier]_.

    Parameters
    ----------
    data : xr.DataArray
      A variable that can't usually be bias-adusted by additive methods.
    trans : {'log', 'logit'}
      The transformation to use. See notes.
    lower_bound : float
      The smallest physical value of the variable, excluded.
      The data should only have values strictly larger than this bound.
    upper_bound : float
      The largest physical value of the variable, excluded. Only relevant for the logit transformation.
      The data should only have values strictly smaller than this bound.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, this apply a transformation to a space where
    addtitive methods are sensible. Given :math:`X` the variable, :math:`b_-` the lower physical bound of that variable
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

    See also
    --------
    from_additive_space : for the inverse transformation.
    jitter_under_thresh : Remove values exactly equal to the lower bound.
    jitter_over_thresh : Remove values exactly equal to the upper bound.

    References
    ----------
    .. [AlavoineGrenier] Alavoine M., and Grenier P. (under review) The distinct problems of physical inconsistency and of multivariate bias potentially involved in the statistical adjustment of climate simulations.
                         International Journal of Climatology, Manuscript ID: JOC-21-0789, submitted on September 19th 2021.
    """

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
    if lower_bound != 0:
        out.attrs["sdba_transform_lower"] = lower_bound
    if upper_bound != 1:
        out.attrs["sdba_transform_upper"] = upper_bound
    if "units" in out.attrs:
        out.attrs["sdba_transform_units"] = out.attrs.pop("units")
        out.attrs["units"] = ""
    return out


@update_xclim_history
def from_additive_space(data, trans=None, lower_bound=None, upper_bound=None):
    r"""Transform back a to the physical space a variable that was transformed with `to_addtitive_space`.

    Based on [AlavoineGrenier]_.

    Parameters
    ----------
    data : xr.DataArray
      A variable that can't usually be bias-adusted by additive methods.
    trans : {'log', 'logit'}, optional
      The transformation to use. See notes.
      If None (the default), the `sdba_transform` attribute is looked up on `data`.
    lower_bound : float, optional
      The smallest physical value of the variable.
      The final data will have no value smaller or equal to this bound.
      If None (default), the `sdba_transform_lower` attribute is looked up on `data`.
    upper_bound : float, optional
      The largest physical value of the variable, only relevant for the logit transformation.
      The final data will have no value larger or equal to this bound.
      If None (default), the `sdba_transform_upper` attribute is looked up on `data`.

    Returns
    -------
    xr.DataArray
      The physical variable. Attributes are conserved, even if some might be incorrect.
      Except units which are taken from `sdba_transform_units` if available.
      All `sdba_transform*` attributes are deleted.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment,
    :py:func:`to_additive_space` applied a transformation to a space where addtitive
    methods are sensible. Given :math:`Y` the transformed variable, :math:`b_-` the
    lower physical bound of that variable and :math:`b_+` the upper physical bound,
    two back-transformations are currently implemented to get :math:`X`. the physical variable.

    - `log`

        .. math::

            X = e^{Y) + b_-

    - `logit`

        .. math::

            X' = \frac{1}{1 + e^{-Y}}
            X = X * (b_+ - b_-) + b_-

    See also
    --------
    to_additive_space : for the original transformation.

    References
    ----------
    .. [AlavoineGrenier] Alavoine M., and Grenier P. (under review) The distinct problems of physical inconsistency and of multivariate bias potentially involved in the statistical adjustment of climate simulations.
                         International Journal of Climatology, Manuscript ID: JOC-21-0789, submitted on September 19th 2021.
    """
    if trans is None:
        trans = data.attrs["sdba_transform"]
    if lower_bound is None:
        lower_bound = data.attrs.get("sdba_transform_lower", 0)
    if upper_bound is None:
        upper_bound = data.attrs.get("sdba_transform_upper", 1)

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
    if "sdba_transform_units" in out.attrs:
        out.attrs["units"] = out.attrs["sdba_transform_units"]
        out.attrs.pop("sdba_transform_units")
    return out


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
      The transformed variable. Attributes are conserved, even if some might be incorrect.
      Except units, which are replace with "". Old units are stored in `sdba_transformation_units`.
      A `sdba_transform` attribute is added, set to the transformation method.
      `sdba_transform_lower` and `sdba_transform_upper` are also set if the requested bounds are different from the defaults.

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
