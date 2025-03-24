"""
Missing Values Identification
=============================

Indicators may use different criteria to determine whether a computed indicator value should be
considered missing. In some cases, the presence of any missing value in the input time series should result in a
missing indicator value for that period. In other cases, a minimum number of valid values or a percentage of missing
values should be enforced. The World Meteorological Organisation (WMO) suggests criteria based on the number of
consecutive and overall missing values per month.

`xclim` has a registry of missing value detection algorithms that can be extended by users to customize the behavior
of indicators. Once registered, algorithms can be used by setting the global option as
``xc.set_options(check_missing="method")`` or within indicators by setting the `missing` attribute of an
`Indicator` subclass. By default, `xclim` registers the following algorithms:

 * `any`: A result is missing if any input value is missing.
 * `at_least_n`: A result is missing if less than a given number of valid values are present.
 * `pct`: A result is missing if more than a given fraction of its values are missing.
 * `wmo`: A result is missing if 11 days are missing, or 5 consecutive values are missing in a month.

To define another missing value algorithm, subclass :py:class:`MissingBase` and decorate it with
:py:func:`xclim.core.options.register_missing_method`. See subclassing guidelines in ``MissingBase``'s doc.
"""

from __future__ import annotations

import textwrap

import numpy as np
import xarray as xr

from xclim.core.calendar import (
    compare_offsets,
    is_offset_divisor,
    parse_offset,
    select_time,
)
from xclim.core.options import (
    CHECK_MISSING,
    MISSING_METHODS,
    MISSING_OPTIONS,
    OPTIONS,
    register_missing_method,
)

__all__ = [
    "at_least_n_valid",
    "expected_count",
    "missing_any",
    "missing_from_context",
    "missing_pct",
    "missing_wmo",
    "register_missing_method",
]


# Mapping from sub-monthly CFtime freq strings to numpy timedelta64 units
# Only "minute" is different between the two
_freq_to_timedelta = {"min": "m"}


def expected_count(
    time: xr.DataArray,
    freq: str | None = None,
    src_timestep: str | None = None,
    **indexer,
) -> xr.DataArray:
    """
    Get expected number of step of length ``src_timestep`` per each resampling period
    ``freq`` that ``time`` covers.

    The determination of the resampling periods intersecting with the input array are
    done following xarray's and pandas' heuristics. The input coordinate needs not be
    continuous if `src_timestep` is given.

    Parameters
    ----------
    time : xr.DataArray, optional
        Input time coordinate from which the final resample time coordinate is guessed.
    freq : str, optional.
        Resampling frequency. If not given or None, the count for the full time range is returned.
    src_timestep : str, Optional
        The expected input frequency. If not given, it will be inferred from the input array.
    **indexer : Indexer
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
        values, month=1 to select January, or month=[6,7,8] to select summer months.
        If not indexer is given, all values are considered.
        See :py:func:`xc.core.calendar.select_time`.

    Returns
    -------
    xr.DataArray
        Integer array at the resampling frequency with the number of expected elements in each period.
    """
    if src_timestep is None:
        src_timestep = xr.infer_freq(time)
        if src_timestep is None:
            raise ValueError("A src_timestep must be passed when it can't be inferred from the data.")

    if freq is not None and not is_offset_divisor(src_timestep, freq):
        raise NotImplementedError(
            "Missing checks not implemented for timeseries resampled to a frequency that is not "
            f"aligned with the source timestep. {src_timestep} is not a divisor of {freq}."
        )

    # Ensure a DataArray constructed like we expect
    time = xr.DataArray(time.values, dims=("time",), coords={"time": time.values}, name="time")

    if freq:
        # We only want the resulting time index, the actual resampling method is not important.
        resamp = time.resample(time=freq).count()
        resamp_time = resamp.indexes["time"]
        _, _, is_start, _ = parse_offset(freq)
        if is_start:
            start_time = resamp_time
            end_time = start_time.shift(1, freq=freq)
        else:
            end_time = resamp_time
            start_time = end_time.shift(-1, freq=freq)
    else:  # freq=None, means the whole timeseries as a single period
        i = time.indexes["time"]
        start_time = i[:1]
        end_time = i[-1:]

    # don't forget : W is converted to 7D
    mult, base, _, _ = parse_offset(src_timestep)
    if indexer or base in "YAQM":
        # Create a full synthetic time series and compare the number of days with the original series.
        t = xr.date_range(
            start_time[0],
            end_time[-1],
            freq=src_timestep,
            calendar=time.dt.calendar,
            use_cftime=(start_time.dtype == "O"),
        )

        sda = xr.DataArray(data=np.ones(len(t)), coords={"time": t}, dims=("time",))
        if "doy_bounds" in indexer:
            # This is the only case supported by select_time where DataArrays are supported
            # TODO: What happens when this new dim makes sda too large ? How do we involve dask here ?
            da_bnds = [bnd for bnd in indexer["doy_bounds"] if isinstance(bnd, xr.DataArray)]
            sda = xr.broadcast(sda, *da_bnds, exclude=("time",))[0]

        st = select_time(sda, **indexer)
        if freq:
            count = st.notnull().resample(time=freq).sum(dim="time")
        else:
            count = st.notnull().sum(dim="time")
    else:  # simpler way for sub monthly without indexer.
        delta = end_time - start_time
        unit = _freq_to_timedelta.get(base, base)
        n = delta.values.astype(f"timedelta64[{unit}]").astype(float) / mult

        if freq:
            count = xr.DataArray(n, coords={"time": resamp.time}, dims="time")
        else:
            count = xr.DataArray(n[0] + 1)
    return count


class MissingBase:
    r"""
    Base class used to determined where Indicator outputs should be masked.

    Subclasses should implement the ``is_missing``, ``validate`` and ``__init__``
    methods. The ``__init__`` is to be implemented in order to change the docstring
    and signature but is not expected to do anything other than the validation
    of the options, everything else should happen in the call (i.e. ``is_missing``).
    Subclasses can also override the ``_validate_src_timestep`` method to add restrictions
    on allowed values. That method should return False on invalid ``src_timestep``.

    Decorate subclasses with `xclim.core.options.register_missing_method` to add them
    to the registry before using them in an Indicator.
    """

    def __init__(self, **options):
        if not self.validate(**options):
            raise ValueError(f"Options {options} are not valid for {self.__class__.__name__}.")
        self.options = options

    @staticmethod
    def validate(**options):
        r"""
        Validate optional arguments.

        Parameters
        ----------
        **options : dict
            Optional arguments.

        Returns
        -------
        bool
            False if the options are not valid.
        """
        return True

    @staticmethod
    def is_valid(da: xr.DataArray, **indexer) -> xr.DataArray:
        r"""
        Return a boolean array indicating which values are valid.

        Parameters
        ----------
        da : xr.DataArray
            Input data.
        **indexer : {dim: indexer}, optional
            The time attribute and values over which to subset the array. For example, use season='DJF' to select winter
            values, month=1 to select January, or month=[6,7,8] to select summer months.
            See :py:func:`xclim.core.calendar.select_time`.

        Returns
        -------
        xr.DataArray
            Boolean array indicating which values are valid.
        """
        selected = select_time(da, **indexer)
        return selected.notnull()

    def _validate_src_timestep(self, src_timestep):
        return True

    def is_missing(
        self,
        valid: xr.DataArray,
        count: xr.DataArray,
        freq: str | None,
    ) -> xr.DataArray:
        """
        Return whether the values within each period should be considered missing or not.

        Must be implemented by subclasses.

        Parameters
        ----------
        valid : DataArray
            Boolean array of valid values (that has already been indexed).
        count : DataArray
            Indexer-aware integer array of number of expected elements at the resampling frequency.
        freq : str or None
            The resampling frequency, or None if the temporal dimension is to be collapsed.

        Returns
        -------
        DataArray
            Boolean array at the resampled frequency,
            True on the periods that should be considered missing.
        """
        raise NotImplementedError()

    def __call__(
        self,
        da: xr.DataArray,
        freq: str | None = None,
        src_timestep: str | None = None,
        **indexer,
    ) -> xr.DataArray:
        """
        Compute the missing period mask according to the object's algorithm.

        Parameters
        ----------
        da : xr.DataArray
            Input data, must have a "time" coordinate.
        freq : str, optional
            Resampling frequency. If None, a collapse of the temporal dimension is assumed.
        src_timestep : str, optional
            The expected source input frequency. If not given, it will be inferred from the input array.
        **indexer : Indexer
            Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
            values, month=1 to select January, or month=[6,7,8] to select summer months.
            If not indexer is given, all values are considered.
            See :py:func:`xclim.core.calendar.select_time`.

        Returns
        -------
        DataArray
            Boolean array at the resampled frequency,
            True on the periods that should be considered missing or invalid.
        """
        if src_timestep is None:
            src_timestep = xr.infer_freq(da.time)
            if src_timestep is None:
                raise ValueError(
                    "The source timestep can't be inferred from the data, but it is required"
                    " to compute the missing values mask."
                )

        if not self._validate_src_timestep(src_timestep):
            raise ValueError(
                f"Input source timestep {src_timestep} is invalid for missing method {self.__class__.__name__}."
            )

        count = expected_count(da.time, freq=freq, src_timestep=src_timestep, **indexer)
        valid = self.is_valid(da, **indexer)
        return self.is_missing(valid, count, freq)

    def __repr__(self):
        opt_str = ", ".join([f"{k}={v}" for k, v in self.options.items()])
        return f"<{self.__class__.__name__}({opt_str})>"


# -----------------------------------------------
# --- Missing value identification algorithms ---
# -----------------------------------------------


@register_missing_method("any")
class MissingAny(MissingBase):
    """Mask periods as missing if any of its elements is missing or invalid."""

    def __init__(self):
        """Create a MissingAny object."""
        super().__init__()

    def is_missing(self, valid: xr.DataArray, count: xr.DataArray, freq: str | None) -> xr.DataArray:
        if freq is not None:
            valid = valid.resample(time=freq)
        # The number of valid values should fit the expected count.
        return valid.sum(dim="time") != count


# TODO: Make coarser method controllable.
class MissingTwoSteps(MissingBase):
    r"""
    Base class used to determined where Indicator outputs should be masked in a two-step process.

    In addition to what :py:class:`MissingBase` does, subclasses first perform the mask
    determination at some frequency and then resample at the (coarser) target frequency.
    This allows the application of specific methods at a finer resolution than the target one.
    The sub-groups are merged using the "Any" method : a group is invalid if any of its
    sub-groups are invalid.

    The first resampling frequency should be implemented as an additional "subfreq" option.
    A value of None means that only one resampling is done at the request target frequency.
    """

    def __call__(
        self,
        da: xr.DataArray,
        freq: str | None = None,
        src_timestep: str | None = None,
        **indexer,
    ) -> xr.DataArray:
        """
        Compute the missing period mask according to the object's algorithm.

        Parameters
        ----------
        da : xr.DataArray
            Input data, must have a "time" coordinate.
        freq : str, optional
            Target resampling frequency. If None, a collapse of the temporal dimension is assumed.
        src_timestep : str, optional
            The expected source input frequency. If not given, it will be inferred from the input array.
        **indexer : Indexer
            Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
            values, month=1 to select January, or month=[6,7,8] to select summer months.
            If no indexer is given, all values are considered.
            See :py:func:`xclim.core.calendar.select_time`.

        Returns
        -------
        DataArray
            Boolean array at the resampled frequency,
            True on the periods that should be considered missing or invalid.
        """
        subfreq = self.options["subfreq"] or freq
        if subfreq is not None and freq is not None and compare_offsets(freq, "<", subfreq):
            raise ValueError(
                "The target resampling frequency cannot be finer than the first-step "
                f"frequency. Got : {subfreq} > {freq}."
            )
        miss = super().__call__(da, freq=subfreq, src_timestep=src_timestep, **indexer)
        if subfreq != freq:
            miss = MissingAny()(miss.where(~miss), freq, src_timestep=subfreq, **indexer)
        return miss


@register_missing_method("wmo")
class MissingWMO(MissingTwoSteps):
    """
    Mask periods as missing using the WMO criteria for missing days.

    The World Meteorological Organisation recommends that where monthly means are computed from daily values,
    it should be considered missing if either of these two criteria are met:

      – observations are missing for 11 or more days during the month;
      – observations are missing for a period of 5 or more consecutive days during the month.

    Stricter criteria are sometimes used in practice, with a tolerance of 5 missing values or 3 consecutive missing
    values.

    Notes
    -----
    If used at frequencies larger than a month, for example on an annual or seasonal basis, the function will return
    True if any month within a period is masked.
    """

    def __init__(self, nm: int = 11, nc: int = 5):
        """
        Create a MissingWMO object.

        Parameters
        ----------
        nm : int
            Minimal number of missing elements for a month to be masked.
        nc : int
            Minimal number of consecutive missing elements for a month to be masked.
        """
        super().__init__(nm=nm, nc=nc, subfreq="MS")

    @staticmethod
    def validate(nm: int, nc: int, subfreq: str | None = None):
        return nm < 31 and nc < 31

    def _validate_src_timestep(self, src_timestep):
        return src_timestep == "D"

    def is_missing(self, valid: xr.DataArray, count: xr.DataArray, freq: str) -> xr.DataArray:
        from xclim.indices import run_length as rl
        from xclim.indices.helpers import resample_map

        validr = valid.resample(time=freq)

        # Total number of missing or invalid days
        missing_days = count - validr.sum(dim="time")
        # Check if more than threshold is missing
        cond1 = missing_days >= self.options["nm"]

        # Check for consecutive invalid values
        # FIXME: This does not take holes in consideration
        longest_run = resample_map(~valid, "time", freq, rl.longest_run, map_blocks=True)
        cond2 = longest_run >= self.options["nc"]

        return cond1 | cond2


@register_missing_method("pct")
class MissingPct(MissingTwoSteps):
    """Mask periods as missing when there are more than a given percentage of missing days."""

    def __init__(self, tolerance: float = 0.1, subfreq: str | None = None):
        """
        Create a MissingPct object.

        Parameters
        ----------
        tolerance: float
            The maximum tolerated proportion of missing values,
            given as a number between 0 and 1.
        subfreq : str, optional
            If given, compute a mask at this frequency using this method and
            then resample at the target frequency using the "any" method on sub-groups.
        """
        super().__init__(tolerance=tolerance, subfreq=subfreq)

    @staticmethod
    def validate(tolerance: float, subfreq: str | None = None):
        return 0 <= tolerance <= 1

    def is_missing(self, valid: xr.DataArray, count: xr.DataArray, freq: str | None) -> xr.DataArray:
        if freq is not None:
            valid = valid.resample(time=freq)

        # Total number of missing or invalid days
        missing_days = (count - valid.sum(dim="time")).fillna(count)
        return (missing_days / count) >= self.options["tolerance"]


@register_missing_method("at_least_n")
class AtLeastNValid(MissingTwoSteps):
    r"""
    Mask periods as missing if they don't have at least a given number of valid values.

    Ignores the expected count of elements.
    """

    def __init__(self, n: int = 20, subfreq: str | None = None):
        """
        Create a AtLeastNValid object.

        Parameters
        ----------
        n: float
            The minimum number of valid values needed.
        subfreq : str, optional
            If given, compute a mask at this frequency using this method and
            then resample at the target frequency using the "any" method on sub-groups.
        """
        super().__init__(n=n, subfreq=subfreq)

    @staticmethod
    def validate(n: int, subfreq: str | None = None):
        return n > 0

    def is_missing(self, valid: xr.DataArray, count: xr.DataArray, freq: str | None) -> xr.DataArray:
        if freq is not None:
            valid = valid.resample(time=freq)
        nvalid = valid.sum(dim="time")
        return nvalid < self.options["n"]


# --------------------------
# --- Shortcut functions ---
# --------------------------
# These stand-alone functions hide the fact the algorithms are implemented in a class and make their use more
# user-friendly. This can also be useful for testing.


def missing_any(  # noqa: D103 # numpydoc ignore=GL08
    da: xr.DataArray, freq: str, src_timestep: str | None = None, **indexer
) -> xr.DataArray:
    """Return whether there are missing days in the array."""
    return MissingAny()(da, freq, src_timestep, **indexer)


def missing_wmo(  # noqa: D103 # numpydoc ignore=GL08
    da: xr.DataArray,
    freq: str,
    src_timestep: str | None = None,
    nm: int = 11,
    nc: int = 5,
    **indexer,
) -> xr.DataArray:
    return MissingWMO(nm=nm, nc=nc)(da, freq, src_timestep, **indexer)


def missing_pct(  # noqa: D103 # numpydoc ignore=GL08
    da: xr.DataArray,
    freq: str,
    src_timestep: str | None = None,
    tolerance: float = 0.1,
    subfreq: str | None = None,
    **indexer,
) -> xr.DataArray:
    return MissingPct(tolerance=tolerance, subfreq=subfreq)(da, freq, src_timestep, **indexer)


def at_least_n_valid(  # noqa: D103 # numpydoc ignore=GL08
    da: xr.DataArray,
    freq: str,
    src_timestep: str | None = None,
    n: int = 20,
    subfreq: str | None = None,
    **indexer,
) -> xr.DataArray:
    return AtLeastNValid(n=n, subfreq=subfreq)(da, freq, src_timestep, **indexer)


def missing_from_context(da: xr.DataArray, freq: str, src_timestep: str | None = None, **indexer) -> xr.DataArray:
    """
    Mask periods as missing according to the algorithm and options set in xclim's global options.

    The options can be manipulated with :py:func:`xclim.core.options.set_options`.

    Parameters
    ----------
    da : xr.DataArray
        Input data, must have a "time" coordinate.
    freq : str, optional
        Resampling frequency. If absent, a collapse of the temporal dimension is assumed.
    src_timestep : str, optional
        The expected source input frequency. If not given, it will be inferred from the input array.
    **indexer : Indexer
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter
        values, month=1 to select January, or month=[6,7,8] to select summer months.
        If not indexer is given, all values are considered.
        See :py:func:`xclim.core.calendar.select_time`.

    Returns
    -------
    DataArray
        Boolean array at the resampled frequency,
        True on the periods that should be considered missing or invalid.
    """
    method = OPTIONS[CHECK_MISSING]
    MissCls = MISSING_METHODS[method]
    opts = OPTIONS[MISSING_OPTIONS].get(method, {})
    return MissCls(**opts)(da, freq, src_timestep, **indexer)


def _get_convenient_doc(cls):
    maindoc = textwrap.dedent(cls.__doc__)
    initdoc = textwrap.dedent(cls.__init__.__doc__)
    calldoc = textwrap.dedent(cls.__call__.__doc__)

    params = []
    ip = 10000
    for i, line in enumerate(initdoc.split("\n")):
        if line.strip() == "Parameters":
            ip = i
        if i >= ip + 2 and line.strip():
            params.append(line)

    doc = [maindoc]
    if "\n" not in maindoc:
        doc.append("")

    ip = 10000
    for i, line in enumerate(calldoc.split("\n")):
        if line.strip() == "Parameters":
            ip = i
        elif "**indexer" in line:
            doc.extend(params)
        if i >= ip:
            doc.append(line)
    return "\n".join(doc)


missing_any.__doc__ = _get_convenient_doc(MissingAny)
missing_wmo.__doc__ = _get_convenient_doc(MissingWMO)
missing_pct.__doc__ = _get_convenient_doc(MissingPct)
at_least_n_valid.__doc__ = _get_convenient_doc(AtLeastNValid)
