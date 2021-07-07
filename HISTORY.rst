=======
History
=======

0.28.0 (2021-07-07)
-------------------

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Automatic load of translations on import and possibility to pass translations for virtual modules.
* New ``xclim.testing.list_datasets`` function listing all available test datasets in repo ``xclim-testdata``.
* ``spatial_analogs`` accepts multi-indexes as the ``dist_dim`` parameter and will work with candidates and target arrays of different lengths.
* ``humidex`` can be computed using relative humidity instead of dewpoint temperature.
* New ``sdba.construct_moving_yearly_window`` and ``sdba.unpack_moving_yearly_window`` for moving window adjustments.
* New ``sdba.adjustment.NpdfTransform`` which is an adaptation of Alex Cannon's version of Pitié's *N-dimensional probability density function transform*. Uses new ``sdba.utils.rand_rot_matrix``. *Experimental, subject to changes.*
* New ``sdba.processing.standardize``, ``.unstandardize`` and  ``.reordering``. All of them, tools needed to replicate Cannon's MBCn algorithm.
* New ``sdba.processing.escore``, backed by  ``sdba.nbutils._escore`` to evaluate the performance of the N pdf transform.
* New function ``xclim.indices.clausius_clapeyron_scaled_precipitation`` can be used to scale precipitation according to changes in mean temperature.
* Percentile based indices gained a ``bootstrap`` argument that applies a bootstrapping algorithm to reduce biases on exceedance frequencies computed over *in base* and *out of base* periods. *Experimental, subject to changes.*
* Added a `.zenodo.json` file for collecting and maintaining author order and tracking ORCIDs.

Bug fixes
~~~~~~~~~
* Various bug fixes in sdba :

    - in ``QDM.adjust``, fix bug occuring with coords of 'object' dtype and ``interp='nearest'``.
    - in ``nbutils.quantiles``, fix dtype bug when using ``float32`` data.
    - raise a proper error when ``ref`` and ``hist`` have a different calendar for map_blocks-backed adjustments.

Breaking changes
~~~~~~~~~~~~~~~~
* ``spatial_analogs`` does not support sequence of ``dist_dim`` anymore. Users are responsible for stacking dimensions prior to calling ``spatial_analogs``.

New indicators
~~~~~~~~~~~~~~
* ``biologically_effective_degree_days`` (with ``method="gladstones"``) indice computes degree-days between two specific dates, with a capped daily max value as well as latitude and temperature range swing as modifying coefficients (based on Gladstones, J. (1992)). This has also been wrapped as an indicator.
* An alternative implementation of ``biologically_effective_degree_days`` (with ``method="icclim"``, based on ICCLIM formula) ignores latitude and temperature range swing modifiers and uses an alternate ``end_date``. Wrapped and available as an ICCLIM indicator.
* ``cool_night_index`` indice returns the mean minimum temperature in September (``lat >= 0`` deg N) or March (``lat < 0`` deg N), based on Tonietto & Carbonneau, 2004 (10.1016/j.agrformet.2003.06.001). Also available as an indicator (see indices `Notes` section on indicator usage recommendations).
* ``latitude_temperature_index`` indice computes LTI values based on mean temperature of warmest month and a parameterizable latitude coefficient (default: ``lat_factor=75``) based on Jackson & Cherry, 1988, and Kenny & Shao, 1992 (10.1080/00221589.1992.11516243). This has also been wrapped as an indicator.
* ``huglin_index`` indice computes Huglin Heliothermal Index (HI) values based on growing degrees and a latitude-influenced coefficient for day-length (based on Huglin. (1978)). The indice supports several methods of estimating the latitude coefficient:

    - ``method="smoothed"``: Marks latitudes between -40 N and 40 N with ``k=1``, and linearly increases to ``k=1.06`` at ``|lat|==50``.
    - ``method="icclim"``: Uses a stepwise function based on the the original method as presented by Huglin (1978). Identical to the ICCLIM implementation.
    - ``method="jones"``: Uses a more robust calculation for calculating day-lengths, based on Hall & Jones (2010). This method is now also available for ``biologically_effective_degree_days``.

* The generic indice ``day_length``, used for calculating approximate daily day-length in hours per day or, given ``start_date`` and ``end_date``, the total aggregated day-hours over period. Uses axial tilt, start and end dates, calendar, and approximate date of northern hemisphere summer solstice, based on Hall & Jones (2010).

Internal Changes
~~~~~~~~~~~~~~~~
* ``aggregate_between_dates`` (introduced in v0.27.0) now accepts ``DayOfYear``-like strings for supplying start and end dates (e.g. ``start="02-01", end="10-31"``).
* The indicator call sequence now considers "variable" the inputs annoted so. Dropped the ``nvar`` attribute.
* Default cfcheck is now to check metadata according to the variable name, using CMIP6 names in xclim/data/variable.yml.
* ``Indicator.missing`` defaults to "skip" if ``freq`` is absent from the list of parameters.
* Minor modifications to the GitHub Pull Requests template.
* Simplification of some yaml elements for virtual modules.
* Allow injecting ``freq`` without the missing checks failing.


0.27.0 (2021-05-28)
-------------------

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Rewrite of nearly all adjustment methods in ``sdba``, with use of ``xr.map_blocks`` to improve scalability with dask. Rewrite of some parts of the algorithms with numba-accelerated code.
* "GFWED" specifics for fire weather computation implemented back into the FWI module. Outputs are within 3% of GFWED data.
* Addition of the `run_length_ufunc` option to control which run length algorithm gets run. Defaults stay the same (automatic switch dependent of the input array : the 1D version is used with non-dask arrays with less than 9000 points per slice).
* Indicator modules built from YAML can now use custom indices. A mapping or module of them can be given to ``build_indicator_module_from_yaml`` with the ``indices`` keyword.
* Virtual submodules now include an `iter_indicators` function to iterate over the pairs of names and indicator objects in that module.
* The indicator string formatter now accepts a "r" modifier which passes the raw strings instead of the adjective version.
* Addition of the `sdba_extra_output` option to adds extra diagnostic variables to the outputs of Adjustment objects. Implementation of `sim_q` in QuantileDeltaMapping and `nclusters` in ExtremeValues.

Breaking changes
~~~~~~~~~~~~~~~~
* The `tropical_nights` indice is being deprecated in favour of `tn_days_above` with ``thresh="20 degC"``. The indicator remains valid, now wrapping this new indice.
* Results of ``sdba.Grouper.apply`` for ``Grouper``s without a group (ex: ``Grouper('time')``) will contain a ``group`` singleton dimension.
* The `daily_freezethaw_cycles` indice is being deprecated in favour of ``multiday_temperature_swing`` with temp thresholds at 0 degC and ``window=1, op="sum"``. The indicator remains valid, now wrapping this new indice.
* CMIP6 variable names have been adopted whenever possible in xclim. Changes are:

    - ``swe`` is now ``snw`` (``snw`` is the snow amount [kg / m²] and ``swe`` the liquid water equivalent thickness [m])
    - ``rh`` is now ``hurs``
    - ``dtas`` is now ``tdps``
    - ``ws`` (in FWI) is now ``sfcWind``
    - ``sic`` is now ``siconc``
    - ``area`` (of sea ice indicators) is now ``areacello``
    - Indicators ``RH`` and ``RH_FROMDEWPOINT`` have be renamed to ``HURS`` and ``HURS_FROMDEWPOINT``. These are changes in the _identifiers_, the python names (``relative_humidity[...]``) are unchanged.

New indicators
~~~~~~~~~~~~~~
* `atmos.corn_heat_units` computes the daily temperature-based index for corn growth.
* New indices and indicators for `tx_days_below`, `tg_days_above`, `tg_days_below`, and `tn_days_above`.
* `atmos.humidex` returns the Canadian *humidex*, an indicator of perceived temperature account for relative humidity.
* `multiday_temperature_swing` indice for returning general statistics based on spells of doubly-thresholded temperatures (Tmin < T1, Tmax > T2).
* New indicators `atmos.freezethaw_frequency`, `atmos.freezethaw_spell_mean_length`, `atmos.freezethaw_spell_max_length` for statistics of Tmin < 0 degC and Tmax > 0 deg C days now available (wrapped from `multiday_temperature_swing`).
* `atmos.wind_chill_index` computes the daily wind chill index. The default is similar to what Environment and Climate Change Canada does, options are tunable to get the version of the National Weather Service.

Internal Changes
~~~~~~~~~~~~~~~~
* `run_length.rle_statistics` now accepts a `window` argument.
* Common arguments to the `op` parameter now have better adjective and noun formattings.
* Added and adjusted typing in call signatures and docstrings, with grammar fixes, for many `xclim.indices` operations.
* Added internal function ``aggregate_between_dates`` for array aggregation operations using xarray datetime arrays with start and end DayOfYear values.


0.26.1 (2021-05-04)
-------------------
* Bug fix release adding `ExtremeValues` to publicly exposed bias-adjustment methods.


0.26.0 (2021-04-30)
-------------------

Announcements
~~~~~~~~~~~~~
* `xclim` no longer supports Python3.6. Code conventions and new features from Python3.7 (`PEP 537 <https://www.python.org/dev/peps/pep-0537/#features-for-3-7>`_) are now accepted.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `core.calendar.doy_to_days_since` and `days_since_to_doy` to allow meaningful statistics on doy data.
* New bias second-order adjustment method "ExtremeValues", intended for re-adjusting extreme precipitation values.
* Virtual indicators modules can now be built from YAML files.
* Indicators can now be built from dictionaries.
* New generic indices, implementation of `clix-meta`'s index functions.
* On-the-fly generation of climate and forecasting convention (CF) checks with `xc.core.cfchecks.generate_cfcheck`, for a few known variables only.
* New `xc.indices.run_length.rle_statistics` for min, max, mean, std (etc) statistics on run lengths.
* New virtual submodule `cf`, with CF standard indices defined in `clix-meta <https://github.com/clix-meta/clix-meta>`_.
* Indices returning day-of-year data add two new attributes to the output: `is_dayofyear` (=1) and `calendar`.

Breaking changes
~~~~~~~~~~~~~~~~
* `xclim` now requires `xarray>=0.17`.
* Virtual submodules `icclim` and `anuclim` are not available at the top level anymore (only through `xclim.indicators`).
* Virtual submodules `icclim` and `anuclim` now provide *Indicators* and not indices.
* Spatial analog methods "KLDIV" and "Nearest Neighbor" now require `scipy>=1.6.0`.

Bug fixes
~~~~~~~~~
* `from_string` object creation in sdba has been removed. Now replaced with use of a new dependency, `jsonpickle`.

Internal Changes
~~~~~~~~~~~~~~~~
* `pre-commit` linting checks now run formatting hook `black==21.4b2`.
* Code cleaning (more accurate call signatures, more use of https links, docstring updates, and typo fixes).

0.25.0 (2021-03-31)
-------------------

Announcements
~~~~~~~~~~~~~
* Deprecation: Release 0.25.0 of `xclim` will be the last version to explicitly support Python3.6 and `xarray<0.17.0`.

New indicators
~~~~~~~~~~~~~~
* `land.winter_storm` computes days with snow accumulation over threshold.
* `land.blowing_snow` computes days with both snow accumulation over last days and high wind speeds.
* `land.snow_melt_we_max` computes the maximum snow melt over n days, and `land.melt_and_precip_max` the maximum combined snow melt and precipitation.
* `snd_max_doy` returns the day of the year where snow depth reaches its maximum value.
* `atmos.high_precip_low_temp` returns days with freezing rain conditions (low temperature and precipitations).
* `land.snow_cover_duration` computes the number of days snow depth exceeds some minimal threshold.
* `land.continuous_snow_cover_start` and `land.continuous_snow_cover_end` identify the day of the year when snow depth crosses a threshold for a given period of time.
* `days_with_snow`, counts days with snow between low and high thresholds, e.g. days with high amount of snow (`indice` and `indicator` available).
* `fire_season`, creates a fire season mask from temperature and, optionally, snow depth time-series.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `generic.count_domain` counts values within low and high thresholds.
* `run_length.season` returns a dataset storing the start, end and length of a *season*.
* Fire Weather indices now support dask-backed data.
* Objects from the `xclim.sdba` submodule can be created from their string repr or from the dataset they created.
* Fire Weather Index submodule replicates the R code of `cffdrs`, including fire season determination and overwintering of the drought_code.
* New `run_bounds` and `keep_longest_run` utilities in `xclim.indices.run_length`.
* New bias-adjustment method: `PrincipalComponent` (based on Hnilica et al. 2017 https://doi.org/10.1002/joc.4890).

Internal changes
~~~~~~~~~~~~~~~~
* Small changes in the output of `indices.run_length.rle`.

0.24.0 (2021-03-01)
-------------------

New indicators
~~~~~~~~~~~~~~
* `days_over_precip_thresh`, `fraction_over_precip_thresh`, `liquid_precip_ratio`, `warm_spell_duration_index`,  all from eponymous indices.
* `maximum_consecutive_warm_days` from indice `maximum_consecutive_tx_days`.

Breaking changes
~~~~~~~~~~~~~~~~
* Numerous changes to `xclim.core.calendar.percentile_doy`:

    * `per` now accepts a sequence as well as a scalar and as such the output has a percentiles axis.
    * `per` argument is now expected to between 0-100 (not 0-1).
    * input data must have a daily (or coarser) time frequency.

* Change in unit handling paradigm for indices, which as a result will lead to some indices returning values with different units. Note that related `Indicator` objects remain unchanged and will return units consistent with CF Convention. If you are concerned with code stability, please use `Indicator` objects. The change was necessary to resolve inconsistencies with xarray's `keep_attrs=True` context.

    * Indice functions now return output units that preserve consistency with input units. That is, feeding inputs in Celsius will yield outputs in Celsius instead of casting to Kelvin. In all cases the dimensionality is preserved.
    * Indice functions now accept non-daily data, but daily frequency is assumed by default if the frequency cannot be inferred.

* Removed the explicitly-installed `netCDF4` python library from the base installation, as this is never explicitly used (now only installed in the `docs` recipe for sdba documented example).
* Removed `xclim.core.checks`, which was deprecated since v0.18.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Indicator now have docstrings generated from their metadata.
* Units and fixed choices set are parsed from indice docstrings into `Indicator.parameters`.
* Units of indices using the `declare_units` decorator are stored in `indice.in_units` and `indice.out_units`.
* Changes to `Indicator.format` and `Indicator.json` to ensure the resulting json really is serializable.

Internal changes
~~~~~~~~~~~~~~~~
* Leave `missing_options` undefined in `land.fit` indicator to allow control via `set_options`.
* Modified `xclim.core.calendar.percentile_doy` to improve performance.
* New `xclim.core.calendar.compare_offsets` for comparing offset strings.
* New `xclim.indices.generic.get_op` to retrieve a function from a string representation of that operator.
* The CI pipeline has been migrated from Travis CI to GitHub Actions. All stages are still built using `tox`.
* Indice functions must always set the units (the `declare_units` decorator does no check anymore).
* New `xclim.core.units.rate2amout` to convert rates like precipitation to amounts.
* `xclim.core.units.pint2cfunits` now removes ' * ' symbols and changes `Δ°` to `delta_deg`.
* New `xclim.core.units.to_agg_units` and `xclim.core.units.infer_sampling_units` for unit handling involving aggregation operations along the time dimension.
* Added an indicators API page to the docs and links to there from the `Climate Indicators` page.

Bug fixes
~~~~~~~~~
* The unit handling change resolved a bug that prevented the use of `xr.set_options(keep_attrs=True)` with indices.

0.23.0 (2021-01-22)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Renamed indicator `atmos.degree_days_depassment_date` to `atmos.degree_days_exceedance_date`.
* In `degree_days_exceedance_date` : renamed argument `start_date` to `after_date`.
* Added cfchecks for Pr+Tas-based indicators.
* Refactored test suite to now be available as part of the standard library installation (`xclim.testing.tests`).
* Running `pytest` with `xdoctest` now requires the `rootdir` to point at `tests` location (`pytest --rootdir xclim/testing/tests/ --xdoctest xclim`).
* Development checks now require working jupyter notebooks (assessed via the `pytest --nbval` command).

New indicators
~~~~~~~~~~~~~~
* `rain_approximation` and `snowfall_approximation` for computing `prlp` and `prsn` from `pr` and `tas` (or `tasmin` or `tasmax`) according to some threshold and method.
* `solid_precip_accumulation` and `liquid_precip_accumulation` now accept a `thresh` parameter to control the binary snow/rain temperature threshold.
* `first_snowfall` and `last_snowfall` to compute the date of first/last snowfall exceeding a threshold in a period.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New `kind` entry in the `parameters` property of indicators, differentiating between [optional] variables and parameters.
* The git pre-commit hooks (`pre-commit run --all`) now clean the jupyter notebooks with `nbstripout` call.

Bug fixes
~~~~~~~~~
* Fixed a bug in `indices.run_length.lazy_indexing` that occurred with 1D coords and 0D indexes when using the dask backend.
* Fixed a bug with default frequency handling affecting `fit` indicator.
* Set missing method to 'skip' for `freq_analysis` indicator.
* Fixed a bug in `ensembles._ens_align_datasets` that occurred when inputs are `.nc` filepaths but files lack a time dimension.

Internal changes
~~~~~~~~~~~~~~~~
* `core.cfchecks.check_valid` now accepts a sequence of strings as its `expected` argument.
* Clean up in the tests to speed up testing. Addition of a marker to include "slow" tests when desired (`-m slow`).
* Fixes in the tests to support `sklearn>=0.24`, `clisops>=0.5` and build xarray@master against python 3.7.
* Moved the testing suite to within xclim and simplified `tox` to manage its own tempdir.
* Indicator class now has a `default_freq` method.


0.22.0 (2020-12-07)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Statistical functions (`frequency_analysis`, `fa`, `fit`, `parametric_quantile`) are now solely accessible via `indices.stats`.

New indicators
~~~~~~~~~~~~~~
* `atmos.degree_days_depassment_date`, the day of year when the degree days sum exceeds a threshold.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Added unique titles to `atmos` calculations employing wrapped_partials.
* `xclim.core.calendar.convert_calendar` now accepts a `missing` argument.
* Added `xclim.core.calendar.date_range` and `xclim.core.calendar.date_range_like` wrapping pandas' `date_range` and xarray's `cftime_range`.
* `xclim.core.calendar.get_calendar` now accepts many different types of data, including datetime object directly.
* New module `xclim.analog` and method `xclim.analog.spatial_analogs` to compute spatial analogs.
* Indicators can now accept dataset in their new `ds` call argument. Variable arguments (that use the `DataArray` annotation) can now be given with strings that correspond to variable names in the dataset, and default to their own name.
* Clarification to `frequency_analysis` notebook.
* Now officially supporting PEP596 (Python3.9).
* New methods `xclim.ensembles.change_significance` and `xclim.ensembles.knutti_sedlacek` to qualify climate change agreement among members of an ensemble.

Bug fixes
~~~~~~~~~
* Fixed bug that prevented the use of `xclim.core.missing.MissingBase` and subclasses with an indexer and a cftime datetime coordinate.
* Fixed issues with metadata handling in statistical indices.
* Various small fixes to the documentation (re-establishment of some internally and externally linked documents).

Internal changes
~~~~~~~~~~~~~~~~
* Passing `align_on` to `xclim.core.calendar.convert_calendar` without using '360_day' calendars will not raise a warning anymore.
* Added formatting utilities for metadata attributes (`update_cell_methods`, `prefix_attrs` and `unprefix_attrs`).
* `xclim/ensembles.py` moved to `xclim/ensembles/*.py`, splitting stats/creation, reduction  and robustness methods.
* With the help of the `mypy` library, added several typing fixes to better identify inputs/outputs, and reduce object type mutations.
* Fixed some doctests in `ensembles` and `set_options`.
* `clisops` v0.4.0+ is now an optional requirements for non-Windows builds.
* New `xclim.core.units.str2pint` method to convert quantity strings to quantity objects. Main improvement is to make "3 degC days" a valid string that converts to "3 K days".


0.21.0 (2020-10-23)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Statistical functions (`frequency_analysis`, `fa`, `fit`, `parametric_quantile`) moved from `indices.generic` to `indices.stats` to make them more visible.

New indicators
~~~~~~~~~~~~~~

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New xclim.testing.open_dataset method to read data from the remote testdata repo.
* Added a notebook, `ensembles-advanced.ipynb`, to the documentation detailing ensemble reduction techniques and showing how to make use of built-in figure-generating commands.
* Added a notebook, `frequency_analysis.ipynb`, with examples showcasing frequency analysis capabilities.

Bug fixes
~~~~~~~~~
* Fixed a bug in the attributes of `frost_season_length`.
* `indices.run_length` methods using dates now respect the array's calendar.
* Worked around an xarray bug in sdba.QuantileDeltaMapping when multidimensional arrays are used with linear or cubic interpolation.

Internal changes
~~~~~~~~~~~~~~~~~

0.20.0 (2020-09-18)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~
* `xclim.subset` has been deprecated and now relies on `clisops` to perform specialized spatio-temporal subsetting.
  Install with `pip install xclim[gis]` in order to retain the same functionality.
* The python library `pandoc` is no longer listed as a docs build requirement. Documentation still requires a current
  version of `pandoc` binaries installed at system-level.
* ANUCLIM indices have seen their `input_freq` parameter renamed to `src_timestep` for clarity.
* A clean-up and harmonization of the indicators metadata has changed some of the indicator identifiers, long_names, abstracts and titles. `xclim.atmos.drought_code` and `fire_weather_indexes` now have indentifiers "dc" and "fwi" (lowercase version of the previous identifiers).
* `xc.indices.run_length.run_length_with_dates` becomes `xc.indices.run_length.season_length`. Its argument `date` is now optional and the default changes from "07-01" to `None`.
* `xc.indices.consecutive_frost_days` becomes `xc.indices.maximum_consecutive_frost_days`.
* Changed the `history` indicator output attribute to `xclim_history` in order to respect CF conventions.

New indicators
~~~~~~~~~~~~~~
* `atmos.max_pr_intensity` acting on hourly data.
* `atmos.wind_vector_from_speed`, also the `wind_speed_from_vector` now also returns the "wind from direction".
* Richards-Baker flow flashiness indicator (`xclim.land.rb_flashiness_index`).
* `atmos.max_daily_temperature_range`.
* `atmos.cold_spell_frequency`.
* `atmos.tg_min` and `atmos.tg_max`.
* `atmos.frost_season_length`, `atmos.first_day_above`. Also, `atmos.consecutive_frost_days` now takes a `thresh` argument (default : 0 degC).

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `sdba.loess` submodule implementing LOESS smoothing tools used in `sdba.detrending.LoessDetrend`.
* xclim now depends on clisops for subsetting, offloading several heavy GIS dependencies. This improves
  maintainability and reduces the size of a "vanilla" xclim installation considerably.
* New `generic.parametric_quantile` function taking parameters estimated by `generic.fit` as an input.
* Add support for using probability weighted moments method in `generic.fit` function. Requires the
  `lmoments3` package, which is not included in dependencies because it is unmaintained. Install manually if needed.
* Implemented `_fit_start` utility function providing initial conditions for statistical distribution parameters estimation, reducing the likelihood of poor fits.
* Added support for indicators based on hourly (1H) inputs, and a first hourly indicator called `max_pr_intensity`
  returning hourly precipitation intensity.
* Indicator instances can be retrieved through their class with the `get_instance()` class method.
  This allows the use of `xclim.core.indicator.registry` as an instance registry.
* Indicators now have a `realm` attribute. It must be given when creating indicators outside xclim.
* Better docstring parsing for indicators: parameters description, annotation and default value are accessible in the json output and `Indicator.parameters`.
* New command line interface `xclim` for simple indicator computing tasks.
* New `sdba.processing.jitter_over_thresh` for variables with a upper bound.
* Added `op` parameter to `xclim.indices.daily_temperature_range` to allow resample reduce operations other than mean
* `core.formatting.AttrFormatter` (and thus, locale dictionaries) can now use glob-like pattern for matching values to translate.

Bug fixes
~~~~~~~~~
The ICCLIM module was identified as `icclim` in the documentation but the module available under `ICCLIM`. Now `icclim == ICCLIM` and `ICCLIM will be deprecated in a future release`.


Internal changes
~~~~~~~~~~~~~~~~
* `xclim.subset` now attempts to load and expose the functions of `clisops.core.subset`. This is an API workaround preserving backwards compatibility.
* Code styling now conforms to the latest release of black (v0.20.8).
* New `IndicatorRegistrar` class that takes care of adding indicator classes and instances to the
  appropriate registries. `Indicator` now inherits from it.


0.19.0 (2020-08-18)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Refactoring of the `Indicator` class. The `cfprobe` method has been renamed to `cfcheck` and the `validate`
  method has been renamed to `datacheck`. More importantly, instantiating `Indicator` creates a new subclass on
  the fly and stores it in a registry, allowing users to subclass existing indicators easily. The algorithm for
  missing values is identified by its registered name, e.g. "any", "pct", etc, along with its `missing_options`.
* xclim now requires xarray >= 0.16, ensuring that xclim.sdba is fully functional.
* The dev requirements now include `xdoctest` -- a rewrite of the standard library module, `doctest`.
* `xclim.core.locales.get_local_attrs` now uses the indicator's class name instead of the indicator itself and no
  longer accepts the `fill_missing` keyword. Behaviour is now the same as passing `False`.
* `Indicator.cf_attrs` is now a list of dictionaries. `Indicator.json` puts all the metadata attributes in the key "outputs" (a list of dicts).
  All variable metadata (names in `Indicator._cf_names`) might be strings or lists of strings when accessed as object attributes.
* Passing doctests are now strictly enforced as a build requirement in the Travis CI testing ensemble.

New features and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New `ensembles.kkz_reduce_ensemble` method to select subsets of an ensemble based on the KKZ algorithm.
* Create new Indicator `Daily`, `Daily2D` subclasses for indicators using daily input data.
* The `Indicator` class now supports outputing multiple indices for the same inputs.
* `xclim.core.units.declare_units` now works with indices outputting multiple DataArrays.
* Doctests now make use of the `xdoctest_namespace` in order to more easily access modules and testdata.

Bug fixes
~~~~~~~~~
* Fix `generic.fit` dimension ordering. This caused errors when "time" was not the first dimension in a DataArray.

Internal changes
~~~~~~~~~~~~~~~~
* `datachecks.check_daily` now uses `xr.infer_freq`.
* Indicator subclasses `Tas`, `Tasmin`, `Tasmax`, `Pr` and `Streamflow` now inherit from `Daily`.
* Indicator subclasses `TasminTasmax` and `PrTas` now inherit from `Daily2D`.
* Docstring style now enforced using the `pydocstyle` with `numpy` doctsring conventions.
* Doctests are now performed for all docstring `Examples` using `xdoctest`. Failing examples must be explicitly skipped otherwise build will now fail.
* Indicator methods `update_attrs` and `format` are now classmethods, attrs to update must be passed.
* Indicators definitions without an accompanying translation (presently French) will cause build failures.
* Major refactoring of the internal machinery of `Indicator` to support multiple outputs.

0.18.0 (2020-06-26)
-------------------
* Optimization options for `xclim.sdba` : different grouping for the normalization steps of DQM and save training or fitting datasets to temporary files.
* `xclim.sdba.detrending` objects can now act on groups.
* Replaced `dask[complete]` with `dask[array]` in basic installation and added `distributed` to `docs` build dependencies.
* `xclim.core.locales` now supported in Windows build environments.
* `ensembles.ensemble_percentiles` modified to compute along a `percentiles` dimension by default, instead of creating different variables.
* Added indicator `first_day_below` and run length helper `first_run_after_date`.
* Added ANUCLIM model climate indices mappings.
* Renamed `areacella` to `areacello` in sea ice tests.
* Sea ice extent and area outputs now have units of m2 to comply with CF-Convention.
* Split `checks.py` into `cfchecks.py`, `datachecks.py` and `missing.py`. This change will only affect users creating custom indices using utilities previously located in `checks.py`.
* Changed signature of `daily_freeze_thaw_cycles`, `daily_temperature_range`, `daily_temperature_range_variability` and `extreme_temperature_range` to take (tasmin, tasmax) instead of (tasmax, tasmin) and match signature of other similar multivariate indices.
* Added `FromContext` subclass of `MissingBase` to have a uniform API for missing value operations.
* Remove logging commands that captured all xclim warnings. Remove deprecated xr.set_options calls.

0.17.0 (2020-05-15)
-------------------
* Added support for operations on dimensionless variables (`units = '1'`).
* Moved `xclim.locales` to `xclim.core.locales` in a batch of internal changes aimed to removed most potential cyclic imports cases.
* Missing checks and input validation refactored with addition of custom missing class registration (`xclim.core.checks.register_missing_method`) and simple validation method decorator (`xclim.core.checks.check`).
* New `xclim.set_options` context to control the missing checks, input validation and locales.
* New `xclim.sdba` module for statistical downscaling and bias-adjustment of climate data.
* Added `convert_calendar` and `interp_calendar` to help in the conversion between calendars.
* Added `at_least_n_valid` function, identifying null calculations based on minimum threshold.
* Added support for `freq=None` in missing calculations.
* Fixed outdated code examples in the docs and docstrings.
* Doctests are now run as part of the test suite.

0.16.0 (2020-04-23)
-------------------
* Added `vectorize` flag to `subset_shape` and `create_mask_vectorize` function based on `shapely.vectorize` as default backend for mask creation.
* Removed `start_yr` and `end_yr` flags from subsetting functions.
* Add multi gridpoints support in `subset.subset_gridpoint`.
* Better `wrapped_partial` for more meaningful inspection.
* Add indices for relative humidity, specific humidity and saturation vapor pressure with a few choices of method.
* Allow lazy units conversion.
* CRS definitions of projected DataSets are now written to file according to Climate and Forecast-convention standards.
* Add utilities to merge attributes and update history in xclim.core.formatting.
* Ensembles : Allow alignment of datasets with same frequency but different offsets.
* Bug fixes in run_length for run-with-dates methods when the date is not found in the run.
* Remove deepcopy from subset.subset_shape to improve memory usage.
* Add `missing_wmo` function, identifying null calculations based on criteria from WMO.
* Add `missing_pct` function, identifying null calculations based on percentage of missing values.

0.15.x (2020-03-12)
-------------------
* Improvement in FWI: Vectorization of DC, DMC and FFMC with numba and small code refactoring for better maintainability.
* Added example notebook for creating a catalog of selected indices
* Added `growing_season_end`, `last_spring_frost`, `dry_days`,  `hot_spell_frequency`, `hot_spell_max_length`, and `maximum_consecutive_frost_free_days` indices.
* Dropped use of `fiona.crs` class in lieu of the newer pyproj CRS handler for `subset_shape` operations.
* Complete internal reorganization of xclim.
* Internationalization of xclim : add `locales` submodule for localized metadata.
* Add feature to retrieve coordinate values instead of index in `run_length.first_run`. Add `run_length.last_run`.
* Fix bug in subset_gridpoint to work on lat/lon coords of any dimension when they are not a dimension of the data.

0.14.x (2020-02-21)
-------------------
* Refactoring of the documentation.
* Added support for pint 0.10
* Add `atmos.heat_wave_total_length` (fixing a namespace issue)
* Fixes in `utils.percentile_doy` and `indices.winter_rain_ratio` for multidimensionnal datasets.
* Rewrote the `subset.subset_shape` function to allow for dask.delayed (lazy) computation.
* Added utility functions to compute `time_bnds` when resampling data encoded with `CFTimeIndex` (non-standard calendars).
* Fix in `subset.subset_gridpoint` for dask array coordinates.
* Modified `subset_shape` to support subsetting with GeoPandas datatypes directly.
* Fix in `subset.wrap_lons_and_split_at_greenwich` to preserve multi-region dataframes.
* Improve the memory use of `indices.growing_season_length`.
* Better handling of data with atypically named `lat` and `lon` dimensions.
* Added six Fire Weather indices.

0.13.x (2020-01-10)
-------------------
* Documentation improvements: list of indicators, RTD theme, notebook example.
* Added `sea_ice_extent` and `sea_ice_area` indicators.
* Reverted #311, removing the `_rolling` util function. Added optimal keywords to `rolling()` calls.
* Fixed `ensembles.create_ensemble` errors for builds against xarray master branch.
* Reformatted code to make better use of Python3.6 conventions (f-strings and object signatures).
* Fixed randomly failing tests of `checks.missing_any`.
* Improvement of `ensemble.ensemble_percentile` and `ensemble.create_ensemble`.

0.12.x-beta (2019-11-18)
------------------------
* Added a distance function computing the geodesic distance to a point.
* Added a `tolerance` argument to `subset_gridpoint` raising an error if distance to closest point is larger than tolerance.
* Created land module for standardized access to streamflow indices.
* Enhancement to utils.Indicator to have more dynamic attributes using callables.
* Added indices `heat_wave_total_length` and `tas` / `tg` to average tasmin and tasmax into tas.
* Fixed a bug with typed call signatures that caused downstream failures on library import.
* Added a `_rolling` util function to fix memory issues on large dask datasets.
* Added the `subset_shape` function to subset utilities for clipping region-masked datasets via polygons.
* Fixed a bug where certain dependencies caused ReadTheDocs builds to fail.
* Added many statically typed function signatures for better function documentation.
* Improved `DeprecationWarnings` and `UserWarnings` ensemble for xclim subsetting functions.
* Dropped support for Python3.5.

0.11.x-beta (2019-10-17)
------------------------
* Added type hinting to call signatures of many functions for more explicit type-checking.
* Added Kmeans clustering ensemble reduction algorithms.
* Added utilities for converting between wind velocity (sfcWind) and wind components (uas, vas) arrays.
* Added type hinting to call signatures of many functions for more explicit type-checking.
* Now supporting explicit builds for Windows OS via Travis CI.
* Fix failing test with Python 3.7.
* Fixed bug in subset.subset_bbox that could add unwanted coordinates/dims to some variables when applied to an entire dataset.
* Reformatted packaging configuration to pure Py3 wheel that ignore tests and test data.
* Now officially supporting Python3.8!
* Enhancement to precip_accumulation() to allow estimated amounts solid (or liquid) phase precipitation.
* Bugfix for frequency analysis choking on time series with NaNs only.

0.10.x-beta (2019-06-18)
------------------------
* Added indices to ICCLIM module.
* Added indices `days_over_precip_thresh` and `fraction_over_precip_thresh`.
* Migrated to a `major.minor.patch-release` semantic versioning system.
* Removed attributes in netCDF output from Indicators that are not in the CF-convention.
* Added `fit` indicator to fit the parameters of a distribution to a series.
* Added utilities with ensemble, run length, and subset algorithms to the documentation.
* Source code development standards now implement Python Black formatting.
* Pre-commit is now used to launch code formatting inspections for local development.
* Documentation now includes more detailed usage and an example workflow notebook.
* Development build configurations are now available via both Anaconda and pip install methods.
* Modified create_ensembles() to allow creation of ensemble dataset without a time dimension as well as from xr.Datasets.
* Modified create ensembles() to pad input data with nans when time dimensions are unequal.
* Updated subset_gridpoint() and subset_bbox() to use .sel method if 'lon' and 'lat' dims are present.
* *Added Azure Pipelines to automatically build xclim in Microsoft Windows environments.* -- **REMOVED**
* Now employing PEP8 + Black compatible autoformatting.
* Added Windows and macOS images to Travis CI build ensemble.
* Added variable thresholds for tasmax and tasmin in daily_freezethaw_events.
* Updated subset.py to use date formatted strings ("%Y", "%Y%m" etc.) in temporal subsetting.
* Clean-up of day-of-year resampling. Precipitation percentile threshold will work without a doy index.
* Addressed deprecations for xarray 0.13.0.
* Added a decorator function that verifies validity and reformats subset calls using start_date or end_date signatures.
* Fixed a bug where 'lon' or 'lon_bounds' would return false values if either signatures were set to 0.

0.10-beta (2019-06-06)
----------------------
* Dropped support for Python 2.
* Added support for *period of the year* subsetting in ``checks.missing_any``.
* Now allow for passing positive longitude values when subsetting data with negative longitudes.
* Improved runlength calculations for small grid size arrays via ``ufunc_1dim`` flag.

0.9-beta (2019-05-13)
---------------------
This is a significant jump in the release. Many modifications have been made and will be added to the documentation in the coming days. Among the many changes:

* New indices have been added with documentation and call examples.
* Run_length based operations have been optimized.
* Support for CF non-standard calendars.
* Automated/improved unit conversion and management via pint library.
* Added ensemble utilities for creation and analysis of muti-model climate ensembles.
* Added subsetting utilities for spatio-temporal subsets of xarray data objects.
* Added streamflow indicators.
* Refactoring of the code : separation of indices.py into a directory with sub-files (simple, threshold and multivariate); ensembles and subset utilities separated into distinct modules (pulled from utils.py).
* Indicators are now split into packages named by realms. import xclim.atmos to load indicators related to atmospheric variables.

0.8-beta (2019-02-11)
---------------------
*This was a staging release and is functionally identical to 0.7-beta*.

0.7-beta (2019-02-05)
---------------------
Major Changes:

* Support for resampling of data structured using non-standard CF-Time calendars.
* Added several ICCLIM and other indicators.
* Dropped support for Python 3.4.
* Now under Apache v2.0 license.
* Stable PyPI-based dependencies.
* Dask optimizations for better memory management.
* Introduced class-based indicator calculations with data integrity verification and CF-Compliant-like metadata writing functionality.

Class-based indicators are new methods that allow index calculation with error-checking and provide on-the-fly metadata checks for CF-Compliant (and CF-compliant-like) data that are passed to them. When written to NetCDF, outputs of these indicators will append appropriate metadata based on the indicator, threshold values, moving window length, and time period / resampling frequency examined.

0.6-alpha (2018-10-03)
----------------------
* File attributes checks.
* Added daily downsampler function.
* Better documentation on ICCLIM indices.

0.5-alpha (2018-09-26)
----------------------
* Added total precipitation indicator.

0.4-alpha (2018-09-14)
----------------------
* Fully PEP8 compliant and available under MIT License.

0.3-alpha (2018-09-4)
---------------------
* Added icclim module.
* Reworked documentation, docs theme.

0.2-alpha (2018-08-27)
----------------------
* Added first indices.

0.1.0-dev (2018-08-23)
----------------------
* First release on PyPI.
