=======
History
=======

0.19.0 (2020-08-18
------------------

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
* Doctests now make use of the `xdoctest_namespace` in order to more easily access mdoules and tesdata.

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
* Major refactoring of the internal marchinery of `Indicator` to support multiple outputs.

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
