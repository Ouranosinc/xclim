=======
History
=======

0.10.x-beta (2019-06-18)
------------------------
* Migrated to a `major.minor.patch-release` semantic versioning system.
* Removed attributes in netCDF output from Indicators that are not in the CF-convention.
* Added `fit` indicator to fit the parameters of a distribution to a series.
* Added utilities with ensemble, run length, and subset algorithms to the documentation.
* Source code development standards now implement Python Black formatting.
* Pre-commit is now used to launch code formatting inspections for local development.

0.10-beta (2019-06-06)
----------------------
* Indicators are now split into packages named by *realms*. ``import xclim.atmos`` is now the method for loading indicators related to atmospheric variables.
* Removed support for Python 2 compatibility.
* Added support for *period of the year* subsetting in ``checks.missing_any``.
* Now allow for passing positive longitude values when subsetting data with negative longitudes.
* Improved runlength calculations for small grid size arrays via ``ufunc_1dim`` flag.

0.9-beta (2019-05-13)
---------------------
TODO

0.8-beta (2019-02-11)
---------------------
TODO

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
