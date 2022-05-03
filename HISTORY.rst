=======
History
=======

0.36.1 (unreleased)
------------------

Contributors to this version:  Abel Aoun (:user:`bzah`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Add "Celsius" to aliases of "celsius" unit.(:issue:`1067`, :pull:`1068`).


0.36.0 (29-04-2022)
-------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), David Huard (:user:`huard`).

Bug fixes
^^^^^^^^^
* Invoking ``lazy_indexing`` twice in row (or more) using the same indexes (using dask) is now fixed. (:issue:`1048`, :pull:`1049`).
* Filtering out the nans before choosing the first and last values as ``fill_value`` in ``_interp_on_quantiles_1D``. (:issue:`1056`, :pull:`1057`).
* Translations from virtual indicator modules do not override those of the base indicators anymore. (:issue:`1053`, :pull:`1058`).
* Fix mmday unit definition (factor 1000 error). (:issue:`1061`, :pull:`1063`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``xclim.sdba.measures.rmse`` and ``xclim.sdba.measures.mae`` now use `numpy` instead of `sklearn`. This improves their performances when using `dask`. (:pull:`1051`).
* Argument ``append_ends`` added to ``sdba.unpack_moving_yearly_window`` (:pull:`1059`).

Internal changes
^^^^^^^^^^^^^^^^
* Ipython was unpinned as version 8.2 fixed the previous issue. (:issue:`1005`, :pull:`1064`).

0.35.0 (01-04-2022)
-------------------
Contributors to this version: David Huard (:user:`huard`), Trevor James Smith (:user:`Zeitsperre`) and Pascal Bourgault (:user:`aulemahal`).

New indicators
^^^^^^^^^^^^^^
* New indicator ``specific_humidity_from_dewpoint``, computing specific humidity from the dewpoint temperature and air pressure. (:issue:`864`, :pull:`1027`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New spatial analogues method "szekely_rizzo" (:pull:`1033`).
* Loess smoothing (and detrending) now skip NaN values, instead of propagating them. This can be controlled through the `skipna` argument. (:pull:`1030`).

Bug fixes
^^^^^^^^^
* ``xclim.analog.spatial_analogs`` is now compatible with dask-backed DataArrays. (:pull:`1033`).
* Parameter ``dmin`` added to spatial analog method "zech_aslan", to avoid singularities on identical points. (:pull:`1033`).
* `xclim` is now compatible with changes in `xarray` that enabled explicit indexing operations. (:pull:`1038`, `xarray PR <https://github.com/pydata/xarray/pull/5692>`_).

Internal changes
^^^^^^^^^^^^^^^^
* `xclim` now uses the ``check-json`` and ``pretty-format-json`` pre-commit checks to validate and format JSON files. (:pull:`1032`).
* The few `logging` artifacts in the ``xclim.ensembles`` module have been replaced with `warnings.warn` calls or removed. (:issue:`1039`, :pull:`1044`).

0.34.0 (25-02-2022)
-------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), David Huard (:user:`huard`), Aoun Abel (:user:`bzah`).

Announcements
^^^^^^^^^^^^^
* `xclim` now officially supports Python3.10. (:pull:`1013`).

Breaking changes
^^^^^^^^^^^^^^^^
* The version pin for `bottleneck` (<1.4) has been lifted. (:pull:`1013`).
* `packaging` has been removed from the `xclim` run dependencies. (:pull:`1013`).
* Quantile mapping adjustment objects (EQM, DQM and QDM) and ``sdba.utils.equally_spaced_nodes`` will not add additional endpoints to the quantile range. With those endpoints, variables are capped to the reference's range in the historical period, which can be dangerous with high variability in the extremes (ex: pr), especially if the reference doesn't reproduce those extremes credibly. (:issue:`1015`, :pull:`1016`). To retrieve the same functionality as before use:

.. code-block:: python

    from xclim import sdba
    # NQ is the the number of equally spaced nodes, the argument previously given to nquantiles directly.
    EQM = sdba.EmpiricalQuantileMapping.train(ref, hist, nquantiles=sdba.equally_spaced_nodes(NQ, eps=1e-6), ...)

* The "history" string attribute added by xclim has been modified for readability: (:issue:`963`, :pull:`1018`).
    - The trailing dot (``.``) was dropped.
    - ``None`` inputs are now printed as "None" (and not "<NoneType>").
    - Arguments are now always shown as keyword-arguments. This mostly impacts ``sdba`` functions, as it was already the case for ``Indicators``.
* The `cell_methods` string attribute appends only the operation from the indicator itself. In previous version, some indicators also appended the input data's own `cell_method`. The clix-meta importer has been modified to follow the same convention. (:issue:`983`, :pull:`1022`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `publish_release_notes` now leverages much more regular expression logic for link translations to markdown. (:pull:`1023`).
* Improve performances of percentile bootstrap algorithm by using ``xarray.map_block`` (:issue:`932`, :pull:`1017`).

Bug fixes
^^^^^^^^^
* Loading virtual python modules with ``build_indicator_module_from_yaml`` is now fixed on some systems where the current directory was not part of python's path. Furthermore, paths of the python and json files can now be passed directly to the ``indices`` and ``translations`` arguments, respectively. (:issue:`1020`, :pull:`1021`).

Internal changes
^^^^^^^^^^^^^^^^
* Due to an upstream bug in `bottleneck`'s support of virtualenv, `tox` builds for Python3.10 now depend on a patched fork of `bottleneck`. This workaround will be removed once the fix is merged upstream. (:pull:`1013`, see: `bottleneck PR/397 <https://github.com/pydata/bottleneck/pull/397/>`_).
    - This has been removed with the release of `bottleneck version 1.3.4 <https://pypi.org/project/Bottleneck/1.3.4/>`_. (:pull:`1025`).
* GitHub CI actions now use the `deadsnakes python PPA Action <https://github.com/deadsnakes/action>`_ for gathering the Python3.10 development headers. (:pull:`1013`).
* The "is_dayofyear" attribute added by several indices is now a ``numpy.int32`` instance, instead of python's ``int``. This ensures a THREDDS server can read it when the variable is saved to a netCDF file with `xarray`/`netCDF4-python`. (:issue:`980`, :pull:`1019`).
* The `xclim` git repository now offers `Issue Forms <https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository#creating-issue-forms>`_ for some general issue types.

0.33.2 (2022-02-09)
-------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`).

Announcements
^^^^^^^^^^^^^
* `xclim` no longer supports Python3.7. Code conventions and new features for Python3.8 (`PEP 569 <https://www.python.org/dev/peps/pep-0569/>`_) are now accepted. (:issue:`966`, :pull:`1000`).

Breaking changes
^^^^^^^^^^^^^^^^
* Python3.7 (`PEP 537 <https://www.python.org/dev/peps/pep-0537/>`_) support has been officially deprecated. Continuous integration testing is no longer run against this version of Python. (:issue:`966`, :pull:`1000`).

Bug fixes
^^^^^^^^^
* Adjusted behaviour in ``dataflags.ecad_compliant`` to remove `data_vars` of invalids checks that return `None`, causing issues with `dask`. (:pull:`1002`).
* Temporarily pinned `ipython` below version 8.0 due to behaviour causing hangs in GitHub Actions and ReadTheDocs. (:issue:`1005`, :pull:`1006`).
* ``indices.stats`` methods where adapted to handle dask-backed arrays. (:issue:`1007`, :`pull:`1011`).
* ``sdba.utils.interp_on_quantiles``, with ``extrapolation='constant'``, now interpolates the limits of the interpolation along the time grouping index, fixing a issue with "time.month" grouping. (:issue:`1008`, :pull:`1009`).

Internal changes
^^^^^^^^^^^^^^^^
* `pre-commit` now uses Black 22.1.0 with Python3.8 style conventions. Existing code has been adjusted. (:pull:`1000`).
* `tox` builds for Python3.7 have been deprecated. (:pull:`1000`).
* Docstrings and documentation has been adjusted for grammar and typos. (:pull:`1000`).
* ``sdba.utils.extrapolate_qm`` has been removed, as announced for xclim 0.33. (:pull:`1009`).

0.33.0 (2022-01-28)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Tom Keel (:user:`Thomasjkeel`), Jeremy Fyke (:user:`JeremyFyke`), David Huard (:user:`huard`), Abel Aoun (:user:`bzah`), Juliette Lavoie (:user:`juliettelavoie`), Yannick Rousseau (:user:`yrouranos`).

Announcements
^^^^^^^^^^^^^
* Deprecation: Release 0.33.0 of `xclim` will be the last version to explicitly support Python3.7 and `xarray<0.21.0`.
* `xclim` now requires yaml files to pass `yamllint` checks on Pull Requests. (:pull:`981`).
* `xclim` now requires docstrings have valid ReStructuredText formatting to pass basic linting checks. (:pull:`993`). Checks generally require:
    - Working hyperlinks and reference tags.
    - Valid content references (e.g. `:py:func:`).
    - Valid NumPy-formatted docstrings.
* The `xclim` developer community has now adopted the 'Contributor Covenant' Code of Conduct v2.1 (`text <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`_). (:issue:`948`, :pull:`996`).

New indicators
^^^^^^^^^^^^^^
* ``jetstream_metric_woollings`` indicator returns latitude and strength of jet-stream in u-wind field. (:issue:`923`, :pull:`924`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Features added and modified to allow proper multivariate adjustments. (:pull:`964`).
    - Added ``xclim.sdba.processing.to_additive_space`` and ``xclim.sdba.processing.from_additive_space`` to transform "multiplicative" variables to the additive space. An example of multivariate adjustment using this technique was added to the "Advanced" sdba notebook.
    - ``xclim.sdba.processing.normalize`` now also returns the norm. ``xclim.sdba.processing.jitter`` was created by combining the "under" and "over" methods.
    - ``xclim.sdba.adjustment.PrincipalComponent`` was modified to have a simpler signature. The "full" method for finding the best PC orientation was added. (:issue:`697`).
* New ``xclim.indices.stats.parametric_cdf`` function to facilitate the computation of return periods over DataArrays of statistical distribution parameters (:issue:`876`, :pull:`984`).
* Add ``copy`` parameter to ``percentile_doy`` to control if the array input can be dumped after computing percentiles (:issue:`932`, :pull:`985`).
* New improved algorithm for ``dry_spell_total_length``, performing the temporal indexing at the right moment and with control on the aggregation operator (``op``) for determining the dry spells.
* Added ``properties.py`` and ``measures.py`` in order to perform diagnostic tests of sdba (:issue:`424`, :pull:`967`).
* Update how ``percentile_doy`` rechunk the input data to preserve the initial chunk size. This should make the computation memory footprint more predictable (:issue:`932`, :pull:`987`).

Breaking changes
^^^^^^^^^^^^^^^^
* To reduce import complexity, `select_time` has been refactored/moved from ``xclim.indices.generic`` to ``xclim.core.calendar``. (:issue:`949`, :pull:`969`).
* The stacking dimension of ``xclim.sdba.stack_variables`` has been renamed to "multivar" to avoid name conflicts with the "variables" property of xarray Datasets. (:pull:`964`).
* `xclim` now requires `cf-xarray>=0.6.1`. (:issue:`923`, :pull:`924`).
* `xclim` now requires `statsmodels`. (:issue:`424`, :pull:`967`).

Internal changes
^^^^^^^^^^^^^^^^
* Added a CI hook in ``.pre-commit-config.yaml`` to perform automated `pre-commit` corrections with GitHub CI. (:pull:`965`).
* Adjusted CI hooks to fail earlier if `lint` checks fail. (:pull:`972`).
* `TrainAdjust` and `Adjust` object have a new `skip_input_checks` keyword arg to their `train` and  `adjust` methods. When `True`, all unit-, calendar- and coordinate-related input checks are skipped. This is an ugly solution to disappearing attributes when using `xr.map_blocks` with dask. (:pull:`964`).
* Some slow tests were marked `slow` to help speed up the standard test ensemble. (:pull:`969`).
    - Tox testing ensemble now also reports slowest tests using the ``--durations`` flag.
* `pint` no longer emits warnings about redefined units when the `logging` module is loaded. (:issue:`990`, :pull:`991`).
* Added a CI step for cancelling running workflows in pull requests that receive multiple pushes. (:pull:`988`).

Bug fixes
^^^^^^^^^
* Fix mistake in the units of spell_length_distribution. (:issue:`1003`, :pull:`1004`)

0.32.1 (2021-12-17)
-------------------

Bug fixes
^^^^^^^^^
* Adjusted a test (``test_cli::test_release_notes``) that prevented conda-forge test ensemble from passing. (:pull:`962`).

0.32.0 (2021-12-17)
-------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Travis Logan (:user:`tlogan2000`), Trevor James Smith (:user:`Zeitsperre`), Abel Aoun (:user:`bzah`), David Huard (:user:`huard`), Clair Barnes (:user:`clairbarnes`), Raquel Alegre (:user:`raquel-ucl`), Jamie Quinn (:user:`JamieJQuinn`), Maliko Tanguy (:user:`malngu`), Aaron Spring (:user:`aaronspring`).

Announcements
^^^^^^^^^^^^^
* Code coverage (`coverage/coveralls`) is now a required CI check for merging Pull Requests. Requirements are now:
    - No individual run may report *<80%* code coverage.
    - Some drop in coverage is now tolerable, but runs cannot dip below *-0.25%* relative to the main branch.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added an optimized pathway for ``xclim.indices.run_length`` functions when ``window=1``. (:pull:`911`, :issue:`910`).
* The data input frequency expected by ``Indicator`` is now in the ``src_freq`` attribute and is thus controllable by subclassing existing indicators. (:issue:`898`, :pull:`927`).
* New ``**indexer`` keyword args added to many indicators, it accepts the same arguments as ``xclim.indices.generic.select_time``, which has been improved. Unless otherwise specified, the time selection is done before any computation. (:pull:`934`, :issue:`899`).
* Rewrite of ``xclim.sdba.ExtremeValues``, now fixed with a correct algorithm. It has not been tested extensively and should be considered experimental. (:pull:`914`, :issue:`789`, :issue:`790`).
* Added `days_over_precip_doy_thresh` and `fraction_over_precip_doy_thresh` indicators to distinguish between WMO and ECAD definition of the Rxxp and RxxpTot indices. (:issue:`931`, :pull:`940`).
* Update `xclim.core.utils.nan_calc_percentiles` to improve maintainability. (:pull:`942`).
* Added `heat_index` indicator. Added `heat_index` indicator. This is similar to `humidex` but uses a different dew point as well as heat balance equations which account for variables other than vapor pressure. (:issue:`807`) and (:pull:`915`).
* Added alternative method for ``xclim.indices.potential_evapotranspiration`` based on `mcguinnessbordne05` (from Tanguay et al. 2018). (:pull:`926`, :issue:`925`).
* Added `snw_max` and `snw_max_doy` indicators to compute the maximum snow amount and the day of year of the maximum snow amount respectively. (:issue:`776`, :pull:`950`).
* Added index for calculating ratio of convective to total precipitation. (:issue:`920`, :pull:`921`).
* Added `wetdays_prop` indicator to calculate the proportion of days in a period where the precipitation is greater than a threshold. (:pull:`919`, :issue:`918`).

Breaking changes
^^^^^^^^^^^^^^^^
* Following version 1.9 of the CF Conventions, published in September 2021, the calendar name "gregorian" is deprecated. ``core.calendar.get_calendar`` will return "standard", even if the underlying cftime objects still use "gregorian" (cftime <= 1.5.1). (:pull:`935`).
* ``xclim.sdba.utils.extrapolate_qm`` is now deprecated and will be removed in version 0.33. (:pull:`941`).
* Dependency ``pint`` minimum necessary version is now 0.10. (:pull:`959`).

Internal changes
^^^^^^^^^^^^^^^^
* Removed some logging configurations in ``xclim.core.dataflags`` that were polluting python's main logging configuration. (:pull:`909`).
* Synchronized logging formatters in ``xclim.ensembles`` and ``xclim.core.utils``. (:pull:`909`).
* Added a helper function for generating the release notes with dynamically-generated ReStructuredText or Markdown-formatted hyperlinks (:pull:`922`, :issue:`907`).
* Split of resampling-related functionality of ``Indicator`` into new ``ResamplingIndicator`` and ``ResamplingIndicatorWithIndexing`` subclasses. The use of new (private) methods makes it easier to inject functionality in indicator subclasses. (:issue:`867`, :pull:`927`, :pull:`934`).
* French translation metadata fields are now cleaner and much more internally consistent, and many empty metadata fields (e.g. ``comment_fr``) have been removed. (:pull:`930`, :issue:`929`).
* Adjustments to the ``tox`` builds so that slow tests are now run alongside standard tests (for more accurate coverage reporting). (:pull:`938`).
* Use ``xarray.apply_ufunc`` to vectorize statistical functions. (:pull:`943`).
* Refactor of ``xclim.sdba.utils.interp_on_quantiles`` so that it now handles the extrapolation directly and to better handle missing values. (:pull:`941`).
* Updated `heating_degree_days` and `fraction_over_precip_thresh` documentations. (:issue:`952`, :pull:`953`).
* Added an intersphinx mapping to xarray. (:pull:`955`).
* Added a CodeQL security analysis GitHub CI hook on push to master and on Friday nights. (:pull:`960`).

Bug fixes
^^^^^^^^^
* Fix bugs in the `cf_attrs` and/or `abstract` of `continuous_snow_cover_end` and `continuous_snow_cover_start`. (:pull:`908`).
* Remove unnecessary `keep_attrs` from `resample` call which would raise an error in futur Xarray version. (:pull:`937`).
* Fixed a bug in the regex that parses usernames in the history. (:pull:`945`).
* Fixed a bug in ``xclim.indices.generic.doymax`` and ``xclim.indices.generic.doymin`` that prevented the use of the functions on multidimensional data. (:pull:`950`, :issue:`951`).
* Skip all missing values in ``xclim.sdba.utils.interp_on_quantiles``, drop them from both the old and new coordinates, as well as from the old values. (:pull:`941`).
* "degrees_north" and "degrees_east" (and their variants) are now considered independent units, so that ``pint`` and ``xclim.core.units.ensure_cf_units`` don't convert them to "deg". (:pull:`959`).
* Fixed a bug in ``xclim.core.dataflags`` that would misidentify the "extra" variable to be called when running multivariate checks. (:pull:`957`, :issue:`861`).

0.31.0 (2021-11-05)
-------------------
Contributors to this version: Abel Aoun (:user:`bzah`), Pascal Bourgault (:user:`aulemahal`), David Huard (:user:`huard`), Juliette Lavoie (:user:`juliettelavoie`), Travis Logan (:user:`tlogan2000`), Trevor James Smith (:user:`Zeitsperre`).

New indicators
^^^^^^^^^^^^^^
* ``thawing_degree_days`` indicator returns degree-days above a default of `thresh="0 degC"`. (:pull:`895`, :issue:`887`).
* ``freezing_degree_days`` indicator returns degree-days below a default of `thresh="0 degC"`. (:pull:`895`, :issue:`887`).
* Several frost-free season calculations are now available as both indices and indicators. (:pull:`895`, :issue:`887`):
    - ``frost_free_season_start``
    - ``frost_free_season_end``
    - ``frost_free_season_length``
* ``growing_season_start`` is now offered as an indice and as an indicator to complement other growing season-based indicators (threshold calculation with `op=">="`). (:pull:`895`, :issue:`887`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Improve cell_methods checking to search the wanted method within the whole string. (:pull:`866`, :issue:`863`).
* New ``align_on='random`` option for ``xclim.core.calendar.convert_calendar``, for conversions involving '360_day' calendars. (:pull:`875`, :issue:`841`).
* ``dry_spell_frequency`` now has a parameter `op: {"sum", "max"}` to choose if the threshold is compared against the accumulated or maximal precipitation, over the given window. (:pull:`879`).
* ``maximum_consecutive_frost_free_days`` is now checking that the minimum temperature is above or equal to the threshold ( instead of only above). (:pull:`883`, :issue:`881`).
* The ANUCLIM virtual module has been updated to accept weekly and monthly inputs and with improved metadata. (:pull:`885`, :issue:`538`)
* The ``sdba.loess`` algorithm has been optimized to run faster in all cases, with an even faster special case (``equal_spacing=True``) when the x coordinate is equally spaced. When activated, this special case might return results different from without, up to around 0.1%. (:pull:`865`).
* Add support for group's window and additional dimensions in ``LoessDetrend``. Add new ``RollingMeanDetrend`` object. (:pull:`865`).
* Missing value algorithms now try to infer the source timestep of the input data when it is not given. (:pull:`885`).
* On indices, `bootstrap` parameter documentation has been updated to explain when and why it should be used. (:pull:`893`, :issue:`846`).

Breaking changes
^^^^^^^^^^^^^^^^
* Major changes in the YAML schema for virtual submodules, now closer to how indicators are declared dynamically, see the doc for details. (:pull:`849`, :issue:`848`).
* Removed ``xclim.generic.daily_downsampler``, as it served no purpose now that xarray's resampling works with cftime (:pull:`888`, :issue:`889`).
* Refactor of ``xclim.core.calendar.parse_offset``, output types were changed to useful ones (:pull:`885`).
* Major changes on how parameters are passed to indicators. (:pull:`873`):
    - Their signature is now consistent : input variables (DataArrays, optional or not) are positional or keyword arguments and all other parameters are keyword only. (:issue:`855`, :issue:`857`)
    - Some indicators have modified signatures because we now rename variables when wrapping generic indices. This is the case for the whole cf module, for example.
    - ``Indicator.parameters`` is now a property generated from ``Indicator._all_parameters``, as the latter includes the injected parameters. The keys of the former are instances of new ``xclim.core.indicator.Parameter``, and not dictionaries as before.
    - New ``Indicator.injected_parameters`` to see which compute function arguments will be injected at call time.
    - See the pull request (:pull:`873`) for all information.
* The call signature for ``huglin_index`` has been modified to reflect the correct variables used in its formula (`tasmin` -> `tas`; `thresh_tasmin` -> `thresh`). (:pull:`903`, :issue:`902`).

Internal changes
^^^^^^^^^^^^^^^^
* Pull Request contributions now require hyperlinks to the issue and pull request pages on GitHub listed alongside changess in HISTORY.rst. (:pull:`860`, :issue:`854`).
* Updated the contribution guidelines to better give credit to contributors and more easily track changes. (:pull:`869`, :issue:`868`).
* Enabled coveralls code coverage reporting for GitHub CI. (:pull:`870`).
* Added automated TestPyPI and PyPI-publishing workflows for GitHub CI. (:pull:`872`).
* Changes on how indicators are constructed. (:pull:`873`).
* Added missing algorithms tests for conversion from hourly to daily. (:pull:`888`).
* Updated pre-commit hooks to use black v21.10.b0. (:pull:`896`).
* Moved ``stack_variables``, ``unstack_variables``, ``construct_moving_yearly_window`` and ``unpack_moving_yearly_window`` from ``xclim.sdba.base`` to ``xclim.sdba.processing``. They still are imported in ``xclim.sdba`` as before. (:pull:`892`).
* Many improvements to the documentation. (:pull:`892`, :issue:`880`).
* Added regex replacement handling in setup.py to facilitate publishing contributor/contribution links on PyPI. (:pull:`906`).

Bug fixes
^^^^^^^^^
* Fix a bug in bootstrapping where computation would fail when the dataset time coordinate is encoded using `cftime.datetime`. (:pull:`859`).
* Fix a bug in ``build_indicator_module_from_yaml`` where bases classes (Daily, Hourly, etc) were not usable with the `base` field. (:pull:`885`).
* ``percentile_doy`` alpha and beta parameters are now properly transmitted to bootstrap calls of this function. (:pull:`893`, :issue:`846`).
* When called with a 1D da and ND index, ``xclim.indices.run_length.lazy_indexing`` now drops the auxiliary coordinate corresponding to da's index. This fixes a bug with ND data in ``xclim.indices.run_length.season``. (:pull:`900`).
* Fix name of heating degree days in French (`"chauffe"` -> "`chauffage`"). (:pull:`895`).
* Corrected several French indicator translation description strings (bad usages of `"."` in `description` and `long_name` fields). (:pull:`895`).
* Fixed an error with the formula for ``huglin_index`` where `tasmin` was being used in the calculation instead of `tas`. (:pull:`903`, :issue:`902`).

0.30.1 (2021-10-01)
-------------------

Bug fixes
^^^^^^^^^
* Fix a bug in ``xclim.sdba``'s ``map_groups`` where 1D input including an auxiliary coordinate would fail with an obscure error on a reducing operation.

0.30.0 (2021-09-28)
-------------------

New indicators
^^^^^^^^^^^^^^
* ``climatological_mean_doy`` indice returns the mean and standard deviation across a climatology according to day-of-year (`xarray.DataArray.groupby("time.dayofyear")`). A moving window averaging of days can also be supplied (default:`window=1`).
* ``within_bnds_doy`` indice returns a boolean array indicating whether or not array's values are within bounds for each day of the year.
* Added ``atmos.wet_precip_accumulation``, an indicator accumulating precipitation over wet days.
* Module ICCLIM now includes ``PRCPTOT``, which accumulates precipitation for days with precipitation above 1 mm/day.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``xclim.core.utils.nan_calc_percentiles`` now uses a custom algorithm instead of ``numpy.nanpercentiles`` to have more flexibility on the interpolation method. The performance is also improved.
* ``xclim.core.calendar.percentile_doy`` now uses the 8th method of Hyndman & Fan for linear interpolation (alpha = beta = 1/3). Previously, the function used Numpy's percentile, which corresponds to the 7th method. This change is motivated by the fact that the 8th is recommended by Hyndman & Fay and it ensures consistency with other climate indices packages (`climdex`, `icclim`). Using `alpha = beta = 1` restores the previous behaviour.
* ``xclim.core.utils._cal_perc`` is now only a proxy for ``xc.core.utils.nan_calc_percentiles`` with some axis moves.
* `xclim` now implements many data quality assurance flags (``xclim.core.dataflags``) for temperature and precipitation based on `ICCLIM documentation guidelines <https://eca.knmi.nl/documents/atbd.pdf>`_. These checks include the following:
    - Temperature (variables: ``tas``, ``tasmin``, ``tasmax``): ``tasmax_below_tasmin``, ``tas_exceeds_tasmax``, ``tas_below_tasmin``, ``temperature_extremely_low`` (`thresh="-90 degC"`), ``temperature_extremely_high`` (`thresh="60 degC"`).
    - Precipitation-specific (variables: ``pr``, ``prsn``, ):  ``negative_accumulation_values``, ``very_large_precipitation_events`` (`thresh="300 mm d-1"`).
    - Wind-specific (variables: ``sfcWind``, ``wsgsmax``/``sfcWindMax``): ``wind_values_outside_of_bounds``
    - Generic: ``outside_n_standard_deviations_of_climatology``, ``values_repeating_for_n_or_more_days``, ``values_op_thresh_repeating_for_n_or_more_days``, ``percentage_values_outside_of_bounds``.

    These quality-assurance checks are selected according to CF-standard variable names, and can be triggered via ``xclim.core.dataflags.data_flags(xarray.DataArray, xarray.Dataset)``. These checks are separate from the Indicator-defined `datachecks` and must be launched manually. They'll return an array of data_flags as boolean variables.
    If called with `raise_flags=True`, will raise an Exception with comments for each quality control check raised.
* A convenience function (``xclim.core.dataflags.ecad_compliant``) is also offered as a method for asserting that data adheres to all relevant ECAD/ICCLIM checks. For more information on usage, consult the docstring/documentation.
* A new utility "``dataflags``" is also available for performing fast quality control checks from the command-line (`xclim dataflags --help`). See the CLI documentation page for usage examples.
* Added missing typed call signatures, expected returns and docstrings for many ``xclim.core.calendar`` functions.

Breaking changes
^^^^^^^^^^^^^^^^
* All "ANUCLIM" indices and indicators have lost their `src_timestep` argument. Most of them were not using it and now every function infers the frequency from the data directly. This may add stricter constraints on the time coordinate, the same as for ``xarray.infer_freq``.
* Many functions found within ``xclim.core.cfchecks`` (``generate_cfcheck`` and ``check_valid_*``) have been removed as existing indicator CF-standard checks and data checks rendered them redundant/obsolete.

Bug fixes
^^^^^^^^^
* Fixes in ``sdba`` for (1) inputs with dimensions without coordinates, for (2) ``sdba.detrending.MeanDetrend`` and for (3) ``DetrendedQuantileMapping`` when used with dask's distributed scheduler.
* Replaced instances of `'◦'` ("White bullet") with `'°'` ("Degree Sign") in ``icclim.yaml`` as it was causing issues for non-UTF8 environments.
* Addressed an edge case where ``test_sdba::test_standardize`` randomness could generate values that surpass the test error tolerance.
* Added a missing `.txt` file to the MANIFEST of the source distributable in order to be able to run all tests.
* ``xc.core.units.rate2amount`` is now exact when the sampling frequency is monthly, seasonal or yearly. Earlier, monthly and yearly data were computed using constant month and year length. End-of-period frequencies are also correctly understood (ex: "M" vs "MS").
* In the ``potential_evapotranspiration`` indice, add abbreviated ``method`` names to docstring.
* Fixed an issue that prevented using the default ``group`` arg in adjustment objects.
* Fix bug in ``missing_wmo``, where a period would be considered valid if all months met WMO criteria, but complete months in a year were missing. Now if any month does not meet criteria or is absent, the period will be considered missing.
* Fix bootstrapping with dask arrays. Dask does not support using ``loc`` with multiple indexes to set new values so a workaround was necessary.
* Fix bootstrapping when the bootstrapped year must be converted to a 366_day calendar.
* Virtual modules and translations now use 'UTF-8' by default when reading yaml or json file, instead of a machine-dependent encoding.

Internal Changes
^^^^^^^^^^^^^^^^
* `xclim` code quality checks now use the newest `black` (v21.8-beta). Checks launched via `tox` and `pre-commit` now run formatting modifications over Jupyter notebooks found under `docs`.

0.29.0 (2021-08-30)
-------------------

Announcements
^^^^^^^^^^^^^
* It was found that the ``ExtremeValues`` adjustment algorithm was not as accurate and stable as first thought. It is now hidden from ``xclim.sdba`` but can still be accessed via ``xclim.sdba.adjustment``, with a warning. Work on improving the algorithm is ongoing, and a better implementation will be in a future version.
* It was found that the ``add_dims`` argument of ``sdba.Grouper`` had some caveats throughout ``sdba``. This argument is to be used with care before a careful analysis and more testing is done within ``xclim``.

Breaking changes
^^^^^^^^^^^^^^^^
* `xclim` has switched back to updating the ``history`` attribute (instead of ``xclim_history``). This impacts all indicators, most ensemble functions, ``percentile_doy`` and ``sdba.processing`` (see below).
* Refactor of ``sdba.processing``. Now all functions take one or more DataArrays as input, plus some parameters. And output one or more dataarrays (not Datasets). Units and metadata is handled. This impacts ``sdba.processing.adapt_freq`` especially.
* Add unit handling in ``sdba``. Most parameters involving quantities are now expecting strings (and not numbers). Adjustment objects will ensure ref, hist and sim all have the same units (taking ref as reference).
* The Adjustment` classes of ``xclim.sdba`` have been refactored into 2 categories:

    - ``TrainAdjust`` objects (most of the algorithms), which are created **and** trained in the same call:
      ``obj = Adj.train(ref, hist, **kwargs)``. The ``.adjust`` step stays the same.

    - ``Adjust`` objects (only ``NpdfTransform``), which are never initialized. Their ``adjust``
      class method performs all the work in one call.
* ``snowfall_approximation`` used a `"<"` condition instead of `"<="` to determine the snow fraction based on the freezing point temperature. The new version sticks to the convention used in the Canadian Land Surface Scheme (CLASS).
* Removed the `"gis"`, `"docs"`, `"test"` and `"setup"`extra dependencies from ``setup.py``. The ``dev`` recipe now includes all tools needed for xclim's development.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``snowfall_approximation`` has gained support for new estimation methods used in CLASS: 'brown' and 'auer'.
* A ``ValidationError`` will be raised if temperature units are given as 'deg C', which is misinterpreted by pint.
* Functions computing run lengths (sequences of consecutive `"True"` values) now take the ``index`` argument. Possible values are ``first`` and ``last``, indicating which item in the run should be used to index the run length. The default is set to `"first"`, preserving the current behavior.
* New ``sdba_encode_cf`` option to workaround a cftime/xarray performance issue when using dask.

New indicators
^^^^^^^^^^^^^^
* ``effective_growing_degree_days`` indice returns growing degree days using dynamic start and end dates for the growing season (based on Bootsma et al. (2005)). This has also been wrapped as an indicator.
* ``qian_weighted_mean_average`` (based on Qian et al. (2010)) is offered as an alternate method for determining the start date using a weighted 5-day average (``method="qian"``). Can also be used directly as an indice.
* ``cold_and_dry_days`` indicator returns the number of days where the mean daily temperature is below the 25th percentile and the mean daily precipitation is below the 25th percentile over period. Added as ``CD`` indicator to ICCLIM module.
* ``warm_and_dry_days`` indicator returns the number of days where the mean daily temperature is above the 75th percentile and the mean daily precipitation is below the 25th percentile over period. Added as ``WD`` indicator to ICCLIM module.
* ``warm_and_wet_days`` indicator returns the number of days where the mean daily temperature is above the 75th percentile and the mean daily precipitation is above the 75th percentile over period. Added as ``WW`` indicator to ICCLIM module.
* ``cold_and_wet_days`` indicator returns the number of days where the mean daily temperature is below the 25th percentile and the mean daily precipitation is above the 75th percentile over period. Added as ``CW`` indicator to ICCLIM module.
* ``calm_days`` indicator returns the number of days where surface wind speed is below threshold.
* ``windy_days`` indicator returns the number of days where surface wind speed is above threshold.

Bug fixes
^^^^^^^^^
* Various bug fixes in bootstrapping:
   - in ``percentile_bootstrap`` decorator, fix the popping of bootstrap argument to propagate in to the function call.
   - in ``bootstrap_func``, fix some issues with the resampling frequency which was not working when anchored.
* Made argument ``thresh`` of ``sdba.LOCI`` required, as not giving it raised an error. Made defaults explicit in the adjustments docstrings.
* Fixes in ``sdba.processing.adapt_freq`` and ``sdba.nbutils.vecquantiles`` when handling all-nan slices.
* Dimensions in a grouper's ``add_dims`` are now taken into consideration in function wrapped with ``map_blocks/groups``. This feature is still not fully tested throughout ``sdba`` though, so use with caution.
* Better dtype preservation throughout ``sdba``.
* "constant" extrapolation in the quantile mappings' adjustment is now padding values just above and under the target's max and min, instead of ``±np.inf``.
* Fixes in ``sdba.LOCI`` for the case where a grouping with additionnal dimensions is used.

Internal Changes
^^^^^^^^^^^^^^^^
* The behaviour of ``xclim.testing._utils.getfile`` was adjusted to launch file download requests for web-hosted md5 files for every call to compare against local test data.
  This was done to validate that locally-stored test data is identical to test data available online, without resorting to git-based actions. This approach may eventually be revised/optimized in the future.

0.28.1 (2021-07-29)
-------------------

Announcements
^^^^^^^^^^^^^
* The `xclim` binary package available on conda-forge will no longer supply ``clisops`` by default. Installation of ``clisops`` must be performed explicitly to preserve subsetting and bias correction capabilities.

New indicators
^^^^^^^^^^^^^^
* ``snow_depth`` indicator returns the mean snow depth over period. Added as ``SD`` to ICCLIM module.

Internal Changes
^^^^^^^^^^^^^^^^
* Minor modifications to many function call signatures (type hinting) and docstrings (numpy docstring compliance).

0.28.0 (2021-07-07)
-------------------

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^
* Various bug fixes in sdba :

    - in ``QDM.adjust``, fix bug occuring with coords of 'object' dtype and ``interp='nearest'``.
    - in ``nbutils.quantiles``, fix dtype bug when using ``float32`` data.
    - raise a proper error when ``ref`` and ``hist`` have a different calendar for map_blocks-backed adjustments.

Breaking changes
^^^^^^^^^^^^^^^^
* ``spatial_analogs`` does not support sequence of ``dist_dim`` anymore. Users are responsible for stacking dimensions prior to calling ``spatial_analogs``.

New indicators
^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Rewrite of nearly all adjustment methods in ``sdba``, with use of ``xr.map_blocks`` to improve scalability with dask. Rewrite of some parts of the algorithms with numba-accelerated code.
* "GFWED" specifics for fire weather computation implemented back into the FWI module. Outputs are within 3% of GFWED data.
* Addition of the `run_length_ufunc` option to control which run length algorithm gets run. Defaults stay the same (automatic switch dependent of the input array : the 1D version is used with non-dask arrays with less than 9000 points per slice).
* Indicator modules built from YAML can now use custom indices. A mapping or module of them can be given to ``build_indicator_module_from_yaml`` with the ``indices`` keyword.
* Virtual submodules now include an `iter_indicators` function to iterate over the pairs of names and indicator objects in that module.
* The indicator string formatter now accepts a "r" modifier which passes the raw strings instead of the adjective version.
* Addition of the `sdba_extra_output` option to adds extra diagnostic variables to the outputs of Adjustment objects. Implementation of `sim_q` in QuantileDeltaMapping and `nclusters` in ExtremeValues.

Breaking changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^
* `atmos.corn_heat_units` computes the daily temperature-based index for corn growth.
* New indices and indicators for `tx_days_below`, `tg_days_above`, `tg_days_below`, and `tn_days_above`.
* `atmos.humidex` returns the Canadian *humidex*, an indicator of perceived temperature account for relative humidity.
* `multiday_temperature_swing` indice for returning general statistics based on spells of doubly-thresholded temperatures (Tmin < T1, Tmax > T2).
* New indicators `atmos.freezethaw_frequency`, `atmos.freezethaw_spell_mean_length`, `atmos.freezethaw_spell_max_length` for statistics of Tmin < 0 degC and Tmax > 0 deg C days now available (wrapped from `multiday_temperature_swing`).
* `atmos.wind_chill_index` computes the daily wind chill index. The default is similar to what Environment and Climate Change Canada does, options are tunable to get the version of the National Weather Service.

Internal Changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^
* `xclim` no longer supports Python3.6. Code conventions and new features from Python3.7 (`PEP 537 Features <https://www.python.org/dev/peps/pep-0537/#features-for-3-7>`_) are now accepted.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^
* `xclim` now requires `xarray>=0.17`.
* Virtual submodules `icclim` and `anuclim` are not available at the top level anymore (only through `xclim.indicators`).
* Virtual submodules `icclim` and `anuclim` now provide *Indicators* and not indices.
* Spatial analog methods "KLDIV" and "Nearest Neighbor" now require `scipy>=1.6.0`.

Bug fixes
^^^^^^^^^
* `from_string` object creation in sdba has been removed. Now replaced with use of a new dependency, `jsonpickle`.

Internal Changes
^^^^^^^^^^^^^^^^
* `pre-commit` linting checks now run formatting hook `black==21.4b2`.
* Code cleaning (more accurate call signatures, more use of https links, docstring updates, and typo fixes).

0.25.0 (2021-03-31)
-------------------

Announcements
^^^^^^^^^^^^^
* Deprecation: Release 0.25.0 of `xclim` will be the last version to explicitly support Python3.6 and `xarray<0.17.0`.

New indicators
^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `generic.count_domain` counts values within low and high thresholds.
* `run_length.season` returns a dataset storing the start, end and length of a *season*.
* Fire Weather indices now support dask-backed data.
* Objects from the `xclim.sdba` submodule can be created from their string repr or from the dataset they created.
* Fire Weather Index submodule replicates the R code of `cffdrs`, including fire season determination and overwintering of the drought_code.
* New `run_bounds` and `keep_longest_run` utilities in `xclim.indices.run_length`.
* New bias-adjustment method: `PrincipalComponent` (based on Hnilica et al. 2017 https://doi.org/10.1002/joc.4890).

Internal changes
^^^^^^^^^^^^^^^^
* Small changes in the output of `indices.run_length.rle`.

0.24.0 (2021-03-01)
-------------------

New indicators
^^^^^^^^^^^^^^
* `days_over_precip_thresh`, `fraction_over_precip_thresh`, `liquid_precip_ratio`, `warm_spell_duration_index`,  all from eponymous indices.
* `maximum_consecutive_warm_days` from indice `maximum_consecutive_tx_days`.

Breaking changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Indicator now have docstrings generated from their metadata.
* Units and fixed choices set are parsed from indice docstrings into `Indicator.parameters`.
* Units of indices using the `declare_units` decorator are stored in `indice.in_units` and `indice.out_units`.
* Changes to `Indicator.format` and `Indicator.json` to ensure the resulting json really is serializable.

Internal changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^
* The unit handling change resolved a bug that prevented the use of `xr.set_options(keep_attrs=True)` with indices.

0.23.0 (2021-01-22)
-------------------

Breaking changes
^^^^^^^^^^^^^^^^
* Renamed indicator `atmos.degree_days_depassment_date` to `atmos.degree_days_exceedance_date`.
* In `degree_days_exceedance_date` : renamed argument `start_date` to `after_date`.
* Added cfchecks for Pr+Tas-based indicators.
* Refactored test suite to now be available as part of the standard library installation (`xclim.testing.tests`).
* Running `pytest` with `xdoctest` now requires the `rootdir` to point at `tests` location (`pytest --rootdir xclim/testing/tests/ --xdoctest xclim`).
* Development checks now require working jupyter notebooks (assessed via the `pytest --nbval` command).

New indicators
^^^^^^^^^^^^^^
* `rain_approximation` and `snowfall_approximation` for computing `prlp` and `prsn` from `pr` and `tas` (or `tasmin` or `tasmax`) according to some threshold and method.
* `solid_precip_accumulation` and `liquid_precip_accumulation` now accept a `thresh` parameter to control the binary snow/rain temperature threshold.
* `first_snowfall` and `last_snowfall` to compute the date of first/last snowfall exceeding a threshold in a period.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New `kind` entry in the `parameters` property of indicators, differentiating between [optional] variables and parameters.
* The git pre-commit hooks (`pre-commit run --all`) now clean the jupyter notebooks with `nbstripout` call.

Bug fixes
^^^^^^^^^
* Fixed a bug in `indices.run_length.lazy_indexing` that occurred with 1D coords and 0D indexes when using the dask backend.
* Fixed a bug with default frequency handling affecting `fit` indicator.
* Set missing method to 'skip' for `freq_analysis` indicator.
* Fixed a bug in `ensembles._ens_align_datasets` that occurred when inputs are `.nc` filepaths but files lack a time dimension.

Internal changes
^^^^^^^^^^^^^^^^
* `core.cfchecks.check_valid` now accepts a sequence of strings as its `expected` argument.
* Clean up in the tests to speed up testing. Addition of a marker to include "slow" tests when desired (`-m slow`).
* Fixes in the tests to support `sklearn>=0.24`, `clisops>=0.5` and build xarray@master against python 3.7.
* Moved the testing suite to within xclim and simplified `tox` to manage its own tempdir.
* Indicator class now has a `default_freq` method.


0.22.0 (2020-12-07)
-------------------

Breaking changes
^^^^^^^^^^^^^^^^
* Statistical functions (`frequency_analysis`, `fa`, `fit`, `parametric_quantile`) are now solely accessible via `indices.stats`.

New indicators
^^^^^^^^^^^^^^
* `atmos.degree_days_depassment_date`, the day of year when the degree days sum exceeds a threshold.

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^
* Fixed bug that prevented the use of `xclim.core.missing.MissingBase` and subclasses with an indexer and a cftime datetime coordinate.
* Fixed issues with metadata handling in statistical indices.
* Various small fixes to the documentation (re-establishment of some internally and externally linked documents).

Internal changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^
* Statistical functions (`frequency_analysis`, `fa`, `fit`, `parametric_quantile`) moved from `indices.generic` to `indices.stats` to make them more visible.

New indicators
^^^^^^^^^^^^^^

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New xclim.testing.open_dataset method to read data from the remote testdata repo.
* Added a notebook, `ensembles-advanced.ipynb`, to the documentation detailing ensemble reduction techniques and showing how to make use of built-in figure-generating commands.
* Added a notebook, `frequency_analysis.ipynb`, with examples showcasing frequency analysis capabilities.

Bug fixes
^^^^^^^^^
* Fixed a bug in the attributes of `frost_season_length`.
* `indices.run_length` methods using dates now respect the array's calendar.
* Worked around an xarray bug in sdba.QuantileDeltaMapping when multidimensional arrays are used with linear or cubic interpolation.

Internal changes
^^^^^^^^^^^^^^^^^

0.20.0 (2020-09-18)
-------------------

Breaking changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^
* `atmos.max_pr_intensity` acting on hourly data.
* `atmos.wind_vector_from_speed`, also the `wind_speed_from_vector` now also returns the "wind from direction".
* Richards-Baker flow flashiness indicator (`xclim.land.rb_flashiness_index`).
* `atmos.max_daily_temperature_range`.
* `atmos.cold_spell_frequency`.
* `atmos.tg_min` and `atmos.tg_max`.
* `atmos.frost_season_length`, `atmos.first_day_above`. Also, `atmos.consecutive_frost_days` now takes a `thresh` argument (default : 0 degC).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^
The ICCLIM module was identified as `icclim` in the documentation but the module available under `ICCLIM`. Now `icclim == ICCLIM` and `ICCLIM will be deprecated in a future release`.


Internal changes
^^^^^^^^^^^^^^^^
* `xclim.subset` now attempts to load and expose the functions of `clisops.core.subset`. This is an API workaround preserving backwards compatibility.
* Code styling now conforms to the latest release of black (v0.20.8).
* New `IndicatorRegistrar` class that takes care of adding indicator classes and instances to the
  appropriate registries. `Indicator` now inherits from it.


0.19.0 (2020-08-18)
-------------------

Breaking changes
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New `ensembles.kkz_reduce_ensemble` method to select subsets of an ensemble based on the KKZ algorithm.
* Create new Indicator `Daily`, `Daily2D` subclasses for indicators using daily input data.
* The `Indicator` class now supports outputing multiple indices for the same inputs.
* `xclim.core.units.declare_units` now works with indices outputting multiple DataArrays.
* Doctests now make use of the `xdoctest_namespace` in order to more easily access modules and testdata.

Bug fixes
^^^^^^^^^
* Fix `generic.fit` dimension ordering. This caused errors when "time" was not the first dimension in a DataArray.

Internal changes
^^^^^^^^^^^^^^^^
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
