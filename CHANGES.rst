=========
Changelog
=========

v0.44.0 (2023-06-23)
--------------------
Contributors to this version: Éric Dupuis (:user:`coxipi`), Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Ludwig Lierhammer (:user:`ludwiglierhammer`), David Huard (:user:`huard`).

Announcements
^^^^^^^^^^^^^
* `xclim: xarray-based climate data analytics` has been published in the Journal of Open Source Software (`DOI:10.21105/joss.05415 <https://doi.org/10.21105/joss.05415>`_). Users can now make use of the `Cite this repository` button in the sidebar for academic purposes. Many thanks to our core developers and user base for their fine contributions over the years! (:issue:`95`, :pull:`250`).
* `xclim` now officially supports Python3.11. (:pull:`1388`).

New indicators
^^^^^^^^^^^^^^
* Several new indices and indicators:
    * ``snowfall_{frequency | intensity}`` for calculating the {percentage of | mean snowfall intensity on} days with snowfall above a threshold. (:issue:`1352`, :pull:`1358`)
    * ``{sfcWind | sfcWindmax}_{max | mean | min}`` for calculating the {max | mean | min} daily {mean | max} wind speed. (:issue:`1352`, :pull:`1358`)
    * ``{precip | liquid_precip | solid_precip}_average}`` for calculating the mean daily {total precipitation | liquid precipitation | solid precipitation } amount. (:issue:`1352`, :pull:`1358`)
    * ``{cold | dry}_spell_max_length`` for calculating maximum length of {cold | dry} spell events. (:issue:`1352`, :pull:`1359`).
    * ``dry_spell_frequency`` for calculating total number of dry spells. (:issue:`1352`, :pull:`1359`).
    * ``hardiness_zones`` with supported methods `"usda"` (USA) and `"anbg"` (Australia) for calculating hardiness classifications from climatologies. (:issue:`1290`, :pull:`1396`).
* New indicator ``late_frost_days`` for calculating the number of days where the daily minimum temperature is below a threshold over a given time period. (:issue:`1352`, :pull:`1361`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``xclim.sdba.processing.escore`` performance was improved with a faster runtime (:pull:`1360`).
* New generic function (``flux_and_rate_converter``) converting flux to a rate (and vice-versa) using a density. ``snw_to_snd`` and ``snd_to_snw`` were refactored using this function. (:issue:`1352`, :pull:`1358`)
* New function (``prsn_to_prsnd``) to convert snowfall flux ([mass] / [area] / [time]) to snowfall rate ([length] / [time]) using snow density ([mass] / [volume]). (:issue:`1352`, :pull:`1358`)
* New variables: Snowfall rate ``prsnd`` and surface maximum wind speed ``sfcWindmax``. (:issue:`1352`, :pull:`1358`).
* Docstring for ``freq`` links to pandas offset aliases documentation. (:issue:`1310`, :pull:`1392`).
* New function ``xclim.indces.run_length.extract_events`` for determining runs whose starting and stopping points are defined through run length conditions. (:pull:`1256`).
* Stats functions `frequency_analysis` now takes `method` parameter to select other fitting methods such as PWM or MOM. (:issue:`1168`, :pull:`1398`).
* ``xclim.indices.frost_days`` now accepts an ``**indexer`` parameter for calculating frost days over a temporal subset of the given dataset. (:issue:`1352`, :pull:`1361`).
* New function ``xclim.indices.generic.get_zones`` attributing a histogram bin index (a zone) to each value in an input array. (:issue:`1290`, :pull:`1396`).

Bug fixes
^^^^^^^^^
* Fixed a bug in ``xclim.core.calendar.time_bnds`` when using ``DataArrayResample`` objects, caused by an upstream change in xarray 2023.5.0. (:issue:`1368`, :pull:`1377`).
* ``ensembles.change_significance`` will returns NaNs when the input values are all NaNs, instead of failing. (:issue:`1379`, :pull:`1380`).
* Accelerated import of xclim by caching the compilation of `guvectorize` functions. (:pull:`1378`).
* Fixed many issues with ``xclim.indices.helpers.cosine_of_solar_zenith_angle``, the signature changed. (:issue:`1110`, :pull:`1399`).

Internal changes
^^^^^^^^^^^^^^^^
* In order to ensure documentation can be rebuilt at a later time, errors raised by `sphinx` linkcheck are now set to be ignored when building the documentation. (:pull:`1375`).
* With the publication of `xclim`, the code repository now offers a `CITATION.cff` configuration for users to properly cite the software (APA formatted and raw BibTeX) for academic purposes. (:issue:`95`, :pull:`250`).
* Logging messages emitted when redefining units via `pint` (caused by `logging` interactions with dependencies) have been silenced. (:issue:`1373`, :pull:`1384`).
* Fixed some annotations and `dev` recipe dependencies issues to allow for the development of xclim inside a python3.11 environment. (:issue:`1376`, :pull:`1381`).
* The deprecated `mamba-org/provision-with-micromamba` GitHub Action has been replaced with `mamba-org/setup-micromamba`. (:pull:`1388`).
* `xclim` GitHub CI workflows now run builds against Python3.11. (:pull:`1388`).
* In indices, verify that all parameters of type `Quantified` that have a default value have their dimension declared. (:issue:`1293`, :pull:`1393`).
* Updated `roy_extremeprecip_2021` to the newly published paper. (:pull:`1394`).
* Two new GitHub CI Actions have been added to the existing Workflows (:pull:`1390`):
    * `actions/add-to-project`: Automatically adds issues to the `xclim` project.
    * `saadmk11/github-actions-version-updater`: Updates GitHub Action versions in all Workflows (triggered monthly).
* Added `method` parameter to `frequency_analysis` and `fa`. (:issue:`1168`, :pull:`1398`).

Breaking changes
^^^^^^^^^^^^^^^^
* Signature of `hot_spell_{frequency | max_length | total_length}` : `thresh_tasmax` modified to `thresh`. (:issue:`1352`, :pull:`1359`).

v0.43.0 (2023-05-09)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Ludwig Lierhammer (:user:`ludwiglierhammer`), Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), Alexis Beaupré (:user:`Beauprel`), Éric Dupuis (:user:`coxipi`).

Announcements
^^^^^^^^^^^^^
* `xclim` has passed the peer-review process and been officially accepted as a project associated with both `pyOpenSci <https://www.pyopensci.org>`_ and `PANGEO <https://pangeo.io/>`_. Additionally, `xclim` has been accepted to be published in the `Journal of Open Source Software <https://joss.theoj.org/>`_. Our review process can be consulted here: `PyOpenSci Software Review <https://github.com/pyOpenSci/software-review/issues/73>`_. (:pull:`1350`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New ``xclim.sdba`` measure ``xclim.sdba.measures.taylordiagram``. (:pull:`1360`).

New indicators
^^^^^^^^^^^^^^
* ``ensembles.change_significance`` now supports the Brown-Forsythe test. (:pull:`1292`).

Bug fixes
^^^^^^^^^
* Fixed a bug in the `pyproject.toml` configuration that excluded the changelog (`CHANGES.rst`) from the packaged source distribution. (:pull:`1349`).
* When summing an all-`NaN` period with `resample`, `xarray` v2023.04.0 now returns `NaN`, whereas earlier versions returned `0`. This broke ``fraction_over_precip_thresh``, but is now fixed. (:pull:`1354`, :issue:`1337`).
* In ``xclim.sdba``'s Quantile Delta Mapping algorithm, the quantiles of the simulation to adjust were computed slightly differently than when creating the adjustment factor. The ``xclim.sdba.utils.rank`` function has been fixed to return "percentage-ranks" (quantiles) in the proper range. (:issue:`1334`, :pull:`1355`).
* The radiation converters (``longwave_upwelling_radiation_from_net_downwelling`` and ``shortwave_upwelling_radiation_from_net_downwelling``) were hard-coded to redefine output units as `W m-2`, regardless of input units, so long as unit dimensions checks cleared. Units are now set directly from inputs. (:issue:`1365`, :pull:`1366`).

Breaking changes
^^^^^^^^^^^^^^^^
* Many previously deprecated indices and indicators have been removed from `xclim` (:pull:`1318`), with replacement indicators suggested as follows:
    * ``xclim.indicators.atmos.first_day_above`` ->  ``xclim.indicators.atmos.first_day_{tn | tg | tx}_above``
    * ``xclim.indicators.atmos.first_day_below`` -> ``xclim.indicators.atmos.first_day_{tn | tg | tx}_below``
    * ``xclim.indicators.land.continuous_snow_cover_end`` -> ``xclim.indicators.land.snd_season_end``
    * ``xclim.indicators.land.continuous_snow_cover_start`` -> ``xclim.indicators.land.snd_season_start``
    * ``xclim.indicators.land.fit`` -> ``xclim.indicators.generic.fit``
    * ``xclim.indicators.land.frequency_analysis`` -> ``xclim.indicators.generic.return_level``
    * ``xclim.indicators.land.snow_cover_duration`` -> ``xclim.indicators.land.snd_season_length``
    * ``xclim.indicators.land.stats`` -> ``xclim.indicators.generic.stats``
    * ``xclim.indices.continuous_snow_cover_end`` -> ``xclim.indices.snd_season_end``
    * ``xclim.indices.continuous_snow_cover_start`` -> ``xclim.indices.snd_season_start``
    * ``xclim.indices.snow_cover_duration`` -> ``xclim.indices.snd_season_length``
* Several `_private` functions within ``xclim.indices.fire._cffwis`` that had been exposed publicly have now been rendered as hidden functions. Affected functions are: ``_day_length``, ``_day_length_factor``, ``_drought_code``, ``_duff_moisture_code``, ``_fine_fuel_moisture_code``, ``_overwintering_drought_code``. (:pull:`1159`, :pull:`1369`).

Internal changes
^^^^^^^^^^^^^^^^
* The testing suite has been adjusted to ensure calls are made to existing functions using non-deprecated syntax. The volume of warnings emitted during testing has been significantly reduced. (:pull:`1318`).
* In order to follow best practices and reduce the installed size of the `xclim` wheel, the `tests` folder containing the testing suite has been split from the package and placed in the top-level of the code repository. (:issue:`1348`, :pull:`1349`, suggested from `PyOpenSci Software Review <https://github.com/pyOpenSci/software-review/issues/73>`_). Submodules that were previously called within ``xclim.testing.tests`` have been refactored as follows:
    * ``xclim.testing.tests.data`` → ``xclim.testing.helpers``
    * ``xclim.testing.tests.test_sdba.utils`` → ``xclim.testing.sdba_utils``
* Added a "Conventions" section to the README. (:issue:`1342`, :pull:`1351`).
* New helper function ``xclim.testing.helpers.test_timeseries`` for generating timeseries objects with specified variable names and units. (:pull:`1356`).
* `tox` recipes and documentation now refer to the official build of `SBCK`, available on PyPI. (:issue:`1362`, :pull:`1364`).
* Excluded some URLs from `sphinx linkcheck` that were causing issues on ReadTheDocs. (:pull:`1364`).
* Tagged versions of `xclim-testdata` now follow a `calendar-based versioning <https://calver.org/>`_ scheme for easier determination of compatibility between `xclim` and testing data. (:pull:`1367`, `xclim-testdata discussion <https://github.com/Ouranosinc/xclim-testdata/pull/24>`_).
* `flake8`, `pycodestyle`, and `pydocstyle` checks have been significantly changed in order to clean up the code base of redundant `# noqa` markers. Linting checks for Makefile and `tox` recipes have been synchronized as well. (:pull:`1369`).
* `flake8` plugin `flake8-alphabetize` has been added to development recipes in order to check order of `__all__` entries and Exceptions. (:pull:`1369`).
* Corrected translations of ``cold_spell_{frequency | days}`` (:pull:`1372`).

v0.42.0 (2023-04-03)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`), Éric Dupuis (:user:`coxipi`), Pascal Bourgault (:user:`aulemahal`).

Announcements
^^^^^^^^^^^^^
* `xclim` now supports testing against tagged versions of `Ouranosinc/xclim-testdata <https://github.com/Ouranosinc/xclim-testdata>`_ in order to support older versions of `xclim`. For more information, see the `Contributing Guide <https://xclim.readthedocs.io/en/stable/contributing.html>`_ for more details. (:pull:`1339`).
* `xclim v0.42.0` will be the last version to explicitly support Python3.8. (:issue:`1268`, :pull:`1344`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Two previously private functions for selecting a day of year in a time series when performing calendar conversions are now exposed. (:issue:`1305`, :pull:`1317`). New functions are:
    * ``xclim.core.calendar.yearly_interpolated_doy``
    * ``xclim.core.calendar.yearly_random_doy``
* `scipy` is no longer pinned below v1.9 and `lmoments3>=1.0.5` is now a core dependency and installed by default with `pip`. (:issue:`1142`, :pull:`1171`).
* Fix bug on number of bins in ``xclim.sdba.propeties.spatial_correlogram``. (:pull:`1336`)
* Add `resample_before_rl` argument to control when resampling happens in `maximum_consecutive_{frost|frost_free|dry|tx}_days` and in heat indices (in `_threshold`)  (:issue:`1329`, :pull:`1331`)
* Add ``xclim.ensembles.make_criteria`` to help create inputs for the ensemble-reduction methods. (:issue:`1338`, :pull:`1341`).

New indicators
^^^^^^^^^^^^^^
* Rain season index implemented (default parameters for West Africa). (:issue:`842`, :pull:`1256`)

Bug fixes
^^^^^^^^^
* Warnings emitted from regular usage of some indices (``snowfall_approximation`` with ``method="brown"``, ``effective_growing_degree_days``) due to successive ``convert_units_to`` calls within their logic have been silenced. (:pull:`1319`).
* Fixed a bug that prevented the use of the `sdba_encode_cf` option with xarray 2023.3.0 (:pull:`1333`).
* Fixed bugs in ``xclim.core.missing`` and ``xclim.sdba.base.Grouper`` when using pandas 2.0. (:pull:`1344`).

Breaking changes
^^^^^^^^^^^^^^^^
* The call signatures for ``xclim.ensembles.create_ensemble`` and ``xclim.ensembles._base._ens_align_dataset`` have been deprecated. Calls to these functions made with the original signature will emit warnings. Changes will become breaking in `xclim>=0.43.0`.(:issue:`1305`, :pull:`1317`). Affected variable:
    * `mf_flag` (bool) -> `multifile` (bool)
* The indice and indicator for ``last_spring_frost`` has been modified to use ``tasmin`` by default, reflecting its docstring and literature definition (:issue:`1324`, :pull:`1325`).
* following indices now accept the `op` argument for modifying the threshold comparison operator (:pull:`1325`):
    * ``snw_season_length``, ``snd_season_length``, ``growing_season_length``, ``frost_season_length``, ``frost_free_season_length``, ``rprcptot``, ``daily_pr_intensity``
* In order to support older environments, `pandas` is now conditionally pinned below v2.0 when installing `xclim` on systems running Python3.8. (:pull:`1344`).

Bug fixes
^^^^^^^^^
* ``xclim.indices.run_length.last_run`` nows works when ``freq`` is not ``None``. (:issue:`1321`, :pull:`1323`).

Internal changes
^^^^^^^^^^^^^^^^
* Added `xclim` to the `ouranos Zenodo community <https://zenodo.org/communities/ouranos/>`_ . (:pull:`1313`).
* Significant documentation adjustments. (:issue:`1305`, :pull:`1308`):
    * The CONTRIBUTING page has been moved to the top level of the repository.
    * Information concerning the licensing of xclim is clearly indicated in README.
    * `sphinx-autodoc-typehints` is now used to simplify call signatures generated in documentation.
    * The SDBA module API is now found with the rest of the User API documentation.
    * `HISTORY.rst` has been renamed `CHANGES.rst`, to follow `dask`-like conventions.
    * Hyperlink targets for individual `indices` and `indicators` now point to their entries under `API` or `Indices`.
    * Module-level docstrings have migrated from the library scripts directly into the documentation RestructuredText files.
    * The documentation now includes a page explaining the reasons for developing `xclim` and a section briefly detailing similar and related projects.
    * Markdown explanations in some Jupyter Notebooks have been edited for clarity
* Removed `Mapping` abstract base class types in call signatures (`dict` variables were always expected). (:pull:`1308`).
* Changes in testing setup now prevent ``test_mean_radiant_temperature`` from sometimes causing a segmentation fault. (:issue:`1303`, :pull:`1315`).
* Addressed a formatting bug that caused `Indicators` with multiple variables returned to not be properly formatted in the documentation. (:issue:`1305`, :pull:`1317`).
* `tox` now include `sbck` and `eofs` flags for easier testing of dependencies. CI builds now test against `sbck-python` @ master.  (:pull:`1328`).
* `upstream` CI tests are now run on push to master, at midnight, and can also be triggered via `workflow_dispatch`. Failures from upstream build will open issues using `xarray-contrib/issue-from-pytest-log`. (:pull:`1327`).
* Warnings from set ``_version_deprecated`` within Indicators now emit ``FutureWarning`` instead of ``DeprecationWarning`` for greater visibility. (:pull:`1319`).
* The `Graphics` section of the `Usage` notebook has been expanded upon while grammar and spelling mistakes within the notebook-generated documentation have been reduced. (:issue:`1335`, :pull:`1338`, suggested from `PyOpenSci Software Review <https://github.com/pyOpenSci/software-review/issues/73>`_).
* The Contributing guide now lists three separate subsections to help users understand the gains from optional dependencies. (:issue:`1335`, :pull:`1338`, suggested from `PyOpenSci Software Review <https://github.com/pyOpenSci/software-review/issues/73>`_).

v0.41.0 (2023-02-28)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Ludwig Lierhammer (:user:`ludwiglierhammer`), Éric Dupuis (:user:`coxipi`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New properties ``xclim.sdba.properties.decorrelation_length`` and ``xclim.sdba.properties.transition_probability``. (:pull:`1252`)

New indicators
^^^^^^^^^^^^^^
* ``ensembles.change_significance`` now supports Mann-whitney U-test and flexible ``realization``. (:pull:`1285`).
* New indices and indicators for converting from snow water equivalent to snow depth (``snw_to_snd``) and snow depth to snow water equivalent (``snd_to_snw``) using snow density [kg/m^3]. (:pull:`1271`).
* New indices and indicators for determining upwelling radiation (`shortwave_upwelling_radiation_from_net_downwelling` and `longwave_upwelling_radiation_from_net_downwelling`; CF variables `rsus` and `rlus`) from net and downwelling radiation (shortwave: `rss` and `rsds`; longwave: `rls` and `rlds`). (:pull:`1271`).
* New indice and indicator ``{snd | snw}_season_{length | start | end}`` which generalize ``snow_cover_duration`` and  ``continuous_snow_cover_{start | end}`` to allow using these functions with variable `snw` (:pull:`1275`).
* New indice and indicator (``dryness_index``) for estimating soil humidity classifications for winegrowing regions (based on Riou et al. (1994)). (:issue:`355`, :pull:`1235`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xclim` testing default behaviours have been changed (:issue:`1295`, :pull:`1297`):
   * Running `$ pytest` will no longer use `pytest-xdist` distributed testing be default (can be set with ``-n auto|logical|#``. Coverage is also no longer gathered/reported by default.
   * Running `$ tox` will now set `pytest-xdist` to use ``-n logical`` processes (with a max of 10).
   * Default behaviour for testing is to no longer always fetch `xclim-testdata`. If testdata is found in ``$HOME/.xclim_testing_data``, files will be copied to individual processes, otherwise, will be fetched as needed.
* Environment variables evaluated when running pytest have been changed (:issue:`1295`, :pull:`1297`):
   * For testing against specific branches of `xclim-testdata`: ``MAIN_TESTDATA_BRANCH`` -> ``XCLIM_TESTDATA_BRANCH``
   * The option to skip fetching of testdata (``SKIP_TEST_DATA``) has been removed
   * A new environment variable (``XCLIM_PREFETCH_TESTING_DATA``) is now available to gather `xclim-testdata` before running test ensemble (default: `False`).
   * Environment variables are now passed to `tox` on execution.

Bug fixes
^^^^^^^^^
* ``build_indicator_module_from_yaml`` now accepts a ``reload`` argument. When re-building a module that already exists, ``reload=True`` removes all previous indicator before creating the new ones. (:issue:`1192`, :pull:`1284`).
* The test for french translations of official indicators was fixed and translations for CFFWIS indices, FFDI, KDBI, DF and Jetstream metric woollings have been added or fixed. (:pull:`1271`).
* ``use_ufunc`` in ``windowed_run_count`` is now supplied with argument ``freq`` to warn users that the 1d method does not support resampling after run length operations (:issue:`1279`, :pull:`1291`).
* ``{snd|snw}_max_doy`` now avoids an error due to `xr.argmax` when there are all-NaN slices. (:pull:`1277`).

Internal changes
^^^^^^^^^^^^^^^^
* `xclim` has adopted `PEP 517 <https://peps.python.org/pep-0517/>`_ and `PEP 621 <https://peps.python.org/pep-0621/>`_ (``pyproject.toml`` using the `flit <https://flit.pypa.io/en/stable/>`_ backend) to replace the legacy ``setup.py`` used to manage package organisation and building. Many tooling configurations that already supported the ``pyproject.toml`` standard have been migrated to this file. CI and development tooling documentation has been updated to reflect these changes. (:pull:`1278`, suggested from `PyOpenSci Software Review <https://github.com/pyOpenSci/software-review/issues/73>`_).
* Documentation source files have been moved around to remove some duplicated image files. (:pull:`1278`).
* Coveralls GitHub Action removed as it did not support ``pyproject.toml``-based configurations. (:pull:`1278`).
* Add a remark about how `xclim`'s CFFWIS is different from the original 1982 implementation. (:issue:`1104`, :pull:`1284`).
* Update CI runs to use Python3.9 when examining upstream dependencies. Replace `setup-conda` action with `provision-with-micromamba` action. (:pull:`1286`).
* Update CI runs to always use `tox~=4.0` and the `latest` virtual machine images (now `ubuntu-22.04`). (:pull:`1288`, :pull:`1297`).
* `SBCK` installation command now points to the official development repository. (:pull:`1288`).
* Some references in the BibTeX were updated to point to better resources. (:pull:`1288`).
* Add a GitHub CI workflow for performing dependency security review scanning. (:pull:`1287`).
* Grammar and spelling corrections were applied to some docstrings. (:pull:`1271`).
* Added `[radiation]` (`[power] / [area]`) to list of defined acceptable units. (:pull:`1271`).
* Updated testing data used to generate the `atmosds` dataset to use more reproducibly-converted ERA5 data, generated with the `miranda` Python package. (:pull:`1269`).
* Updated testing dependencies to use `pytest-xdist>=3.2`, allowing for the new `--dist=worksteal` scheduler for distributing the pool of remaining tests across workers after individual workers have exhausted their own queues. (:pull:`1235`).
* Adding infer context to the unit conversion in of the training of ExtremeValues. (:pull:`1299`).
* Added `sphinxcontrib-svg2pdfconverter` for converting SVG graphics within documentation to PDF-compatible images. (:pull:`1296`).
* README badges for supported Python versions and repository health have been added. (:issue:`1304`, :pull:`1307`).

v0.40.0 (2023-01-13)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), David Huard (:user:`huard`), Juliette Lavoie (:user:`juliettelavoie`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Virtual modules can add variables to ``xclim.core.utils.VARIABLES`` through the new `variables` section of the yaml files. (:issue:`1129`, :pull:`1231`).
* ``xclim.core.units.convert_units_to`` can now perform automatic conversions based on the standard name of the input when needed. (:issue:`1205`, :pull:`1206`).
    - Conversion from amount (thickness) to flux (rate), using ``amount2rate`` and ``rate2amount``.
    - Conversion from amount to thickness for liquid water quantities, using the new ``amount2lwethickness`` and ``lwethickness2amount``. This is similar to the implicit transformations enabled by the "hydro" unit context.
    - Passing ``context='infer'`` will activate the "hydro" context if the source or the target are DataArrays with a standard name that is compatible, as decided by the new ``xclim.core.units.infer_context`` function.
* New `generic` indicator realm. Now holds indicators previously meant for streamflow analysis in the `land` realm: `fit`, `return_level` (previously `freq_analysis`) and `stats`. (:issue:`1130`, :pull:`1225`).
* Thresholds and other quantities passed as parameters of indicators can now be multi-dimensional `DataArray`s. `xarray` broadcasting mechanisms will apply. These parameters are now annotated as "Quantity" in the signatures (``xclim.core.utils.Quantity``), instead of "str" as before. Attributes where such thresholds where included will now read "<an array>" (french: "<une matrice>") for these new cases. Multi-dimensional quantities are still largely unsupported, except where documented in the docstring. (:issue:`1093`, :pull:`1236`).

Breaking changes
^^^^^^^^^^^^^^^^
* Rewrite of ``xclim.core.calendar.time_bnds``. It should now be more resilient and versatile, but all ``cftime_*`` and ``cfindex_*`` functions were removed. (:issue:`74`, :pull:`1207`).
* `hydro` context is not always enabled, as it led to unwanted unit conversions. Unit conversion operations now need to explicitly declare the `hydro` context to support conversions from `kg / m2 /s` to `mm/day`. (:issue:`1208`, :pull:`1227`).
* Many previously deprecated indices and indicators have been removed from `xclim` (:pull:`1228`), with replacement indices/indicators suggested as follows:
    - ``xclim.indicators.atmos.fire_weather_indexes`` → ``xclim.indicators.atmos.cffwis_indices``
    - ``xclim.indices.freshet_start`` → ``xclim.indices.first_day_temperature_above``
    - ``xclim.indices.first_day_above`` → ``xclim.indices.first_day_temperature_above``
    - ``xclim.indices.first_day_below`` → ``xclim.indices.first_day_temperature_below``
    - ``xclim.indices.tropical_nights`` → ``xclim.indices.tn_days_above``
    - ``xclim.indices.generic.degree_days`` → ``xclim.indices.generic.cumulative_difference``
* The following *modules* have been removed (:pull:`1228`):
    - `xclim.indices.fwi` → functions migrated to `xclim.indices.fire`
    - `xclim.subset` (mock submodule) → functions migrated to `clisops.core.subset`
* Indicators ``standardized_precipitation_index`` and ``standardized_precipitation_evapotranspiration_index`` will now require ``pr_cal`` and ``wb_cal`` as keyword arguments only. (:pull:`1236`).
* The internal object ``PercentileDataArray`` has been removed. (:pull:`1236`).
* The ``xclim.testing.utils.get_all_CMIP6_variables`` and ``xclim.testing.utils.update_variable_yaml`` function were removed as the former was extremely slow and unusable. (:pull:`1258`).
* The wind speed input of ``atmos.potential_evapotranspiration`` and ``atmos.water_budget`` was renamed to ``sfcWind`` (capital W) as this is the correct CMIP6 name. (:pull:`1258`).
* Indicator `land.stats`, `land.fit` and `land.freq_analysis` are now deprecated and will be removed in version 0.43. They are being phased out in favor of generic indicators `generic.stats`, `generic.fit` and `generic.return_level` respectively. (:issue:`1130`, :pull:`1225`).

Bug fixes
^^^^^^^^^
* The weighted ensemble statistics are now performed within a context in order to preserve data attributes. (:issue:`1232`, :pull:`1234`).
* The `make docs` Makefile recipe was failing with an esoteric error. This has been resolved by splitting the `linkcheck` and `docs` steps into separate actions. (:issue:`1248`. :pull:`1251`).
* The setup step for `pytest` needed to be addressed due to the fact that files were being accessed/modified by multiple tests at a time, causing segmentation faults in some tests. This has been resolved by splitting functions into those that fetch or generate test data (under `xclim.testing.tests.data`) and the fixtures that supply accessors to them (under `xclim.testing.tests.conftest`). (:issue:`1238`, :pull:`1254`).
* Relaxed the expected output for ``test_spatial_analogs[friedman_rafsky]`` to support expected results from `scikit-learn` 1.2.0.
* The MBCn example in documentation has been fixed to properly imitate the source. (:issue:`1249`, :pull:`1250`).
* Streamflow indicators relying on indices defined in `xclim.indices.stats` were not checking input variable units. These indicators will now raise an error if input data units are not m^3/s. (:issue:`1130`, :pull:`1225`).
* Adjusted some documentation examples were not being rendered properly. (:issue:`1264`, :pull:`1266`).

Internal changes
^^^^^^^^^^^^^^^^
* Minor adjustments to GitHub Actions workflows (newest Ubuntu images, updated actions version, better CI triggering). (:pull:`1221`).
* Pint units `context` added to various operations, tests and `Indicator` attributes. (:issue:`1208`, :pull:`1227`).
* Updated article from Alavoine & Grenier (2022) within documentation. Many article reference URLs have been updated to use HTTPS where possible. (:issue:`1246`, :pull:`1247`).
* Added relevant variable dataflag checks for potential evaporation, convective precipitation, and air pressure at sea level. (:pull:`1241`).
* Documentation restructured to include `ReadMe` page (as `About`) with some minor changes to documentation titles. (:pull:`1233`).
* `xclim` development build now uses `nbqa` to effectively run black checks over notebook cells. (:pull:`1233`).
* Some `tox` recipes (``opt-slow``, ``conda``) are temporarily deactivated until a `tox>=4.0`-compatible `tox-conda` plugin is released. (:pull:`1258`).
* A notebook (``extendingxclim.ipynb``) has been updated to remove mentions of obsolete `xclim.subset` module. (:pull:`1258`).
* Merge of sdba documentation from the module and the rst files, some cleanup and addition of a section referring to GitHub issues. (:pull:`1230`).

v0.39.0 (2022-11-02)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Abel Aoun (:user:`bzah`), Éric Dupuis (:user:`coxipi`), Travis Logan (:user:`tlogan2000`), Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* The general ``xclim`` description and ReadMe have been updated to reflect recent enhancements. (:issue:`1185`, :pull:`1209`).
* Documentation now supports intersphinx mapping references within code examples via `sphinx-codeautolink` and copying of code blocks via `sphinx-copybutton`. (:pull:`1182`).
* Log-logistic distribution added to `stats.py` for use with ``standardized_precipitation_index`` and ``standardized_precipitation_evapotranspiration_index``. (:issue:`1141`, :pull:`1183`).
* New option in many indices allowing for resampling in given periods after ``run_length`` operations. (:issue:`505`, :issue:`916`, :issue:`917`, :pull:`1161`).
* New base indicator class for sdba: ``StatisticalPropertyMeasure``, those measures that also reduce the time (as a property does). (:pull:`1198`).
* ``xclim.core.calendar.common_calendar`` to find the best calendar to use when uniformizing an heterogeneous collection of data. (:pull:`1217`).
* ``xclim.ensembles.create_ensemble`` now accepts ``calendar=None``, and uses the above function to guess the best one. It also now accepts ``cal_kwargs`` to fine tune the calendar conversion. (:issue:`1190`, :pull:`1217`).
* New data check : ``xclim.core.datachecks.check_common_time`` that ensures all inputs of multivariate indicators have the same frequency (and the same time anchoring for daily and hourly data). (:issue:`1111`, :pull:`1217`).

New indicators
^^^^^^^^^^^^^^
* New indices ``first_day_temperature_{above | below}`` and indicators ``xclim.indices.first_day_{tn | tg | tx}_{above | below}``. These indices/indicators accept operator (``op``) keyword for finer threshold comparison controls. (:issue:`1175`, :pull:`1186`).
* New generic indice ``cumulative_difference`` for calculating difference between values and thresholds across time (e.g. temperature: degree-days, precipitation: moisture deficit), with or without resampling/accumulating by frequency. (:pull:`1202`).
* New spatial sdba properties and measures : ``spatial_correlogram``, ``scorr`` and ``first_eof``. The later needs the optional dependency `eofs <https://ajdawson.github.io/eofs/>`_. (:pull:`1198`).

Breaking changes
^^^^^^^^^^^^^^^^
* Indices that accept `lat` or `lon` coordinates in their call signatures will now use `cf-xarray` accessors to gather these variables in the event that they are not explicitly supplied. (:pull:`1180`). This affects the following:
    - ``huglin_index``, ``biologically_effective_degree_days``, ``cool_night_index``, ``latitude_temperature_index``, ``water_budget``, ``potential_evapotranspiration``
* ``cool_night_index`` now optionally accepts ``lat: str = "north" | "south"`` for calculating CNI over DataArrays lacking a latitude coordinate. (:pull:`1180`).
* The offset value in ``standardized_precipitation_evapotranspiration_index`` is changed to better reproduce results in the reference library ``monocongo/climate_indices``. (:issue:`1141`, :pull:`1183`).
* The ``first_day_below`` and ``first_day_above`` indices are now deprecated in order to clearly communicate the variables they act upon (:issue:`1175`, :pull:`1186`). The suggested migrations are as follows:
    - ``xclim.indices.first_day_above`` -> ``xclim.indices.first_day_temperature_above``
    - ``xclim.indices.first_day_below`` -> ``xclim.indices.first_day_temperature_below``
* The ``first_day_below`` and ``first_day_above`` atmos indicators are now deprecated in order to clearly communicate the variables they act upon (:issue:`1175`, :pull:`1186`). The suggested migrations are as follows:
    - ``xclim.atmos.first_day_above`` -> ``xclim.indices.first_day_{tn | tg | tx}_above``
    - ``xclim.atmos.first_day_below`` -> ``xclim.indices.first_day_{tn | tg | tx}_below``
* The ``degree_days`` generic indice has been deprecated in favour of ``cumulative_difference`` that is not limited only to temperature variables (:issue:`1200`, :pull:`1202`). The indices for ``atmos.{heating | cooling | growing}_degree_days`` are now built from ``generic.cumulative_difference``.
* Running `pytest` now requires the `pytest-xdist` distributed testing dependency. This library has been added to the `dev` requirements and conda environment configuration. (:pull:`1203`).
* Parameters ``reducer`` and ``window`` in ``xclim.indices.rle_statistics`` are now positional. (:pull:`1161`).
* The ``relative_annual_cycle_amplitude`` and ``annual_cycle_amplitude`` have been rewritten to match the version defined in the VALUE project, outputs will change drastically (for the better) (:pull:`1198`).
* English indicator metadata has been adjusted to remove frequencies from fields in the `long_name` of indicators. English indicators now have an explicit `title` and `abstract`. (:issue:`936`, :pull:`1123`).
* French indicator metadata translations are now more uniform and more closely follow agreed-upon grammar conventions, while also removing frequency fields in `long_name_fr`. (:issue:`936`, :pull:`1123`).
* The ``freshet_start`` indice is now deprecated in favour of ``first_day_temperature_above`` with `thresh='0 degC', window=5`. The `freshet_start` indicator is now based on ``first_day_temperature_above``, but is otherwise unaffected. (:issue:`1195`, :pull:`1196`).
* Call signatures for several indices/indicators have been modified to optionally accept `op` for manually setting threshold comparison operators (:issue:`1194`, :pull:`1197`). The affected indices and indicators as follows:
   - ``hot_spell_max_length``, ``hot_spell_frequency``, ``cold_spell_days``, ``cold_spell_frequency``, ``heat_wave_index``, ``warm_day_frequency`` (indice only), ``warm_night_frequency`` (indice only), ``dry_days``, ``wetdays``, ``wetdays_prop``.
* Cleaner ``xclim.core.calendar.parse_offset`` : fails on invalid frequencies, return implicit anchors (YS -> JAN, Y -> DEC) and implicit ``is_start_anchored`` (D -> True). (:issue:`1213`, , :pull:`1217`).

Bug fixes
^^^^^^^^^
* The docstring of ``cool_night_index`` suggested that `lat` was an optional parameter. This has been corrected. (:issue:`1179`, :pull:`1180`).
* The ``mean_radiant_temperature`` indice was accessing hardcoded `lat` and `lon` coordinates from passed DataArrays. This now uses `cf-xarray` accessors. (:pull:`1180`).
* Adopt (and adapt) unit registry declaration and preprocessors from `cf-xarray` to circumvent bugs caused by a refactor in `pint` 0.20. It also cleans the code a little bit. (:issue:`1211`, :pull:`1212`).

Internal changes
^^^^^^^^^^^^^^^^
* The documentation build now relies on `sphinx-codeautolink` and `sphinx-copybutton`. (:pull:`1182`).
* Many docstrings did not fully adhere to the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_. Fields and entries for many classes and functions have been adjusted to adhere better. (:pull:`1182`).
* The xdoctest namespace now provides access to session-scoped ``{variable}_dataset`` accessors, as well as a ``path_to_atmos_file`` object. These can be used for running doctests on all variables made in the pytest ``atmosds()`` fixture. (:pull:`1882`).
* Upgrade CodeQL GitHub Action to v2. (:issue:`1188`, :pull:`1189`).
* New generic index ``first_day_threshold_reached`` is now used to compose all ``first_day_XYZ`` indices. (:issue:`1175`, :pull:`1186`).
* In order to reduce computation footprint, the GitHub CI full testing suite and doctests are now only run once a pull request has been reviewed and approved. The number of simultaneously triggered builds has also been reduced. (:issue:`1155`, :pull:`1203`).
* ReadTheDocs now only builds full documentation (including running notebooks) when pull requests are merged to the main branch. (:issue:`1155`, :pull:`1203`).
* `xclim` now leverages `pytest-xdist` to distribute tests among Python workers and significantly speed up the testing suite. (:pull:`1203`).
* ``show_versions`` can now accept a list of dependencies so that other libraries can make use of this utility. (:pull:`1215`).
* Pull Requests now are automatically tagged (``CI``, ``docs``, ``indicators``, and/or ``sdba``) according to files modified using the `GitHub Labeler Action <https://github.com/actions/labeler>`_. (:pull:`1214`).

v0.38.0 (2022-09-06)
--------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Éric Dupuis (:user:`coxipi`), Trevor James Smith (:user:`Zeitsperre`), Abel Aoun (:user:`bzah`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Dougie Squire (:user:`dougiesquire`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Adjustment methods of `SBCK <https://github.com/yrobink/SBCK>`_ are wrapped into xclim when that package is installed. (:issue:`1109`, :pull:`1115`).
    - Wrapped SBCK tests are also properly run in the tox testing ensemble. (:pull:`1119`).
* Method ``FAO_PM98`` (based on Penman-Monteith formula) to compute potential evapotranspiration. (:pull:`1122`).
* New indices for droughts: SPI (standardized precipitations) and SPEI (standardized water budgets). (:issue:`131`, :pull:`1096`).
* Most numba functions of ``sdba.nbutils`` now use the "lazy" compilation mode. This significantly accelerates the import time of xclim. (:issue:`1135`, :pull:`1167`).
* Statistical properties and measures from ``xclim.sdba`` are now ``Indicator`` subclasses (:pull:`1149`).

New indicators
^^^^^^^^^^^^^^
* `xclim` now has the McArthur Forest Fire Danger Index and related indices under a new ``xclim.indices.fire`` module. These indices are also available as indicators. (:issue:`1152`, :pull:`1159`)
* Drought-related indicators: SPI (standardized precipitations) and SPEI (standardized water budgets). (:issue:`131`, :pull:`1096`).
* ``ensembles.create_ensembles`` now accepts a ``realizations`` argument to assign a coordinate to the "realization" axis. It also accepts a dictionary as input so that keys are used as that coordinate. (:pull:`1153`).
* ``ensembles.ensemble_percentiles``, ``ensembles.ensemble_mean_std_max_min`` and ``ensembles.change_significance`` now support weights (:pull:`1151`).
* Many generic indicators that compare arrays or against thresholds or now accept an `op` keyword for specifying the logical comparison operation to use in their calculations (i.e. `{">", ">=", "<", "<=, "!=", "=="}`). (:issue:`389`, :pull:`1157`).
    - In order to prevent user error, many of these generic indices now have a ``constrain`` variable that prevents calling an indice with an inappropriate comparison operator. (e.g. The following will raise an error: ``op=">", constrain=("<", "<=")``). This behaviour has been added to indices accepting ``op`` where appropriate.

Breaking changes
^^^^^^^^^^^^^^^^
* `scipy` has been pinned below version 1.9 until `lmoments3` can be adapted to the new API. (:issue:`1142`, :pull:`1143`).
* `xclim` now requires `xarray>=2022.06.0`. (:pull:`1151`).
* Documentation CI (ReadTheDocs) builds will now fail if there are any misconfigured pages, internal link/reference warnings, or broken external hyperlinks. (:issue:`1094`, :pull:`1131`, :issue:`1139`, :pull:`1140`, :pull:`1160`).
* Call signatures for generic indices have been reordered and/or modified to accept `op`, and optionally `constrain`, in many cases, and `condition`/`conditional`/`operation` has been renamed to `op` for consistency. (:issue:`389`, :pull:`1157`). The affected indices are as follows:
    - `get_op`, `compare`, `threshold_count`, `get_daily_events`, `count_level_crossings`, `count_occurrences`, `first_occurrence`, `last_occurrence`, `spell_length`, `thresholded_statistics`, `temperature_sum`, `degree_days`.
* All indices in `xclim.indices.generic` now use `threshold` in lieu of `thresh` for consistency. (:pull:`1157`).
* Existing function ``xclim.indices.generic.compare`` can now be used to construct operations with `op` and `constrain` variables to allow for dynamic comparisons with user input handling. (:issue:`389`, :pull:`1157`).
* Two deprecated indices have been removed from `xclim`. (:pull:`1157`):
    - ``xclim.indices._multivariate.daily_freezethaw_cycles`` -> Replaceable with the generic ``multiday_temperature_swing`` with `thresh_tasmax='0 degC'`, `thresh_tasmin='0 degC'`, `window=1`, and `op='sum'`. The indicator version (``xclim.atmos.daily_freezethaw_cycles``) is unaffected.
    - ``xclim.indices.generic.select_time`` -> Was previously moved to ``xclim.core.calendar``.
* The `clix-meta` indicator table parsing function (``xclim.core.utils.adapt_clix_meta_yaml``) has been adapted to support the new "op" operator handler. (:pull:`1157`).
* Because they have been re-implemented as ``Indicator`` subclasses, statistical properties and measures of ``xclim.sdba`` no longer preserve attributes of their inputs by default. Use ``xclim.set_options(keep_attrs=True)`` to get the previous behaviour. (:pull:`1149`).
* The ``xclim.indices.generic.extreme_temperature_range`` function has been fixed so it now does what its definition says. Results from ``xclim.indicators.cf.etr`` will change. (:issue:`1172`, :pull:`1173`).
* `xclim` now has a dedicated ``indices.fire`` submodule that houses all fire-related indices. The previous ``xclim.indices.fwi`` submodule is deprecated and will be removed in a future version. (:issue:`1152`, :pull:`1159`).
* The indicator ``xclim.indicators.atmos.fire_weather_indexes`` and indice ``xclim.indices.fire_weather_indexes`` have both been deprecated and renamed to ``cffwis_indices``. Calls using the previous naming will be removed in a future version. (:pull:`1159`).
* `xclim` now explicitly requires `pybtex` in order to generate documentation. (:pull:`1176`).

Bug fixes
^^^^^^^^^
* Fixed ``saturation_vapor_pressure`` for temperatures in other units than Kelvins (also fixes ``relative_humidity_from_dewpoint``). (:issue:`1125`, :pull:`1127`).
* Indicators that do not care about the input frequency of the data will not check the cell methods of their inputs. (:pull:`1128`).
* Fixed the signature and docstring of ``heat_index`` by changing ``tasmax`` to ``tas``. (:issue:`1126`, :pull:`1128`).
* Fixed a formatting issue with virtual indicator modules (`_gen_returns_section`) that was creating malformed `Returns` sections in `sphinx`-generated documentation. (:pull:`1131`).
* Fix ``biological_effective_degree_days`` for non-scalar latitudes, when using method "gladstones". (:issue:`1136`, :pull:`1137`).
* Fixed some ``extlink`` warnings found in `sphinx` and configured ReadTheDocs to use `mamba` as the dependency solver. (:issue:`1139`, :pull:`1140`).
* Fixed some broken hyperlinks to articles, users, and external documentation throughout the code base and jupyter notebooks. (:pull:`1160`).
* Removed some artefact reference roles introduced in :pull:`1131` that were causing LaTeX builds of the documentation to fail. (:issue:`1154`, :pull:`1156`).
* Fix ``biological_effective_degree_days`` for non-scalar latitudes, when using method "gladstones". (:issue:`1136`, :pull:`1137`).
* Fixed some ``extlink`` warnings found in `sphinx` and configured ReadTheDocs to use `mamba` as the dependency solver. (:issue:`1139`, :pull:`1140`).
* Fixed some broken hyperlinks to articles, users, and external documentation throughout the code base and jupyter notebooks. (:pull:`1160`).
* Addressed a bug that was causing `pylint` to stackoverflow by removing it from the tox configuration. `pylint` should only be called from an active environment. (:pull:`1163`)
* Fixed an issue with ``xclim.ensembles.kmeans_reduce_ensemble`` which caused it to fail when using dask arrays. (:pull:`1170`).
* Addressed a bug that was causing `pylint` to stackoverflow by removing it from the tox configuration. `pylint` should only be called from an active environment. (:pull:`1163`)

Internal changes
^^^^^^^^^^^^^^^^
* Marked a test (``test_release_notes_file_not_implemented``) that can only pass when source files are available so that it can easily be skipped on conda-forge build tests. (:issue:`1116`, :pull:`1117`).
* Split a few YAML strings found in the virtual modules that regularly issued warnings on the code checking CI steps. (:pull:`1118`).
* Function ``xclim.core.calendar.build_climatology_bounds`` now exposed via `__all__`. (:pull:`1146`).
* Clarifications added to docstring of ``xclim.core.bootstrapping.bootstrap_func``. (:pull:`1146`).
* Bibliographic references for supporting scientific articles are now found in a bibtex file (`docs/references.bib`). These are now made available within the generated documentation using ``sphinxcontrib-bibtex``. (:issue:`1094`, :pull:`1131`).
* Added information URLs to ``setup.py`` in order to showcase issue tracker and other sites on PyPI page (:pull:`1156`).
* Configured the LaTeX build of the documentation to ignore the custom bibliographies, as they were redundant in the generated PDF. (:pull:`1158`).
* Run length encoding (``xclim.indices.run_length.rle``) has been optimized. (:issue:`956`, :pull:`1122`).
* Added a `sphinx-build -b linkcheck` step to the `tox`-based `"docs"` build as well as to the ReadTheDocs configuration. (:pull:`1160`).
* `pylint` is now setup to use a `pylintrc` file, allowing for more granular control of warnings and exceptions. Many errors are still present, so addressing them will need to occur gradually. (:pull:`1163`).
* The generic indices `count_level_crossings`, `count_occurrences`, `first_occurrence`, and `last_occurrence` are now fully tested. (:pull:`1157`).
* Adjusted the ANUCLIM indices by removing "ANUCLIM" from their titles, modifying their docstrings, and handling `"op"` input in a more user-friendly way. (:issue:`1055`, :pull:`1169`).
* Documentation for fire-based indices/indicators has been reorganized to reflect the new submodule structure. (:pull:`1159`).

v0.37.0 (2022-06-20)
--------------------
Contributors to this version: Abel Aoun (:user:`bzah`), Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Ludwig Lierhammer (:user:`ludwiglierhammer`).

Announcements
^^^^^^^^^^^^^
* `xclim` is now compliant with `PEP 563 <https://peps.python.org/pep-0563>`_. Python3.10-style annotations are now permitted. (:issue:`1065`, :pull:`1071`).
* `xclim` is now fully compatible with `xarray`'s `flox`-enabled ``GroupBy`` and ``resample`` operations. (:pull:`1081`).
* `xclim` now (properly) enforces docstring compliance checks using `pydocstyle` with modified `numpy`-style docstrings. Docstring errors will now cause build failures. See the `pydocstyle documentation <http://www.pydocstyle.org/en/stable/error_codes.html>`_ for more information. (:pull:`1074`).
* `xclim` now uses GitHub Actions to manage patch version bumping. Merged Pull Requests that modify `xclim` code now trigger version-bumping automatically when pushed to the main development branch. Running `$ bump2version patch` within development branches is no longer necessary. (:pull:`1102`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Add "Celsius" to aliases of "celsius" unit. (:issue:`1067`, :pull:`1068`).
* All indicators now have indexing enabled, except those computing statistics on spells. (:issue:`1069`, :pull:`1070`).
* A convenience function for returning the version numbers for relevant xclim dependencies (``xclim.testing.show_versions``) is now offered. (:pull:`1073`).
    - A CLI version of this function is also available from the command line (`$ xclim show_version_info`). (:pull:`1073`).
* New "keep_attrs" option to control the handling of the attributes within the indicators. (:issue:`1026`, :pull:`1076`).
* Added a notebook showcasing some simple examples of Spatial Analogues. (:issue:`585`, :pull:`1075`).
* ``create_ensembles`` now accepts a glob string to find datasets. (:pull:`1081`).
* Improved percentile based indicators metadata with the window, threshold and climatology period used to compute percentiles. (:issue:`1047`, :pull:`1050`).
* New ``xclim.core.calendar.construct_offset``, the inverse operation of ``parse_offset``. (:pull:`1090`).
* Rechunking operations in ``xclim.indices.run_length.rle`` are now synchronized with dask's options. (:pull:`1090`).
* A mention of the "missing" checks and options is added to the history attribute of indicators, where appropriate. (:issue:`1100`, :pull:`1103`).

Breaking changes
^^^^^^^^^^^^^^^^
* ``xclim.atmos.water_budget`` has been separated into ``water_budget`` (calculated directly with 'evspsblpot') and ``water_budget_from_tas`` (original function). (:pull:`1086`).
* Injected parameters in indicators are now left out of a function's signature and will not be included in the history attribute. (:pull:`1086`).
* The signature for the following Indicators have been modified (:pull:`1050`):
    - cold_spell_duration_index, tg90p, tg10p, tx90p, tx10p, tn90p, tn10p, warm_spell_duration_index, days_over_precip_doy_thresh, days_over_precip_thresh, fraction_over_precip_doy_thresh, fraction_over_precip_thresh, cold_and_dry_days, warm_and_dry_days, warm_and_wet_days, cold_and_wet_days
* The parameter for percentile values is now named after the variable it is supposed to be computed upon. (:pull:`1050`).
* `pytest-runner` has been removed as a dependency (it was never needed for `xclim` development). (:pull:`1074`).
* `xclim.testing._utils.py` has been renamed to `xclim.testing.utils.py` for added documentation visibility. (:pull:`1074`).
    - Some unused functions and classes (``as_tuple``, ``TestFile``, ``TestDataSet``) have been removed. (:pull:`1107`).

New indicators
^^^^^^^^^^^^^^
* ``universal_thermal_climate_index`` and ``mean_radiant_temperature`` for computing the universal thermal climate index from the near-surface temperature, relative humidity, near-surface windspeed and radiation. (:issue:`1060`, :pull:`1062`).
    - A new method ``ITS90`` has also been added for calculating saturation water vapour pressure. (:issue:`1060`, :pull:`1062`).

Internal changes
^^^^^^^^^^^^^^^^
* Typing syntax has been updated within pre-commit via `isort`. Pre-commit hooks now append `from __future__ import annotations` to all python module imports for backwards compatibility. (:issue:`1065`, :pull:`1071`)
* `isort` project configurations are now set in `setup.cfg`. (:pull:`1071`).
* Many function docstrings, external target links, and internal section references have been adjusted to reduce warnings when building the docs. (:pull:`1074`).
* Code snippets within documentation are now checked and reformatted to `black` conventions with `blackdoc`. A `pre-commit` hook is now in place to run these checks. (:pull:`1098`).
* Test coverage statistic no longer includes coverage of the test files themselves. Coverage now reflects lines of usable code covered. (:pull:`1101`).
* Reordered listed authors alphabetically. Promoted :user:`bzah` to core contributor. (:pull:`1105`).
* Tests have been added for some functions in `xclim.testing.utils.py`; some previously uncaught bugs in ``list_input_variables``, ``publish_release_notes``, and ``show_versions`` have been patched. (:issue:`1078`, :pull:`1107`).
* A convenience command for installing xclim with key development branches of some dependencies has been added (`$ make upstream`). (:issue:`1088`, :pull:`1092`; amended in :issue:`1113`, :pull:`1114`).
    - This build configuration is also available in `tox` for local development purposes (`$ tox -e pyXX-upstream`).

Bug fixes
^^^^^^^^^
* Clean the `bias_adjustement` and `history` attributes created by `xclim.sdba.adjust` (e.g. when an argument  is an `xr.DataArray`, only print the name instead of the whole array). (:issue:`1083`, :pull:`1087`).
* `pydocstyle` checks were silently failing in the `pre-commit` configuration due to a badly-formed regex. This has been adjusted. (:pull:`1074`).
* `adjust_doy_calendar` was broken when the source or the target were seasonal. (:issue:`1097`, :issue:`1091`, :pull:`1099`)

v0.36.0 (2022-04-29)
--------------------
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

v0.35.0 (2022-04-01)
--------------------
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

v0.34.0 (2022-02-25)
--------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), David Huard (:user:`huard`), Aoun Abel (:user:`bzah`).

Announcements
^^^^^^^^^^^^^
* `xclim` now officially supports Python3.10. (:pull:`1013`).

Breaking changes
^^^^^^^^^^^^^^^^
* The version pin for `bottleneck` (<1.4) has been lifted. (:pull:`1013`).
* `packaging` has been removed from the `xclim` run dependencies. (:pull:`1013`).
* Quantile mapping adjustment objects (EQM, DQM and QDM) and ``sdba.utils.equally_spaced_nodes`` will not add additional endpoints to the quantile range. With those endpoints, variables are capped to the reference's range in the historical period, which can be dangerous with high variability in the extremes (ex: pr), especially if the reference doesn't reproduce those extremes credibly. (:issue:`1015`, :pull:`1016`). To retrieve the same functionality as before use:

.. autolink-skip::
.. code-block:: python

    from xclim import sdba

    # NQ is the the number of equally spaced nodes, the argument previously given to nquantiles directly.
    EQM = sdba.EmpiricalQuantileMapping.train(
        ref, hist, nquantiles=sdba.equally_spaced_nodes(NQ, eps=1e-6), ...
    )

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

v0.33.2 (2022-02-09)
--------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`).

Announcements
^^^^^^^^^^^^^
* `xclim` no longer supports Python3.7. Code conventions and new features for Python3.8 (`PEP 569 <https://peps.python.org/pep-0569/>`_) are now accepted. (:issue:`966`, :pull:`1000`).

Breaking changes
^^^^^^^^^^^^^^^^
* Python3.7 (`PEP 537 <https://peps.python.org/pep-0537/>`_) support has been officially deprecated. Continuous integration testing is no longer run against this version of Python. (:issue:`966`, :pull:`1000`).

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

v0.33.0 (2022-01-28)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Tom Keel (:user:`Thomasjkeel`), Jeremy Fyke (:user:`JeremyFyke`), David Huard (:user:`huard`), Abel Aoun (:user:`bzah`), Juliette Lavoie (:user:`juliettelavoie`), Yannick Rousseau.

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

v0.32.1 (2021-12-17)
--------------------

Bug fixes
^^^^^^^^^
* Adjusted a test (``test_cli::test_release_notes``) that prevented conda-forge test ensemble from passing. (:pull:`962`).

v0.32.0 (2021-12-17)
--------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Travis Logan (:user:`tlogan2000`), Trevor James Smith (:user:`Zeitsperre`), Abel Aoun (:user:`bzah`), David Huard (:user:`huard`), Clair Barnes (:user:`clairbarnes`), Raquel Alegre (:user:`raquelalegre`), Jamie Quinn (:user:`JamieJQuinn`), Maliko Tanguy (:user:`malngu`), Aaron Spring (:user:`aaronspring`).

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

v0.31.0 (2021-11-05)
--------------------
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

v0.30.1 (2021-10-01)
--------------------

Bug fixes
^^^^^^^^^
* Fix a bug in ``xclim.sdba``'s ``map_groups`` where 1D input including an auxiliary coordinate would fail with an obscure error on a reducing operation.

v0.30.0 (2021-09-28)
--------------------

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
* `xclim` now implements many data quality assurance flags (``xclim.core.dataflags``) for temperature and precipitation based on `ICCLIM documentation guidelines <https://www.ecad.eu/documents/atbd.pdf>`_. These checks include the following:
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

v0.29.0 (2021-08-30)
--------------------

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

v0.28.1 (2021-07-29)
--------------------

Announcements
^^^^^^^^^^^^^
* The `xclim` binary package available on conda-forge will no longer supply ``clisops`` by default. Installation of ``clisops`` must be performed explicitly to preserve subsetting and bias correction capabilities.

New indicators
^^^^^^^^^^^^^^
* ``snow_depth`` indicator returns the mean snow depth over period. Added as ``SD`` to ICCLIM module.

Internal Changes
^^^^^^^^^^^^^^^^
* Minor modifications to many function call signatures (type hinting) and docstrings (numpy docstring compliance).

v0.28.0 (2021-07-07)
--------------------

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

    - in ``QDM.adjust``, fix bug occurring with coords of 'object' dtype and ``interp='nearest'``.
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

v0.27.0 (2021-05-28)
--------------------

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

v0.26.1 (2021-05-04)
--------------------
* Bug fix release adding `ExtremeValues` to publicly exposed bias-adjustment methods.

v0.26.0 (2021-04-30)
--------------------

Announcements
^^^^^^^^^^^^^
* `xclim` no longer supports Python3.6. Code conventions and new features from Python3.7 (`PEP 537 Features <https://peps.python.org/pep-0537/#features-for-3-7>`_) are now accepted.

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

v0.25.0 (2021-03-31)
--------------------

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

v0.24.0 (2021-03-01)
--------------------

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

v0.23.0 (2021-01-22)
--------------------

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

v0.22.0 (2020-12-07)
--------------------

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

v0.21.0 (2020-10-23)
--------------------

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

v0.20.0 (2020-09-18)
--------------------

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

v0.19.0 (2020-08-18)
--------------------

Breaking changes
^^^^^^^^^^^^^^^^
* Refactoring of the `Indicator` class. The `cfprobe` method has been renamed to `cfcheck` and the `validate`
  method has been renamed to `datacheck`. More importantly, instantiating `Indicator` creates a new subclass on
  the fly and stores it in a registry, allowing users to subclass existing indicators easily. The algorithm for
  missing values is identified by its registered name, e.g. "any", "pct", etc, along with its `missing_options`.
* xclim now requires xarray >= 0.16, ensuring that xclim.sdba is fully functional.
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
* The `Indicator` class now supports outputting multiple indices for the same inputs.
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
* Docstring style now enforced using the `pydocstyle` with `numpy` docstring conventions.
* Doctests are now performed for all docstring `Examples` using `xdoctest`. Failing examples must be explicitly skipped otherwise build will now fail.
* Indicator methods `update_attrs` and `format` are now classmethods, attrs to update must be passed.
* Indicators definitions without an accompanying translation (presently French) will cause build failures.
* Major refactoring of the internal machinery of `Indicator` to support multiple outputs.

v0.18.0 (2020-06-26)
--------------------
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

v0.17.0 (2020-05-15)
--------------------
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

v0.16.0 (2020-04-23)
--------------------
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

v0.15.x (2020-03-12)
--------------------
* Improvement in FWI: Vectorization of DC, DMC and FFMC with numba and small code refactoring for better maintainability.
* Added example notebook for creating a catalog of selected indices
* Added `growing_season_end`, `last_spring_frost`, `dry_days`,  `hot_spell_frequency`, `hot_spell_max_length`, and `maximum_consecutive_frost_free_days` indices.
* Dropped use of `fiona.crs` class in lieu of the newer pyproj CRS handler for `subset_shape` operations.
* Complete internal reorganization of xclim.
* Internationalization of xclim : add `locales` submodule for localized metadata.
* Add feature to retrieve coordinate values instead of index in `run_length.first_run`. Add `run_length.last_run`.
* Fix bug in subset_gridpoint to work on lat/lon coords of any dimension when they are not a dimension of the data.

v0.14.x (2020-02-21)
--------------------
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

v0.13.x (2020-01-10)
--------------------
* Documentation improvements: list of indicators, RTD theme, notebook example.
* Added `sea_ice_extent` and `sea_ice_area` indicators.
* Reverted #311, removing the `_rolling` util function. Added optimal keywords to `rolling()` calls.
* Fixed `ensembles.create_ensemble` errors for builds against xarray master branch.
* Reformatted code to make better use of Python3.6 conventions (f-strings and object signatures).
* Fixed randomly failing tests of `checks.missing_any`.
* Improvement of `ensemble.ensemble_percentile` and `ensemble.create_ensemble`.

v0.12.x-beta (2019-11-18)
-------------------------
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

v0.11.x-beta (2019-10-17)
-------------------------
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

v0.10.x-beta (2019-06-18)
-------------------------
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

v0.10-beta (2019-06-06)
-----------------------
* Dropped support for Python 2.
* Added support for *period of the year* subsetting in ``checks.missing_any``.
* Now allow for passing positive longitude values when subsetting data with negative longitudes.
* Improved runlength calculations for small grid size arrays via ``ufunc_1dim`` flag.

v0.9-beta (2019-05-13)
----------------------
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

v0.8-beta (2019-02-11)
----------------------
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

v0.6-alpha (2018-10-03)
-----------------------
* File attributes checks.
* Added daily downsampler function.
* Better documentation on ICCLIM indices.

v0.5-alpha (2018-09-26)
-----------------------
* Added total precipitation indicator.

v0.4-alpha (2018-09-14)
-----------------------
* Fully PEP8 compliant and available under MIT License.

v0.3-alpha (2018-09-4)
----------------------
* Added icclim module.
* Reworked documentation, docs theme.

v0.2-alpha (2018-08-27)
-----------------------
* Added first indices.

v0.1.0-dev (2018-08-23)
-----------------------
* First release on PyPI.
