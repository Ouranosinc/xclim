==============
Why use xclim?
==============

Purpose
=======

.. important::

    The content of this section is actively being developed in the forthcoming paper submission to JOSS.
    This section will be updated and finalized when the wording has been agreed upon in :pull:`250`

`xclim` aims to position itself as a climate services tool for any researchers interested in using Climate and Forecast Conventions (`CF-Conventions <https://cfconventions.org/>`_) compliant datasets to perform climate analyses. This tool is optimized for working with Big Data in the climate science domain and can function as an independent library for one-off analyses in *Jupyter Notebooks* or as a backend engine for performing climate data analyses via **Web Processing Services** (`WPS <https://www.ogc.org/standard/wps/>`_; e.g. `Finch <https://github.com/bird-house/finch>`_). It was primarily developed targeting Earth and Environmental Science audiences and researchers, originally for calculating climate indicators for the Canadian government web service `ClimateData.ca <https://climatedata.ca/>`_.

The primary domains that `xclim` is built for are in calculating climate indicators, performing statistical correction / bias adjustment of climate model output variables or simulations, and in performing climate model simulation ensemble statistics.

Other Python projects similar to xclim
======================================

`xclim` has been developed within an ecosystem of several existing projects that deal with climate and statistical correction/downscaling and has both influenced and been influenced by their approaches:

* `icclim` (`icclim Source Code <https://github.com/cerfacs-globc/icclim>`_; `icclim Documentation <https://icclim.readthedocs.io/en/stable/index.html>`_)
    - `xclim` aimed to reimplement `icclim` using `xarray`-natives for the computation of climate indices. Starting from version 5.0 of `icclim`, `xclim` has become a core dependency for this project.
    - The `icclim` developers have prepared a documentation page comparing xclim and icclim (`xclim_and_icclim <https://icclim.readthedocs.io/en/stable/explanation/xclim_and_icclim.html>`_).

* `climate_indices` (`climate_indices Source Code <https://github.com/monocongo/climate_indices>`_; `climate_indices Documentation <https://climate-indices.readthedocs.io/en/latest/index.html>`_)
    - Provides several moisture- and drought-related indicators not implemented at-present in `xclim` (SPI, SPEI, PDSI, etc.). It also offers a robust command-line interface and uses `xarray` in its backend.
    - There is currently an ongoing discussion about the merging of `climate_indices` and `xclim`: :issue:`1273`.

* `MetPy` (`MetPy Source Code <https://github.com/Unidata/MetPy>`_; `MetPy Documentation <https://unidata.github.io/MetPy/latest/index.html>`_)
    - `MetPy` is built for reading, visualizing, and performing calculations specifically on standards-compliant, operational weather data. Like `xclim`, it makes use of `xarray`.
    - `xclim` adopted its standards and unit-handling approaches from `MetPy` and associated project `cf-xarray`.

* `climpred` (`climpred Source Code <https://github.com/pangeo-data/climpred>`_; `climpred Documentation <https://climpred.readthedocs.io/en/stable/index.html>`_)
    - `climpred` is designed to analyze and validate weather and climate forecast data against observations, reconstructions, and simulations. Similar to `xclim`, it leverages `xarray`, `dask`, and `cf_xarray` for object handling, distributed computation, and metadata validation, respectively.

* `pyet` (`pyet Source Code <https://github.com/pyet-org/pyet>`_; `pyet Documentation <https://pyet.readthedocs.io/en/latest/>`_)
    - `pyet` is a tool for calculating/estimating evapotranspiration using many different accepted methodologies and employs a similar design approach as `xclim`, based on `xarray`-natives.

* `xcdat` (`xcdat Source Code <https://github.com/xCDAT/xcdat>`_; `xcdat Documentation <https://xcdat.readthedocs.io/en/latest/>`_)

* `GeoCAT` (`GeoCAT Documentation <https://geocat.ucar.edu/>`_)
    - `GeoCAT` is an ensemble of tools developed specifically for scalable data analysis and visualization of structures and unstructured gridded earth science datasets. `GeoCAT` tools rely on many of the same tools that `xclim` uses in its stack (notably, `xarray`, `dask`, and `jupyter notebooks`).

* `scikit-downscale` (`scikit-downscale Source Code <https://github.com/pangeo-data/scikit-downscale>`_, `scikit-downscale Documentation <https://scikit-downscale.readthedocs.io/en/latest/>`_)
    - `scikit-downscale` offers algorithms for statistical downscaling. `xclim` drew inspiration from its fit/predict architecture API approach. The suite of downscaling algorithms offered between both projects differs.

R-language specific projects
----------------------------

* `climdex.pcic` (`climdex.pcic Source Code <https://github.com/pacificclimate/climdex.pcic>`_; `climdex.pci R-CRAN Index <https://cran.r-project.org/web/packages/climdex.pcic/index.html>`_)
* `climind` (`climind Source Code <https://github.com/ECA-D/climind>`_; `climind Documentation <https://cran.r-project.org/package=ClimInd>`_)
