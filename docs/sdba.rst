==========================================
Bias Adjustment and Downscaling Algorithms
==========================================

.. warning::

    The `xclim.sdba` module was split from the library in `xclim==0.57` in order to facilitate development and effective maintenance of the SDBA utilities. This functionality is now available in the `xsdba` package. While the package aims to maintain compatibility with `xclim`, some algorithms have been slightly modified.

    For convenience, the `xclim.sdba` module remains available as a reimport of `xsdba`, exposing the functionality of the `xsdba` package, though users are encouraged to import `xsdba` directly. This behaviour may eventually change.

.. note::

    For more information on how to transition to using `xsdba`, a guide is available at the following link: :doc:`xclim Migration Guide <xsdba:xclim_migration_guide>`.


The `xclim.sdba` submodule provides a collection of bias-adjustment methods meant to correct for systematic biases found in climate model simulations relative to observations.
Almost all adjustment algorithms conform to the `train` - `adjust` scheme, meaning that adjustment factors are first estimated on training data sets, then applied in a distinct step to the data to be adjusted.
Given a reference time series (ref), historical simulations (hist) and simulations to be adjusted (sim),
any bias-adjustment method would be applied by first estimating the adjustment factors between the historical simulation
and the observation series, and then applying these factors to `sim`, which could be a future simulation:

.. code-block:: python

    # Create the adjustment object by training it with reference and model data, plus certain arguments
    Adj = Adjustment.train(ref, hist, group="time.month")
    # Get a scenario by applying the adjustment to a simulated timeseries.
    scen = Adj.adjust(sim, interp="linear")
    Adj.ds.af  # adjustment factors.

Most method support both additive and multiplicative correction factors.
Also, the `group` argument allows adjustment factors to be estimated independently for different periods: the full
time series,  months, seasons or day of the year.  For monthly groupings, the `interp` argument then allows for interpolation between
adjustment factors to avoid discontinuities in the bias-adjusted series.
See :ref:`Grouping` below.

The same interpolation principle is also used for quantiles. Indeed, for methods extracting adjustment factors by
quantile, interpolation is also done between quantiles. This can help reduce discontinuities in the adjusted time
series, and possibly reduce the number of quantile bins that needs to be used.

Modular Approach
================
The module attempts to adopt a modular approach instead of implementing published and named methods directly.
A generic bias adjustment process is laid out as follows:

- preprocessing on ``ref``, ``hist`` and ``sim`` (using methods in :py:mod:`xclim.sdba.processing` or :py:mod:`xclim.sdba.detrending`)
- creating and training the adjustment object ``Adj = Adjustment.train(obs, sim, **kwargs)`` (from :py:mod:`xclim.sdba.adjustment`)
- adjustment ``scen = Adj.adjust(sim, **kwargs)``
- post-processing on ``scen`` (for example: re-trending)

The train-adjust approach allows to inspect the trained adjustment object. The training information is stored in
the underlying `Adj.ds` dataset and usually has a `af` variable with the adjustment factors. Its layout and the
other available variables vary between the different algorithm, refer to :ref:`Adjustment methods <sdba-user-api>`.

Parameters needed by the training and the adjustment are saved to the ``Adj.ds`` dataset as a `adj_params` attribute.
Parameters passed to the `adjust` call are written to the history attribute in the output scenario DataArray.

.. _grouping:

Grouping
========
For basic time period grouping (months, day of year, season), passing a string to the methods needing it is sufficient.
Most methods acting on grouped data also accept a `window` int argument to pad the groups with data from adjacent ones.
Units of `window` are the sampling frequency of the main grouping dimension (usually `time`). For more complex grouping,
one can pass an instance of :py:class:`xclim.sdba.base.Grouper` directly. For example, if one wants to compute the factors
for each day of the year but across all realizations of an ensemble : ``group = Grouper("time.dayofyear", add_dims=['realization'])``.
In a conventional empirical quantile mapping (EQM), this will compute the quantiles for each day of year and all realizations together, yielding a single set of adjustment factors for all realizations.

.. warning::

    If grouping according to the day of the year is needed, the :py:mod:`xclim.core.calendar` submodule contains useful
    tools to manage the different calendars that the input data can have. By default, if 2 different calendars are
    passed, the adjustment factors will always be interpolated to the largest range of day of the years but this can
    lead to strange values, so we recommend converting the data beforehand to a common calendar.

Application in multivariate settings
====================================
When applying univariate adjustment methods to multiple variables, some strategies are recommended to avoid introducing unrealistic artifacts in adjusted outputs.

Minimum and maximum temperature
-------------------------------
When adjusting both minimum and maximum temperature, adjustment factors sometimes yield minimum temperatures larger than the maximum temperature on the same day, which of course, is nonsensical.
One way to avoid this is to first adjust maximum temperature using an additive adjustment, then adjust the diurnal temperature range (DTR) using a multiplicative adjustment, and then determine minimum temperature by subtracting DTR from the maximum temperature :cite:p:`thrasher_technical_2012,agbazo_characterizing_2020`.

Relative and specific humidity
------------------------------
When adjusting both relative and specific humidity, we want to preserve the relationship between both.
To do this, :cite:t:`grenier_two_2018` suggests to first adjust the relative humidity using a multiplicative factor, ensure values are within 0-100%, then apply an additive adjustment factor to the surface pressure before estimating the specific humidity from thermodynamic relationships.

Radiation and precipitation
---------------------------
In theory, short wave radiation should be capped when precipitation is not zero, but there is as of yet no mechanism proposed to do that, see :cite:t:`hoffmann_meteorologically_2012`.

Usage examples
==============
The usage of this module is documented in two example notebooks: `SDBA <notebooks/sdba.ipynb>`_ and `SDBA advanced <notebooks/sdba-advanced.ipynb>`_.

Discussion topics
=================
Some issues were also discussed on the Github repository. Most of these are still open questions, feel free to participate to the discussion!

* Number quantiles to use in quantile mapping methods: :issue:`1162`
* How to choose the quantiles: :issue:`1015`
* Bias-adjustment when the trend goes to zero: :issue:`1145`
* Spatial downscaling: :issue:`1150`

User API
========

.. note::

    For more information, the documentation is available at the following link: :doc:`xsdba API <xsdba:api>`

Notes for Developers
====================

.. warning::

    The `xclim.sdba` module was split from the library in `xclim==0.57` in order to facilitate development and effective maintenance of the SDBA utilities. This functionality is now available in the `xsdba` package. While the package aims to maintain compatibility with `xclim`, some algorithms have been slightly modified.

.. note::

    For more information, the contributor documentation is available at the following link: :doc:`xsdba contributor guidelines <xsdba:contributing>`

Developer API
=============

.. note::

    For more information, the documentation is available at the following link: :doc:`xsdba modules API <xsdba:apidoc/modules>`

.. only:: html or text

    .. _sdba-footnotes:

    SDBA Footnotes
    ==============

    .. bibliography::
       :style: xcstyle
       :labelprefix: SDBA-
       :keyprefix: sdba-
