Usage
=====

Basic example
-------------
To use xclim in a project:

.. code-block:: python

    import xclim

To open NetCDF (`.nc`) climate data sets, use xarray:

.. code-block:: python

    import xarray as xr

    data_file = "tas-daily-historical-data.nc"
    ds = xr.open_dataset(data_file)

.. note::
    Calculations are performed on the variable of interest present within the data set (in this example, **tas**) and not the data set itself, so be sure to specify this when using xclim's indices and other calls.

To perform a simple climate indice calculation:

.. code-block:: python

    gdd = xclim.indices.growing_degree_days(ds.tas, thresh='10.0 degC', freq='YS')

To plot a time slice of the output, we recommend the following:

.. code-block:: python

    import matplotlib.pyplot

    # for a summary statistics histogram
    gdd.plot()

    # for a specific geographical coordinate
    gdd.isel(lon=-75, lat=47).plot()

    # for a specific time period
    gdd.isel(time=25).plot()

For more examples, see the directions suggested by `Xarray's Plotting Documentation <https://xarray.pydata.org/en/stable/plotting.html>`_

To save the data as a new NetCDF

.. code-block:: python

    gdd.to_netcdf('viticultural_growing_degree_days-data.nc')

.. note::
    You can also save as different formats such as by leveraging the functions already built into  the xarray dataset instance. For more information see: `Xarray's Documentation <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_

Slicing and subsetting with xarray
----------------------------------
.. warning::

    This section is presently under development!

Using the xclim.icclim module
-----------------------------
`ICCLIM <https://github.com/cerfacs-globc/icclim>`_ is a software platform / python library for performing climate indice calculations. It includes roughly 50 types of indicators as defined by the `European Climate Assessment & Dataset project <https://www.ecad.eu/>`_.

For those familiar with ICCLIM, xclim has created similar mappings for many of their indices. Effectively, these are simply indice calculations that have been renamed to be consistent with ICCLIM's terminology.

Where thresholds may differ between xclim's thresholds for indices and those of ICCLIM, those called from `xclim.icclim` will ensure that the indice follows the ICCLIM threshold standards. Other extras that the ICCLIM library performs, such as error-handling and variable integrity checks modeled after ICCLIM's standards are currently planned but not yet implemented.

The list of all presently-available ICCLIM indices can be found `here <icclim>`_.

.. code-block:: python

    from xclim import icclim
    from xclim import indices

    gd4 = icclim.GD4(ds.tas)
    gdd_base_4 = indices.growing_degree_days(ds.tas, thresh='4 degC', freq='YS')

    gd4.sum() == gdd_base_4.sum()

.. code-block:: pycon

    <xarray.DataArray 'tas' ()>
    array(True)
    Coordinates:
    height   float64 2.0

Using the xclim.atmos module
----------------------------
The `xclim.atmos` module is tool for running error-detection checks and ensuring consistent metadata when running an indice calculation. Some of the processes involve:

* Checking for missing data and incomplete periods.
* Writing `Climate and Forecast Convention <http://cfconventions.org/>`_ compliant metadata based on the variables and indices calculated.
* Identifying periods where missing data significantly impacts the calculation and omits calculations for those periods.
* Appending process history and maintaining the historical provenance of file metadata.

This module is best used for producing NetCDF that will be shared with users and is called using the following methods:

.. code-block:: python

    import xclim.atmos

    tmax = xclim.atmos.tx_max(ds.tas)
    tmax.to_netcdf('tmax.nc')

Resampling frequencies
----------------------
.. warning::

    This section is under development!

Use `Q-NOV` to resample into climatological seasons (DJF, MAM, JJA, SON).
