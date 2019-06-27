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
    Calculations are performed on the variable of interest present within the data set (in this example, **ds.tas**) and not the data set itself, so be sure to specify this when using xclim's indices and other calls.

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


Using the xclim.icclim module
-----------------------------
`ICCLIM <https://github.com/cerfacs-globc/icclim>`_ is a software platform / python library for performing climate indice calculations. It includes roughly 50 types of indicators as defined by the `European Climate Assessment & Dataset project <https://www.ecad.eu/>`_.

For those familiar with ICCLIM, xclim has created one-to-one mappings of their indices. Effectively, these are simply indice calculations methods that have been renamed to be consistent with ICCLIM's terminology.

Where thresholds may differ between xclim's own indices and those of ICCLIM, those called from `xclim.icclim` will ensure that the indice follows the ICCLIM standards. Other extras that the ICCLIM library performs, such as error-handling and specific variable integrity checks are currently planned but not implemented yet.

The list of all currently-implented ICCLIM indices can be found `here <icclim>`_.

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
checks for missing data, incomplete periods, CF-metadata. Useful for producing NetCDFs that will be shared with others. Omits calculations for period with bad coverage.


Resampling frequencies
----------------------
Use `Q-NOV` to resample into climatological seasons (DJF, MAM, JJA, SON).
