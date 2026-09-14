Differences between v0 and v1
=============================
In September 2026, a major overhaul of `xclim` was implemented. While most of the functionalities were preserved, the internals were modified in many breaking ways.
The goal was to modernize the lower-level API, aiming to ease development and favour contributions to `xclim` by the larger community.
As of `xclim` v1.0, not all planned changes have been performed, but only non-breaking changes remain.

This page aims to summarize the largest and most breaking changes, to help transition to v1.

Indicators calculations return Datasets
---------------------------------------
The ``as_dataset`` option of :py:class:`~xclim.core.set_options` was changed to `True`, meaning that indicator calculations now return :py:class:`~xarray.Dataset` objects by default instead of :py:class:`~xarray.DataArray` objects or tuples of those.
Re-enabling the previous behaviour is simply done with ``xclim.set_options(as_dataset=False)``, either globally or as a context.

Renamed `indices` to `compute`
------------------------------
The ``xclim.indices`` module was renamed as :py:mod:`xclim.compute`. This was part of a general move to stop using the words "index" or "indices" which were always quite amibugous with "indicator".
In the documentation, functions that perform the actual computation, which were previously named "indices", are now usually named "compute functions" or "index-like compute functions".
"Indicator" still denotes the larger object that performs all the checks and metadata formatting in adititon to the computation.

Major reimplementation of the generic compute functions
-------------------------------------------------------
The :py:mod:`xclim.compute.generic` submodule holds compute functions (previously "indices") that are not variable-specific and can be reused in many indicators.
All functions and their arguments were modified and renamed to follow a more systematic naming convention, heavily inspired by the work of `clix-meta <github.com/clix-meta/clix-meta/>`_.
See `this GitHub comment <https://github.com/Ouranosinc/xclim/pull/2258#issuecomment-3473430173>`_ for an exhaustive list of the changes.

The names of the indicators in :py:mod:`xclim.indicators` were largerly unchanged, but some arguments might have been renamed.

Indicator constructor and "virtual submodules"
----------------------------------------------
The :py:mod:`xclim.core.indicator` submodule was refactored in hopes of making the internals of the :py:class:`~xclim.core.indicator.Indicator` object easier to maintain and extend.
In addition, the "virtual submodules" concept was rewritten as the :py:class:`~xclim.core.collection.IndicatorCollection` object.
These are dictionary-like in structure, holding a collection of indicators and typically created from a YAML file, similar to the previous implementation.

The main breaking changes are:

- Attribute `cf_attrs` was renamed `attrs` and is now a list of :py:class:`~xclim.core.indicator.Output` objects.
  These act like dictionaries of attributes, but have also four properties : `var_name`, `units`, `units_metadata` and `dimensionality`, which are accessed as properties (ex: ``ind.attrs[0].var_name``) and not as dictionary items. This name change is to be applied in the indicator's constructor and in the YAML files.
- Function ``xclim.build_indicator_module_from_yaml`` is changed to :py:meth:`xclim.core.collection.IndicatorCollection.from_yaml`. Collections are standalone objects, they don't automatically register as python submodules of ``xclim.indicators`` anymore.
- Indicators defined within a collection are not registered by default into the indicators registry.
- When creating an indicator from an existing one, the ``var_name`` of the output of an indicator with a single output now only defaults to the indicator's identifier if the parent does not have a ``var_name``. Previously, the identifier would always be used as the ``var_name``, regardless of the parent's attributes.
- The :py:meth:`xclim.core.indicator.Indicator.from_dict` method is deprecated. Use :py:meth:`xclim.core.indicator.Indicator.copy` instead, calling it on the indicator object you want to subclass/copy.
  This new method doesn't parse ``compute`` function names given as string, instead passing the function directly.

See the updated :ref:`notebooks/extendxclim:Extending xclim` page for more details.

Indicator registry
------------------
The indicator registry :py:data:`xclim.core.indicator.registry` is now case-insensitive and holds indicator *instances*, not *classes*. No need to call ``ind.get_instance()`` to get a working object anymore. As noted above, creating a derived indicator from another is now done by calling :py:meth:`~xclim.core.indicator.Indicator.copy()` on the parent instance.

The base classes registry :py:data:`xclim.core.indicator_base_registry` is preserved.
