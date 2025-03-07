==============
Support Policy
==============

Support Channels
----------------

* `xclim` Issues: https://github.com/Ouranosinc/xclim/issues
* `xclim` Discussions: https://github.com/Ouranosinc/xclim/discussions
* PAVICS-related Questions: `<pavics@ouranos.ca>`_

API Compatibility
-----------------

`xclim` aims to maintain backwards compatibility as much as possible. New features that are considered "breaking changes" are adopted gradually while deprecation notices are issued for the older features. Dropping support for older versions of support libraries is considered a breaking change.

Significant `xclim` API changes are documented in the changelog. When modules are significantly modified, they are marked as such in the documentation, while deprecation warnings are issued in the code. Support for deprecated features is often maintained for a reasonable period of time (two or three stable releases), but users are encouraged to update their code to the new API as soon as possible.

Scientific Python Ecosystem Compatibility
-----------------------------------------

`xclim` closely follows the compatibility of the `xarray` and `dask` libraries. The `xclim` library is tested against the latest stable versions of `xarray` and `dask` and is expected to work with the latest stable versions of `numpy`, `scipy`, and `pandas`. These projects tend to follow either the `NumPy Enhancement Protocols (NEP-29) <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ or the `Scientific Python SPEC-0 <https://scientific-python.org/specs/spec-0000/>`_ for deprecation policies; `xclim` generally follows the same policies.

The lowest supported versions of libraries listed in the `xclim` package metadata are expected to be compatible with the latest stable versions of `xclim`. From time to time, these minimum supported versions will be updated to follow the Scientific Python SPEC-0 recommendations. In the event that a significant breaking change is made to the `xarray` or `dask` libraries, `xclim` may adopt a newer minimum supported version of those libraries than SPEC-0 might recommend.

`xclim` tends to support older Python versions until one or many of the following events occur:
- The Python version no longer receives security patches by the Python Software Foundation (EoL).
- The Python version is no longer supported by the last stable releases of the `xarray` or `dask` libraries.
- Maintaining support for an older Python versions becomes a burden on the development team.

Versioning
----------

`xclim` mostly adheres to the `Semantic Versioning v2.0 <https://semver.org/spec/v2.0.0.html>`_ convention, which means that the version number is composed of three or four numbers: `MAJOR.MINOR.PATCH-DEV.#`. The version number is incremented according to the following rules:

- `MAJOR` version is incremented when incompatible changes are made to the API.
- `MINOR` version is incremented when new features are added in a backwards-compatible manner.
- `PATCH` version is incremented when backwards-compatible bug fixes are made.
- `DEV.#` (development) version is incremented when new features are added or bug fixes are made.

The development version is used for testing new features and bug fixes before they are released in a stable version. The development version is not considered stable and should not be used in production environments.

Modifications to continuous integration (CI) pipelines, linting tools, documentation, and other non-code changes do not affect the version number.
