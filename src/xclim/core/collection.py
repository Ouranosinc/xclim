"""
Indicator collections
=====================

An indicator collection is a structure holding multiple indicators. It can be created through a yaml configuration file.

YAML file structure
~~~~~~~~~~~~~~~~~~~

Indicator-defining yaml files are structured in the following way. Most entries of the `indicators` section are
mirroring attributes of the :py:class:`xclim.core.indicator.Indicator`, please refer to its documentation for more
details on each.

.. code-block:: yaml

    module: <module name>  # Defaults to the file name
    realm: <realm>  # If given here, applies to all indicators that do not already provide it.
    keywords:
      - <keyword>  # Merged with indicator-specific keywords (appended to the list)
    references: <references> # Merged with indicator-specific references (joined with a new line)
    base: <base indicator class>  # Defaults to "Daily" and applies to all indicators that do not give it.
    doc: <module docstring>  # Defaults to a minimal header, only valid if the module doesn't already exist.
    variables:  # Optional section if indicators declared below rely on variables unknown to xclim
                # (not in `xclim.core.VARIABLES`)
                # The variables are not module-dependent and will overwrite any already existing with the same name.
      <varname>:
        canonical_units: <units> # required
        description: <description> # required
        standard_name: <expected standard_name> # optional
        cell_methods: <expected cell_methods> # optional
    indicators:
      <identifier>:  # The actual indicator identifier will be prepended by the module name.
        # From which Indicator to inherit
        base: <base indicator class>  # Defaults to module-wide base class
                                      # If the name startswith a '.', the base class is taken from the current module
                                      # (thus an indicator declared _above_).
                                      # Available indicators are listed in `xclim.core.indicator.registry` and
                                      # other base classes in `xclim.core.indicator.base_registry`.

        # General metadata, usually parsed from the `compute`s docstring when possible.
        realm: <realm>  # defaults to module-wide realm. One of "atmos", "land", "seaIce", "ocean".
        title: <title>
        abstract: <abstract>
        keywords:
          - <keyword>  # merged to module-wide keywords.
        references: <references>  # newline-seperated, merged to module-wide references.
        notes: <notes>

        # Other options (not all indicator classes support them)
        missing: <missing method name>
        missing_options: <missing options mapping>
        allowed_periods: [<list>, <of>, <allowed>, <periods>]
        context: <context> # A unit context enabled during the conversion of the compute's output to the requested units

        # Compute function
        compute: <function name>  # Referring to a function in `compute` module
                                  # (xclim.compute.generic or xclim.compute)
                                  # Or to a function declared in the mapping passed to the collection constructor.
        input:  # When "compute" is a generic function, this is a mapping from argument name to the expected variable.
                # It will change the expected name of the variable as well as its units/dimensionality.
                # Can refer to a variable declared in the `variables` section above or in `xclim.core.VARIABLES`.
          <var name in compute> : <variable official name>
          ...

        # Parameters
        <param name>: <param data>  # Simplest case, to inject parameters in the compute function.
                                    # Kwargs-like parameters like ``indexer`` must be injected as a dictionary here.
        <param name>:  # To change parameters metadata or to declare units when "compute" is a generic function.
          default : <param default>
          description: <param description>
          name : <param name>  # Change the name of the parameter (similar to what `input` does for variables)
          kind: <param kind> # Override the parameter kind. This is mostly useful for transforming an
                             # optional variable into a required one by passing ``kind: 0``.
        ...
      ...  # and so on.

All fields are optional. Other fields found in the yaml file will trigger errors when validation is activated.

When a module is built from a yaml file, the yaml is first validated against the schema (see xclim/data/schema.yml)
using the YAMALE library (:cite:p:`lopker_yamale_2022`). See the "Extending xclim" notebook for more info.

Inputs
~~~~~~
As xclim has strict definitions of possible input variables (see :py:data:`xclim.core.VARIABLES`),
the mapping of `indicators.<identifier>.input` simply links an argument name from the function given in "compute"
to one of those official variables.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Literal

import yamale
from yaml import safe_load

import xclim.compute
import xclim.compute.generic
from xclim.core import VARIABLES, raise_warn_or_log
from xclim.core.indicator import Daily, Indicator, base_registry, registry
from xclim.core.locales import load_locale, read_locale_file
from xclim.core.utils import load_module


class IndicatorCollection(dict):  # numpydoc ignore=PR01
    """A collection of indicators."""

    def __init__(self, indicators: dict[str, Indicator], name: str | None = None, doc: str | None = None):
        """
        Create an IndicatorCollection.

        Parameters
        ----------
        indicators : dict of Indicator
            Indicators to put in the new module.
        name : str, optional
            The name of the module.
        doc : str, optional
            Documentation of the collection. Defaults to a simple header.
        """
        self.name = name
        self.__doc__ = doc or f"{name.capitalize()} indicators\n" + "=" * (len(name) + 11)
        super().__init__(**indicators)

    def iter_indicators(self):
        """Iterate over the (name, indicator) pairs in this collection."""
        yield from self.items()

    @classmethod
    def from_yaml(  # noqa: C90
        cls,
        filename: PathLike,
        name: str | None = None,
        computes: dict[str, Callable] | ModuleType | PathLike | None = None,
        translations: dict[str, dict | PathLike] | None = None,
        mode: Literal["raise", "warn", "ignore"] = "raise",
        encoding: str = "UTF8",
        validate: bool | PathLike = True,
        register: bool = False,
    ):
        """
        Build an indicator collection from a YAML file.

        When given only a base filename (no 'yml' extension), this tries to find custom indicators in a module
        of the same name (*.py) and translations in json files (*.<lang>.json), see Notes.

        Indicator created here will have the name of the module prepended to their identifier (ex: `{mod}.{baseId}`).
        The base identifier being the key name within the `indicators` mapping in the yaml.

        Parameters
        ----------
        filename : PathLike
            Path to a YAML file or to the stem of all module files.
            See Notes for behaviour when passing a basename only.
        name : str, optional
            The name of the new or existing module, defaults to the basename of the file (e.g: `atmos.yml` -> `atmos`).
        computes : Mapping of callables or module or path, optional
            A mapping or module of compute functions or a python file declaring such a module. When creating the
            indicator, the name in the `compute` field is first sought here, then the indicator class will search
            in :py:mod:`xclim.compute.generic` and finally in :py:mod:`xclim.compute`.
        translations : Mapping of dicts or path, optional
            Translated metadata for the new indicators. Keys of the mapping must be two-character language tags.
            Values can be translations dictionaries as defined in :ref:`internationalization:Internationalization`.
            They can also be a path to a JSON file defining the translations.
        mode : {'raise', 'warn', 'ignore'}
            How to deal with broken indicator definitions.
        encoding : str
            The encoding used to open the `.yaml` and `.json` files.
            It defaults to UTF-8, overriding python's mechanism which is machine dependent.
        validate : bool or PathLike
            If True (default), the yaml module is validated against the `xclim` schema.
            Can also be the path to a YAML schema against which to validate;
            Or False, in which case validation is simply skipped.
        register : bool
            If True, the indicators created here are registered in xclim's indicators registry
            :py:data:`~xclim.core.indicator.registry` upon creation, using the collection's name
            prepended to their identifier as key, as explained above.
            Defaults to False, making collections independent from xclim's registry.
            This does not change the behaviour of registering new variables, which are always added
            to xclim's central :py:data:`xclim.core.VARIABLES`.

        Returns
        -------
        IndicatorCollection
            A collection of indicators.

        See Also
        --------
        xclim.core.indicator : Indicator build logic.

        Notes
        -----
        When the given `filename` has no suffix (usually '.yaml' or '.yml'), the function will try to load
        custom compute functions definitions from a file with the same name but with a `.py` extension. Similarly,
        it will try to load translations in `*.<lang>.json` files, where `<lang>` is the IETF language tag.
        Note that the file name *can not* contain a dot (``.``) for this logic to work.

        For example. a set of custom indicators could be fully described by the following files:

            - `example.yml` : defining the indicator's metadata.
            - `example.py` : defining a few compute functions.
            - `example.fr.json` : French translations
        """
        filepath = Path(filename)
        # A stem was passed, try to load module, functions and translations with same name but different suffixes
        is_stem = filepath.suffix not in [".yml", ".yaml"]

        if is_stem:
            yml_path = filepath.with_suffix(".yml")
        else:
            yml_path = filepath

        # Read YAML file
        with yml_path.open(encoding=encoding) as f:
            yml = safe_load(f)

        if validate is not False:
            # Read schema
            if validate is not True:
                schema = yamale.make_schema(validate)
            else:
                schema = yamale.make_schema(Path(__file__).parent.parent / "data" / "schema.yml")

            # Validate - a YamaleError will be raised if the module does not comply with the schema.
            yamale.validate(schema, yamale.make_data(content=yml_path.read_text(encoding=encoding)))

        # Load values from top-level in yml.
        # Priority of arguments differ.
        coll_name = name or yml.get("name", filepath.stem)
        default_base = registry.get(yml.get("base"), base_registry.get(yml.get("base"), Daily))
        doc = yml.get("doc")

        if is_stem and computes is None and (ind_file := filepath.with_suffix(".py")).is_file():
            # No suffix means we try to automatically detect the python file
            computes = ind_file

        if isinstance(computes, str | Path):
            computes = load_module(computes, name=coll_name)

        _translations: dict[str, dict] = {}
        if is_stem and translations is None:
            # No suffix mean we try to automatically detect the json files.
            for loc_file in filepath.parent.glob(f"{filepath.stem}.*.json"):
                locale = loc_file.suffixes[0][1:]
                _translations[locale] = read_locale_file(loc_file, module=coll_name, encoding=encoding)
        elif translations is not None:
            # A mapping was passed, we read paths is any.
            _translations = {
                lng: (
                    read_locale_file(trans, module=coll_name, encoding=encoding)
                    if isinstance(trans, str | Path)
                    else trans
                )
                for lng, trans in translations.items()
            }

        # Module-wide default values for some attributes
        defkwargs = {
            # Only used in case the indicator definition does not give them.
            "realm": yml.get("realm", "atmos"),
            # Merged with a space
            "keywords": yml.get("keywords", []),
            # Merged with a new line
            "references": yml.get("references"),
        }

        # Parse the variables:
        for varname, vardata in yml.get("variables", {}).items():
            if varname in VARIABLES and VARIABLES[varname] != vardata:
                warnings.warn(
                    f"Variable {varname} from collection {coll_name} "
                    "will overwrite the one already defined in `xclim.core.VARIABLES`"
                )
            VARIABLES[varname] = vardata.copy()

        # Parse the indicators:
        mapping = {}
        for identifier, data in yml["indicators"].items():
            try:
                # Get base class
                base = default_base
                if (basename := data.pop("base", None)) is not None:
                    base = cls._find_base_class(basename, mapping)

                if (funcname := data.pop("compute", None)) is not None:
                    data["compute"] = cls._find_compute_function(funcname, computes)

                if data.get("references") and defkwargs.get("references"):
                    data["references"] = f"{data['references']}\n{defkwargs['references']}"
                elif defkwargs.get("references"):
                    data["references"] = defkwargs["references"]
                data["keywords"] = [*defkwargs.get("keywords", []), *data.get("keywords", [])]
                data.setdefault("realm", defkwargs.get("realm"))

                mapping[identifier] = base(
                    identifier=f"{coll_name}.{identifier}", module=coll_name, register=register, **data
                )

            except Exception as err:  # pylint: disable=broad-except
                raise_warn_or_log(err, mode, msg=f"Constructing {identifier} failed with {err!r}")

        coll = cls(mapping, name=coll_name, doc=doc)
        # If there are translations, load them
        if _translations:
            for locale, loc_dict in _translations.items():
                load_locale(loc_dict, locale)
        return coll

    @staticmethod
    def _find_base_class(name, mapping):
        if name.startswith("."):
            # A point means the base has been declared above.
            base = mapping[name[1:]].__class__
        elif name in base_registry:
            base = base_registry[name]
        elif name in registry:
            base = registry[name].__class__
        else:
            raise ValueError(f"Can't find requested base class {name}.")
        return base

    @staticmethod
    def _find_compute_function(name, computes):
        func = None
        if computes is not None:
            func = getattr(computes, name, None)
        if func is None:
            if hasattr(computes, "__getitem__") and name in computes:
                func = computes[name]
            elif "." in name:
                modname, name = name.split(".")
                submod = getattr(xclim.compute, modname, None)
                func = getattr(submod, name, None)
            else:
                func = getattr(xclim.compute.generic, name, getattr(xclim.compute, name, None))
        if func is None:
            raise ValueError(f"Can't find requested compute function {name}.")
        return func

    def __dir__(self):
        """Autocompletion support for indicators."""
        return self.keys()

    def __getattr__(self, k):
        """Access indicators as properties of a module (Obj.name)."""
        if k in self.keys():
            return self[k]
        super().__getattribute__(k)
