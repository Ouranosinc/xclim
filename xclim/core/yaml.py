# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
YAML parser submodule
=====================

Helper functions to construct indicators from yaml files.

YAML file structure
-------------------

Indicator-defining yaml files are structured in the following way:

    indices:
      <identifier>:
        realm: <realm>  # Optional, defaults to "atmos".
        output:
          var_name: <var_name>  # Optional, defaults to "identifier",
          standard_name: <standard_name>  # Optional
          long_name: <long_name>  # Optional
          units: <units>  # Optional, defaults to ""
          cell_methods:
            - <dim1> : <method 1>
            ...

        index_functions:
          name: <function name>  # Refering to a function in xclim.indices.generic
          parameters:
            <param name>  # Refering to a parameter of the function above.
              kind: <param kind>  # One of quantity, operator or reducer
              data: <param data>  # If kind = quantity, the magnitude of the quantity
              units: <param units>  # If kind = quantity, the units of the quantity
              # If kind = quantity, the value passed to the function is : "<param data> <param units>".
              operator: <param value>  # If kind = operator, the operator to use, see below.
              reducer: <param value>  # If kind = reducer, the reducing operation to use, see below.
            ...

        input:
          <var1> : <variable type 1>  # <var1> refers to a name in the function above, see below.
          ...
      ...  # and so on.

Other fields can be found in the yaml file, but they will not be used by xclim. In the following,
the section under `<identifier>` is refered to as `data`.

Parameters
----------
Mappings passed in the `data.index_functions.parameters` section can be of 3 kinds:

    - "quantity", a quantity with a magnitude and some units,
    - "operator", one of "<", "<=", ">", ">=", "==", "!=", an operator for conditional computations.
    - "reducer", one of "maximum", "minimum", "mean", "sum", a reducing method name.

Inputs
------
As xclim has strict definitions of possible input variables (see :py:data:`xclim.core.yaml.variables`),
the mapping of `data.input` simply links a variable name from the function in `data.index_function.name`
to one of those official variables.
"""
from typing import Type

from pkg_resources import resource_stream
from yaml import safe_load

from ..indices import generic
from .indicator import Daily, Indicator
from .units import declare_units
from .utils import wrapped_partial

# Official variables definitions
variables = safe_load(resource_stream("xclim.data", "variables.yml"))["variables"]


def create_indicator(
    data: dict, base: Type[Indicator] = Daily, identifier=None, realm="atmos"
):
    """Create an indicator subclass and instance from a dictionary.

    See the submodule's doc for more details in how that dictionary should be structured.
    """
    # Get identifier
    # Priority is : passed arg -> givin in data -> var_name in data.output
    identifier = identifier or data.get("identifier", data["output"]["var_name"])

    # Make cell methods. YAML will generate a list-of-dict structure, put it back in a space-divided string
    if "cell_methods" in data["output"]:
        methods = []
        for cellmeth in data["output"]["cell_methods"]:
            methods.append(
                "".join([f"{dim}: {meth}" for dim, meth in cellmeth.items()])
            )
        cell_methods = " ".join(methods)
    else:
        cell_methods = None

    # Override input metadata
    params = {}
    input_units = {}
    for varname, name in data["input"].items():
        # Indicator's new will put the name of the variable as its default,
        # we override this with the real variable name.
        # Also take the dimensionaliy and description from the yaml of official variables.
        # Description overrides the one parsed from the generic compute docstring
        # Dimensionality goes into the declare_units wrapper.
        params[varname] = {
            "default": name,
            "description": variables[name]["description"],
        }
        input_units[varname] = f"[{variables[name]['dimensionality']}]"

    # Generate compute function
    # data.index_function.name refers to a function in xclim.indices.generic
    # data.index_function.parameters is a list of injected arguments.
    compute = getattr(generic, data["index_function"]["name"])
    injected_params = {}
    for name, param in data["index_function"].get("parameters", {}):
        if param["kind"] == "quantity":
            # A string with units
            injected_params[name] = f"{param['data']} {param['units']}"
        else:  # "reducer", "condition"
            # Simple string-like parameters, value is stored in a field of the same name as the kind.
            injected_params[name] = param[param["kind"]]

    compute = wrapped_partial(declare_units(**input_units)(compute), **injected_params)

    indicator = base(
        # General
        realm=data.get("realm", realm),
        identifier=identifier,
        # Output meta
        var_name=data["output"].get("var_name", identifier),
        standard_name=data["output"].get("standard_name"),
        long_name=data["output"].get("long_name"),
        units=data["output"].get("units"),
        cell_methods=cell_methods,
        # Input data, override defaults given in generic compute's signature.
        parameters=params,
        nvar=len(data["input"]),
        compute=compute,
    )

    return indicator
