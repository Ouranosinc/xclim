# -*- coding: utf-8 -*-
"""
xclim units module

Defines the `xclim.units.units` pint UnitRegistry.
"""
import pint.converters
import pint.unit
from packaging import version


units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
units.define(
    pint.unit.UnitDefinition(
        "percent", "%", ("pct",), pint.converters.ScaleConverter(0.01)
    )
)

# Define commonly encountered units not defined by pint
if version.parse(pint.__version__) >= version.parse("0.10"):
    units.define("@alias degC = C = deg_C")
    units.define("@alias degK = deg_K")
    units.define("@alias day = d")
    units.define("@alias hour = h")  # Not the Planck constant...
    units.define(
        "@alias degree = degrees_north = degrees_N = degreesN = degree_north = degree_N = degreeN"
    )
    units.define(
        "@alias degree = degrees_east = degrees_E = degreesE = degree_east = degree_E = degreeE"
    )

else:
    units.define("degC = kelvin; offset: 273.15 = celsius = C = deg_C")
    units.define("d = day")
    units.define("h = hour")
    units.define(
        "degrees_north = degree = degrees_N = degreesN = degree_north = degree_N "
        "= degreeN"
    )
    units.define(
        "degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE"
    )

units.define("[speed] = [length] / [time]")

# Default context.
null = pint.Context("none")
units.add_context(null)

# Precipitation units. This is an artificial unit that we're using to verify that a given unit can be converted into
# a precipitation unit. Ideally this could be checked through the `dimensionality`, but I can't get it to work.
units.define("[precipitation] = [mass] / [length] ** 2 / [time]")
units.define("mmday = 1000 kg / meter ** 2 / day")

units.define("[discharge] = [length] ** 3 / [time]")
units.define("cms = meter ** 3 / second")

hydro = pint.Context("hydro")
hydro.add_transformation(
    "[mass] / [length]**2",
    "[length]",
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3),
)
hydro.add_transformation(
    "[mass] / [length]**2 / [time]",
    "[length] / [time]",
    lambda ureg, x: x / (1000 * ureg.kg / ureg.m ** 3),
)
hydro.add_transformation(
    "[length] / [time]",
    "[mass] / [length]**2 / [time]",
    lambda ureg, x: x * (1000 * ureg.kg / ureg.m ** 3),
)
units.add_context(hydro)
units.enable_contexts(hydro)

# These are the changes that could be included in a units definition file.

# degrees_north = degree = degrees_N = degreesN = degree_north = degree_N = degreeN
# degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE
# degC = kelvin; offset: 273.15 = celsius = C
# day = 24 * hour = d
# @context hydro
#     [mass] / [length]**2 -> [length]: value / 1000 / kg / m ** 3
#     [mass] / [length]**2 / [time] -> [length] / [time] : value / 1000 / kg * m ** 3
#     [length] / [time] -> [mass] / [length]**2 / [time] : value * 1000 * kg / m ** 3
# @end
