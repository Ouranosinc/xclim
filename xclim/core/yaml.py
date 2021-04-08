# -*- coding: utf-8 -*-
# noqa: D205,D400
"""

"""
from types import ModuleType
from typing import Type

from pkg_resources import resource_stream
from yaml import safe_load

from ..indices import generic
from .indicator import Daily, Indicator
from .units import declare_units
from .utils import wrapped_partial
