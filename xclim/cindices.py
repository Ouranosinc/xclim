"""Class version of indices."""

from functools import wraps
from .checks import *
from .indices import *

class UnivariateFunc():
    """
    Base class for climate indices.

    The basic indice computation is done in the compute method.
    The __call__ method wraps the compute method and assigns metadata attributes to the output.

    """
    standard_name = ''
    long_name = ''
    units = ''
    comments = ''

    def __init__(self):
        self.meta_wrap(self.compute)
        self.__doc__ = self.compute.__doc__

    def add_meta(self, arr):
        arr.attrs.update(self.attrs)
        return arr


    def meta_wrap(self, func):
        @wraps(func)
        def wrapper(*args, **kwds):
            out = func(*args, **kwds)
            return self.add_meta(out)

        self.__call__ = wrapper

    @staticmethod
    def compute(*args, **kwds):
        raise NotImplementedError


class CDD(UnivariateFunc):
    attrs = {'standard_name': 'cooling_degree_days',
             'long_name': 'cooling degree days',
             'units': 'K*day'
             }

    compute = staticmethod(cooling_degree_days)

    @staticmethod
    def validate(tas):
        check_valid_temperature(tas)
        check_valid(tas, 'cell_methods', 'time: mean within days')
