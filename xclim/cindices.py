"""Class version of indices.

This does nothing more that assigning attributes to the output array. I suspect there is a more compact way to do this
and avoid these classes entirely.

This will be deleted eventually.
"""

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
    description = ''
    units = ''

    def __init__(self):
        self.meta_wrap(self.compute)
        self.__doc__ = self.compute.__doc__

    def add_meta(self, arr):
        arr.attrs.update(self.attrs)
        return arr

    def meta_wrap(self, func):
        @wraps(func)
        def wrapper(*args, **kwds):
            self.validate(*args)
            out = func(*args, **kwds)
            return self.add_meta(out)

        self.__call__ = wrapper

    @staticmethod
    def compute(*args, **kwds):
        """Function taking one or more DataArray arguments and
        optional parameters and returning a new DataArray storing the
        computed indicator."""
        raise NotImplementedError

    def validate(*args):
        """Function validating the input data. Metadata attributes
        non-conformity will raise a warning, while unmet data
        requirements such as non consecutive dates in a time
        series will raise an error.


        Notes
        -----
        Ideally, long running validating functions would only be run once,
        and a store of checksums and the validation outputs would be maintained.
        """
        raise NotImplementedError


class CDD(UnivariateFunc):
    attrs = {'standard_name': 'cooling_degree_days',
             'long_name': 'cooling degree days',
             'units': 'K*day'
             }

    compute = staticmethod(cooling_degree_days)

    def validate(self, tas):
        check_valid_temperature(tas)
        check_valid(tas, 'cell_methods', 'time: mean within days')
