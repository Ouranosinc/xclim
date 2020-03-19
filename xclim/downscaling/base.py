"""Base classes for the downscaling module"""


class ParametrizableClass(object):
    """Helper base class that sets as attribute every kwarg it receives in __init__.

    Parameters are all public attributes. Subclasses should use private attributes (starting with _).
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def parameters(self):
        """Return all parameters as a dictionary"""
        return {
            key: val for key, val in self.__dict__.items() if not key.startswith("_")
        }
