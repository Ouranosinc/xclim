"""
Scaling method
============

TODO

References
----------

"""
from .utils import add_cyclic_bounds
from .utils import apply_correction
from .utils import broadcast
from .utils import get_correction
from .utils import group_apply
from .utils import parse_group


def train(x, y, group="time.month", kind="+", window=1):
    """Compute mean adjustment factors."""
    sx = group_apply("mean", x, group, window)
    sy = group_apply("mean", y, group, window)

    return get_correction(sx, sy, kind)


def predict(x, obj, interp=False):
    """Apply correction to data.
    """
    dim, prop = parse_group(obj.group)

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        obj = add_cyclic_bounds(obj, prop)

    factor = broadcast(obj, x, interp)

    out = apply_correction(x, factor, obj.kind)
    out["bias_corrected"] = True
    return out
