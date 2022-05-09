"""
Ensemble tools.

This submodule defines some useful methods for dealing with ensembles of climate
simulations. In xclim, an "ensemble" is a `Dataset` or a `DataArray` where multiple
climate realizations or models are concatenated along the `realization` dimension.
"""
from __future__ import annotations

from ._base import create_ensemble
from ._base import ensemble_mean_std_max_min
from ._base import ensemble_percentiles
from ._reduce import kkz_reduce_ensemble
from ._reduce import kmeans_reduce_ensemble
from ._reduce import plot_rsqprofile
from ._robustness import change_significance
from ._robustness import robustness_coefficient
