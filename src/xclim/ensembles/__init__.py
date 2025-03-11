"""
Ensemble tools
==============

This submodule defines some useful methods for dealing with ensembles of climate simulations.
In xclim, an "ensemble" is a `Dataset` or a `DataArray` where multiple climate realizations
or models are concatenated along the `realization` dimension.
"""

from __future__ import annotations

from xclim.ensembles._base import (
    create_ensemble,
    ensemble_mean_std_max_min,
    ensemble_percentiles,
)
from xclim.ensembles._partitioning import (
    fractional_uncertainty,
    general_partition,
    hawkins_sutton,
    lafferty_sriver,
)
from xclim.ensembles._reduce import (
    kkz_reduce_ensemble,
    kmeans_reduce_ensemble,
    make_criteria,
    plot_rsqprofile,
)
from xclim.ensembles._robustness import (
    robustness_categories,
    robustness_coefficient,
    robustness_fractions,
)
