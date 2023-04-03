from __future__ import annotations

from xclim.ensembles import (
    concat_hist,
    hawkins_sutton,
    model_in_all_scens,
    single_member,
)


def test_hawkins_sutton(open_dataset):
    """Just a smoke test - looking for data for a hard validation."""
    dims = {"run": "member", "scen": "scenario"}
    da = (
        open_dataset(
            "uncertainty_partitioning/cmip5_pr_global_mon.nc", branch="hawkins_sutton"
        )
        .pr.sel(time=slice("1950", None))
        .rename(dims)
    )
    da1 = model_in_all_scens(da)
    dac = concat_hist(da1, scenario="historical")
    das = single_member(dac)
    hawkins_sutton(das)
