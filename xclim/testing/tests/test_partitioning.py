from xclim.ensembles import (
    concat_hist,
    hawkins_sutton,
    model_in_all_scens,
    single_member,
)


def test_hawkins_sutton(open_dataset):
    """Just a smoke test - looking for data for a hard validation."""
    dims = {"member": "run", "scenario": "scen"}
    da = open_dataset(
        "uncertainty_partitioning/cmip5_pr_global_mon.nc", branch="hawkins_sutton"
    ).pr
    da1 = model_in_all_scens(da, dimensions=dims)
    dac = concat_hist(da1, scen="historical")
    das = single_member(dac, dimensions=dims)
    hawkins_sutton(das, dimensions=dims)
