"""Pipeline objects"""
from xarray import DataArray

from .detrending import NoDetrend
from .grouping import EmptyGrouping
from .mapping import DeltaMapping


def basicpipeline(
    obs: DataArray,
    sim: DataArray,
    fut: DataArray,
    detrender=NoDetrend(),
    grouper=EmptyGrouping(),
    mapper=DeltaMapping(),
):
    obs_trend = detrender.fit(obs)
    sim_trend = detrender.fit(sim)
    fut_trend = detrender.fit(fut)

    obs = obs_trend.detrend(obs)
    sim = sim_trend.detrend(sim)
    fut = fut_trend.detrend(fut)

    mapper.fit(grouper.group(obs), grouper.group(sim))
    fut_corr = mapper.predict(grouper.add_group_axis(fut))

    fut_corr = fut_trend.retrend(fut_corr)

    return fut_corr.drop_vars("group")
