"""Grouping classes"""
import xarray as xr

from .base import ParametrizableClass


class BaseGrouping(ParametrizableClass):
    def group(self, da: xr.DataArray):
        return self._group(da)

    def add_group_axis(self, da: xr.DataArray):
        return self._add_group_axis(da)

    def _group(self, da):
        raise NotImplementedError

    def _add_group_axis(self, da):
        raise NotImplementedError


class EmptyGrouping(BaseGrouping):
    def _group(self, da):
        return da.expand_dims(group=xr.DataArray([1], dims=("group",), name="group"))

    def _add_group_axis(self, da):
        return da.assign_coords(
            group=xr.DataArray([1.0] * da["time"].size, dims=("time",), name="group")
        )


class MonthGrouping(BaseGrouping):
    def _group(self, da):
        group = da.time.dt.month
        group.name = "group"
        group.attrs.update(group_name="month")
        return da.groupby(group)

    def _add_group_axis(self, da):
        group = da.time.dt.month - 0.5 + da.time.dt.day / da.time.dt.daysinmonth
        group.name = "group"
        group.attrs.update(group_name="month")
        return da.assign_coords(group=group)


class DOYGrouping(BaseGrouping):
    def __init__(self, window=None):
        super().__init__(self, window=window)

    def _group(self, da):
        group = da.time.dt.dayofyear
        group.name = "group"
        group.attrs.update(group_name="dayofyear")
        if self.window is not None:
            da = da.rolling(time=self.window, center=True).construct(
                window_dim="window"
            )
            group = xr.concat([group] * self.window, da.window)
            da.rename(time="old_time").stack(time=("old_time", "window"))
            group.rename(time="old_time").stack(time=("old_time", "window"))
        return da.groupby(group)

    def _add_group_axis(self, da):
        group = da.time.dt.dayofyear
        group.name = "group"
        group.attrs.update(group_name="dayofyear")
        return da.assign_coords(group=group)
