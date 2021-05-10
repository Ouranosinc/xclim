#!/usr/bin/python
# This script downloads data from PAVICS into a subfolder of docs/notebooks/
# It makes one file per realization, merging tasmax, tasmin and pr.
import xarray as xr

url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/birdhouse/cccma/CanESM2/rcp85/day/atmos/r{i}i1p1/{var}/{var}_day_CanESM2_rcp85_r{i}i1p1_20060101-21001231.nc"

for real in range(1, 6):
    print(f"Opening realization {real}")
    ds = xr.open_mfdataset(
        [url.format(i=real, var=var) for var in ["tasmin", "tasmax", "pr"]],
        chunks={"lat": 4, "lon": 4},
    )
    print("Selection of a point over a large metropolis: Saint-Jean-Port-Joli, QC")
    sub = ds.sel(lat=47 + 13 / 60, lon=360 - 70 - 16 / 60, method="nearest")
    print("Writing the file")
    sub.drop_vars(["time_bnds", "height", "lat_bnds", "lon_bnds"]).to_netcdf(
        f"../docs/notebooks/earthcube_data/CanESM2_rcp85_r{real}i1p1_2006-2100_singlept.nc"
    )
