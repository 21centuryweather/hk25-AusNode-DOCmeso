import pandas as pd
import datetime as dt
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

import imageio
import glob
import os

import warnings
warnings.simplefilter("ignore")


files_2020 = sorted(glob.glob("/g/data/ia39/aus-ref-clim-data-nci/gpm/data/V07/2020/3B-HHR.MS.MRG*.nc"))
files_2021 = sorted(glob.glob("/g/data/ia39/aus-ref-clim-data-nci/gpm/data/V07/2021/3B-HHR.MS.MRG*.nc"))
all_files = files_2020 + files_2021

#Dask
ds = xr.open_mfdataset(
    all_files,
    combine='by_coords',
    parallel=True,
    chunks={'time': 1}
)


ds_subset = ds.sel(time=slice("2020-04-01T00:00:00", "2021-03-31T23:00:00"))

rain_mc = ds_subset.precipitation.sel(
    lon=slice(-180, 180),
    lat=slice(-30, 30)
)

# Daily mean rain rate (mm/hr)
rain_daily = rain_mc.resample(time="1D").mean()

# Shift longitudes from [0, 360] to [-180, 180]
rain_daily = rain_daily.assign_coords(
    lon=(((rain_daily.lon + 180) % 360) - 180)
).sortby('lon')


rain_daily = rain_daily.sortby("lat")

#mean over all days (1-year average of daily mean rain rate)
mean_precip = rain_daily.mean(dim='time').compute()

fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

vmin = 0
vmax = 10  


mesh = ax.pcolormesh(
    mean_precip['lon'], mean_precip['lat'], mean_precip.values.T,
    cmap="Blues", vmin=vmin, vmax=vmax,
    transform=ccrs.PlateCarree()
)


ax.set_title("Mean Daily Rain Rate (Apr 2020 – Mar 2021)", fontsize=14)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')


cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('Mean Daily Precipitation (mm/hr)', fontsize=12)


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}


output_dir = "/scratch/nf33/gs5098/imerg/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "mean_daily_precipitation_map.png"), dpi=150, bbox_inches='tight')
plt.show()
