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

# Open all files with Dask
ds = xr.open_mfdataset(
    all_files,
    combine='by_coords',
    parallel=True,
    chunks={'time': 1}
)

# Select time and spatial region
ds_subset = ds.sel(time=slice("2020-04-01T00:00:00", "2021-03-31T23:00:00"))

rain_mc = ds_subset.precipitation.sel(
    lon=slice(-180, 180),  # full range for proper shift
    lat=slice(-30, 30)
)

# Daily mean rain rate (mm/hr)
rain_daily = rain_mc.resample(time="1D").mean()

# Shift longitudes from [0, 360] to [-180, 180]
rain_daily = rain_daily.assign_coords(
    lon=(((rain_daily.lon + 180) % 360) - 180)
).sortby('lon')

# Ensure ascending latitude
rain_daily = rain_daily.sortby("lat")

# Output folder
output_dir = "/scratch/nf33/gs5098/imerg/"
os.makedirs(output_dir, exist_ok=True)

# Loop over all days
for i in range(rain_daily.time.size):
    day = rain_daily.isel(time=i).compute()
    date_str = np.datetime_as_string(day.time.values, unit='D')

    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    vmin = 0
    vmax = 17

    # Plot data
    mesh = ax.pcolormesh(
        day['lon'], day['lat'], day.values.T,
        cmap="Blues", vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    # Title and map features
    ax.set_title(f"Daily Mean Precipitation: {date_str}", fontsize=14)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Horizontal colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Daily Mean Precipitation (mm/hr)', fontsize=12)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Save figure
    filename = os.path.join(output_dir, f"daily_mean_{date_str}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
