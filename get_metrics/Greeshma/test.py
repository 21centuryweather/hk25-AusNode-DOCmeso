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

# Dask
ds = xr.open_mfdataset(
    all_files,
    combine='by_coords',
    parallel=True,      
    chunks={'time': 1} 
)

ds_subset = ds.sel(time=slice("2020-04-01T00:00:00", "2021-03-31T23:00:00"))


rain_mc = ds_subset.precipitation.sel(
    lon=slice(0, 360),
    lat=slice(-13, 13)
)

rain_mc_mm_day = rain_mc * 24
rain_daily = rain_mc_mm_day.resample(time="1D").mean()


rain_mean = rain_daily.mean(dim="time")

# Output directory
save_dir = "/scratch/nf33/gs5098/data/"
os.makedirs(save_dir, exist_ok=True) 

filename = os.path.join(save_dir, "precip_mean.nc")
rain_mean.to_netcdf(filename)

print(f"Saved {filename}")
