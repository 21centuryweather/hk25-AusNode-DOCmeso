import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings


ds = xr.open_mfdataset(
    "/g/data/ia39/aus-ref-clim-data-nci/gpm/data/V07/202[0-1]/3B-HHR.MS.MRG*.nc",
    combine='by_coords',
    parallel=True
    #chunks={'time': 24, 'lat': 180, 'lon': 360}  # tune based on memory
)



ds_subset = ds.precipitation.sel(time=slice("2020-04-01", "2021-03-31"))

rain_mc = ds_subset.sel(lat=slice(-30, 30), lon=slice(-180, 180))

# Daily mean
rain_daily = rain_mc.resample(time="1D").mean()

# Convert DataArray to Dataset
precip_ds = rain_daily.to_dataset(name="precipitation")

# Save to NetCDF file
precip_ds.to_netcdf("/scratch/nf33/gs5098/data/daily_mean_precip_2020_2021.nc")