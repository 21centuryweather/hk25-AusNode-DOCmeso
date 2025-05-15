import xarray as xr
import healpy as hp
import easygems.healpix as egh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd

# start and end time 
start_time = '2020-04-01T00:00:00'
end_time = '2021-03-31T23:00:00'  

# get hourly time series
time_series = pd.date_range(start=start_time, end=end_time, freq='H')

# convert to ISO 
time_strings = time_series.strftime('%Y-%m-%dT%H:%M:%S').tolist()

path='/g/data/nf33/hk25_DOCmeso/temp_data/observations/IMERG/doc_metrics/mean_area/IMERG/mean_area_IMERG_hrly_100-149_-13-13_3600x1800_2020-04_2021-03/'
fh = xr.open_dataset(path+'mean_area_IMERG_hrly_100-149_-13-13_3600x1800_2020-04_2021-03_var_2020_4_1.nc')
pr_mean = fh['var']
#lat = fh.lat
#lon = fh.lon
time = fh.time
fh.close()

# read ICON model output
zoom = '10'
#file = '/g/data/qx55/germany_node/d3hp003.zarr/PT1H_point_z' + zoom + '.zarr'
file = '/g/data/qx55/germany_node/d3hp003.zarr/PT1H_point_z10_atm.zarr/'
ds2d = xr.open_zarr(file)
pr_icon = ds2d['pr'] #precipitation

this_time = '2020-01-26T00:00:00'
#pr_icon_month = pr_icon.groupby("time.month").mean() # Calculate monthly mean.
this_month = 2

# read imerg data
file_IMERG = '/g/data/ia39/aus-ref-clim-data-nci/gpm/data/V07/2020/3B-HHR.MS.MRG.3IMERG.*.V07A.nc'

pr_imerg = xr.open_mfdataset(file_IMERG, combine = 'nested', concat_dim = 'time')
pr_imerg_monthly = pr_imerg['precipitation'].groupby('time.month').mean() # Calculate monthly mean.

#regridding for maritime continent
pr_imerg_monthly_mari = pr_imerg_monthly.sel(lon = slice(100, 150), lat = slice(-13, 13))

def get_nn_lon_lat_index(nside, lons, lats):
    """
    nside: integer, power of 2. The return of hp.get_nside()
    lons: uniques values of longitudes
    lats: uniques values of latitudes
    returns: array with the HEALPix cells that are closest to the lon/lat grid
    """
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest = True, lonlat = True),
        coords=[("lat", lats), ("lon", lons)],
    )
# Find the HEALPix pixels that are closest to the imerg grid

# longitudes and latitudes for the imerg grid
lon = pr_imerg_monthly_mari['lon'].values
lat = pr_imerg_monthly_mari['lat'].values
# nside for icon simulation, it should be equal to 2**zoom
for time in time_series[:10]:
    pr_icon_snap = pr_icon.sel(time = time) # select the time point
    this_nside = hp.get_nside(pr_icon)
    cells = get_nn_lon_lat_index(this_nside, lon, lat) 
    pr_icon_regrided = pr_icon_snap.isel(cell = cells)# regriding
    ds = pr_icon_regrided.to_dataset(name = 'pr')
    if 'hiopy::enable' in ds['pr'].attrs:
        del ds['pr'].attrs['hiopy::enable']
    ds['lat'] = lat   #!!this is not right:ds['xloc_final'] = np.sort(xloc_final)
    ds['lon'] = lon
    ds.to_netcdf('/g/data/w28/ym7079/Hackthon/DOC/hk25-AusNode-DOCmeso/get_metrics/Yinglin/ICON_regridding/'+'ICON'+str(time)[:-9]+'_'+str(time)[:-9:-6]+'.nc')

#ds.to_netcdf('/g/data/w28/ym7079/Hackthon/DOC/hk25-AusNode-DOCmeso/get_metrics/Yinglin/ICON_regridding/test.nc')
   

    
    
