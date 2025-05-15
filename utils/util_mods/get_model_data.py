'''
# ----------------------------------------------------------------------
#  Function to save model data regridded to IMERG grid for certain region
# ----------------------------------------------------------------------

'''
    
# == imports ==
# -- packages --
import xarray as xr
import os
import pandas as pd
from easygems import healpix as egh
import healpy as hp
import numpy as np

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())

# == post-process ==
def process_data_further(da, process_request):
    ''' Function to save regridded data '''
    ''
    # var_name, dataset, time_period, t_freq, lon_area, lat_area, resolution, name, _ = process_request
    # path_save = f"/scratch/nf33/hk25_DOCmeso/{dataset}_interp/{dataset}_{var_name}_{resolution}_{time_period.replace(' ','')}_{name}.nc"    
    
    # try: del da.attrs["hiopy::enable"]
    # except: pass

    # # Save as netcdf
    # da.to_netcdf(path_save)
    
    # print (f"{dataset} regridded file was saved")
    ''
    del da

# == pre-process ==
def pre_process(ds, process_request):
    var_name, dataset, time_period, t_freq, lon_area, lat_area, resolution, _, _ = process_request

    # -- pick out variable --    
    da = ds[var_name]#.load()
    
    # -- temporally resample --
    if t_freq == 'hrly':
        da = da.resample(time='1h').mean()
    elif t_freq == '3hrly':
        da = da.resample(time='3h').mean()
    else:
        pass

    # -- select region of interest --
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    
    return da

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

def regrid(ds, process_request):
    var_name, dataset, time_period, t_freq, lon_area, lat_area, resolution, _, ds_regrid = process_request
    
    # -- pick out variable --    
    da = ds[var_name]#.load()

    # -- temporally resample --
    if t_freq == 'hrly':
        da = da.resample(time='1h').mean()
    elif t_freq == '3hrly':
        da = da.resample(time='3h').mean()
    else:
        pass

    # Find the HEALPix pixels that are closest to the IMERG grid
    lon = ds_regrid['lon'].values
    lat = ds_regrid['lat'].values

    # nside for um simulation, it should be equal to 2**zoom
    this_nside = hp.get_nside(da) # I have to do it to the whole domain
    cells = get_nn_lon_lat_index(this_nside, lon, lat) 

    # Regridded
    ds_regrided = da.isel(cell = cells) # regriding

    return ds_regrided


def get_data(process_request, process_data_further):
    var, dataset, time_str, t_freq, lon_area, lat_area, resolution, _, ds_regrid = process_request

    # -- get file and open data --
    if dataset == "IMERG":
        year, month, day = time_str.split('-')
        folder = f'/g/data/ia39/aus-ref-clim-data-nci/gpm/data/V07/{year}'
        if int(month) == 12 and int(day) == 23:                                     # day is missing, give previous days value
            filename = f'3B-HHR.MS.MRG.3IMERG.{year}{month}22.V07A.nc'
            da = xr.open_dataset(f'{folder}/{filename}')
            da['time'] = da['time'] + pd.Timedelta(days=1)
        elif int(month) == 12 and int(day) == 24:                                   # day is missing, give previous days value
            filename = f'3B-HHR.MS.MRG.3IMERG.{year}{month}25.V07A.nc'
            da = xr.open_dataset(f'{folder}/{filename}')
            da['time'] = da['time'] - pd.Timedelta(days=1)
        else:
            filename = f'3B-HHR.MS.MRG.3IMERG.{year}{month}{day}.V07A.nc'
        da = xr.open_dataset(f'{folder}/{filename}')
    
    elif dataset == "UM":
        #folder = f'/scratch/nf33/Healpix_data/{dataset}'
        #filename = f'data.healpix.PT1H.{resolution}.zarr'
        folder = f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/'
        filename = f'data.healpix.PT{t_freq}.{resolution}.zarr'
        da  = xr.open_zarr(f'{folder}/{filename}')
        da  = da.sel(time = time_str) if time_str != "All" else da 
        
    else: # For ICON
        #folder   = f'/scratch/nf33/Healpix_data/{dataset}'
        #filename = f'data.healpix.PT1H.{resolution}.zarr'
        folder = f'/g/data/qx55/germany_node/d3hp003.zarr/'
        filename = f'PT{t_freq}_point_{resolution}_atm.zarr'
        da  = xr.open_zarr(f'{folder}/{filename}')
        da  = da.sel(time = time_str) if time_str != "All" else da 
        
    # -- pre-process --
    if dataset == "IMERG":
        da = pre_process(da, process_request)
    else:
        da = regrid(da, process_request)
        # da = process_data_further(da, process_request)
        
    return da


if __name__ == '__main__':
    # exit()

    # Get IMERG data: any date
    var =           'precipitation'
    dataset =       'IMERG'
    time_str =      '2020-03-01'
    t_freq =        'hourly'
    lon_area =      '100:149'
    lat_area =      '-10:10'
    resolution =    0.1

    process_request = [var, dataset, time_str, t_freq, lon_area, lat_area, resolution, None, None]
    ds_imerg = get_data(process_request, process_data_further)
    
    print ("IMERG data was read")
    
    # Get model data and interpolate to IMERG grid
    var   =         'pr'
    dataset =       'ICON' # "UM" or "ICON"
    time_str =      '2020-03-01' # or "All"
    t_freq =        '1H'
    lon_area =      None
    lat_area =      None
    resolution =    'z10'
    name       =    "MarCont" # Name to add to the saved file
    
    process_request = [var, dataset, time_str, t_freq, lon_area, lat_area, resolution, name, ds_imerg]
    ds_model = get_data(process_request, process_data_further)
    
    print(ds_model)
    exit()


