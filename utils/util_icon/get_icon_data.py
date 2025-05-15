'''
# --------------------
# ICON: icon_d3hp003
# --------------------

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
import utils.util_obs.get_imerg_data as gI


# == post-process ==
def process_data_further(da):
    ''' This function can be defined and given externally '''
    return da


# == pre-process ==
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

def pre_process(ds, process_request):
    var_name, dataset, time_period, t_freq, lon_area, lat_area, resolution = process_request
    # -- pick out variable --    
    da = ds[var_name].load()

    # -- temporally resample --
    if t_freq == 'hrly':
        da = da.resample(time='1h').mean()
    elif t_freq == '3hrly':
        da = da.resample(time='3h').mean()
    elif t_freq == 'daily':
        da = da.resample(time='1d').mean()
    else:
        pass

    # -- load IMERG data for interpolation --
    var =           'precipitation'
    dataset =       'IMERG'
    time_str =      '2020-03-01'
    t_freq =        ''                  # don't need to resample
    resolution =    0.1
    process_request = ['precipitation', 'IMERG', time_str, t_freq, lon_area, lat_area, resolution]
    da_imerg = gI.get_data(process_request, process_data_further = process_data_further)  
    da_imerg = da_imerg.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    
    lon = da_imerg['lon'].values
    lat = da_imerg['lat'].values

    # print(da)
    # print(da_imerg)
    # exit()

    # -- interpolate healpix grid --
    this_nside = hp.get_nside(da) # I have to do it to the whole domain
    cells = get_nn_lon_lat_index(this_nside, lon, lat) 
    da = da.isel(cell = cells)

    return da


# == get raw data ==
def get_data(process_request, process_data_further):
    var, dataset, time_str, t_freq, lon_area, lat_area, resolution = process_request

    # -- get file and open data --
    zoom = 'z10'                                            
    time_freq = '1H'
    folder = f'/g/data/qx55/germany_node/d3hp003.zarr/'
    filename = f'PT{time_freq}_point_{zoom}_atm.zarr'
    ds  = xr.open_zarr(f'{folder}/{filename}')
    ds  = ds.sel(time = time_str)
    # print(ds)
    # exit()

    # -- pre-process --
    da = pre_process(ds, process_request)

    # -- post-process --
    da = process_data_further(da)

    da.attrs = {}
    return da


if __name__ == '__main__':
    # == specify data and data process ==
    var =           'pr'
    dataset =       'icon_d3hp003'
    time_str =      '2020-03-01'
    t_freq =        '3hrly'
    lon_area =      '100:149'
    lat_area =      '-13:13'
    resolution =    0.1

    process_request = [var, dataset, time_str, t_freq, lon_area, lat_area, resolution]
    da = get_data(process_request, process_data_further)
    print(da)
    exit()










