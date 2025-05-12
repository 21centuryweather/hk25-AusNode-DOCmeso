'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == calculate metric ==
def calculate_metric(data_objects):
    da, i, lon_area, lat_area = data_objects

    # -- check data --
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    # print(da)

    # -- calculate metric --
    da = da.mean(dim = ('lat', 'lon'))
    # print(da)

    # -- fill xr.Dataset --
    ds = xr.Dataset()
    ds['pr_mean'] = da
    ds = ds.expand_dims(dim = 'time')
    ds = ds.assign_coords(time=[da.time.data])
    # print(ds)
    # exit()
    return ds



