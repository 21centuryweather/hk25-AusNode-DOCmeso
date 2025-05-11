'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
# -- packages --
import numpy as np
import xarray as xr
import skimage.measure as skm

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_calc.connect_boundaries.connect_lon_boundary  as cB


# == calc ==
def get_contiguous_regions(da, connect_lon = True):
    ''' da is 2D array with [0, 1] '''
    # -- convective objects --
    labels_np = skm.label(da, background = 0, connectivity = 2)                     # returns numpy array
    labels_np = cB.connect_boundary(labels_np)                                      # connect objects across boundary
    labels = np.unique(labels_np)[1:]                                               # first unique value (zero) is background
    labels_xr = xr.DataArray(                                                       # convective objects
        data = labels_np,
        dims=["lat", "lon"],
        coords={"lat": da.lat, "lon": da.lon},
        )
    return labels_xr, labels



# == when this script is ran ==
if __name__ == '__main__':
    ''



