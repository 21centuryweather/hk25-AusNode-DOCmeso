'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
# -- packages --
import numpy as np
import xarray as xr


# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_calc.connect_boundaries.contiguous_regions    as cR
import utils.util_calc.distance_matrix.distance_matrix          as dM
import utils.util_calc.area_weighting.globe_area_weight         as gW


# == Calculate metric ==
def calc_rome(da, distance_matrix = None, da_area = None, check_input = False, connect_lon = True):
    '''         
    rome = (1 / N) * sum(q_i(a, b)) for i = 1, 2, 3, n, when n > 1
    rome = A_a                                          when n = 1
    Where
        q_i = A_a + min(1, A_b / A_d) * A_b 
            A_a - larger area of unique pair
            A_b - smaller area of unique pair
            A_d - squared shortest distance between unique pair boundaries (km)
        N = n
    ref: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD031801
    
    Function takes:
    da              - best: 2D binary xarray data array with coords = ['lat', 'lon'], elements: [0, 1], numpy array works too
    distance_matrix - from utils
    da_area         - from utils
    '''
    # -- check input --
    if check_input:
        # -- check elements --
        if not np.array_equal(np.unique(da), [0, 1]):
            print('check the input')
            print('needs da with only [0, 1]')
            print('exiting')
            exit()
        # -- check shape --
        try:
            lat_dim, lon_dim = np.shape(da)
            if not lat_dim == len(da.lat) or not lon_dim == len(da.lon):
                print('check the input')
                print('needs da in shape [len(lat), len(lon)]')
                print('exiting')
                exit()                
        except:
            print('check the input')
            print('needs da as xarray object with [lat, lon]')
            print('exiting')
            exit()

    # -- contiguous regions --
    da, labels = cR.get_contiguous_regions(da, connect_lon)                  # each object gets a unique number specified by labels

    # -- distance matrix --
    if distance_matrix is None:
        distance_matrix = dM.create_distance_matrix(da.lat, da.lon)                 # quicker to define this before, and give to functino

    # -- area_weghting --
    if da_area is None:
        da_area = gW.get_area_matrix(da.lat, da.lon)                                # quicker to define this before, and give to functino

    # -- rome --
    if not len(labels) == 1:
        # -- q-weights --
        weights = []
        for i, label_i in enumerate(labels[0:-1]): 
            obj_i = da.isin(label_i)                                                # has the label number
            obj_i = xr.where(obj_i > 0, 1, 0)                                       # [0, 1]
            obj_i_dist_to = dM.find_distance(obj_i, distance_matrix)                # distance from obj_i to rest of domain
            obj_i_area = (obj_i * da_area).sum()                                    # area of object
            for _, label_j in enumerate(labels[i+1:]):
                obj_j = da.isin(label_j)
                obj_j = xr.where(obj_j > 0, 1, np.nan)
                obj_j_area = (obj_j * da_area).sum()
                A_a, A_b = sorted([obj_i_area, obj_j_area], reverse=True)
                A_d = (obj_i_dist_to * obj_j).min()**2                              # squared closest distance to obj_j
                weights.append(A_a + np.minimum(1, A_b / A_d) * A_b)
        return np.mean(np.array(weights))
    else:
        return np.sum((da.isin(labels) > 0) * 1 * da_area)                          # area of object


# == when this script is ran ==
if __name__ == '__main__':
    ''


