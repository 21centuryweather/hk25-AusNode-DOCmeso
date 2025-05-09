''' 
# -----------------
#     I_org
# -----------------
paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016ms000802

'''

# -- Packages --
import os
import sys
import numpy as np
import xarray as xr
import skimage.measure as skm
from scipy.integrate import simps

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_calc.connect_boundaries.contiguous_regions    as cR
import utils.util_calc.distance_matrix.distance_matrix          as dM
import utils.util_calc.area_weighting.globe_area_weight         as gW
import utils.util_calc.distance_matrix                          as dM


# == convective-core evaluation funcs ==
def get_cdf(lat_coords, lon_coords):
    n = len(lat_coords)                                                                         # number of convective points
    # -- find nearest neighbour distance between convective cores --
    NN_distances = []
    for a in np.arange(0, n):
        conv_a_lat = np.array([lat_coords[a]] * n)
        conv_a_lon = np.array([lon_coords[a]] * n)
        pair_distances = dM.haversine_dist(conv_a_lat, conv_a_lon, lat_coords, lon_coords)      # distance to other convective points
        NN_distances.append(np.min(pair_distances[pair_distances > 0]))                         # distance to closest other convective gridbox
    NN_distances = np.array(NN_distances)                                                       # minimum distance of touching points is about 270 km
    # -- cdf of "observed" distribution --
    r = np.linspace(np.min(NN_distances), np.max(NN_distances), 100)                            # Range of distances
    sorted_NN_distances = np.sort(NN_distances)
    cumulative_sum = np.zeros_like(r)
    for i, val in enumerate(r):
        cumulative_sum[i] = np.sum(sorted_NN_distances <= val)
    obs_cdf = cumulative_sum / len(sorted_NN_distances)
    return obs_cdf, n, NN_distances, r

def get_poisson_cdf(conv_regions, n, NN_distances, r):
    ''' Expected cdf from random poisson process '''
    # -- cdf of theoretical distribution --
    lamda = n / gW.get_area_matrix(conv_regions).sum().data                                     # normalization factor
    poisson_cdf = 1 - np.exp(-lamda * np.pi * (r - np.min(NN_distances))**2)                    # cdf of random distribution
    return poisson_cdf


# == metric calc ==
def get_i_org(conv_regions, lat_coords, lon_coords):
    obs_cdf, n, NN_distances, r = get_cdf(lat_coords, lon_coords)
    poisson_cdf = get_poisson_cdf(conv_regions, n, NN_distances, r)
    # i_org = simps(obs_cdf, poisson_cdf)
    i_org = np.trapz(obs_cdf, poisson_cdf)


# == when this script is ran ==
if __name__ == '__main__':
    ''




