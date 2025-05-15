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
# from scipy.integrate import simps

# -- imported scripts --

def get_area_matrix(lat, lon):
    ''' # area of domain: cos(lat) * (dlon * dlat) R^2 (area of gridbox decrease towards the pole as gridlines converge) '''
    lonm, latm = np.meshgrid(lon, lat)
    dlat = lat.diff(dim='lat').data[0]
    dlon = lon.diff(dim='lon').data[0]
    R = 6371     # km
    area =  np.cos(np.deg2rad(latm))*np.float64(dlon * dlat * R**2*(np.pi/180)**2) 
    da_area = xr.DataArray(data = area, dims = ["lat", "lon"], coords = {"lat": lat, "lon": lon}, name = "area")
    return da_area

# == helper funcs ==
def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula)
    input: 
    lon range: [-180, 180]
    lat range: [-90, 90]
    (Takes vectorized input) 

    Formula:
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 
    where 
    phi -latitutde
    lambda - longitude
    '''
    R = 6371                                                                                    # radius of earth in km
    lat1 = np.deg2rad(lat1)                                                                     # function requires degrees in radians 
    lon1 = np.deg2rad(lon1-180)                                                                 # and lon in range [-180, 180]
    lat2 = np.deg2rad(lat2)                                                                     #
    lon2 = np.deg2rad(lon2-180)                                                                 #
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2     # Haversine formula
    h = np.clip(h, 0, 1)                                                                        # float point precision sometimes give error
    result =  2 * R * np.arcsin(np.sqrt(h))                                                     # formula rearranged for spherical distance
    return result


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
    return i_org


# == when this script is ran ==
if __name__ == '__main__':
    ''




