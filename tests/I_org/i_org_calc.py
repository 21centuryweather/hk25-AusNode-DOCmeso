''' 
# -----------------
#     I_org
# -----------------
paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016ms000802

'''

# == imports ==
# -- packages --
import numpy as np
import xarray as xr
from scipy.ndimage import convolve
from scipy.ndimage import maximum_filter


# == smoothing function --
def exponential_kernel(size, decay_distance):
    ''' Creates grid to smooth over, and the magnitude of decay '''
    x, y = np.meshgrid(np.arange(size), np.arange(size))        # matrix to smooth over
    dist = np.sqrt((x - size//2)**2 + (y - size//2)**2)         # distance to smooth over
    kernel = np.exp(-dist / decay_distance)                     # rate of decay
    kernel /= kernel.sum()                                      # smoothing
    return kernel

def apply_smoothing(da_orig, kernel_size, decay_distance):
    kernel = exponential_kernel(size = kernel_size, decay_distance = decay_distance)
    da = convolve(da_orig, kernel, mode='nearest')                   # returns numpy, so put in xarray again later
    da = xr.DataArray(                                          
        data = da,
        dims=["lat", "lon"],
        coords={"lat": da_orig.lat, "lon": da_orig.lon},
        )
    return da

def find_conv_cores(da, threshold, exceed_threshold = True):
    # -- find local maxima --
    local_max = maximum_filter(da, size=3)
    local_maxima = (da == local_max) * 1

    # -- only consider points above threshold --
    if exceed_threshold:
        local_maxima = (local_maxima * da) > threshold

    # -- find associated lat, lon --
    latitudes, longitudes = np.where(local_maxima)
    lat_values = local_maxima.lat.values[latitudes]
    lon_values = local_maxima.lon.values[longitudes]
    return da, lat_values, lon_values


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
        pair_distances = haversine_dist(conv_a_lat, conv_a_lon, lat_coords, lon_coords)         # distance to other convective points
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

def get_poisson_cdf(da, n, NN_distances, r):
    ''' Expected cdf from random poisson process '''
    # -- cdf of theoretical distribution --
    lamda = n / get_area_matrix(da.lat, da.lon).sum().data                                     # normalization factor
    poisson_cdf = 1 - np.exp(-lamda * np.pi * (r - np.min(NN_distances))**2)                    # cdf of random distribution
    return poisson_cdf

# == metric calc ==
def get_i_org(da, lat_coords, lon_coords):
    obs_cdf, n, NN_distances, r = get_cdf(lat_coords, lon_coords)
    poisson_cdf = get_poisson_cdf(da, n, NN_distances, r)
    # i_org = simps(obs_cdf, poisson_cdf)
    i_org = np.trapz(obs_cdf, poisson_cdf)
    return i_org


# == when this script is ran ==
if __name__ == '__main__':
    # -- change these --
    path_to_data = '/home/565/cb4968/hk25-AusNode-DOCmeso/tests/pr_percentiles_icon_d3hp003_3hrly_0-360_-30-30_3600x1800_2020-04_2021-03_var_2020_4_1.nc'
    folder_for_plots = '/home/565/cb4968/hk25-AusNode-DOCmeso/tests/I_org/plots'

    # -- data --
    ds = xr.open_dataset(path_to_data)
    da = ds['var'].isel(time = 0)
    lon_area = '100:149'
    lat_area = '-13:13'  
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    da = da.fillna(0)
    da_orig = da

    # -- "example convective threshold" --
    threshold = da.quantile(0.90).data

    # -- smoothing --
    kernel_size, decay_distance = 8, 10
    da_smooth = apply_smoothing(da_orig, kernel_size, decay_distance)

    # -- cores --
    da, lat_coords, lon_coords = find_conv_cores(da_smooth, threshold, exceed_threshold = True)

    # -- cdf of "observed" distribution --
    obs_cdf, n, NN_distances, r = get_cdf(lat_coords, lon_coords)

    # -- cdf expected by random distribution --
    poisson_cdf = get_poisson_cdf(da, n, NN_distances, r)

    # -- i_org --
    i_org = get_i_org(da, lat_coords, lon_coords)
    # print(i_org)
    # exit()

    # == plots ==
    # -- cdf distributions --
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    fig = plt.figure(figsize=(6, 4))
    plt.plot(r, obs_cdf,        label="Observed CDF", color="red",  linestyle = "--")
    plt.plot(r, poisson_cdf,    label="Poisson CDF",  color='blue', linestyle = '--')
    plt.xlabel("NN Distance [km]")
    plt.ylabel("Cumulative Distribution Function (NNCDF)")
    plt.legend()
    plt.grid(True)
    plt.legend()
    # -- save figure --
    filename = f'precip_smooth_cores_NN_cdf_and_random_cdf'
    path = f'{folder_for_plots}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)

    # -- metric visualizaiton --
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(poisson_cdf, obs_cdf, label="NNCDF as a function of expected from random", color="red", linestyle="-")
    ax.fill_between(poisson_cdf, obs_cdf, color="red", alpha=0.3, label="Area under CDF (I_org)")
    ax.plot([0, 1], [0, 1], label="1:1 Line (Random)", color="k", linestyle="--")  # Reference line for perfect match
    ax.set_xlabel("Poisson CDF")
    ax.set_ylabel("Observed CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.text(0.375, 0.675, f"$I_{{org}} = {i_org:.2f}$", transform=ax.transAxes, fontsize=30, ha='center', va='center')

    # -- save figure --
    filename = f'precip_smooth_cores_metric_illustration'
    path = f'{folder_for_plots}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)




















