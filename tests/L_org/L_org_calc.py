'''
# -----------------
#     L_org
# -----------------
Calculation inspired by
paper - https://journals.ametsoc.org/view/journals/atsc/80/12/JAS-D-23-0103.1.xml
code - https://github.com/giobiagioli/organization_indices/blob/main/org_indices.py, also in: utils/util_calc/doc_metrics/L_org/L_org_rectangular.py

'''

# == imports ==
# -- Packages --
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

def get_area_matrix(lat, lon):
    ''' # area of domain: cos(lat) * (dlon * dlat) R^2 (area of gridbox decrease towards the pole as gridlines converge) '''
    lonm, latm = np.meshgrid(lon, lat)
    dlat = lat.diff(dim='lat').data[0]
    dlon = lon.diff(dim='lon').data[0]
    R = 6371     # km
    area =  np.cos(np.deg2rad(latm))*np.float64(dlon * dlat * R**2*(np.pi/180)**2) 
    da_area = xr.DataArray(data = area, dims = ["lat", "lon"], coords = {"lat": lat, "lon": lon}, name = "area")
    return da_area

# == convective-core evaluation funcs ==
def find_counts_in_radius_bins(lat_c, lon_c, lat_coords, lon_coords, r_bin_edges):
    dist_c = haversine_dist(np.array([lat_c] * len(lat_coords)), 
                            np.array([lon_c] * len(lon_coords)), 
                            np.array(lat_coords), 
                            np.array(lon_coords))                                                                       # distance to other cores 
    dist_c = dist_c[(dist_c > 0) & (dist_c < np.max(r_bin_edges))]                                                      # keep distances that are not distance from the core to itself, and the distances greater than r_max        
    bin_indices = np.digitize(dist_c, r_bin_edges, right=True)                                                          # identifies associated bin (last bin edge is inclusive right, others not) (firt bin edge not inclusive left) (note: zero is outside range below, len(r_bins) is outside range above)
    counts = np.array([np.sum(bin_indices == i) for i in range(1, len(r_bin_edges))])                                   # number of points in different r_bins
    return np.cumsum(counts)  

def calculate_overlap(lat_c, lon_c, r_dist, dist_to_y_edges, dist_to_x_edges):
    """
    Calculate the area of overlap when both latitude and longitude boundaries are exceeded.
    This is a rough approximation and may require adjustment based on the specific geometry.
    """
    # Get the angle of the overlap using the distances to both the latitude and longitude edges
    cos_lat = np.clip(dist_to_y_edges[0] / r_dist, -1, 1)
    cos_lon = np.clip(dist_to_x_edges[0] / r_dist, -1, 1)

    # Calculate the angle
    angle_lat = 2 * np.arccos(cos_lat)
    angle_lon = 2 * np.arccos(cos_lon)
    
    # Approximate the overlap area as a sector intersection
    overlap_area = (angle_lat * angle_lon / (2 * np.pi)) * np.pi * r_dist**2  # Sector intersection area
    return overlap_area

def find_weights(lat_c, lon_c, lat, lon, r_bin_edges, periodic_lon):
    for r, r_dist in enumerate(r_bin_edges[1:]):                                                            # first r_bins entry is zero
        dist_to_y_edges = haversine_dist(np.array([lat_c] * 2), 
                                            np.array([lon_c] * 2), 
                                            np.array([lat[0], lat[-1]]),                                    # top and bottom edge
                                            np.array([lon_c] * 2))                                          # distance from core to meridional edges    
        area_circle_bin = np.pi * r_dist**2                                                                 # area of circle considered
        area_inside     = np.pi * r_dist**2                                                                 # area of circle included in domain (updated in loop)
        for i, edge_dist in enumerate(dist_to_y_edges):                                                     # check both edges
            if r_dist > edge_dist:                                                                          # if the circle extends past the boundary
                theta = 2 * np.arccos(edge_dist / r_dist)                                                   # angle of segment exceeding boundary (cosine rule: adjacent / hypo.)
                area_circsec = (theta / 2) * r_dist**2                                                      # circle sector area: (theta / 2pi) * pi r^2  
                area_triangle = edge_dist * np.sqrt(r_dist**2 - edge_dist**2)                               # triangle of circle segment (pyth. th and area of triangle * 2)   
                area_inside -= (area_circsec - area_triangle)                                               # subtract the area of the circle excluded due to the boundary
        if not periodic_lon:
            dist_to_x_edges = haversine_dist(np.array([lat_c] * 2), 
                                            np.array([lon_c] * 2), 
                                            np.array([lat_c] * 2),                                          # Same latitude, different longitudes
                                            np.array([lon[0], lon[-1]]))                                    # West and East edges (longitude)
            for i, edge_dist in enumerate(dist_to_x_edges):  
                if r_dist > edge_dist:                                                                      # If the circle extends past the longitude boundary
                    theta = 2 * np.arccos(edge_dist / r_dist)                                               # Angle of segment exceeding boundary
                    area_circsec = (theta / 2) * r_dist**2                                                  # Circle sector area
                    area_triangle = edge_dist * np.sqrt(r_dist**2 - edge_dist**2)                           # Triangle area of segment
                    area_inside -= (area_circsec - area_triangle)                                           # Subtract excluded area
            if r_dist > min(dist_to_y_edges[0], dist_to_x_edges[0]):                                        # if both boundaries are exceeded
                overlap_area = calculate_overlap(lat_c, lon_c, r_dist, dist_to_y_edges, dist_to_x_edges)
                area_inside += overlap_area                                                                 # Add back the overlap area that was subtracted twice
        yield area_circle_bin / area_inside                                                                 # This weighs the number of included points with a greater weight if part of the circle is outside the domain


# == metric calc ==
def get_L_org(lat_coords, lon_coords, da_area, r_bin_edges, periodic_lon = False):
    # -- core evaluation -- 
    n_c = len(lat_coords)                                                                                                           # number of "cores"
    cum_counting = np.zeros((n_c, len(r_bin_edges) - 1))                                                                            # considering bin edges (one less bin than bin edges)
    weights = np.ones((n_c, len(r_bin_edges) - 1))                                                                                  # This weights the number of included points with a greater weight if part of the circle is outside the domain
    for c, (lat_c, lon_c) in enumerate(zip(lat_coords, lon_coords)):                                                                # position of selected core
        cum_counting[c,:] = find_counts_in_radius_bins(lat_c, lon_c, lat_coords, lon_coords, r_bin_edges)                           # how many cores fall in each circle radius
        weights[c, :] = np.fromiter(find_weights(lat_c, lon_c, da_area.lat, da_area.lon, r_bin_edges, periodic_lon), dtype=float)   # weight for each bin      
    cum_counting_weighted = cum_counting * weights                                                                                  # apply weights to the count

    # -- Observational Bsag function (L^hat) -- 
    Besag_obs = np.sqrt((da_area.sum().data / (np.pi * n_c * (n_c - 1))) * np.sum(cum_counting_weighted, axis = 0))                 # one value for each r in r_bins
    Besag_obs = np.concatenate(([0], Besag_obs))                                                                                    # add zero at begining for integral

    # -- Theoretical Bsag function (L) -- 
    Besag_theor = r_bin_edges                                                                                                       # linearly increasing with r

    # -- L_org calc --                           
    L_org = np.trapz(Besag_obs - Besag_theor, x = r_bin_edges)                                                                      # integral between observed- and theoretical curve
    L_ir = np.sqrt(np.trapz((Besag_obs - Besag_theor)**2, x = r_bin_edges))                                                         # irregularity index (standard deviation between observed- and theoretical curve)
    return L_org, L_ir, Besag_obs, Besag_theor, r_bin_edges



if __name__ == '__main__':
    # -- change these --
    path_to_data = '/home/565/cb4968/hk25-AusNode-DOCmeso/tests/pr_percentiles_icon_d3hp003_3hrly_0-360_-30-30_3600x1800_2020-04_2021-03_var_2020_4_1.nc'
    folder_for_plots = '/home/565/cb4968/hk25-AusNode-DOCmeso/tests/L_org/plots'

    import xarray as xr

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
    # short range clustering, long range regularity
    random_noise = np.random.normal(loc=0.0, scale=0.01, size=(260, 490))
    clustered_field = np.copy(random_noise)
    cluster_centers = [(50, 100), (150, 300), (200, 400)]  # example cluster centers
    for center in cluster_centers:
        x, y = center
        radius = 20  # radius for the cluster
        for i in range(x - radius, x + radius):
            for j in range(y - radius, y + radius):
                if 0 <= i < 260 and 0 <= j < 490:
                    clustered_field[i, j] = np.random.uniform(0, 2)
    example_field = clustered_field + random_noise
    da_custom = xr.DataArray(                                          
        data = example_field,
        dims=["lat", "lon"],
        coords={"lat": da_orig.lat, "lon": da_orig.lon},
        )
    da = da_custom
    da_orig = da
    # print(da)
    # exit()

    # -- "example convective threshold" --
    threshold = da.quantile(0.90).data

    # -- smoothing --
    kernel_size, decay_distance = 8, 10
    da_smooth = apply_smoothing(da_orig, kernel_size, decay_distance)

    # -- cores --
    da, lat_coords, lon_coords = find_conv_cores(da_smooth, threshold, exceed_threshold = True)

    # -- bins of radius' to compare random distribution in circles with "pbserved" distribution --
    r_bin_edges = np.arange(0, 2500, 10)    # km
    # print(len(r_bin_edges))

    # -- domain area --
    da_area = get_area_matrix(da.lat, da.lon)

    # -- get observed and expected by random distribution curves --
    L_org, L_ir, Besag_obs, Besag_theor, r_bin_edges = get_L_org(lat_coords, lon_coords, da_area, r_bin_edges, periodic_lon = False)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(r_bin_edges, Besag_obs,                         label = 'L_obs',    color = "red",  linestyle = "-")
    ax.plot(r_bin_edges, Besag_theor,                       label = 'L_random', color = "blue", linestyle="--")
    ax.fill_between(r_bin_edges, Besag_obs, Besag_theor,    label = 'L_org',    color = "red",  alpha=0.3)
    ax.set_xlabel("r [km]")
    ax.set_ylabel("L_Besag")
    plt.legend()
    plt.grid(True)
    ax.text(0.25, 0.725, f"$I_{{org}} = {L_org:.2f}$", transform=ax.transAxes, fontsize=15, ha='center', va='center')
    ax.text(0.25, 0.65, f"$I_{{ir}} = {L_ir:.2f}$", transform=ax.transAxes, fontsize=15, ha='center', va='center')

    # -- save figure --
    filename = f'precip_smooth_cores_L_org'
    path = f'{folder_for_plots}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)

