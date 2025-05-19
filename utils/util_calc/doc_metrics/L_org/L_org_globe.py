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
import os
import sys
import numpy as np
import xarray as xr

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_calc.distance_matrix.distance_matrix          as dM
import utils.util_calc.distance_matrix                          as dM


# == convective-core evaluation funcs ==
def find_counts_in_radius_bins(lat_c, lon_c, lat_coords, lon_coords, r_bin_edges):
    dist_c = dM.haversine_dist(np.array([lat_c] * len(lat_coords)), 
                            np.array([lon_c] * len(lon_coords)), 
                            np.array(lat_coords), 
                            np.array(lon_coords))                                                                       # distance to other cores 
    dist_c = dist_c[(dist_c > 0) & (dist_c < np.max(r_bin_edges))]                                                      # keep distances that are not distance from the core to itself, and the distances greater than r_max        
    bin_indices = np.digitize(dist_c, r_bin_edges, right=True)                                                          # identifies associated bin (last bin edge is inclusive right, others not) (firt bin edge not inclusive left) (note: zero is outside range below, len(r_bins) is outside range above)
    counts = np.array([np.sum(bin_indices == i) for i in range(1, len(r_bin_edges))])                                   # number of points in different r_bins
    return np.cumsum(counts)  

def find_weights(lat_c, lon_c, lat, r_bin_edges):
    for r, r_dist in enumerate(r_bin_edges[1:]):                                                                        # first r_bins entry is zero
        dist_to_y_edges = dM.haversine_dist(np.array([lat_c] * 2), 
                                            np.array([lon_c] * 2), 
                                            np.array([lat[0], lat[-1]]),                                                # top and bottom edge
                                            np.array([lon_c] * 2))                                                      # distance from core to meridional edges    
        area_circle_bin = np.pi * r_dist**2                                                                             # area of circle considered
        area_inside     = np.pi * r_dist**2                                                                             # area of circle included in domain (updated in loop)
        for i, edge_dist in enumerate(dist_to_y_edges):                                                                 # check both edges
            if r_dist > edge_dist:                                                                                      # if the circle extends past the boundary
                theta = 2 * np.arccos(edge_dist / r_dist)                                                               # angle of segment exceeding boundary (cosine rule: adjacent / hypo.)
                area_circsec = (theta / 2) * r_dist**2                                                                  # circle sector area: (theta / 2pi) * pi r^2  
                area_triangle = edge_dist * np.sqrt(r_dist**2 - edge_dist**2)                                           # triangle of circle segment (pyth. th and area of triangle * 2)   
                area_inside -= (area_circsec - area_triangle)                                                           # subtract the area of the circle excluded due to the boundary
        yield area_circle_bin / area_inside                                                                             # This weights the number of included points with a greater weight if part of the circle is outside the domain


# == metric calc ==
def get_L_org(lat_coords, lon_coords, da_area, r_bin_edges):
    # -- core evaluation -- 
    n_c = len(lat_coords)                                                                                               # number of "cores"
    cum_counting = np.zeros((n_c, len(r_bin_edges) - 1))                                                                # considering bin edges (one less bin than bin edges)
    weights = np.ones((n_c, len(r_bin_edges) - 1))                                                                      # This weights the number of included points with a greater weight if part of the circle is outside the domain
    for c, (lat_c, lon_c) in enumerate(zip(lat_coords, lon_coords)):                                                    # position of selected core
        cum_counting[c,:] = find_counts_in_radius_bins(lat_c, lon_c, lat_coords, lon_coords, r_bin_edges)               # how many cores fall in each circle radius
        weights[c, :] = np.fromiter(find_weights(lat_c, lon_c, lat, r_bin_edges), dtype=float)                          # weight for each bin      
    cum_counting_weighted = cum_counting * weights                                                                      # apply weights to the count

    # -- Observational Bsag function (L^hat) -- 
    Besag_obs = np.sqrt((da_area.sum().data / (np.pi * n_c * (n_c - 1))) * np.sum(cum_counting_weighted, axis = 0))     # one value for each r in r_bins
    Besag_obs = np.concatenate(([0], Besag_obs))                                                                        # add zero at begining for integral

    # -- Theoretical Bsag function (L) -- 
    Besag_theor = r_bin_edges                                                                                           # linearly increasing with r

    # -- L_org calc --                           
    L_org = np.trapz(Besag_obs - Besag_theor, x = r_bin_edges)                                                          # integral between observed- and theoretical curve
    L_ir = np.sqrt(np.trapz((Besag_obs - Besag_theor)**2, x = r_bin_edges))                                             # irregularity index (standard deviation between observed- and theoretical curve)
    return L_org, L_ir, Besag_obs, Besag_theor, r_bin_edges


# == when this script is ran ==
if __name__ == '__main__':
    ''







