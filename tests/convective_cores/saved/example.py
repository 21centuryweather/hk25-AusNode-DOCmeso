
import os
import sys
import importlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from scipy.ndimage import convolve
from scipy.ndimage import maximum_filter

# -- Imported scripts --
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, plot_path):
    ''' import module from relative path '''
    if plot_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        relative_path = plot_path.replace(os.getcwd(), "").lstrip("/")
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                               'utils')
pF_M = import_relative_module('util_plot.get_subplot.map_subplot',      'utils')

# == smoothing function ==
def exponential_kernel(size, decay_distance):
    ''' Creates grid to smooth over, and the magnitude of decay '''
    x, y = np.meshgrid(np.arange(size), np.arange(size))        # matrix to smooth over
    dist = np.sqrt((x - size//2)**2 + (y - size//2)**2)         # distance to smooth over
    kernel = np.exp(-dist / decay_distance)                     # rate of decay
    kernel /= kernel.sum()                                      # smoothing
    return kernel


if __name__ == '__main__':
    # == precipitation field ==
    ds = xr.open_dataset('/Users/cbla0002/Desktop/pr_percentiles_icon_d3hp003_3hrly_0-360_-30-30_3600x1800_2020-04_2021-03_var_2020_4_1.nc')
    da = ds['var'].isel(time = 0)
    lon_area = '100:149'
    lat_area = '-13:13'  
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    da = da.fillna(0)
    threshold = da.quantile(0.90).data
    da_orig = da

    # == apply smoothing of field ==
    kernel_size = 8
    decay_distance = 100
    kernel = exponential_kernel(size = kernel_size, decay_distance = decay_distance)
    da = convolve(da, kernel, mode='nearest')
    da = xr.DataArray(                                                   # convective objects
        data = da,
        dims=["lat", "lon"],
        coords={"lat": da_orig.lat, "lon": da_orig.lon},
        )
    da_smooth = da
    # da = da_orig

    # == cores, from local maxima exceednig threshold ==
    local_max = maximum_filter(da, size=3)
    local_maxima = (da == local_max) * 1
    local_maxima_above_threshold = (local_maxima * da) > threshold
    latitudes, longitudes = np.where(local_maxima_above_threshold)
    lat_values = local_maxima.lat.values[latitudes]
    lon_values = local_maxima.lon.values[longitudes]

    # == plots ==
    # -- precipitation field -- 
    # -- create figure --
    width, height = 6.27, 9.69                      # max size (for 1 inch margins)
    width, height = 1.25 * width, 0.45 * height      # modulate size and subplot distribution
    ncols, nrows  = 1, 1
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
    # -- format subplot --
    xticks = [110, 120, 130, 140]
    yticks = [-10, 0, 10]
    ds_map = xr.Dataset({'var': da_orig})
    ds_map.attrs.update({ 'scale': 1.2, 'move_row':-0.075, 'move_col': -0.09,                                                                    # format axes
    'name': f'var',                                                                                                                              # plot
    'vmin': 0, 'vmax': 1, 'cmap': 'Blues', 'cbar_height': 0.025, 'cbar_pad': 0.1,                                                                     # colorbar: position
    'hide_colorbar': True, 'cbar_label': f'', 'cbar_fontsize': 6.25, 'cbar_numsize': 6, 'cbar_label_pad': 0.085,         # colorbar: label                
    'hide_xticks': True, 'xticks': xticks, 'xticks_fontsize': 6.5,                                                                                 # x-axis:   ticks
    'hide_xlabel': True, 'xlabel_label': 'longitude', 'xlabel_pad': 0.0785, 'xlabel_fontsize': 6,                                                 # x-axis:   label
    'hide_yticks': True, 'yticks': yticks, 'yticks_fontsize':  5.5,                                                                                # y-axis:   ticks
    'hide_ylabel': True, 'ylabel_label': 'latitude', 'ylabel_pad': 0.055, 'ylabel_fontsize': 5,                                                  # y-axis:   label
    'axtitle_label': f'ICON precip: MTC', 'axtitle_xpad': 0, 'axtitle_ypad': 0.01, 'axtitle_fontsize': 10,
    'line_dots_size': 0.1,
    'coastline_width': 0.6,
    })                                                     # subplot   title
    row = 0
    col = 0
    ax = pF_M.plot(fig, nrows, ncols, row, col, ax = axes, ds = ds_map)
    # -- save figure --
    folder = '/Users/cbla0002/Desktop/scratch/plots'
    filename = 'precip'
    path = f'{folder}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)

    # -- precipitation field, smoothed -- 
    # -- create figure --
    width, height = 6.27, 9.69                      # max size (for 1 inch margins)
    width, height = 1.25 * width, 0.45 * height      # modulate size and subplot distribution
    ncols, nrows  = 1, 1
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
    # -- format subplot --
    xticks = [110, 120, 130, 140]
    yticks = [-10, 0, 10]
    ds_map = xr.Dataset({'var': da_smooth})
    ds_map.attrs.update({ 'scale': 1.2, 'move_row':-0.075, 'move_col': -0.09,                                                                    # format axes
    'name': f'var',                                                                                                                              # plot
    'vmin': 0, 'vmax': 1, 'cmap': 'Blues', 'cbar_height': 0.025, 'cbar_pad': 0.1,                                                                     # colorbar: position
    'hide_colorbar': True, 'cbar_label': f'', 'cbar_fontsize': 6.25, 'cbar_numsize': 6, 'cbar_label_pad': 0.085,         # colorbar: label                
    'hide_xticks': True, 'xticks': xticks, 'xticks_fontsize': 6.5,                                                                                 # x-axis:   ticks
    'hide_xlabel': True, 'xlabel_label': 'longitude', 'xlabel_pad': 0.0785, 'xlabel_fontsize': 6,                                                 # x-axis:   label
    'hide_yticks': True, 'yticks': yticks, 'yticks_fontsize':  5.5,                                                                                # y-axis:   ticks
    'hide_ylabel': True, 'ylabel_label': 'latitude', 'ylabel_pad': 0.055, 'ylabel_fontsize': 5,                                                  # y-axis:   label
    'axtitle_label': f'ICON precip: MTC, smoothed: kernel_size = {kernel_size}, decay_distance = {decay_distance}', 'axtitle_xpad': 0, 'axtitle_ypad': 0.01, 'axtitle_fontsize': 10,
    'line_dots_size': 0.1,
    'coastline_width': 0.6,
    })                                                     # subplot   title
    row = 0
    col = 0
    ax = pF_M.plot(fig, nrows, ncols, row, col, ax = axes, ds = ds_map)
    # -- save figure --
    folder = '/Users/cbla0002/Desktop/scratch/plots'
    filename = f'precip_smooth'
    path = f'{folder}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)

    # -- precipitation field, with cores --
    # -- create figure --
    width, height = 6.27, 9.69                      # max size (for 1 inch margins)
    width, height = 1.25 * width, 0.45 * height      # modulate size and subplot distribution
    ncols, nrows  = 1, 1
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
    # -- format subplot --
    xticks = [110, 120, 130, 140]
    yticks = [-10, 0, 10]
    ds_map = xr.Dataset({'var': da_smooth})
    ds_map.attrs.update({ 'scale': 1.2, 'move_row':-0.075, 'move_col': -0.09,                                                                    # format axes
    'name': f'var',                                                                                                                              # plot
    'vmin': 0, 'vmax': 1, 'cmap': 'Blues', 'cbar_height': 0.025, 'cbar_pad': 0.1,                                                                     # colorbar: position
    'hide_colorbar': True, 'cbar_label': f'', 'cbar_fontsize': 6.25, 'cbar_numsize': 6, 'cbar_label_pad': 0.085,         # colorbar: label                
    'hide_xticks': True, 'xticks': xticks, 'xticks_fontsize': 6.5,                                                                                 # x-axis:   ticks
    'hide_xlabel': True, 'xlabel_label': 'longitude', 'xlabel_pad': 0.0785, 'xlabel_fontsize': 6,                                                 # x-axis:   label
    'hide_yticks': True, 'yticks': yticks, 'yticks_fontsize':  5.5,                                                                                # y-axis:   ticks
    'hide_ylabel': True, 'ylabel_label': 'latitude', 'ylabel_pad': 0.055, 'ylabel_fontsize': 5,                                                  # y-axis:   label
    'axtitle_label': f'ICON precip: MTC, smoothed: kernel_size = {kernel_size}, decay_distance = {decay_distance}, cores 3x3 grid, exceeding threshold', 'axtitle_xpad': 0, 'axtitle_ypad': 0.01, 'axtitle_fontsize': 10,
    'line_dots_size': 0.1,
    'coastline_width': 0.6,
    })                                                     # subplot   title
    row = 0
    col = 0
    ax = pF_M.plot(fig, nrows, ncols, row, col, ax = axes, ds = ds_map)
    ax.scatter(lon_values, lat_values, transform=ccrs.PlateCarree(), color = 'r', s = 2)
    # -- save figure --
    folder = '/Users/cbla0002/Desktop/scratch/plots'
    filename = f'precip_smooth_cores'
    path = f'{folder}/{filename}.png'
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)



