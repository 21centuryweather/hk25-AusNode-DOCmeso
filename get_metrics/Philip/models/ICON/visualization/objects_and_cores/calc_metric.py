'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# -- util- and local scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, file_path):
    ''' import module from relative path '''
    if file_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        cwd = os.getcwd()
        relative_path = os.path.relpath(file_path, cwd) # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                                                   'utils')
pF = import_relative_module('util_plot.map_subplot',                                        'utils')
cC = import_relative_module('util_calc.doc_metrics.conv_cores.find_conv_cores',             'utils')


# == metric funcs ==
def get_metric(da, time_period, metric_var = 'pr_percentiles_95'):
    ''' convective threshold '''
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'models'
    data_type =         'ICON'
    metric_group =      'precip'
    metric_name =       'pr_percentiles'
    dataset =           'icon_d3hp003'
    # t_freq =            'hrly'
    t_freq =            '3hrly'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        0.1
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (
            f'{metric_name}'   
            f'_{dataset}'                                                                                                   
            f'_{t_freq}'                                                                                                    
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
            )       
    path = f'{folder}/{filename}.nc'
    # print(xr.open_dataset(path))
    # -- get metric --
    try:
        threshold = xr.open_dataset(path)[metric_var].mean(dim = 'time')
        da_threshold = threshold.broadcast_like(da.isel(lat = 0, lon = 0))                                                  #
    except:
        print(f'couldnt open metric file: {path}')
        print('try regenerating if it does not exist')
        print('exiting')
        exit()
    return da_threshold

# == metric funcs ==
def get_metric_saved(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, p_id, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                        # base result_filename
            f'{metric_name}'                                                                                            #
            f'_{dataset}'                                                                                               #
            f'_{t_freq}'                                                                                                #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
            f'_{p_id.split(":")[0]}_{p_id.split(":")[1]}'                                                 #
            )       
    path = f'{folder}/{filename}.nc'
    # -- find metric -- 
    metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        # -- get metric variation -- 
        metric = metric[metric_var]
    # try:
    #     cA.detrend_month_anom(metric)                                                                                   # correlate anomalies (won't be able to do this with nan)
    # except:                                                                                                             #
    #     metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
    #     metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
    return metric


# def plot_subplot(title, fig, nrows, ncols, axes, ds, ds_contour, ds_ontop, lines):
#     # print(ds)
#     # print(ds['var'])
#     # exit()

#     # -- add subplot settings --
#     xticks = [60, 120, 180, 240, 300]
#     yticks = [-20, 0, 20]
#     # print(ds)
#     ds.attrs.update({ 
#         # -- format axes --
#         'scale': 1.05, 'move_row': 0.125, 'move_col': 0.025,
#         'hide_colorbar': False, 'cbar_height': 0.035, 'cbar_pad': 0.2, 'cbar_label_pad': 0.175,   
#         'xlabel_pad': 0.15,   
#         'ylabel_pad': 0.085,
#         'axtitle_xpad': 0, 'axtitle_ypad': 0.05,

#         # -- format plot elements --
#         'vmin': -2, 'vmax': 2, 
#         'cmap': 'RdBu', 
    
#         # -- format text --
#         'cbar_label': f'std from mean [Nb]',                    'cbar_fontsize': 8, 'cbar_numsize': 6,             
#         'hide_xticks': False,   'xticks': xticks,               'xticks_fontsize': 6.5,
#         'hide_xlabel': False,   'xlabel_label': 'longitude',    'xlabel_fontsize': 6.5,
#         'hide_yticks': False,   'yticks': yticks,               'yticks_fontsize': 6,
#         'hide_ylabel': False,   'ylabel_label': 'latitude',     'ylabel_fontsize': 6.5,
#         'axtitle_label':        title,                          'axtitle_fontsize': 9,
#         'coastline_width': 0.6,
#         'line_dots_size': 0.1,
#         })
#     # print(ds)
#     # exit()
#     if ds_contour is not None:
#         ds_contour.attrs.update({
#             # -- contour --
#             'name': 'var', 
#             'threshold': [ds_contour["var"].quantile(0.5), ds_contour["var"].quantile(0.9)], 
#             'color': 'k', 
#             'linewidth': 0.5,
#             'contour_text_size': 4.5,
#             })

    # # -- plot subplot --
    # row, col = 0, 0
    # # [print(f) for f in [fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines]]
    # # exit()

    # ax = pF.plot(fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines)

    # return fig

def plot_subplot(title, fig, nrows, ncols, axes, ds = None, ds_contour = None, ds_ontop = None, lines = []):
    # print(ds)
    # print(ds['var'])
    # exit()

    # -- add subplot settings --
    xticks = [110, 120, 130, 140]
    yticks = [-10, 0, 10]

    # print(ds)
    symbol = r'$\sigma$'
    some_text = r'OLR$_{mean}$'
    add_size = 4
    ds.attrs.update({ 
        # -- format axes --
        'scale': 0.9, 'move_row': 0.12, 'move_col': 0.01,
        'hide_colorbar': False, 'cbar_height': 0.035, 'cbar_pad': 0.12, 'cbar_label_pad': 0.1,   
        'xlabel_pad': 0.09,   
        'ylabel_pad': 0.085,
        'axtitle_xpad': 0.015, 'axtitle_ypad': 0.015,

        # -- format plot elements --
        'vmin': -2, 'vmax': 2, 
        # 'cmap': 'RdBu', 
        # 'vmin': -925379.2375, 'vmax': 925379.2375, 
        'cmap': 'GnBu',
        # 'cmap': 'Blues',
        # 'cmap': 'PuBu',

        # -- format text --
        'cbar_label': f'{symbol}(OLR) []', 'cbar_fontsize': 6 + add_size, 'cbar_numsize': 6 + add_size,             
        'hide_xticks': False,   'xticks': xticks,               'xticks_fontsize': 6.5 + add_size,
        'hide_xlabel': False,   'xlabel_label': 'longitude',    'xlabel_fontsize': 6.5 + add_size,
        'hide_yticks': False,   'yticks': yticks,               'yticks_fontsize': 6 + add_size,
        'hide_ylabel': False,   'ylabel_label': 'latitude',     'ylabel_fontsize': 6.5 + add_size,
        'axtitle_label':        title,                          'axtitle_fontsize': 10 + add_size -1,
        'coastline_width': 0.6,
        'line_dots_size': 0.1,
        })
    # print(ds)
    # exit()
    if ds_contour is not None:
        ds_contour.attrs.update({
            # -- contour --
            'name': 'var', 
            'threshold': [ds_contour["var"].quantile(0.25)], # ds_contour["var"].quantile(0.9)], 
            'color': 'g', 
            'linewidth': 0.5,
            'contour_text_size': 4.5,
            })

    row, col = 0, 0
    # [print(f) for f in [fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines]]
    # exit()

    ax = pF.plot(fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines)
    return fig, ax


# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name
    ds = xr.Dataset()

    # == create metric ==
    # -- check data --
    da, process_requestd, count, da2 = data_objects
    # print(da2)
    # exit()

    # == get other metrics to help plot ==
    quant = 0.95
    quant_str = f'pr_percentiles_{int(quant *100)}'
    threshold = get_metric(da, time_period = '2020-03:2021-02', metric_var = quant_str).mean(dim = 'time').data
    t_freq =    '3hrly'
    x1_tfreq,   x1_group,   x1_name,    x1_var,     x1_label,   x1_units =  t_freq, 'precip',      'pr_percentiles',    'pr_percentiles_95',                        r'pr$_{95}$',    r'[mm day$^{-1}$]'
    x2_tfreq,   x2_group,   x2_name,    x2_var,     x2_label,   x2_units =  t_freq, 'doc_metrics', 'area_fraction',     'area_fraction_thres_pr_percentiles_95',    r'A$_f$',       r'[%]'
    x3_tfreq,   x3_group,   x3_name,    x3_var,     x3_label,   x3_units =  t_freq, 'doc_metrics', 'mean_area',         'mean_area_thres_pr_percentiles_95',        r'A$_m$',       r'[km$^2$]'
    x4_tfreq,   x4_group,   x4_name,    x4_var,     x4_label,   x4_units =  t_freq, 'doc_metrics', 'i_org',             'i_org_thres_pr_percentiles_95',            r'I$_{org}$',     r'[]'
    # x5_tfreq,   x5_group,   x5_name,    x5_var,     x5_label,   x5_units =  t_freq, 'doc_metrics', 'L_org',             'L_org_thres_pr_percentiles_95',            r'I$_org$',     r'[]'

    data_type_group, data_type, dataset  =   'models', 'ICON', 'icon_d3hp003'
    res =       0.1         
    lon_area =  '0:360'  
    lat_area =  '-30:30'       
    p_id = '2020-03:2021-02'
    x1 = get_metric_saved(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var) 
    lat_area =  '-13:13'   
    lon_area =  '100:149'
    x2 = get_metric_saved(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var) 
    x3 = get_metric_saved(data_type_group, data_type, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, p_id, x3_var) 
    x4 = get_metric_saved(data_type_group, data_type, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, p_id, x4_var) 
    # x5 = get_metric(data_type_group, data_type, dataset, x5_tfreq, x5_group, x5_name, lon_area, lat_area, res, p_id, x5_var) 
    pr_mean = xr.open_dataset('/scratch/nf33/hk25_DOCmeso/Mean_prec_all.nc')['pr'].sel(dataset = 'ICON')

    # --loop through timesteps --
    for i, timestep in enumerate(da.time):
        da_timestep = da.isel(time = i)
        da2_timestep = da2.isel(time = i)
        # print(da2_timestep.quantile([0.5, 0.8, 0.9]))
        # exit()

        # -- smoothing --
        kernel_size, decay_distance = 8, 10
        pr_mean = cC.apply_smoothing(pr_mean, kernel_size, decay_distance)

        kernel_size, decay_distance = 8, 10
        da_timestep = cC.apply_smoothing(da_timestep, kernel_size, decay_distance)
        # -- cores --
        _, lat_coords, lon_coords = cC.find_conv_cores(da_timestep, threshold, exceed_threshold = True)
        # -- convective objects --
        conv_regions = (da_timestep > threshold) * 1
        # -- visualize whole tropics --
        # -- visualize MTC --
        plot = True
        if plot:
            # -- create figure --
            width, height = 6.27, 9.69                      # max size (for 1 inch margins)
            width, height = 1 * width, 0.4 * height         # modulate size and subplot distribution
            ncols, nrows  = 1, 1
            fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
            # -- data for plot --
            spatial_mean = da2_timestep.mean(dim = ('lat', 'lon'))
            da_plot = ((da2_timestep - spatial_mean)).drop('time')  # anomalies from the spatial-mean
            da_plot =  da_plot / da_plot.std()
            # da_plot = da2_timestep
            title = (
                f'time:{str(timestep.data)[2:18]}    '
                f'{x4_label}: {x4.sel(time = timestep).data:.2e} {x4_units}\n'
                f'{x2_label}: {x2.sel(time = timestep).data:.2e} {x2_units}               '
                f'{x3_label}: {x3.sel(time = timestep).data:.2e} {x3_units}'
                )
            da_ontop = xr.where(conv_regions!= 0, 1, np.nan)    # .drop('time') 
            fig, ax = plot_subplot(title,
                            fig = fig,
                            nrows = nrows,
                            ncols = ncols,
                            axes = axes,
                            ds = xr.Dataset({'var': -da_plot}), 
                            # ds_contour = xr.Dataset({'var': pr_mean}), 
                            ds_ontop = xr.Dataset({'var': da_ontop}), 
                            lines = [],
                            )
            ax.scatter(lon_coords, lat_coords, transform=ccrs.PlateCarree(), color = 'r', s = 0.25)
            # -- save figure --
            folder = f'/scratch/nf33/cb4968/plots/icon_snapshots'
            filename = f'snapshot_{count * len(da.time) + i}.png'
            path = f'{folder}/{filename}'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.remove(path) if os.path.exists(path) else None
            fig.savefig(path)
            print(f'plot saved at: {path}')
            plt.close(fig)
            # exit()
    # exit()
    return ds


