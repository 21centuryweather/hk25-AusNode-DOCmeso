'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import skimage.measure as skm
import numpy as np

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
mS = import_relative_module('user_specs',                                           'utils')
pF = import_relative_module('plot_func.map_subplot',                               __file__)
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')
dM = import_relative_module('util_calc.distance_matrix.distance_matrix',            'utils')
doc = import_relative_module('util_calc.doc_metrics.rome.rome',                     'utils')

# == metric funcs ==
def get_metric(da, time_period, metric_var = 'pr_percentiles_95'):
    ''' convective threshold '''
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'observations'
    data_type =         'IMERG'
    metric_group =      'precip'
    metric_name =       'pr_percentiles'
    dataset =           'IMERG'
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

# def get_saved_mean_pr():
#     ''' /scratch/nf33/gs5098/data ''' 
#     path = '/scratch/nf33/gs5098/data/precip_mean.nc'
#     ds = xr.open_dataset(path)
#     print(ds['precipitation'].data)
#     exit()
#     return ds


def plot_subplot(title, fig, nrows, ncols, axes, ds, ds_contour, ds_ontop, ds_ontop2, lines):
    # print(ds)
    # print(ds['var'])
    # exit()

    # -- add subplot settings --
    xticks = [110, 120, 130, 140]
    yticks = [-10, 0, 10]

    # print(ds)
    add_size = 4
    ds.attrs.update({ 
        # -- format axes --
        'scale': 0.9, 'move_row': 0.12, 'move_col': 0.01,
        'hide_colorbar': False, 'cbar_height': 0.035, 'cbar_pad': 0.12, 'cbar_label_pad': 0.1,   
        'xlabel_pad': 0.09,   
        'ylabel_pad': 0.085,
        'axtitle_xpad': 0, 'axtitle_ypad': 0.025,

        # -- format plot elements --
        'vmin': -2, 'vmax': 2, 
        'cmap': 'RdBu', 
    
        # -- format text --
        'cbar_label': f'std from mean [Nb]',                    'cbar_fontsize': 8 + add_size, 'cbar_numsize': 6 + add_size,             
        'hide_xticks': False,   'xticks': xticks,               'xticks_fontsize': 6.5 + add_size,
        'hide_xlabel': False,   'xlabel_label': 'longitude',    'xlabel_fontsize': 6.5 + add_size,
        'hide_yticks': False,   'yticks': yticks,               'yticks_fontsize': 6 + add_size,
        'hide_ylabel': False,   'ylabel_label': 'latitude',     'ylabel_fontsize': 6.5 + add_size,
        'axtitle_label':        title,                          'axtitle_fontsize': 9 + add_size -1,
        'coastline_width': 0.6,
        'line_dots_size': 0.1,
        })
    # print(ds)
    # exit()
    if ds_contour is not None:
        ds_contour.attrs.update({
            # -- contour --
            'name': 'var', 
            'threshold': [ds_contour["var"].quantile(0.5), ds_contour["var"].quantile(0.9)], 
            'color': 'k', 
            'linewidth': 0.5,
            'contour_text_size': 4.5,
            })

    row, col = 0, 0
    # [print(f) for f in [fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines]]
    # exit()

    ax = pF.plot(fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, ds_ontop2, lines)
    return fig

# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name
    ds = xr.Dataset()

    # == create metric ==
    # -- check data --
    da, process_requestd, count = data_objects
    
    # -- for area weighting --
    da_area = cW.get_area_matrix(da.lat, da.lon)   
    distance_matrix = dM.create_distance_matrix(da.lat, da.lon)  
    
    # -- convective threshold variations --
    quantile_thresholds = [0.95, 0.97, 0.99]
    for quant in quantile_thresholds:
        quant_str = f'pr_percentiles_{int(quant *100)}'
        threshold = get_metric(da, time_period = '2020-04:2021-03', metric_var = quant_str).mean(dim = 'time').data
        # --loop through timesteps --
        metric_calc = []
        for i, timestep in enumerate(da.time):
            da_timestep = da.isel(time = i)
            # -- calculate metric --
            # -- convective objects --
            conv_regions = (da_timestep > threshold) * 1
            labels_np = skm.label(conv_regions, background = 0, connectivity = 2)       # returns numpy array
            # labels_np = cB.connect_boundary(labels_np)                                  # connect objects across boundary
            labels = np.unique(labels_np)[1:]                                           # first unique value (zero) is background
            labels_xr = xr.DataArray(                                                   # convective objects
                data = labels_np,
                dims=["lat", "lon"],
                coords={"lat": da.lat, "lon": da.lon},
                )
            metric_timestep = doc.calc_rome(conv_regions, distance_matrix, da_area, connect_lon = True)
            metric_timestep = xr.DataArray(metric_timestep)
            metric_timestep = metric_timestep.expand_dims(dim = 'time')
            metric_timestep = metric_timestep.assign_coords(time=[da_timestep.time.values])
            metric_calc.append(metric_timestep)

            # -- visualize metric --
            plot = False
            if plot and quant_str == 'pr_percentiles_95':
                # print(f'threshold is: {threshold}')
                # -- create figure --
                width, height = 6.27, 9.69                      # max size (for 1 inch margins)
                width, height = 1 * width, 0.4 * height      # modulate size and subplot distribution
                ncols, nrows  = 1, 1
                fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))

                # -- plot data --
                metric_plot = metric_timestep.data[0]
                spatial_mean = da_timestep.mean(dim = ('lat', 'lon'))
                da_plot = ((da_timestep - spatial_mean)).drop('time')  # anomalies from the spatial-mean
                da_plot =  da_plot / da_plot.std()
                units = r'km$^2$'
                title = (
                    f'time:{str(metric_timestep.time.data)[2:18]}\n'   
                    f'{metric_name}: {metric_plot:.2e} {units}, '
                    f'areafraction: {(metric_plot * len(labels) / da_area.sum()).data:.2e}'
                    )
                da_ontop = xr.where(conv_regions.drop('time') != 0, 1, np.nan)

                # -- add an extra threshold --
                quant_str2 = f'pr_percentiles_99'
                # quant_str2 = f'pr_percentiles_97'
                threshold2 = get_metric(da, time_period = '2020-04:2021-03', metric_var = quant_str2).mean(dim = 'time').data
                # print(f'threshold2 is: {threshold2}')
                conv_regions2 = (da_timestep > threshold2) * 1
                da_ontop2 = xr.where(conv_regions2.drop('time') != 0, 1, np.nan)
                fig = plot_subplot(title,
                                fig = fig,
                                nrows = nrows,
                                ncols = ncols,
                                axes = axes,
                                ds = xr.Dataset({'var': da_plot}), 
                                ds_contour = None, 
                                ds_ontop = xr.Dataset({'var': da_ontop}), 
                                ds_ontop2 = xr.Dataset({'var': da_ontop2}), 
                                lines = [],
                                )
                # -- save figure --
                folder = f'{os.path.dirname(__file__)}/plots/snapshots'
                filename = f'snapshot_{count * len(da.time) + i}.png'
                path = f'{folder}/{filename}'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                os.remove(path) if os.path.exists(path) else None
                fig.savefig(path)
                print(f'plot saved at: {path}')
                plt.close(fig)
                # exit()
        # exit()

        # -- concatenate timesteps --
        metric_calc =  xr.concat(metric_calc, dim = 'time')

        # -- fill xr.dataset with metric --
        ds[f'{metric_name}_thres_{quant_str}'] = metric_calc

    # print(ds)
    # exit()
    return ds
