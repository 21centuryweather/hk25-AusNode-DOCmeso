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
mS = import_relative_module('user_specs',                                                       'utils')
# metric_script = import_relative_module('util_calc.doc_metrics.rome.rome',                     'utils')
pF = import_relative_module('util_plot.map_subplot',                                            'utils')


def plot_subplot(title, fig, nrows, ncols, axes, ds, ds_contour, ds_ontop = None, ds_ontop2 = None, lines = []):
    # print(ds)
    # print(ds['var'])
    # exit()

    # -- add subplot settings --
    # xticks = [110, 120, 130, 140]
    # yticks = [-10, 0, 10]
    xticks = [60, 120, 180, 240, 300]
    yticks = [-20, 0, 20]
    # print(ds)
    units = r'mm day$^{-1}$'
    add_size = 4
    ds.attrs.update({ 
        # -- format axes --
        'scale': 1.175, 'move_row': 0.075, 'move_col': - 0.065,
        'hide_colorbar': False, 'cbar_height': 0.035, 'cbar_pad': 0.15, 'cbar_label_pad': 0.125,   
        'xlabel_pad': 0.125,   
        'ylabel_pad': 0.05,
        'axtitle_xpad': 0, 'axtitle_ypad': 0.025,

        # -- format plot elements --
        'vmin': 0, 'vmax': 1, 
        'cmap': 'Blues', #plt.get_cmap('tab20') , 
    
        # -- format text --
        'cbar_label': f'precip [{units}]',                                'cbar_fontsize': 8 + add_size, 'cbar_numsize': 6 + add_size,             
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

    ax = pF.plot(fig, nrows, ncols, row, col, axes, ds, ds_contour, ds_ontop, lines)
    return fig


# == calculate ds_metric ==
def calculate_metric(data_objects):
    # -- metric name --
    metric_name = Path(__file__).resolve().parents[0].name

    # -- check data --
    da, process_requestd, count = data_objects

    # -- load necessary additional metrics --
    # threshold = get_metric(da, time_period = '2020-03:2021-03')
    threshold = 0

    # --loop through timesteps --
    metric_calc = []
    for i, timestep in enumerate(da.time):
        da_timestep = da.isel(time = i)
        # -- calculate metric --
        quantile_thresholds = [
            0.5,                           
            0.9,                   
            0.95,                          # 5% of the domain 
            0.97,  
            0.99      
            ]
        metric_timestep = da_timestep.quantile(quantile_thresholds, dim = ('lat', 'lon'))
        metric_timestep = metric_timestep.expand_dims(dim = 'time')
        metric_timestep = metric_timestep.assign_coords(time=[da_timestep.time.values])
        metric_calc.append(metric_timestep)

        # -- visualize metric start --
        plot = False
        if plot:
            # -- create figure --
            width, height = 6.27, 9.69                      # max size (for 1 inch margins)
            width, height = 1.85 * width, 0.35 * height      # modulate size and subplot distribution
            ncols, nrows  = 1, 1
            fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))

            # -- plot data --
            metric_text = metric_timestep.data[0]
            spatial_mean = da_timestep.mean(dim = ('lat', 'lon'))
            da_plot = da_timestep #((da_timestep - spatial_mean)).drop('time')  # anomalies from the spatial-mean
            # da_plot =  da_plot / da_plot.std()
            title = (
                # f'time:{str(metric_timestep.time.data)[2:18]}'   
                f'{metric_name}' #: {metric_text:.2e}'
                )
            fig = plot_subplot(title,
                            fig = fig,
                            nrows = nrows,
                            ncols = ncols,
                            axes = axes,
                            ds = xr.Dataset({'var': da_plot}), 
                            ds_contour = None, 
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

        #     ''
    # 0.05 mm/hr
    # -- concatenate timesteps --
    metric_calc = xr.concat(metric_calc, dim = 'time')
    # print(metric_calc.mean(dim = 'time').data)
    # exit()

    # -- fill xr.dataset with metric --
    ds = xr.Dataset()
    for quant in quantile_thresholds:
        name = f'{metric_name}_{int(quant * 100)}'
        ds[name] = metric_calc.sel(quantile = quant)

    # print(ds)
    # exit()

    return ds

