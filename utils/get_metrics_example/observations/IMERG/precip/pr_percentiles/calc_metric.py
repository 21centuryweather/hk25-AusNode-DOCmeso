'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import xarray as xr
from pathlib import Path

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


# == metric funcs ==
def get_metric(da, time_period):
    ''' convective threshold '''
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'observations'
    data_type =         'IMERG'
    metric_group =      'precip'
    metric_name =       'precip_prctiles'
    metric_var =        'precip_prctiles_95'
    dataset =           dataset
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8
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


# == calculate ds_metric ==
def calculate_metric(data_objects):
    # -- metric name --
    metric_name = Path(__file__).resolve().parents[0].name

    # -- check data --
    da, process_requestd = data_objects

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
        # plot = True
        # if plot:
        #     ''

    # -- concatenate timesteps --
    metric_calc = xr.concat(metric_calc, dim = 'time')

    # -- fill xr.dataset with metric --
    ds = xr.Dataset()
    for quant in quantile_thresholds:
        name = f'{metric_name}_{int(quant * 100)}'
        ds[name] = metric_calc.sel(quantile = quant)

    # print(ds)
    # exit()

    return ds

