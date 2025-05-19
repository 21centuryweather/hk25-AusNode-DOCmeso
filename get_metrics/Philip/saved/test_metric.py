'''
# -------------------
#  Testing metric
# -------------------

'''

# == imports ==
# -- Packages --
import xarray as xr
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


# == metric funcs ==
def get_metric(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, p_id, metric_var):
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


def plot_monthly():
    ''


if __name__ == '__main__':
    # path = '/g/data/nf33/cb4968/metrics/observations/IMERG/precip/pr_percentiles/IMERG/pr_percentiles_IMERG_3hrly_0-360_-30-30_3600x1800_2020-04_2021-03.nc'
    # path = '/g/data/nf33/cb4968/metrics/observations/IMERG/doc_metrics/area_fraction/IMERG/area_fraction_IMERG_3hrly_100-149_-13-13_3600x1800_2020-04_2021-03.nc'
    # path = '/g/data/nf33/cb4968/metrics/observations/IMERG/doc_metrics/mean_area/IMERG/mean_area_IMERG_3hrly_100-149_-13-13_3600x1800_2020-04_2021-03.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()

    # -- general settings --
    cmap =      'RdBu'                                                                                                                                # area                                                                                                                                 #
    res =       0.1          

    # -- define metrics --
    t_freq =    '3hrly'
    x1_tfreq,   x1_group,   x1_name,    x1_var,     x1_label,   x1_units =  t_freq,    'precip',      'pr_percentiles',    'pr_percentiles_95',                        r'pr${95}$',    r'[mm day%^{-1}%]'
    x2_tfreq,   x2_group,   x2_name,    x2_var,     x2_label,   x2_units =  t_freq,    'precip',      'pr_percentiles',    'pr_percentiles_97',                        r'pr${97}$',    r'[mm day%^{-1}%]'
    x3_tfreq,   x3_group,   x3_name,    x3_var,     x3_label,   x3_units =  t_freq,    'precip',      'pr_percentiles',    'pr_percentiles_99',                        r'pr${99}$',    r'[mm day%^{-1}%]'
    
    # t_freq =    'hrly'
    t_freq =    '3hrly'
    x4_tfreq,   x4_group,   x4_name,    x4_var,     x4_label,   x4_units =  t_freq,     'doc_metrics', 'area_fraction',     'area_fraction_thres_pr_percentiles_95',    r'A$_f$',       r'[%]'
    x5_tfreq,   x5_group,   x5_name,    x5_var,     x5_label,   x5_units =  t_freq,     'doc_metrics', 'mean_area',         'mean_area_thres_pr_percentiles_95',        r'A$_m$',       r'[km$^2$]'

    # -- general settings --
    p_id = '2020-04:2021-03'       

    # -- IMERG data --
    data_type_group, data_tyoe, dataset = 'observations', 'IMERG', 'IMERG'
    lon_area =  '0:360'  
    lat_area =  '-30:30'       
    x1 = get_metric(data_type_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var) 
    x2 = get_metric(data_type_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var) 
    x3 = get_metric(data_type_group, data_tyoe, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, p_id, x3_var) 
    lat_area =  '-13:13'   
    lon_area =  '100:149'
    x4 = get_metric(data_type_group, data_tyoe, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, p_id, x4_var) 
    x5 = get_metric(data_type_group, data_tyoe, dataset, x5_tfreq, x5_group, x5_name, lon_area, lat_area, res, p_id, x5_var) 

    # -- print values --
    for x, var in zip([x1, x2, x3, x4, x5], [x1_var, x2_var, x3_var, x4_var, x5_var]):
        print(f'{var}: {x.mean(dim = 'time').data}')
        # print(f'{var}:  {np.round(x.mean(dim = 'time').data, 3)}')




