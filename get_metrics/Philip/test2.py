
import xarray as xr


if __name__ == '__main__':
    # precipitation field
    # ds = xr.open_dataset('/scratch/k10/cb4968/temp_data/observations/IMERG/doc_metrics/area_fraction/IMERG/area_fraction_IMERG_hrly_0-360_-30-30_3600x1800_2020-04_2021-03/area_fraction_IMERG_hrly_0-360_-30-30_3600x1800_2020-04_2021-03_var_2020_4_1.nc')
    # print(ds)
    # exit()

    ds = xr.open_dataset('/g/data/k10/cb4968/metrics/observations/IMERG/precip/pr_percentiles/IMERG/pr_percentiles_IMERG_hrly_0-360_-30-30_3600x1800_2020-04_2021-03.nc')
    print(ds)
    exit()





