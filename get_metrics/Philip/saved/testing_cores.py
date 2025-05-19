
import xarray as xr


if __name__ == '__main__':
    f = xr.open_dataset('/scratch/nf33/hk25_DOCmeso/ICON_cores/2020-04-01.nc')
    lat = f['lat']
    lon = f['lon']


    print(lat.data)
    print(lon.data)

    # '2020-04-01'