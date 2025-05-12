
import numpy as np
import xarray as xr
from pathlib import Path
import os
import healpy as hp
import easygems.healpix as egh


def get_nn_lon_lat_index(nside, lons, lats):
    """
    nside: integer, power of 2. The return of hp.get_nside()
    lons: uniques values of longitudes
    lats: uniques values of latitudes
    returns: array with the HEALPix cells that are closest to the lon/lat grid
    """
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest = True, lonlat = True),
        coords=[("lat", lats), ("lon", lons)],
    )

if __name__ == '__main__':
    # -- UM model --
    zoom = '10'
    folder = '/g/data/qx55/uk_node/glm.n2560_RAL3p3'
    filename = 'data.healpix.PT1H.z' + zoom + '.zarr'
    ds_um = xr.open_zarr(f'{folder}/{filename}')
    print(ds_um['pr'])

    # -- imerg --
    folder = '/g/data/ia39/aus-ref-clim-data-nci/frogs/data/1DD_V1/IMERG_V07B_FC'
    filename = 'IMERG_V07B_FC.1DD.2020.nc'
    ds_imerg = xr.open_dataset(f'{folder}/{filename}')
    print(ds_imerg)

    exit()











