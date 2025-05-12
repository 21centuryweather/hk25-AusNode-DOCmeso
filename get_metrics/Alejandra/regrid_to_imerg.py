import intake
from easygems import healpix as egh
import matplotlib.pyplot as plt
import warnings
import xarray as xr
from pathlib import Path
import healpy as hp

# Open UK_node data: 
zoom = "z3"
data_2d = f"/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.{zoom}.zarr" #PT1H is hourly data
ds = xr.open_zarr(data_2d)

# Test: plot one timestep
#egh.healpix_show(ds["pr"].sel(time="2020-05-10T00:00:00"), cmap="inferno", dpi=72);
#plt.savefig("/g/data/up6/ai2733/hackaton/test_fig.png")

# Add lat and lon 
ds = ds.pipe(egh.attach_coords)

# Select maritime Continent subregion
Slim, Nlim = -10.0, 10.0
Elim, Wlim = 100.0, 149.0

ds_mar = ds.where((ds["lat"] > Slim) & (ds["lat"] < Nlim) & (ds["lon"] > Elim) & (ds["lon"] < Wlim),drop=True)

# Test: Plot one time
#egh.healpix_show(ds_mar["pr"].sel(time="2020-05-10T00:00:00"), cmap="inferno", dpi=72);
#plt.savefig("/g/data/up6/ai2733/hackaton/test_fig2.png")

# Open imerg data to interpolate
path_imerg = "/g/data/ia39/aus-ref-clim-data-nci/frogs/data/1DD_V1/IMERG_V07B_FC/IMERG_V07B_FC.1DD.2020.nc"
imerg = xr.open_dataset(path_imerg).isel(time = 0) # Select first time

imerg_mar = imerg.sel(lon = slice(Elim, Wlim), lat = slice(Slim, Nlim))


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

# Find the HEALPix pixels that are closest to the ERA5 grid

# longitudes and latitudes for the ERA5 grid
lon = imerg_mar['lon'].values
lat = imerg_mar['lat'].values

# nside for um simulation, it should be equal to 2**zoom
this_nside = hp.get_nside(ds_mar["pr"])

cells = get_nn_lon_lat_index(this_nside, lon, lat) 

print ("worked")