


# import healpy as hp
# import easygems.healpix as egh


# def get_nn_lon_lat_index(nside, lons, lats):
#     """
#     nside: integer, power of 2. The return of hp.get_nside()
#     lons: uniques values of longitudes
#     lats: uniques values of latitudes
#     returns: array with the HEALPix cells that are closest to the lon/lat grid
#     """
#     lons2, lats2 = np.meshgrid(lons, lats)
#     return xr.DataArray(
#         hp.ang2pix(nside, lons2, lats2, nest = True, lonlat = True),
#         coords=[("lat", lats), ("lon", lons)],
#     )





























