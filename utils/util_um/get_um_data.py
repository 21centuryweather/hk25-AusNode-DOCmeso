


    # # == UM model data ==
    # zoom = '4'
    # folder = '/g/data/qx55/uk_node/glm.n2560_RAL3p3'
    # filename = 'data.healpix.PT1H.z' + zoom + '.zarr'
    # ds_um = xr.open_zarr(f'{folder}/{filename}')
    # print(ds_um)
    # exit()

    # ds_um = ds_um.isel(time = 0)
    # ds_um = ds_um.pipe(egh.attach_coords)
    # print(ds_um['pr'])
    # exit()

    # # -- pick out region of interest in healpix --
    # da_subset = ds_um['pr'].where((ds_um["lat"] > lat_min) & (ds_um["lat"] < lat_max) & (ds_um["lon"] > lon_min) & (ds_um["lon"] < lon_max), drop=True)
    # print(da_subset)
    # exit()



