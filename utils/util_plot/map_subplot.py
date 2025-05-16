'''
# -----------------
#   map_subplot
# -----------------

'''

# == imports ==
# -- Packages --
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

import warnings
warnings.simplefilter("ignore")

# == general plotfuncs ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds           # [left, bottom, width, height]
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def move_col(ax, moveby):
    ax_position = ax.get_position()             
    _, bottom, width, height = ax_position.bounds       # [left, bottom, width, height]
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds         # [left, bottom, width, height]
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def plot_xlabel(ds, fig, ax):
    if ds.attrs.get('hide_xlabel', False):
        return
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - ds.attrs.get('xlabel_pad', 0.1), 
            ds.attrs.get('xlabel_label', ''), 
            ha = 'center', 
            fontsize = ds.attrs.get('xlabel_fontsize', 5), 
            transform=fig.transFigure)
    
def plot_ylabel(ds, fig, ax):
    if ds.attrs.get('hide_ylabel', False):
        return
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - ds.attrs.get('ylabel_pad', 0.1), 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            ds.attrs.get('ylabel_label', ''), 
            va = 'center', 
            rotation='vertical', 
            fontsize = ds.attrs.get('ylabel_fontsize', 5), 
            transform=fig.transFigure)
    
def plot_ax_title(ds, fig, ax):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + ds.attrs.get('axtitle_xpad', 0.01),          # x-start
            ax_position.y1 + ds.attrs.get('axtitle_ypad', 0.01),          # y-start
            ds.attrs.get('axtitle_label', ''),                          
            fontsize = ds.attrs.get('axtitle_fontsize', 5), 
            transform=fig.transFigure,
            linespacing = 0.75
            # fontweight = 'bold'
            )

def plot_ticks(ds, ax):
    # x-ticks
    ax.set_xticks(ds.attrs.get('xticks', ds.lon[::int(len(ds.lon) / 5)].astype(int)), crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = ds.attrs.get('xticks_fontsize', 5))
    ax.xaxis.set_tick_params(length = ds.attrs.get('xtick_length', 2))
    ax.xaxis.set_tick_params(width = ds.attrs.get('xtick_width', 1))
    # y-ticks
    ax.set_yticks(ds.attrs.get('yticks', ds.lat[::int(len(ds.lat) / 3)].astype(int)), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = ds.attrs.get('yticks_fontsize', 5)) 
    ax.yaxis.set_tick_params(length = ds.attrs.get('xtick_length', 2))
    ax.yaxis.set_tick_params(width = ds.attrs.get('xtick_width', 1))
    # both
    ax.yaxis.set_ticks_position('both')
    # remove if requested
    ax.set_xticklabels('') if ds.attrs.get('hide_xticks', False) else None
    ax.set_yticklabels('') if ds.attrs.get('hide_yticks', False) else None

def cbar_ax_below(ds, fig, ax, h):
    if not ds.attrs.get('hide_colorbar', False):
        ax_position = ax.get_position()
        cbar_ax = fig.add_axes([ax_position.x0,                                                                     # left 
                                ax_position.y0 - ds.attrs.get('cbar_height', 0.1) - ds.attrs.get('cbar_pad', 0.1),  # bottom
                                ax_position.width,                                                                  # width
                                ds.attrs.get('cbar_height', 0.1)                                                    # height
                                ])
        cbar = fig.colorbar(h, cax = cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize = ds.attrs.get('cbar_numsize', 5))
        ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
                ax_position.y0 - ds.attrs.get('cbar_height', 0.1) - ds.attrs.get('cbar_pad', 0.1) - ds.attrs.get('cbar_label_pad', 0.025), 
                ds.attrs.get('cbar_label', ''), 
                ha = 'center', 
                fontsize = ds.attrs.get('cbar_fontsize', 5), 
                transform=fig.transFigure)
        
        # if tick_values:
        #     cbar.set_ticks(tick_values)
        return cbar

def plot_ref_line(axes, ds_line, ds):
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()  # Flatten if it's a 2D array of axes
    else:
        axes = [axes]  
    for dataset, ax in zip(list(ds_line.data_vars.keys()), axes):
        if 'lon' in ds_line.dims:
            lon_values = ds_line['lon']
            lat_values = ds_line[dataset]
            ax.scatter(lon_values, lat_values, transform=ccrs.PlateCarree(), color = 'r', s = ds.attrs.get('line_dots_size', 2))
        if 'lat' in ds_line.dims:
            lon_values = ds_line[dataset]
            lat_values = ds_line['lat']
            ax.scatter(lon_values, lat_values, transform=ccrs.PlateCarree(), color = 'r', s = ds.attrs.get('line_dots_size', 2))
    # for dataset, ax in zip(list(ds_line.data_vars.keys()), axes):
    #     if 'lon' in ds_line.dims:
    #         lon_values = ds_line['lon']
    #         lat_values = ds_line[dataset]
    #         # ax.scatter(lon_values, lat_values, transform=ccrs.PlateCarree(), color = 'g', s = ds.attrs['marker_size'])
    #         ax.plot(lon_values, lat_values, transform=ccrs.PlateCarree(), color=ds_line.attrs['color'], linestyle="--", dashes=ds_line.attrs['dashes'], linewidth = ds_line.attrs['linewidth'], alpha=1)
    #     if 'lat' in ds_line.dims:
    #         lon_values = ds_line[dataset]
    #         lat_values = ds_line['lat']
    #         # ax.scatter(lon_values, lat_values, transform=ccrs.PlateCarree(), color = 'g', s = ds.attrs['marker_size'])
    #         ax.plot(lon_values, lat_values, transform=ccrs.PlateCarree(), color=ds_line.attrs['color'], linestyle="--", dashes=ds_line.attrs['dashes'], linewidth = ds_line.attrs['linewidth'], alpha=1)


# == main ==
def plot(fig, nrows, ncols, row, col, ax, ds, ds_contour = None, ds_ontop = None, lines = []):
    # -- replace ax with map projection --
    ax.remove()                                                                             # Replace subplot with projection
    projection = ccrs.PlateCarree(central_longitude = 180)                                  # centre in pecific
    ax = fig.add_subplot(nrows, ncols, (row * ncols) + (col + 1), projection = projection)  # index starts at 1 here
    lat, lon = ds.lat, ds.lon                                                               # create domain
    lonm,latm = np.meshgrid(lon, lat)                                                       #
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())               # plot scene

    # -- plot contour--
    if ds_contour is None:
        pass
    else:
        lat_contour, lon_contour = ds_contour.lat, ds_contour.lon
        lonm_contour,latm_contour = np.meshgrid(lon_contour, lat_contour)
        if ds_contour.attrs.get('contour_text_size', False):
            contours = ax.contour(lonm_contour, latm_contour, ds_contour['var'], 
                                transform=ccrs.PlateCarree(),
                                levels =        ds_contour.attrs.get('threshold'),
                                colors =        ds_contour.attrs.get('color'), 
                                linewidths =    ds_contour.attrs.get('linewidth'))
            ax.clabel(contours, inline = True, fontsize = ds_contour.attrs.get('contour_text_size'), fmt = '%1.0f', colors = ds_contour.attrs.get('color')) 
        else:
            contours = ax.contour(lonm_contour, latm_contour, ds_contour['var'], 
                                transform=ccrs.PlateCarree(),
                                levels =        [ds_contour.attrs.get('threshold')],
                                colors =        ds_contour.attrs.get('color'), 
                                linewidths =    ds_contour.attrs.get('linewidth'))

    # -- plot data --
    h_pcm = ax.pcolormesh(lonm, latm, ds['var'], 
                            transform=ccrs.PlateCarree(),
                            cmap = ds.attrs.get('cmap', 'Blues'), 
                            vmin = ds.attrs.get('vmin', None), 
                            vmax = ds.attrs.get('vmax', None)
                            )

    # -- plot field ontop --
    if ds_ontop is None:
        pass
    else:
        name_ontop = list(ds_ontop.data_vars)[0]
        da_ontop = ds_ontop[name_ontop]
        ax.pcolormesh(lonm, latm, da_ontop, 
                    transform=ccrs.PlateCarree(),
                    cmap = 'Greys', 
                    vmin = 0, 
                    vmax = 1)
        
    # -- plot reference line --
    for ds_line in lines:
        plot_ref_line(ax, ds_line, ds)

    # -- format axes --
    ax.coastlines(resolution = "110m", linewidth = ds.attrs.get('coastline_width', 0.6))
    scale_ax(ax, ds.attrs.get('scale', 1))
    move_row(ax, ds.attrs.get('move_row', 0))     
    move_col(ax, ds.attrs.get('move_col', 0))
    plot_ticks(ds, ax)
    plot_xlabel(ds, fig, ax)
    plot_ylabel(ds, fig, ax)
    plot_ax_title(ds, fig, ax)
    cbar_ax_below(ds, fig, ax, h_pcm)
    return ax


# == when this script is ran / submitted ==
if __name__ == '__main__':
    print('executes')








# -- coastlines --
    # ax.coastlines(resolution = "110m", linewidth = 0.6)
    # ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)  # Borders
    # ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')  # Land



    # ax.add_feature(cfeature.COASTLINE, linewidth=0.6)  # Coastlines