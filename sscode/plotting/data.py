# arrays
import numpy as np

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from .utils import plot_ccrs_nz
from .config import _figsize, _fontsize_title, _fontsize_legend
from ..config import default_location, default_region


def plot_era5(data):
    """
    This funtion plots the previously loaded era5 data

    Args:
        data (list): This is a list with the era5 data
    """

    # figure and axis
    fig, axes = plt.subplots(
        ncols=3,figsize=_figsize, 
        subplot_kw={'projection':ccrs.PlateCarree(
            central_longitude=default_location[0]
        )}
    )    
    # sea-level-pressure fields
    data[0].mean(dim='time').plot(
        ax=axes[0],cmap='bwr',add_colorbar=False,
        vmin=101300-3000,vmax=101300+3000,
        transform=ccrs.PlateCarree()
    )
    # real and projected winds
    axes[1].streamplot(data[1].longitude.values,data[1].latitude.values,
                       data[1]['u10'].mean(dim='time').values,
                       data[1]['v10'].mean(dim='time').values,
                       color=np.sqrt(data[1]['u10'].mean(dim='time').values**2 +
                                     data[1]['v10'].mean(dim='time').values**2),
                       cmap='jet',density=6,transform=ccrs.PlateCarree())
    axes[2].quiver(data[1].longitude.values[::8],data[1].latitude.values[::8],
                   data[1]['wind_proj'].mean(dim='time').values[::8,::8]*\
                       np.cos(data[1]['direc_proj_math'].values[::8,::8]),
                   data[1]['wind_proj'].mean(dim='time').values[::8,::8]*\
                       np.sin(data[1]['direc_proj_math'].values[::8,::8]),
                   transform=ccrs.PlateCarree())
    axes[2].set_facecolor('lightblue')
    # plot map and points
    plot_ccrs_nz(axes.flatten())
    fig.suptitle('ERA5 data available',fontsize=_fontsize_title)


def plot_all_data(geocean_tgs = None, 
                  uhslc_tgs = None, 
                  codec_hind = None, 
                  moana_hind = None):
    """
    Plot all data available

    Args:
        geocean_tgs ([type]): [description]
        uhslc_tgs ([type]): [description]
        moana_hind ([type]): [description]
        codec_hind ([type]): [description]
    """

    # fig general attrs
    fig, ax = plt.subplots(
        figsize=(10,10),
        subplot_kw={'projection':ccrs.PlateCarree(
            central_longitude=180
        )}
    )
    ax.set_facecolor('lightblue')
    # plot all the data available
    if moana_hind:
        ax.scatter(
            moana_hind.lon.values,moana_hind.lat.values,
            transform=ccrs.PlateCarree(),
            label='Moana v2 hindcast',c='orange'
        )
    if codec_hind:
        ax.scatter(
            codec_hind.codec_coords_lon.values,
            codec_hind.codec_coords_lat.values,
            transform=ccrs.PlateCarree(),
            label='CoDEC hindcast',c='red',zorder=20
        )
    if geocean_tgs:
        ax.scatter(
            geocean_tgs.longitude.values,
            geocean_tgs.latitude.values,
            transform=ccrs.PlateCarree(),label='GeoOcean tidal gauges',
            c='darkblue',s=100,alpha=0.7,zorder=20
        )
    if uhslc_tgs:
        ax.scatter(
            uhslc_tgs.longitude.values,
            uhslc_tgs.latitude.values,
            transform=ccrs.PlateCarree(),label='UHSLC tidal gauges',
            c='green',s=100,alpha=0.9,zorder=20
        )
    ax.legend(loc='lower right',fontsize=_fontsize_legend)
    # plot the map
    plot_ccrs_nz([ax])

