# arrays
import numpy as np

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from .utils import plot_ccrs_nz
from .config import _figsize, _fontsize_title, _fontsize_legend
from ..config import default_location, default_region


def plot_pres_winds(data, data_name='CFSR',
                    lat_name: str = 'lat',
                    lon_name: str = 'lon',
                    u_name: str = 'U_GRD_L103',
                    v_name: str = 'V_GRD_L103'):
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
    axes[1].streamplot(data[1][lon_name].values,data[1][lat_name].values,
                       data[1][u_name].mean(dim='time').values,
                       data[1][v_name].mean(dim='time').values,
                       color=np.sqrt(data[1][u_name].mean(dim='time').values**2 +
                                     data[1][v_name].mean(dim='time').values**2),
                       cmap='jet',density=8,transform=ccrs.PlateCarree())
    quiv_step = 5
    axes[2].quiver(data[1][lon_name].values[::quiv_step],
                   data[1][lat_name].values[::quiv_step],
                   data[1]['wind_proj'].mean(dim='time').values[::quiv_step,::quiv_step]*\
                       np.cos(data[1]['direc_proj_math'].values[::quiv_step,::quiv_step]),
                   data[1]['wind_proj'].mean(dim='time').values[::quiv_step,::quiv_step]*\
                       np.sin(data[1]['direc_proj_math'].values[::quiv_step,::quiv_step]),
                   transform=ccrs.PlateCarree())
    axes[2].set_facecolor('lightblue')
    # plot map and points
    plot_ccrs_nz(axes.flatten())
    fig.suptitle(data_name+' data available',fontsize=_fontsize_title)


def plot_all_data(geocean_tgs = None, 
                  uhslc_tgs = None, 
                  codec_hind = None, 
                  moana_hind = None,
                  moana_hind_all = None,
                  pres_cfsr = None):
    """
    Plot all data available, both in a map and the time series

    Args:
        geocean_tgs ([type]): [description]
        uhslc_tgs ([type]): [description]
        moana_hind ([type]): [description]
        codec_hind ([type]): [description]
    """

    # map plot
    fig, ax = plt.subplots(
        figsize=(12,12),
        subplot_kw={'projection':ccrs.PlateCarree(
            central_longitude=180
        )}
    )
    ax.set_facecolor('lightblue')
    # plot all the data available
    if moana_hind:
        ax.scatter(
            moana_hind.lon.values,moana_hind.lat.values,
            transform=ccrs.PlateCarree(),s=30,
            label='Moana v2 hindcast shore - 5 km',
            c='red',alpha=0.8,zorder=12
        )
    if moana_hind_all:
        ax.scatter(
            np.meshgrid(
                moana_hind_all.lon.values,
                moana_hind_all.lat.values
            )[0].reshape(-1),
            np.meshgrid(
                moana_hind_all.lon.values,
                moana_hind_all.lat.values
            )[1].reshape(-1),
            transform=ccrs.PlateCarree(),s=5,
            label='Moana v2 hindcast offshore - 20 km',
            c='orange',alpha=0.5,zorder=4
        )
    if codec_hind:
        ax.scatter(
            codec_hind.codec_coords_lon.values,
            codec_hind.codec_coords_lat.values,
            transform=ccrs.PlateCarree(),s=50,
            label='CoDEC hindcast',c='red',zorder=14
        )
    if geocean_tgs:
        ax.scatter(
            geocean_tgs.longitude.values,
            geocean_tgs.latitude.values,
            transform=ccrs.PlateCarree(),
            label='More tidal gauges',
            c='darkblue',s=80,alpha=0.8,zorder=16
        )
    if uhslc_tgs:
        ax.scatter(
            uhslc_tgs.longitude.values,
            uhslc_tgs.latitude.values,
            transform=ccrs.PlateCarree(),label='UHSLC tidal gauges',
            c='green',s=60,alpha=0.8,zorder=16
        )
    try:
        name = pres_cfsr.name # check xarray existence
        ax.scatter(
            np.meshgrid(
                pres_cfsr.longitude.values,
                pres_cfsr.latitude.values
            )[0].reshape(-1),
            np.meshgrid(
                pres_cfsr.longitude.values,
                pres_cfsr.latitude.values
            )[1].reshape(-1),
            transform=ccrs.PlateCarree(),s=5,
            label='CFSR slp fields - 0.5 º',
            c='darkblue',alpha=0.5,zorder=2
        )
    except:
        pass
    # legend attrs
    ax.legend(
        loc='lower left', # bbox_to_anchor=(-0.1,1.05),
        ncol=2,fancybox=True,shadow=True
    )
    # plot the map
    plot_ccrs_nz([ax],plot_region=(True,default_region))

    # TODO: add time series from forensic.ipynb

