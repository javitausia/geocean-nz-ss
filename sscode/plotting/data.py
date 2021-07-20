# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from .utils import plot_ccrs_nz, custom_cmap
from .config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_title, _fontsize_legend, _mbar_diff
from ..config import default_location, default_region


def plot_pres_winds(data, data_name='CFSR',
                    lat_name: str = 'lat',
                    lon_name: str = 'lon',
                    wind_proj: str = 'wind_proj',
                    u_name: str = 'U_GRD_L103',
                    v_name: str = 'V_GRD_L103'):
    """
    This funtion plots the previously loaded pressure data

    Args:
        data (list): This is a list with the era5 data
    """

    # figure and axis
    fig, axes = plt.subplots(
        ncols=3,figsize=(_figsize_width*4.0,_figsize_height*1.2), 
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
    quiv_step = 8
    axes[2].quiver(data[1][lon_name].values[::quiv_step],
                   data[1][lat_name].values[::quiv_step],
                   data[1][wind_proj].mean(dim='time').values[::quiv_step,::quiv_step]*\
                       np.cos(data[1]['direc_proj_math'].values[::quiv_step,::quiv_step])*10,
                   data[1][wind_proj].mean(dim='time').values[::quiv_step,::quiv_step]*\
                       np.sin(data[1]['direc_proj_math'].values[::quiv_step,::quiv_step])*10,
                   transform=ccrs.PlateCarree())
    axes[2].set_facecolor('lightblue')
    # plot map and points
    plot_ccrs_nz(axes.flatten(),plot_region=(True,default_region),
                 plot_location=(True,default_location),
                 plot_land=False)
    fig.suptitle(data_name+' data available',fontsize=_fontsize_title)

    # show results
    plt.show()


def plot_all_data(private_tgs = None, 
                  uhslc_tgs = None, 
                  codec_hind = None, 
                  moana_hind = None,
                  moana_hind_all = None,
                  pres_cfsr = None):
    """
    Plot all data available, both in a map and the time series if specified

    Args:
        All the args are the data previously loaded in the sscode/data.py file
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
            c='red',alpha=0.8,zorder=14
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
            c='orange',alpha=0.5,zorder=8
        )
    if codec_hind:
        ax.scatter(
            codec_hind.codec_coords_lon.values,
            codec_hind.codec_coords_lat.values,
            transform=ccrs.PlateCarree(),s=50,
            label='CoDEC hindcast',c='red',zorder=15
        )
    if private_tgs:
        ax.scatter(
            private_tgs.longitude.values,
            private_tgs.latitude.values,
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
        ncol=2,fancybox=True,shadow=True,
        fontsize=_fontsize_legend
    ).set_zorder(20)
    # plot the map
    plot_ccrs_nz([ax],plot_region=(True,default_region),
                 plot_coastline=(False,None,None))

    # show results
    plt.show()

    # TODO: add time series from forensic.ipynb


def plot_winds(wind_data, n_times: int = 1,
               quiv_step: int = 2,
               wind_coords: tuple = ('lon','lat'),
               wind_vars: tuple = ('U_GRD_L103','V_GRD_L103','wind_proj_mask')):

    """
    This funtion plots the previously loaded wind data, so we
    can compare the relation between the real u and v components
    of the wind and the calculated projected winds

    Args:
        data (list): This is an xarray.Dataset that is returned
        in the constructor of the data.py file, where the winds
        can be loaded/calculated
    """

    print('\n plotting the projected winds!! \n')
    
    times_to_plot = np.random.randint(0,len(wind_data.time.values),n_times)
    for time in times_to_plot:
        fig, axes = plt.subplots(ncols=2,figsize=_figsize,
            subplot_kw={
                'projection':ccrs.PlateCarree(
                    central_longitude=default_location[0]
                )
            }
        )
        wind_data[wind_vars[2]].isel(time=time).plot(
            ax=axes[0],cmap='jet',vmin=-1,vmax=1,
            transform=ccrs.PlateCarree()
        )
        axes[1].streamplot(
            wind_data[wind_coords[0]].values[::quiv_step],
            wind_data[wind_coords[1]].values[::quiv_step],
            wind_data.isel(time=time)[wind_vars[0]].values[::quiv_step,::quiv_step],
            wind_data.isel(time=time)[wind_vars[1]].values[::quiv_step,::quiv_step],
            transform=ccrs.PlateCarree(),density=12
        )
        axes[1].set_title('REAL stream-plot',fontsize=_fontsize_title)
        # plot map and show
        plot_ccrs_nz(
            axes,plot_land=False,plot_labels=(False,None,None)
        )
        plt.show()


def plot_pres_ibar(slp_data, ss_data, n_times: int = 1):

    """
    This funtion plots the previously loaded pressure data and the
    inverse barometer, to see differences in them

    Args:
        data (list): The real slp and ss data must be provided
        as xarray datasets
    """

    print('\n plotting the inverse barometer!! \n')

    # interp slp to ss
    slp_data = slp_data.interp(
        longitude=ss_data.lon,
        latitude=ss_data.lat
    )
    # check time coherence
    common_times = np.intersect1d(
        pd.to_datetime(slp_data.time.values),
        pd.to_datetime(ss_data.time.values),
        return_indices=True
    )
    # analyze same data
    slp_data = slp_data.isel(time=common_times[1]) / 100.0 # to mbar
    ss_data = ss_data.isel(time=common_times[2])
    ss_data.name = 'Storm surge [m]'

    # calculate the inverse barometer
    inv_bar = -(slp_data-slp_data.mean(dim='time')) / 100.0 # to cm
    inv_bar.name = 'Inverse barometer [m]'

    # plot ALL results

    # seasonality
    p = inv_bar.groupby('time.season').quantile(0.99).plot(
        col='season',subplot_kws={
            'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        },cmap=custom_cmap(15,'YlOrRd',0.15,0.9,'YlGnBu_r',0,0.85),
        vmin=0,vmax=0.4,transform=ccrs.PlateCarree(),figsize=_figsize
    )
    for ax in p.axes.flatten():
        ax.coastlines()
    p = ss_data.groupby('time.season').quantile(0.99).plot(
        col='season',subplot_kws={
            'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        },cmap=custom_cmap(15,'YlOrRd',0.15,0.9,'YlGnBu_r',0,0.85),
        vmin=0,vmax=0.4,transform=ccrs.PlateCarree(),figsize=_figsize
    )
    for ax in p.axes.flatten():
        ax.coastlines()

    # ind. time plots
    times_to_plot = np.random.randint(0,len(slp_data.time.values),n_times)
    for time in times_to_plot:
        fig, axes = plt.subplots(ncols=3,figsize=_figsize,
            subplot_kw={
                'projection':ccrs.PlateCarree(
                    central_longitude=default_location[0]
                )
            }
        )
        # slp_data.isel(time=time).plot(
        #     ax=axes[0],transform=ccrs.PlateCarree(),
        #     cmap='RdBu_r',vmin=1013-_mbar_diff,vmax=1013+_mbar_diff
        # )
        inv_bar.isel(time=time).plot(
            ax=axes[0],transform=ccrs.PlateCarree(),cmap='plasma',
            vmin=-0.2,vmax=0.2
        )
        ss_data.isel(time=time).plot(
            ax=axes[1],transform=ccrs.PlateCarree(),cmap='plasma',
            vmin=-0.2,vmax=0.2
        )
        (inv_bar.isel(time=time)-ss_data.isel(time=time))\
            .to_dataset(name='elevation_difference')\
            .apply(np.abs).elevation_difference.plot(
            ax=axes[2],transform=ccrs.PlateCarree(),cmap='jet',
            vmin=0,vmax=0.2
        )
        plot_ccrs_nz(
            axes,plot_land=False,plot_labels=(False,None,None)
        ) # plot nz map, as always

        # show results
        plt.show()


def plot_uhslc_locations(uhslc_data):

    """
    This funtion plots the previously loaded uhslc data

    Args:
        data (list): This is the uhslc data as loaded with the
        fuction join_load_uhslc_tgs() in data.py
    """

    fig, ax = plt.subplots(figsize=(9,9),subplot_kw={
        'projection':ccrs.PlateCarree(central_longitude=180)
    })
    xr.plot.scatter(
        uhslc_data.max(dim='time'),
        x='longitude',y='latitude',c='red',zorder=110,
        ax=ax,transform=ccrs.PlateCarree(),
        s=100,edgecolor='yellow'
    )
    for i,tg in enumerate(uhslc_data.name):
        if i==6:
            ax.text(
                tg.longitude.values-3.0,tg.latitude.values-0.2,str(tg.values)[2:],
                transform=ccrs.PlateCarree(),zorder=100,size=12,
                bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                )
            )
        elif i==11:
            ax.text(
                tg.longitude.values-2.5,tg.latitude.values-0.2,str(tg.values)[2:],
                transform=ccrs.PlateCarree(),zorder=100,size=12,
                bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                )
            )
        elif i==2:
            ax.text(
                tg.longitude.values-1.0,tg.latitude.values+0.7,str(tg.values)[2:],
                transform=ccrs.PlateCarree(),zorder=100,size=12,
                bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                )
            )
        elif i==4:
            pass
        elif i==1 or i==7 or i==8:
            ax.text(
                tg.longitude.values+0.6,tg.latitude.values-0.5,str(tg.values)[2:],
                transform=ccrs.PlateCarree(),zorder=100,size=12,
                bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                )
            )
        else:
            ax.text(tg.longitude.values+0.6,tg.latitude.values,str(tg.values)[2:],
                    transform=ccrs.PlateCarree(),zorder=100,size=12,
                    bbox=dict(boxstyle="round",
                        ec=(1., 0.5, 0.5),
                        fc=(1., 0.8, 0.8),
                    ))
    plot_ccrs_nz([ax],plot_labels=(True,5,5))
    ax.set_facecolor('lightblue')

