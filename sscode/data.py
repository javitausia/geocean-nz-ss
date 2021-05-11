# basic
import os, glob, sys

# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from .config import data_path, default_location # get config params
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_title, _fontsize_legend
from .utils import calculate_relative_winds
from .validation import compare_datasets
from .plotting.data import plot_pres_winds
from .plotting.utils import plot_ccrs_nz

# warnings
import warnings
warnings.filterwarnings('ignore')


# loader dicts summary
loader_dict_options = {
    'predictor': ['cfsr','era5'],
    'predictand': ['dac','moana','codec'],
    'validator': ['uhslc','geotgs']
}
# dataset attrs
datasets_attrs = {
    'dac': ('longitude','latitude',None,'DAC global reanalysis'),
    'moana': ('lon','lat','site','Moana v2 hindcast'),
    'codec': ('codec_coords_lon','codec_coords_lat','name','CoDEC reanalysis'),
    'uhslc': ('longitude','latitude','name','UHSLC tgs'),
    'geotgs': ('longitude','latitude','name','GeoOcean tgs')
}


class Loader(object):
    """
    This class loads the data that will be used in future parts. This class is
    useful if all the data is wanted to be loaded at the same time, and then the
    methods in the class can be easily used, specifying just the list with all
    the datasets in the correct order

    """


    def __init__(self, data_to_load: list = ['cfsr','moana','uhslc'],
                 location: tuple = default_location,
                 plot: bool = True):
        """
        Loader class constructor

        Args:
            data_to_load (list, optional): List with the predictor, predictand 
            and validator: 
                - Defaults to ['era5','moana','uhslc'].
            location: location if required
            plot: whether to plot or not the loaded data
        """

        # save location
        self.location = location

        # load the predictor
        if data_to_load[0] in loader_dict_options['predictor']:
            if data_to_load[0]=='era5':
                predictor = load_era5(time='1D',load_winds=(True,location))
                if len(predictor)==1:
                    self.predictor_slp = predictor
                else:
                    self.predictor_slp = predictor[0]
                    self.predictor_wind = predictor[1]
            elif data_to_load[0]=='cfsr':
                predictor = load_cfsr(time='1D',load_winds=(True,location))
                if len(predictor)==1:
                    self.predictor_slp = predictor
                else:
                    self.predictor_slp = predictor[0]
                    self.predictor_wind = predictor[1]
            else:
                print('\n data not available for the predictor!! \n')

        # load the predictand
        if data_to_load[1] in loader_dict_options['predictand']:
            if data_to_load[1]=='dac':
                self.predictand = load_dac()
                self.predictand_attrs = datasets_attrs[data_to_load[1]]

            if data_to_load[1]=='moana':
                self.predictand = load_moana_hindcast(plot=plot)
                self.predictand_attrs = datasets_attrs[data_to_load[1]]
            elif data_to_load[1]=='codec':
                self.predictand = load_codec_hindcast(plot=plot)
                self.predictand_attrs = datasets_attrs[data_to_load[1]]
            else:
                print('\n data not available for the predictand!! \n')

        # load the validator
        if data_to_load[2] in loader_dict_options['validator']:
            if data_to_load[2]=='uhslc':
                self.validator = join_load_uhslc_tgs(plot=True)
                self.validator_attrs = datasets_attrs[data_to_load[2]]
            elif data_to_load[2]=='geotgs':
                self.validator = load_geocean_tgs(plot=plot)
                self.validator_attrs = datasets_attrs[data_to_load[2]]
            else:
                print('\n data not available for the validation!! \n')
                
                
    def validate_datasets(self, # this is prepared for UHSLC-Moana
                          comparison_variables: list = [['ss','msea'],['ss','msea']],
                          time_resample = None):
        """
        This method validates the loaded data with the compare_datasets funtion

        """

        self.predictand_reduced, self.ss_stats = compare_datasets(
            self.predictand,self.predictand_attrs,
            self.validator,self.validator_attrs,
            comparison_variables=comparison_variables,
            time_resample=time_resample
        ) # compare datasets


def load_era5(data_path: str = data_path,
              time: str = '1997', # time cropping recommended
              load_winds: tuple = (True,default_location)):
    """
    This function loas era5 data and crops it to a time frame
    of a year, or resamples it daily, as it is very difficult to 
    work will all the data at the same time. The winds can be easily
    loaded, and also cropped and projected in the direction of a
    location if requested

    Args:
        data_path (str, optional): Data path folder in repository. 
            - Defaults to data_path.
        time (str, optional): Year to crop the data. It can also be a time
            step to resample the data as 1H, 6H, 1D...
            - Defaults to '1997'.
        load_winds: this indicates wheter the winds are loaded or not, and
            the location of the projected winds

    Returns:
        [list]: This is a list with the data loaded
    """

    # load/calculate... xarray datasets
    print('\n loading the sea-level-pressure fields... \n')
    if time=='1D' or time=='6H':
        # resample to daily
        if os.path.isfile(data_path+'/era_5/ERA5_MSLP_daily.nc'):
            print('\n loading daily resampled data... \n')
            # loading resampled data
            mslp = xr.open_dataarray(data_path+'/era_5/ERA5_MSLP_daily.nc')
            if load_winds[0]:
                wind = xr.open_dataset(data_path+'/era_5/ERA5_WINDs_daily.nc')
                # plot the data
                plot_pres_winds([mslp,wind],data_name='ERA5',
                                lat_name='latitude',lon_name='longitude',
                                u_name='u10',v_name='v10')
            # return data
            return_data = [mslp] if not load_winds[0] else [mslp,wind]
            return return_data

        else:
            print('\n resampling data to {}... \n'.format(time))
            mslp = xr.open_dataset(data_path+'/era_5/ERA5_MSLP_1H_1979_2021.nc')['msl']\
                .resample(time=time).mean()
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/era_5/ERA5_10mu_1H_1979_2021.nc')['u10']\
                .resample(time=time).mean()
            vw = xr.open_dataset(data_path+'/era_5/ERA5_10mv_1H_1979_2021.nc')['v10']\
                .resample(time=time).mean()
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw,
                                            lat_name='latitude',
                                            lon_name='longitude')
            # plot the data
            plot_pres_winds([mslp,wind],data_name='ERA5',
                            lat_name='latitude',lon_name='longitude',
                            u_name='u10',v_name='v10')
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    else:
        mslp = xr.open_dataset(data_path+'/era_5/ERA5_MSLP_1H_1979_2021.nc')['msl']
        # try year cropping
        if time:
            mslp = mslp.sel(time=time)
            print(' cropping the data to {} \n'.format(int(time)))
        else:
            print('\n LOADING ALL THE MSLP DATA (be careful with memory) \n')
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/era_5/ERA5_10mu_1H_1979_2021.nc')['u10']\
                .sel(time=time)
            vw = xr.open_dataset(data_path+'/era_5/ERA5_10mv_1H_1979_2021.nc')['v10']\
                .sel(time=time)
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw,
                                            lat_name='latitude',
                                            lon_name='longitude')
            # plot the data
            plot_pres_winds([mslp,wind],data_name='ERA5',
                            lat_name='latitude',lon_name='longitude',
                            u_name='u10',v_name='v10')
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')

    # return the loaded datasets
    return_data = [mslp] if not load_winds[0] else [mslp,wind]

    return return_data


def load_cfsr(data_path: str = data_path,
              time: str = '1997', # time cropping recommended
              load_winds: tuple = (False,default_location)):
    """
    This function loas cfsr data and crops it to a time frame
    of a year, or resamples it daily, as it is very difficult to 
    work will all the data at the same time. The winds can be easily
    loaded, and also cropped and projected in the direction of a
    location if requested

    Args:
        data_path (str, optional): Data path folder in repository. 
            - Defaults to data_path.
        time (str, optional): Year to crop the data. It can also be a time
            step to resample the data as 1H, 6H, 1D...
            - Defaults to '1997'.
        load_winds: this indicates wheter the winds are loaded or not, and
            the location of the projected winds

    Returns:
        [list]: This is a list with the data loaded
    """

    # load/calculate... xarray datasets
    print('\n loading the sea-level-pressure fields... \n')
    if time=='1D' or time=='6H':
        # resample to daily
        if os.path.isfile(data_path+'/cfsr/CFSR_MSLP_daily.nc'):
            print('\n loading daily resampled data... \n')
            # loading resampled data
            mslp = xr.open_dataarray(data_path+'/cfsr/CFSR_MSLP_daily.nc')
            if load_winds[0]:
                wind = xr.open_dataset(data_path+'/cfsr/CFSR_WINDs_daily.nc')
                # plot the data
                plot_pres_winds([mslp,wind],data_name='CFSR')
            # return data
            return_data = [mslp] if not load_winds[0] else [mslp,wind]
            return return_data

        else:
            print('\n resampling data to {}... \n'.format(time))
            mslp = xr.open_dataset(data_path+'/cfsr/CFSR_MSLP_1H_1990_2021.nc')['SLP']\
                .resample(time=time).mean()
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/cfsr/CFSR_uwnd_1H_1990_2011.nc')['U_GRD_L103']\
                .resample(time=time).mean()
            vw = xr.open_dataset(data_path+'/cfsr/CFSR_vwnd_1H_1990_2011.nc')['V_GRD_L103']\
                .resample(time=time).mean()
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw)
            # plot the data
            plot_pres_winds([mslp,wind],data_name='CFSR')
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    else:
        mslp = xr.open_dataset(data_path+'/cfsr/CFSR_MSLP_1H_1990_2021.nc')['SLP']
        # try year cropping
        if time:
            mslp = mslp.sel(time=time)
            print(' cropping the data to {} \n'.format(int(time)))
        else:
            print('\n LOADING ALL THE MSLP DATA (be careful with memory) \n')
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/cfsr/CFSR_30mu_1H_1990_2020.nc')['u30']\
                .resample(time=time).mean()
            vw = xr.open_dataset(data_path+'/cfsr/CFSR_30mv_1H_1990_2020.nc')['v30']\
                .resample(time=time).mean()
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw)
            # plot the data
            plot_pres_winds([mslp,wind],data_name='CFSR')
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')

    # return the loaded datasets
    return_data = [mslp] if not load_winds[0] else [mslp,wind]

    return return_data


def join_load_uhslc_tgs(files_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/uhslc/processed/*.nc',
    plot: bool = False):

    """
    Join all the uhslc tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # join files assigning a name to each
    print('\n loading and plotting the UHSLC tidal guages... \n')
    uhslc_tgs_list = []
    for file in glob.glob(files_path):
        uhslc_tg = xr.open_dataset(file)
        uhslc_tgs_list.append(
            uhslc_tg.expand_dims(dim='name').assign_coords(
                {'name':(('name'),[file[100:-13]]),
                'latitude':(('name'),[uhslc_tg.latitude]),
                'longitude':(('name'),[uhslc_tg.longitude])}
            ) 
        )

    # join and plot
    uhslc_tgs = xr.concat(uhslc_tgs_list,dim='name',combine_attrs='drop')
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        hue_plot = uhslc_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('UHSLC tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(uhslc_tgs.name.values),loc='lower left',ncol=6)
        fig, axes = plt.subplots(ncols=2,nrows=6,figsize=(_figsize_width*4,6),
                                sharex=True,sharey=True)
        for axi in range(len(uhslc_tgs.name.values)):
            uhslc_tgs.isel(name=axi).ss.plot(
                ax=axes.flatten()[axi],c=hue_plot[axi].get_color(),alpha=0.6,
                label=uhslc_tgs.name.values[axi].upper()
            )
            axes.flatten()[axi].legend(loc='lower left')
            axes.flatten()[axi].set_title('')
            axes.flatten()[axi].set_xlabel('')
            axes.flatten()[axi].set_ylabel('')
        # show results
        plt.show()

    return uhslc_tgs


def load_geocean_tgs(file_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/geocean/tgs_geocean_NZ.nc',
    plot: bool = False):

    """
    Load all the geocean tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # load and plot all the geocean tgs
    print('\n loading and plotting the GeoOcean tidal guages... \n')
    geocean_tgs = xr.open_dataset(file_path)

    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        geocean_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('GeoOcean tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(geocean_tgs.name.values),loc='lower left',ncol=7)
        geocean_tgs.ss.plot(col='name',col_wrap=3,figsize=(12,7))
        # show results
        plt.show()

    return geocean_tgs


def load_moana_hindcast(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/moana_coast.zarr/',
    plot: bool = False):

    """
    Load and plot the moana hindcast (saved as .zarr) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the moana data
    """

    print('\n loading the Moana v2 hindcast data... \n')

    # load moana
    moana = xr.open_zarr(file_path)

    if plot: # plot if specified
        # calculate 99% quantile
        threshold = moana.ss.load().max(dim='time').mean()
        # values to plot
        moana_to_plot = moana.ss.load().groupby('time.season').max()-threshold
        # plot some stats
        fig, axes = plt.subplots(
            ncols=2,nrows=2,figsize=(_figsize_width*3.6,_figsize_height*2.6),
            subplot_kw={
                'projection':ccrs.PlateCarree(
                    central_longitude=180
                )
            }
        )
        fig.suptitle('Moana shore nodes quantiles anomalies',
                     fontsize=_fontsize_title)
        for seas,ax in zip(moana_to_plot.season.values,axes.flatten()):
            p = ax.scatter(
                moana.lon.values,moana.lat.values,
                c=moana_to_plot.sel(season=seas).values,
                transform=ccrs.PlateCarree(),
                s=20,zorder=40,cmap='jet',
                vmin=-0.3,vmax=0.3
            )
            pos_ax = ax.get_position()
            pos_colbar = fig.add_axes([
                pos_ax.x0 + pos_ax.width + 0.01, pos_ax.y0, 0.02, pos_ax.height
            ])
            fig.colorbar(p,cax=pos_colbar)
            ax.set_facecolor('lightblue')
            ax.set_title(seas,fontsize=_fontsize_title)
        # plot NZ map
        plot_ccrs_nz(axes.flatten(),
                     plot_coastline=(False,None,None),
                     plot_labels=(True,5,5))
        # show results
    plt.show()

    return moana


def load_moana_hindcast_ss(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/',
    plot: bool = False, daily: bool = True):

    """
    Load the moana hindcast (ss) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the ss moana data
    """

    print('\n loading the Moana v2 hindcast data (ss)... \n')

    # TODO: add basic plotting

    if os.path.isfile(file_path+'moana_ss_daily.nc') and daily:
        return xr.open_dataarray(
            file_path+'moana_ss_daily.nc'
        )
    else:
        return xr.open_zarr(file_path+'ss/')


def load_moana_hindcast_msea(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/msea/',
    plot: bool = False):

    """
    Load the moana hindcast (msea) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the msea moana data
    """

    print('\n loading the Moana v2 hindcast data (msea)... \n')

    # TODO: add basic plotting

    return xr.open_zarr(file_path)


def load_dac_hindcast(file_path: str = 
    data_path+'/storm_surge_data/dac_reanalysis/DAC_SS_6H_1993_2020.nc'):

    """
    Load the DAC hindcast in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the DAC data
    """

    print('\n loading the DAC hindcast data... \n')

    # TODO: add basic plotting

    return xr.open_dataarray(file_path)


def load_codec_hindcast(file_path: str = 
    data_path+'/storm_surge_data/codec_reanalysis/codec_geocean_NZ.nc',
    plot: bool = False):

    """
    Load the CoDEC hindcast in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the codec data
    """

    # load and plot codec
    print('\n loading and plotting the CoDEC numerical data... \n')
    codec_hind = xr.open_dataset(file_path).sel(time=slice('1980','2021'))
    codec_hind = codec_hind.assign_coords(
        {'time':pd.to_datetime(codec_hind.time.values).round('H')}
    )

    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        codec_hind.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('CoDEC numerical model',fontsize=_fontsize_title)
        ax.legend(list(codec_hind.name.values),loc='lower left',ncol=7)
        codec_hind.ss.plot(col='name',col_wrap=3,figsize=(12,7))
        # show results
        plt.show()

    return codec_hind

