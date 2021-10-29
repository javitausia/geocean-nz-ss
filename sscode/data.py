# basic
import os, glob, sys
from datetime import datetime

# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from .config import data_path, default_location # get config params
#data_path = os.getenv('SSURGE_DATA_PATH', data_path)
print("DATA PATH", data_path)
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
    'predictor': ['cfsr','era_5'],
    'predictand': ['dac','moana','codec'],
    'validator': ['uhslc','linz','other','privtgs']
}
# dataset attrs
datasets_attrs = {
    'era5': ('longitude','latitude',None,'ERA 5 reanalysis','u10','v10'),
    #'cfsr': ('lon','lat',None,'CFSR reanalysis','U_GRD_L103','V_GRD_L103'),
    'cfsr': ('longitude','latitude',None,'CFSR reanalysis','ugrd10m','vgrd10m'),
    'dac': ('longitude','latitude',None,'DAC global reanalysis'),
    'moana': ('lon','lat','site','Moana v2 hindcast'),
    'codec': ('codec_coords_lon','codec_coords_lat','name','CoDEC reanalysis'),
    'uhslc': ('longitude','latitude','name','UHSLC tgs'),
    'linz': ('longitude','latitude','name','LINZ tgs'),
    'other': ('longitude','latitude','name','OTHER tgs'),
    'privtgs': ('longitude','latitude','name','Private tgs')
}


class Loader(object):
    """
    This class loads the data that will be used in future parts. This class is
    useful if all the data is wanted to be loaded at the same time, and then the
    methods in the class can be easily used, specifying just the list with all
    the datasets in the correct order

    TODO: add Experiment to Loader class??

    """

    def __init__(self, data_to_load: list = ['cfsr','moana','uhslc'],
                 time_resample: str = '1D', load_winds: bool = True,
                 location: tuple = default_location, plot: bool = True):
        """
        Loader class constructor

        ** By default, the Loader loads / calculates the winds projected in
        default_location, but the loaded data can be re-projected using
        the function calculate_relative_winds() in utils.py **

        Most of the parameters in this Loader class refer to the predictor
        loading functions, as those are the ones that might inccur memory
        problems

        Args:
            data_to_load (list, optional): List with the predictor, predictand 
            and validator:
                - Defaults to ['cfsr','moana','uhslc'].
            time_resample (str, optional): Time step to resample the data to, this might
                also by a year / years to crop the data to. Notice that this time
                resample is only made in the predictor, as the other datasets
                do not occupy much memory. Defaults to '1D'.
            load_winds (bool, optional): Load or not the wind data. It is recommended
                to be loaded, but be careful killing the kernel. Defaults to True.
            location (tuple, optional): Location if required (to project the winds).
                It is alsa saved in the class attributes. Defaults to default_location.
            plot (bool, optional): Whether to plot or not the loaded data.
                Defaults to True.
        """

        # save location
        self.location = location

        # load the predictor
        if data_to_load[0] in loader_dict_options['predictor']:
            if data_to_load[0]=='era_5':
                predictor = load_era_5(
                    time=time_resample,
                    load_winds=(load_winds,location),
                    plot=plot
                )
            elif data_to_load[0]=='cfsr':
                predictor = load_cfsr(
                    time=time_resample,
                    load_winds=(load_winds,location),
                    plot=plot
                )
            # check loaded slp / winds and save attributes
            if len(predictor)==1:
                self.predictor_slp = predictor[0]
            else:
                self.predictor_slp = predictor[0]
                self.predictor_wind = predictor[1]
            self.predictor_attrs = datasets_attrs[data_to_load[0]]
        else:
            print('\n data not available for this predictor!! \n')

        # load the predictand
        if data_to_load[1] in loader_dict_options['predictand']:
            if data_to_load[1]=='dac':
                self.predictand = load_dac_hindcast(plot=plot)
            elif data_to_load[1]=='moana':
                self.predictand = load_moana_hindcast(plot=plot)
            elif data_to_load[1]=='codec':
                self.predictand = load_codec_hindcast(plot=plot)
            self.predictand_attrs = datasets_attrs[data_to_load[1]]
        else:
            print('\n data not available for this predictand!! \n')

        # load the validator
        if data_to_load[2] in loader_dict_options['validator']:
            if data_to_load[2]=='uhslc':
                self.validator = join_load_uhslc_tgs(plot=plot)
            elif data_to_load[2]=='privtgs':
                self.validator = load_private_tgs(plot=plot)
            elif data_to_load[2]=='linz':
                self.validator = join_load_linz_tgs(plot=plot)
            elif data_to_load[2]=='other':
                self.validator = join_load_other_tgs(plot=plot)
            self.validator_attrs = datasets_attrs[data_to_load[2]]
        else:
            print('\n data not available for this validator!! \n')
                
                
    def validate_datasets(self, # this is prepared for UHSLC-Moana
                          comparison_variables: list = [['ss','msea'],['ss','msea']],
                          time_resample = None):
        """
        This method validates the loaded data with the compare_datasets function

        Args:
            comparison_variables (List of lists): These are two python lists that save
                the name of the variables to be compared, the first list refer to the
                predictand, and the second list refer to the validator
            time_resample (str, optional): Time resample step in case the comparison
                might be done in a reduced time step. Defaults to None.

        """

        self.predictand_reduced, self.validator_reduced, self.ss_stats = compare_datasets(
            self.predictand,self.predictand_attrs,
            self.validator,self.validator_attrs,
            comparison_variables=comparison_variables,
            time_resample=time_resample
        ) # compare datasets


def load_era_5(data_path: str = data_path+'/era_5/',
               time: str = '1997', # time cropping recommended
               load_winds: tuple = (True,default_location),
               plot: bool = True):
    """
    This function loads the era5 data and crops it to a time frame
    of a year, or resamples it daily, as it is very difficult to 
    work with all the data at the same time. The winds can be easily
    loaded, and also cropped and projected in the direction of a
    location if requested

    Args:
        data_path (str, optional): Data path folder in repository. 
            - Defaults to data_path.
        time (str, optional): Year to crop the data. It can also be a time
            step to resample the data as 1H, 6H, 1D...
            - Defaults to '1997'.
        load_winds (tuple, optional): this indicates wether the winds are loaded or not, and
            the location of the projected winds.
            - Defaults to (True,default_location)
        plot (bool, optional): Whether to plot or not the results.
            - Defaults to True.
            
    Returns:
        [list]: This is a list with the data loaded.
    """

    # load / calculate... xarray datasets
    print('\n loading the sea-level-pressure fields... \n')
    if time not in list(
        np.arange(1988,2022,1).astype(str) # years of data + 2
    ):
        # resample to daily
        if time=='1D' and os.path.isfile(os.path.join(data_path,
                                                      'ERA5_MSLP_daily.nc')) and \
            os.path.isfile(
                os.path.join(data_path,'ERA_5_WINDs_daily.nc')
            ): # check files existance
            print('\n loading daily resampled data... \n')
            # loading resampled data
            mslp = xr.open_dataarray(os.path.join(data_path,
                                                  'ERA5_MSLP_daily.nc'))
            if load_winds[0]:
                wind = xr.open_dataset(os.path.join(data_path,
                                                    'ERA_5_WINDs_daily.nc'))
                # plot the data
                plot_pres_winds( # if plot, slp and winds must be loaded
                    [mslp,wind],data_name=datasets_attrs['era_5'][3],
                    lat_name=datasets_attrs['era_5'][1],
                    lon_name=datasets_attrs['era_5'][0],
                    u_name=datasets_attrs['era_5'][4],
                    v_name=datasets_attrs['era_5'][5],
                    wind_proj='wind_proj_mask'
                ) if plot else None
            # return data
            return_data = [mslp] if not load_winds[0] else [mslp,wind]
            return return_data
        else:
            print('\n resampling data to {}... \n'.format(time))
            mslp = xr.open_dataset(os.path.join(data_path,
                                                'ERA5_MSLP_1H_1979_2021.nc'))['msl']\
                .resample(time=time).mean()
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(os.path.join(data_path,
                                              'ERA5_10mu_1H_1979_2021.nc'))['u10']\
                .resample(time=time).mean()
            vw = xr.open_dataset(os.path.join(data_path,
                                              'ERA5_10mv_1H_1979_2021.nc'))['v10']\
                .resample(time=time).mean()
            wind = calculate_relative_winds(
                location=load_winds[1],uw=uw,vw=vw,
                lat_name=datasets_attrs['era_5'][1],
                lon_name=datasets_attrs['era_5'][0]
            )
            # plot the data
            plot_pres_winds(
                [mslp,wind],data_name=datasets_attrs['era_5'][3],
                lat_name=datasets_attrs['era_5'][1],
                lon_name=datasets_attrs['era_5'][0],
                u_name=datasets_attrs['era_5'][4],
                v_name=datasets_attrs['era_5'][5]
            ) if plot else None
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    else:
        mslp = xr.open_dataset(os.path.join(data_path,
                                            'ERA5_MSLP_1H_1979_2021.nc'))['msl']
        # try year cropping
        if time:
            mslp = mslp.sel(time=time) # year cropping
            print(' cropped data to {} \n'.format(int(time)))
        else:
            print('\n LOADING ALL THE MSLP DATA (be careful with memory) \n')
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(os.path.join(data_path,
                                              'ERA5_10mu_1H_1979_2021.nc'))['u10']\
                .sel(time=time)
            vw = xr.open_dataset(os.path.join(data_path,
                                              'ERA5_10mv_1H_1979_2021.nc'))['v10']\
                .sel(time=time)
            wind = calculate_relative_winds(
                location=load_winds[1],uw=uw,vw=vw,
                lat_name=datasets_attrs['era5'][1],
                lon_name=datasets_attrs['era5'][0]
            )
            # plot the data
            plot_pres_winds(
                [mslp,wind],data_name=datasets_attrs['era5'][3],
                lat_name=datasets_attrs['era5'][1],
                lon_name=datasets_attrs['era5'][0],
                u_name=datasets_attrs['era_5'][4],
                v_name=datasets_attrs['era_5'][5]
            ) if plot else None
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')

    # return the loaded datasets
    return_data = [mslp] if not load_winds[0] else [mslp,wind]

    return return_data


def load_cfsr(data_path: str = data_path+'/cfsr/',
              time: str = '1997', # time cropping recommended
              load_winds: tuple = (False,default_location),
              plot: bool = True):
    """
    This function loads the cfsr data and crops it to a time frame
    of a year, or resamples it daily, as it is very difficult to 
    work with all the data at the same time. The winds can be easily
    loaded, and also cropped and projected in the direction of a
    location if requested

    Args:
        data_path (str, optional): Data path folder in repository. 
            - Defaults to data_path.
        time (str, optional): Year to crop the data. It can also be a time
            step to resample the data as 1H, 6H, 1D...
            - Defaults to '1997'.
        load_winds (tuple, optional): this indicates wether the winds are loaded or not, and
            the location of the projected winds.
            - Defaults to (True,default_location)
        plot (bool, optional): Whether to plot or not the results.
            - Defaults to True.

    Returns:
        [list]: This is a list with the data loaded.
    """

    # load / calculate... xarray datasets
    print('\n loading the sea-level-pressure fields... \n')
    if time not in list(
        np.arange(1988,2022,1).astype(str) # years of data + 2
    ):
        # resample to daily
        if time=='1D' and os.path.isfile(os.path.join(data_path,
                                                      'CFSR_MSLP_daily.nc')) \
            and os.path.isfile(os.path.join(data_path,
                                            'CFSR_WINDs_daily.nc')):
            # loading saved datasets
            print('\n loading daily resampled data... \n')
            # loading resampled data
            mslp = xr.open_dataarray(os.path.join(data_path,
                                                  'CFSR_MSLP_daily.nc')).sel(time=slice(datetime(1990,1,1), None))
            mslp = mslp.sortby(datasets_attrs['cfsr'][0],ascending=True)\
                .sortby(datasets_attrs['cfsr'][1],ascending=True)
            if load_winds[0]:
                wind = xr.open_dataset(os.path.join(data_path,
                                                    'CFSR_WINDs_daily.nc')).sel(time=slice(datetime(1990,1,1), None))
                # plot the data
                plot_pres_winds(
                    [mslp,wind],data_name=datasets_attrs['cfsr'][3],
                    lat_name=datasets_attrs['cfsr'][1],
                    lon_name=datasets_attrs['cfsr'][0],
                    u_name=datasets_attrs['cfsr'][4],
                    v_name=datasets_attrs['cfsr'][5],
                    wind_proj='wind_proj_mask'
                ) if plot else None
            # return data
            return_data = [mslp] if not load_winds[0] else [mslp,wind]
            return return_data
        else:
            print('\n resampling data to {}... \n'.format(time))
            mslp = xr.open_dataarray(os.path.join(data_path,
                                                  'CFSR_MSLP_1H_1990_2021.nc'))\
                .sel(time=slice(datetime(1990,1,1), None)).resample(time=time).mean()
            mslp = mslp.sortby(datasets_attrs['cfsr'][0],ascending=True)\
                .sortby(datasets_attrs['cfsr'][1],ascending=True)
        if load_winds[0]:
            print('\n loading and calculating the winds... \n')
            uw = xr.open_dataset(os.path.join(data_path,
                                              'CFSR_uwnd_6H_1990_2021.nc'))[datasets_attrs['cfsr'][4]]\
                .sel(time=slice(datetime(1990,1,1), None)).resample(time=time).mean()
            vw = xr.open_dataset(os.path.join(data_path,
                                              'CFSR_vwnd_6H_1990_2021.nc'))[datasets_attrs['cfsr'][5]]\
                .sel(time=slice(datetime(1990,1,1), None)).resample(time=time).mean()
            wind = calculate_relative_winds(location=load_winds[1],
                                            lat_name=datasets_attrs['cfsr'][1],
                                            lon_name=datasets_attrs['cfsr'][0],
                                            uw=uw,vw=vw)
            # plot the data
            plot_pres_winds(
                [mslp,wind],data_name=datasets_attrs['cfsr'][3],
                lat_name=datasets_attrs['cfsr'][1],
                lon_name=datasets_attrs['cfsr'][0],
                u_name=datasets_attrs['cfsr'][4],
                v_name=datasets_attrs['cfsr'][5],
                wind_proj='wind_proj_mask'
            )
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    else:
        mslp = xr.open_dataarray(os.path.join(data_path,
                                              'CFSR_MSLP_1H_1990_2021.nc')).sel(time=slice(datetime(1990,1,1), None))
        mslp = mslp.sortby(datasets_attrs['cfsr'][0],ascending=True)\
            .sortby(datasets_attrs['cfsr'][1],ascending=True)
        # try year cropping
        if time:
            mslp = mslp.sel(time=time)
            print(' cropping the data to {} \n'.format(int(time)))
        else:
            print('\n LOADING ALL THE MSLP DATA (be careful with RAM memory) \n')
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataarray(os.path.join(data_path,
                                                'CFSR_uwnd_6H_1990_2021.nc'))\
                .sel(time=slice(datetime(1990,1,1), None)).sel(time=time)
            vw = xr.open_dataarray(os.path.join(data_path,
                                                'CFSR_vwnd_6H_1990_2021.nc'))\
                .sel(time=slice(datetime(1990,1,1), None)).sel(time=time)
            wind = calculate_relative_winds(
                location=load_winds[1],uw=uw,vw=vw,
                lat_name=datasets_attrs['cfsr'][1],
                lon_name=datasets_attrs['cfsr'][0]
            )
            # plot the data
            plot_pres_winds(
                [mslp,wind],data_name=datasets_attrs['cfsr'][3],
                lat_name=datasets_attrs['cfsr'][1],
                lon_name=datasets_attrs['cfsr'][0],
                u_name=datasets_attrs['cfsr'][4],
                v_name=datasets_attrs['cfsr'][5],
            )
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
    uhslc_tgs = xr.concat(uhslc_tgs_list,dim='name') # TODO: add combine_attrs='drop'
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        hue_plot = uhslc_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('UHSLC tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(uhslc_tgs.name.values),loc='lower left',ncol=6)
        fig, axes = plt.subplots(
            ncols=2,nrows=6,figsize=(_figsize_width*4,6),
            sharex=True,sharey=True
        )
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


def join_load_linz_tgs(files_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/linz/processed/*.nc',
    plot: bool = False):

    """
    Join all the linz tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # join files assigning a name to each
    print('\n loading and plotting the LINZ tidal guages... \n')
    linz_tgs_list = []
    for file in glob.glob(files_path):
        linz_tg = xr.open_dataset(file)
        linz_tgs_list.append(
            linz_tg.expand_dims(dim='name').assign_coords(
                {'name':(('name'),[file[95:-13]]),
                'latitude':(('name'),[linz_tg.latitude]),
                'longitude':(('name'),[linz_tg.longitude])}
            )
        )

    # join and plot
    linz_tgs = xr.concat(linz_tgs_list,dim='name') # TODO: add combine_attrs='drop'
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        hue_plot = linz_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('LINZ tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(linz_tgs.name.values),loc='lower left',ncol=10)
        fig, axes = plt.subplots(
            ncols=2,nrows=10,figsize=(_figsize_width*4,10),
            sharex=True,sharey=True
        )
        for axi in range(len(linz_tgs.name.values)):
            linz_tgs.isel(name=axi).ss.plot(
                ax=axes.flatten()[axi],c=hue_plot[axi].get_color(),alpha=0.6,
                label=linz_tgs.name.values[axi].upper()
            )
            axes.flatten()[axi].legend(loc='lower left')
            axes.flatten()[axi].set_title('')
            axes.flatten()[axi].set_xlabel('')
            axes.flatten()[axi].set_ylabel('')
        # show results
        plt.show()

    return linz_tgs


def join_load_other_tgs(files_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/other/processed/*.nc',
    plot: bool = False):

    """
    Join all the other tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # join files assigning a name to each
    print('\n loading and plotting the OTHER tidal guages... \n')
    other_tgs_list = []
    for file in glob.glob(files_path):
        other_tg = xr.open_dataset(file)
        other_tgs_list.append(
            other_tg.expand_dims(dim='name').assign_coords(
                {'name':(('name'),[file[96:-13]]),
                'latitude':(('name'),[other_tg.latitude]),
                'longitude':(('name'),[other_tg.longitude])}
            )
        )

    # join and plot
    other_tgs = xr.concat(other_tgs_list,dim='name') # TODO: add combine_attrs='drop'
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        hue_plot = other_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('OTHER tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(other_tgs.name.values),loc='lower left',ncol=4)
        fig, axes = plt.subplots(
            ncols=2,nrows=4,figsize=(_figsize_width*4,4.4),
            sharex=True,sharey=True
        )
        for axi in range(len(other_tgs.name.values)):
            other_tgs.isel(name=axi).ss.plot(
                ax=axes.flatten()[axi],c=hue_plot[axi].get_color(),alpha=0.6,
                label=other_tgs.name.values[axi].upper()
            )
            axes.flatten()[axi].legend(loc='lower left')
            axes.flatten()[axi].set_title('')
            axes.flatten()[axi].set_xlabel('')
            axes.flatten()[axi].set_ylabel('')
        # show results
        plt.show()

    return other_tgs


def load_private_tgs(file_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/geocean/tgs_geocean_NZ.nc',
    plot: bool = False):

    """
    Load some private tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # load and plot all the geocean tgs
    print('\n loading and plotting the private tidal guages... \n')
    private_tgs = xr.open_dataset(file_path)

    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        private_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('Private tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(private_tgs.name.values),loc='lower left',ncol=7)
        private_tgs.ss.plot(col='name',col_wrap=3,figsize=(12,7))
        # show results
        plt.show()

    return private_tgs


def load_moana_hindcast(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/moana_coast.zarr/',
    plot: bool = False):

    """
    Load and plot the moana hindcast (saved as .zarr) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the moana data
    """

    # print('\n loading the Moana v2 hindcast data... \n')

    # load moana
    moana = xr.open_zarr(file_path)

    if plot: # plot if specified
        # calculate 99% quantile
        # threshold = moana.ss.load().max(dim='time').mean()
        # values to plot
        moana_to_plot = moana.ss.load().groupby('time.season').quantile(0.99)
        # plot some stats
        fig, axes = plt.subplots(
            ncols=2,nrows=2,figsize=(_figsize_width*3.8,_figsize_height*2.6),
            subplot_kw={
                'projection':ccrs.PlateCarree(
                    central_longitude=180
                )
            }
        )
        fig.suptitle('Moana shore nodes -- 0.99 quantiles and season',
                     fontsize=_fontsize_title)
        for seas,ax in zip(moana_to_plot.season.values,axes.flatten()):
            p = ax.scatter(
                moana.lon.values,moana.lat.values,
                c=moana_to_plot.sel(season=seas).values,
                transform=ccrs.PlateCarree(),
                s=20,zorder=40,cmap='jet',
                vmin=0.1,vmax=0.5
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
                     plot_labels=(True,10,10))
        # show results
        plt.show()

    return moana


def load_moana_hindcast_ss(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/',
    plot: bool = False, daily: bool = False):

    """
    Load the moana hindcast (ss) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the ss moana data
    """

    print('\n loading the Moana v2 hindcast data (ss)... \n')

    # TODO: add basic plotting

    if os.path.isfile(os.path.join(file_path,
                                   'moana_ss_daily.nc')) and daily:
        return xr.open_dataarray(
            os.path.join(file_path,
                         'moana_ss_daily.nc')
        )
    else:
        return xr.open_zarr(os.path.join(file_path,'ss/'))


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
    data_path+'/storm_surge_data/dac_reanalysis/DAC_SS_6H_1993_2020.nc',
    plot: bool = False):

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

