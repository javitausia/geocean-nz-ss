# basic
import os, glob, sys

# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt

# custom
from .config import data_path, default_location # get config params
from .utils import calculate_relative_winds
from .validation import compare_datasets
from .plotting.data import plot_era5
from .plotting.config import _figsize, _fontsize_title, _figsize_width

# warnings
import warnings
warnings.filterwarnings('ignore')


# loader dicts summary
loader_dict_options = {
    'predictor': ['cfsr','era5'],
    'predictand': ['dac','moana','codec'],
    'validator': ['uhslc','geotgs']
}
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
    methods in the class can be easily used, especifying just the list with all
    the datasets in the correct order

    Args:
        object ([type]): [description]
    """


    def __init__(self, data_to_load: list = ['era5','moana','uhslc'],
                 location: tuple = default_location):
        """
        Loader class constructor

        Args:
            data_to_load (list, optional): [description]. Defaults to ['era5','moana','uhslc'].
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
            else:
                print('\n data not available for the predictor!! \n')
        # load the predictand
        if data_to_load[1] in loader_dict_options['predictand']:
            if data_to_load[1]=='dac':
                self.predictand = load_dac()
                self.predictand_attrs = datasets_attrs[data_to_load[1]]

            if data_to_load[1]=='moana':
                self.predictand = load_moana_hindcast()
                self.predictand_attrs = datasets_attrs[data_to_load[1]]
            elif data_to_load[1]=='codec':
                self.predictand = load_codec_hindcast()
                self.predictand_attrs = datasets_attrs[data_to_load[1]]
            else:
                print('\n data not available for the predictand!! \n')
        # load the validator
        if data_to_load[2] in loader_dict_options['validator']:
            if data_to_load[2]=='uhslc':
                self.validator = join_load_uhslc_tgs(plot=True)
                self.validator_attrs = datasets_attrs[data_to_load[2]]
            elif data_to_load[2]=='geotgs':
                self.validator = load_geocean_tgs()
                self.validator_attrs = datasets_attrs[data_to_load[2]]
            else:
                print('\n data not available for the validation!! \n')
                
                
    def validate_datasets(self, 
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
              load_winds: bool = (True,default_location)):
    """
    This function loas era5 data and crops it to a time frame
    of a year, or resamples it daily, as it is very difficult to 
    work will all the data at the same time. The winds can be easily
    loaded, and also cropped and projected in the direction of a
    location if requested

    Args:
        data_path (str, optional): Data path folder in repository. 
            - Defaults to data_path.
        time (str, optional): Year to crop the data. Defaults to '1997'.
        load_winds: this indicates wheter the winds are loaded or not

    Returns:
        [list]: This is a list with the data loaded
    """

    # load/calculate... xarray datasets
    print('\n loading the sea-level-pressure fields... \n')
    if time=='1D':
        # resample to daily
        if os.path.isfile(data_path+'/era_5/ERA5_MSLP_daily.nc'):
            print('\n loading daily resampled data... \n')
            # loading resampled data
            mslp = xr.open_dataarray(data_path+'/era_5/ERA5_MSLP_daily.nc')
            if load_winds[0]:
                wind = xr.open_dataset(data_path+'/era_5/ERA5_WINDs_daily.nc')
                # plot the data
                plot_era5([mslp,wind])
            # return data
            return_data = [mslp] if not load_winds[0] else [mslp,wind]
            return return_data

        else:
            print('\n resampling data to daily... \n')
            mslp = xr.open_dataset(data_path+'/era_5/ERA5_MSLP_1H_1979_2021.nc')['msl']\
                .resample(time='1D').mean()
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/era_5/ERA5_10mu_1H_1979_2021.nc')['u10']\
                .resample(time='1D').mean()
            vw = xr.open_dataset(data_path+'/era_5/ERA5_10mv_1H_1979_2021.nc')['v10']\
                .resample(time='1D').mean()
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw)
            # plot the data
            plot_era5([mslp,wind])
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    else:
        mslp = xr.open_dataset(data_path+'/era_5/ERA5_MSLP_1H_1979_2021.nc')['msl']
        # try year cropping
        mslp = mslp.sel(time=time)
        print(' cropping the data to {} \n'.format(int(time)))
        if load_winds[0]:
            print('\n loading the winds... \n')
            uw = xr.open_dataset(data_path+'/era_5/ERA5_10mu_1H_1979_2021.nc')['u10']\
                .sel(time=time)
            vw = xr.open_dataset(data_path+'/era_5/ERA5_10mv_1H_1979_2021.nc')['v10']\
                .sel(time=time)
            wind = calculate_relative_winds(location=load_winds[1],
                                            uw=uw,vw=vw)
            # plot the data
            plot_era5([mslp,wind])
        else:
            print('\n projected winds will not be calculated... returning the SLP... \n')
    # return the loaded datasets
    return_data = [mslp] if not load_winds[0] else [mslp,wind]

    return return_data


def load_cfsr(data_path: str = 
              data_path+'/cfsr/CFSR_MSLP_1H_1979_2020_NZ.nc'):
    """
    This function loads the CFSR sea-level-pressure

    Returns:
        [xarray.Dataset]: xarray dataset with the slp
    """

    return xr.open_dataarray(data_path)


def join_load_uhslc_tgs(files_path: str = 
    data_path+'/storm_surge_data/nz_tidal_gauges/uhslc/processed/*.nc',
    plot: bool = False):

    """
    Join all the uhslc tgs in a single xarrray dataset to play with it

    Returns:
        [xarray.Dataset]: xarray dataset with all the tgs and variables
    """

    # join files assigning a name to each
    print('\n Loading and plotting the UHSLC tidal guages... \n')
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
        fig, axes = plt.subplots(ncols=2,nrows=6,figsize=(_figsize_width,6),
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
    print('\n Loading and plotting the GeoOcean tidal guages... \n')
    geocean_tgs = xr.open_dataset(file_path)
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        geocean_tgs.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('GeoOcean tidal gauges',fontsize=_fontsize_title)
        ax.legend(list(geocean_tgs.name.values),loc='lower left',ncol=7)
        geocean_tgs.ss.plot(col='name',col_wrap=3,figsize=(12,7))

    return geocean_tgs


def load_moana_hindcast(file_path: str = 
    data_path+'/storm_surge_data/moana_hindcast_v2/moana_coast.zarr/',
    plot: bool = False):

    """
    Load the moana hindcast (saved as .zarr) in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the moana data
    """

    print('\n Loading the Moana v2 hindcast data... \n')

    return xr.open_zarr(file_path)


def load_dac(file_path: str = 
    data_path+'/storm_surge_data/dac_reanalysis/DAC_SS_6H_1993_2020.nc'):

    """
    Load the DAC hindcast in a single xarray dataset

    Returns:
        [xarray.Dataset]: xarray dataset with all the DAC data
    """

    print('\n Loading the DAC hindcast data... \n')

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
    print('\n Loading and plotting the CoDEC numerical data... \n')
    codec_hind = xr.open_dataset(file_path).sel(time=slice('1980','2021'))
    if plot: # plot if specified
        fig, ax = plt.subplots(figsize=_figsize)
        codec_hind.ss.plot(hue='name',alpha=0.6,ax=ax) # plot the ss
        fig.suptitle('CoDEC numerical model',fontsize=_fontsize_title)
        ax.legend(list(codec_hind.name.values),loc='lower left',ncol=7)
        codec_hind.ss.plot(col='name',col_wrap=3,figsize=(12,7))

    return codec_hind

