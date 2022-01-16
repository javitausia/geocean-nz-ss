# -*- coding: utf-8 -*-
# arrays
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs

# maths
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# custom
from .config import default_region, default_region_reduced, default_evaluation_metrics
from .pca import PCA_DynamicPred
from .mda import maxdiss_simplified_no_threshold
from .rbf import rbf_reconstruction, rbf_validation
from .validation import generate_stats, calc_closest_data2_in_data1
from .utils import calculate_relative_winds
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_legend, _fontsize_title, real_obs_col, pred_val_col
from .plotting.data import plot_winds
from .plotting.utils import plot_ccrs_nz, get_n_colors
from .plotting.mda import Plot_MDA_Data
from .plotting.validation import scatterplot, qqplot


class MDA_RBF_Model(object):
    """
    This MDA+RBF model allows the user to perform a model which consists in
    reconstructing / interpolating the storm surge in different locations or
    shores around New Zealand.
    
    Two options are available for the reconstruction of the SS:
    
        - First, the storm surge might be directly reconstructed in some
        desired locations (this is what we will be doing to compare with
        the results obtained in the linear and knn models)

        - Moreover, we add here an extra option which consists in giving
        the MDA+RBF interpolating machine the pcs extracted from the ss in 
        several locations from a desired shore. In this case, we will use
        the slp fields pcs to interpolate the non-linear surface of these
        ss-pcs, and then we will reconstruct the real ss in all the locations

    """

    def __init__(self, ss_data, # this should be the loaded moana with Loader class
        slp_data, # this will be used to calculate the local predictors
        dict_to_pca, # this is the parameters dictionary to calculate pca from the slp
        model_metrics=default_evaluation_metrics, # model metrics to evaluate performance
        ss_data_attrs: tuple = ('lon','lat','site','ss'), # ss attrs
        sites_to_analyze = None, # sites to analyze in case individual analysis is performed
        # lons / lats in case shore analysis is required
        lons=[171.2, 169.4, 167.0,  # west-south coast
              168.1, 167.8,         # south-south
              166.0,                # south island
              170.3, 171.6, 173.5,  # east-south
              173.4, 174.5, 175.1,  # middle bays
              176.4, 177.5, 178.2,  # east-north
              176.6, 175.5, 173.7,  # north-north
              172.8, 173.9, 174.9,  # west-north
              183.5                 # east island
        ][:1],
        lats=[-41.5, -43.3, -44.8, 
              -46.0, -47.1, 
              -50.8,
              -46.0, -44.1, -42.8, 
              -40.9, -39.9, -41.2,
              -41.1, -39.1, -37.4, 
              -37.8, -35.8, -34.5, 
              -35.6, -36.8, -38.3,
              -43.9
        ][:1],
        min_dist_th=[110,100,100,
                     100,60,
                     110,
                     100,100,100,
                     90,90,100,
                     120,120,130,
                     100,120,110,
                     100,80,100,
                     110
        ][:1],
        extra_help=[ # this helps to stay in the wanted shore of NZ
            ('lat',0.9),('lat',0.9),('lat',0.9),
            ('lat',0.7),('lat',0.5),
            ('lon',0.5),
            ('lat',0.8),('lat',0.8),('lat',0.7),
            ('lat',0.3),('lat',0.5),('lon',0.5),
            ('lon',0.3),('lat',0.5),('lon',0.8),
            ('lon',0.7),('lon',0.5),('lat',0.5),
            ('lon',0.8),('lon',0.8),('lat',0.8),
            ('lon',0.5)
        ][:1], # be careful when playing with this tool
        time_resample: str = '1D',
        verbose: bool = True, plot: bool = True):

        """
        As always, we have the initializator/constructor function, where the 
        instance of the class is created, giving some initial parameters

        Args:
            ss_data (xarray.Dataset): This is the previously loaded moana dataset.
                This might be just the shore moana dataset or the one including
                the continental platform until 300m of depth, which is now saved
                in data/storm_surge_data/moana_hindcast_v2/moana_ss_300m_daily.nc
            slp_data (xarray.Dataset): This is the slp data, also loaded with the
                usually used, Loader class in data.py
            dict_to_pca (dict): This is the parameters dictionary with all the
                requested information to calculate the slp principal components      
            model_metrics (list): This is a list with all the model metrics to 
                evaluate model performance          
            ss_data_attrs (tuple): This is a tuple with the main info regarding
                the ss data, for posterior calculations
            sites_to_analyze (list): This is the list with all the individual sites
                where the MDA+RBF technique will be performed
            lons (list): Longitudes of the "middle" location of the shores
            lats (list): Latitudes of the "middle" location of the shores
            min_dist_th (list): Distances in km to find the nearest points to
                the lons/lats specified
            extra_help (list): This is extra information, regarding the "dimension"
                in which the shore is projected, so fake nodes are not introduced
                in the pcs analysis
            time_resample (str, optional): As usually, time resample step

        ***** OPTIONS ARE:
            - Give the desired sites_to_analyze!!
            - Give lon/lat locations and extra_help/min_dist_th for 
                shore analysis!!

        """

        # we first save the verbose and plot attrs to print/plot or not logs
        self.verbose = verbose
        self.plot = plot

        # and save also the resample time and the metrics
        self.time_resample = time_resample
        self.model_metrics = model_metrics

        print('\n Initializing the MDA + RBF constructor... \n') \
            if verbose else None

        if sites_to_analyze: # we will study individual locations

            # lets load / save the ss in specified sites
            self.lons = [lon for lon in ss_data.sel(
                {ss_data_attrs[2]:sites_to_analyze}
            )[[ss_data_attrs[0],ss_data_attrs[1]]]\
                .load()[ss_data_attrs[0]].values]
            self.lats = [lat for lat in ss_data.sel(
                {ss_data_attrs[2]:sites_to_analyze}
            )[[ss_data_attrs[0],ss_data_attrs[1]]]\
                .load()[ss_data_attrs[1]].values]
            print('\n in locations: {}, \n\n with coords: ({},{}) \n'.format(
                sites_to_analyze, self.lons, self.lats
            )) if verbose else None
            self.sites_to_analyze = sites_to_analyze

            # save data and data_attrs in the class attributes
            self.raw_ss_data = ss_data # this could be the moana dataset
            self.ss_attrs = ss_data_attrs # moana / other, vars and coords
            self.ss_real_data = [ss_data.sel({ss_data_attrs[2]:[site]}) \
                .ss.load().resample(time=time_resample).max() \
                .dropna(dim='time',how='all') for site in sites_to_analyze
            ] # to mantain code workflow 2.0
            self.ss_pcs_data = self.ss_real_data.copy() 
            # to mantain coding workflow, there are not PCs!!
            self.ss_scalers = None # no SS-pcs will be calculated in this case

        else: # shore analysis will be performed
             
            self.lons, self.lats = lons, lats # save lon/lat as class attrs
            print('\n with longitudes = {} \n \n and \n \n latitudes = {} !! \
                \n\n for the specified middle locations / shores !!'.format(
                lons, lats # output lon/lat values
            )) if verbose else None

            if len(self.lons)==len(self.lats):
                self.num_locs = len(self.lons) # TODO: add RaiseError

            # save data and data_attrs in the class attributes
            self.raw_ss_data = ss_data # this could be the moana dataset
            self.ss_attrs = ss_data_attrs # moana / other, vars and coords
            print('\n lets calculate the storm surge pcs... \n') \
                if True else None
            ss_tuple_data = self.generate_ss_data(
                min_dist_th, extra_help
            ) # generate the lists with the ss pcs...
            # save all the returned lists
            self.ss_pcs_data, self.ss_real_data, self.ss_scalers, self.shore_sites = ss_tuple_data

        if len(self.lons)==len(self.lats):
            self.num_locs = len(self.lons) # TODO: add RaiseError

        # we now save the slp, slp-pcs...
        print('\n data will be resampled to {} \n\n and the PC analysis \
            will be calculated using the parameters \n\n {} \n'.format(
                time_resample, dict_to_pca
            )
        ) if verbose else None
        self.dict_to_pca = dict_to_pca # save pca attrs
        self.raw_slp_data = slp_data # this could be the CFSR slp/winds
        print('\n lets calculate the slp pcs... \n') \
            if verbose else None
        self.slp_pcs_data = self.generate_slp_data() # this uses self.dict_to_pca


    def generate_ss_data(self, min_dist_th, extra_help):
        """
        This class function generates the ss pcs analysis, so the mda+rbf
        interpolation can be performed over all the different indicated
        shores, ex: west, south...

        Args:
            min_dist_th (list): Distances in km to find the nearest points to
                the lons/lats specified
            extra_help (list): This is extra information, regarding the "dimension"
                in which the shore is projected, so fake nodes are not introduced
                in the pcs analysis
        
        Returns:
            (lists): The function returns the list with the data for all the
            shores involved in the calculation
        """

        # calculate closets points and select ss maximums
        sites, dists = calc_closest_data2_in_data1(
            (self.raw_ss_data[self.ss_attrs[0]].values,
             self.raw_ss_data[self.ss_attrs[1]].values),
            (self.lons,self.lats), # this is the tuple with the lon/lat values
            min_dist_th=min_dist_th, extra_help=extra_help
        )
        sites_in_list = [] # list with all the found sites
        for sites_in_shore in sites:
            sites_in_list += list(sites_in_shore)

        # lets find, not founded sites
        sites_out_list = []  # these are the sites which are not in the shores
        for site in self.raw_ss_data.site.values:
            if site not in sites_in_list:
                sites_out_list.append(site)
        # and calculate closest shores to new found nodes
        closest_shores = calc_closest_data2_in_data1(
            (self.lons,self.lats),
            (self.raw_ss_data.sel(site=sites_out_list)[self.ss_attrs[0]].values,
             self.raw_ss_data.sel(site=sites_out_list)[self.ss_attrs[1]].values),
            min_dist_th=200, extra_help=('lon',3) # dont avoid any node
        )[0]
        # for site_out, shore in zip(sites_out_list, closest_shores):
        #     # INFO: we avoid the shore with difficult structures
        #     # for sh in [0,1,2,3,6,7,8,12,14,16,17]:
        #     #     if sh in [0,1,2,3,6,7,8,12] and sh==shore[0]:
        #     #         sites[sh].append(site_out)
        #     #         pass
        #     #     if sh in [14,16,17,19] and sh in shore:
        #     #         sites[sh].append(site_out)
        #     #         pass
        #     if shore[0]==11:
        #         sites[12].append(site_out)
        #     else:
        #         sites[shore[0]].append(site_out)
        # # check all lists are with unique elements
        # sites = [
        #     list(np.unique(sites_in_shore)) for sites_in_shore in sites
        # ]

        # save as list with all the data for the pcs
        raw_ss_locs = [self.raw_ss_data.sel(
                {self.ss_attrs[2]:sites[i]}
            ).ss.load().resample(time=self.time_resample).max() \
            .dropna(dim='time',how='all') for i in range(self.num_locs)
        ] # num_locs==len(sites) if working

        # calculate the pcs of all the shores
        ss_pcs_locs, ss_scalers_locs = [], [] # lists to save the data
        for loc in range(self.num_locs):
            # standarize the features
            ss_scaler = StandardScaler()
            ss_stan = ss_scaler.fit_transform(raw_ss_locs[loc].values.T) # TODO: check .T!!
            ss_stan[np.isnan(ss_stan)] = 0.0 # check additional nans
            ss_scalers_locs.append(ss_scaler)
            # calculate de PCAs
            pca_fit = PCA(n_components=min(ss_stan.shape[0],ss_stan.shape[1]))
            PCs = pca_fit.fit_transform(ss_stan)
            # return data
            ss_pcs_locs.append(
                xr.Dataset(
                    data_vars = {
                        'PCs': (('time','n_components'), PCs),
                        'EOFs': (('n_components','n_features'), pca_fit.components_),
                        'variance': (('n_components'), pca_fit.explained_variance_),
                        'total_variance': ((), np.sum(pca_fit.explained_variance_)),
                        'site': (('site'), raw_ss_locs[loc][self.ss_attrs[2]].values)
                    },
                    coords = {
                        'time': raw_ss_locs[loc].time.values
                    }
                )
            )
            # TODO: plot pcs analysis briefly
            # raw_ss_locs[loc].plot(hue=self.ss_attrs[2],figsize=(_figsize))
            # plt.plot(raw_ss_locs[loc].time.values,PCs[:,0])
            # plt.plot(raw_ss_locs[loc].time.values,PCs[:,1])
        
        if self.plot:
            print('\n All the SS pcs have been calculated, lets plot what we got!! \n') \
                if self.verbose else None
            # plot selected points to check coherence
            fig, axes = plt.subplots(
                ncols=2,figsize=(20,10), #_figsize,
                subplot_kw={
                    'projection':ccrs.PlateCarree(central_longitude=180)
                }
            )
            cmap, loc_colors = get_n_colors('gist_rainbow',self.num_locs)
            loc_colors = [
                'navy','blue','royalblue','darkorange','orange','gold','indianred','red','darkred',
                'purple','blueviolet','mediumslateblue','pink','palevioletred','mediumorchid',
                'yellowgreen','limegreen','darkgreen','darkolivegreen','greenyellow','lightgreen','yellow'
            ] # set shore colors manually
            for axi,ax in enumerate(axes):
                if axi==0:
                    xr.plot.scatter(
                        self.raw_ss_data.max(dim='time'),
                        x=self.ss_attrs[0],y=self.ss_attrs[1],
                        hue=self.ss_attrs[3],ax=ax,
                        transform=ccrs.PlateCarree()
                    ) # plot the maximum ss in data
                    ax.set_title(
                        'Maximum SS for each station',
                        fontsize=_fontsize_title
                    )
                else:
                    for ishore in range(self.num_locs):
                        ax.scatter( # total selected points for each shore
                            x=self.raw_ss_data.sel(
                                {self.ss_attrs[2]:ss_pcs_locs[ishore][self.ss_attrs[2]].values}
                            )[self.ss_attrs[0]].values,
                            y=self.raw_ss_data.sel(
                                {self.ss_attrs[2]:ss_pcs_locs[ishore][self.ss_attrs[2]].values}
                            )[self.ss_attrs[1]].values,
                            transform=ccrs.PlateCarree(),s=20,
                            #c=np.array(loc_colors[ishore],ndmin=2)
                            c=loc_colors[ishore]
                        )
                        # ax.scatter( # "middle" point as a star
                        #     x=self.lons[ishore],y=self.lats[ishore],s=100,zorder=112,
                        #     marker='*',c='k',transform=ccrs.PlateCarree(),edgecolors='yellow'
                        # )
                        if ishore==0:
                            ax.set_title(
                                'Input locations and closest stations',
                                fontsize=_fontsize_title
                            )
                ax.set_facecolor('white')
            plot_ccrs_nz(axes,plot_labels=(False,5,5))
            plt.show() # debug results as calculated
        else:
            print('\n All the SS pcs have been calculated!! \n') \
                if self.verbose else None

        return ss_pcs_locs, raw_ss_locs, ss_scalers_locs, sites


    def generate_slp_data(self, pres_vars: tuple = ('SLP','longitude','latitude'),
        calculate_gradient: bool = False, wind = None,
        wind_vars: tuple = ('wind_proj_mask','lon','lat'),
        time_lapse: int = 1, # 1 equals to NO time delay 
        time_resample: str = '1D', region: tuple = (True,default_region),
        pca_plot: tuple = (True,False,1), verbose: bool = True,
        pca_ttls = None, pca_borders = None):

        """
        This class function generates the slp pcs analysis, so the mda+rbf
        interpolation can be performed over all the different indicated
        shores, ex: west, south...

        Args:
            The fucntion includes all the arguments used in PCA_DynamicPred,
            but the way this function will be used is calling the self.dict_to_pca
            attribute loaded in the __init__.py

        Returns:
            (list): This is a python list with all the pcs
        """

        # all the attrs are saved in self.dict_to_pca

        # lets save all the pcs (for each shore) in a list
        pcs_for_each_shore = []

        print(' \n\n \
            The slp pcs will be calculated for all the locations / shores... \
            \n'
        ) if self.verbose else None

        if self.dict_to_pca['region'][0]!='local':
            return [PCA_DynamicPred(
                        self.raw_slp_data, # this is always the same
                        **self.dict_to_pca # extra arguments without the winds
                    ).pcs_get()[0]
                ] * self.num_locs # same PCs in all sites

        for ipc in range(self.num_locs):

            print('pcs matrix calculation for site / shore {}'.format(
                ipc+1 # number of shore / site
            ), end='\r') if self.verbose else None

            # save site_location attribute
            site_location = (self.lons[ipc],self.lats[ipc])

            # save the dict in a local copy
            dict_to_pca = self.dict_to_pca.copy()

            if self.dict_to_pca['region'][0]=='local':
                # save region to calculate new grid
                region_coords = dict_to_pca.pop('region')[1]
                local_region = (True,(
                    self.lons[ipc]-region_coords[0], # new lon / lat region
                    self.lons[ipc]+region_coords[0],
                    self.lats[ipc]-region_coords[1],
                    self.lats[ipc]+region_coords[1],
                ))
                pca_data, pca_scaler = PCA_DynamicPred(
                    self.raw_slp_data, # this is always the same
                    region=local_region, # pass the calculated local region
                    site_location=site_location, # pass the site location
                    **dict_to_pca # extra arguments without the winds
                ).pcs_get()
            else:
                pca_data, pca_scaler = PCA_DynamicPred(
                    self.raw_slp_data, # this is always the same
                    site_location=site_location, # pass the site location
                    **dict_to_pca # extra arguments without the winds
                ).pcs_get()

            # append calculated pcs to list
            pcs_for_each_shore.append(pca_data)

        return pcs_for_each_shore


    def calc_MDA_RBF(self, selected_shores=None,
                     percentage_pcs_ini=[0.99],
                     num_samples_ini=[750], ss_pcs=1,
                     try_all=False, append_extremes=None,
                     plot: bool = False, verbose: bool = True):

        """
        This is the calculation function, where given the wanted
        locations / shores, the MDA+RBF interpolation model is performed,
        plotting different results and statistics

        Args:
            selected_shores (list): List with ilocs for the shores, but could
                be range(n) being n the number of individual locations
            percentage_pcs_ini (list): List with all the percentage of pcs
                that will be used to interpolate
            num_samples_ini (list): List with all the number of samples that
                will be used to create the subset
            ss_pcs (int): This indicates the number of SS pcs that will be used
                when interpolating, set this parameter to 1 if we will directly
                interpolate the SS in individual locations
            try_all (bool): Whether to try all the different combinations or not
            append_extremes (int,None): This indicates the number of extremes that
                will be used to create the subset, but can be set to None
            plot / verbose (bools): Whether to plot / log or not results
        """

        print('\n Lets interpolate the SS using all the calculated \
            pcs (slp + ss) and the MDA+RBF technique!! \
            This interpolation will be performed in all the locations!! \n'
        )

        final_datasets = []

        for sel_loc in selected_shores: # could be individual locations

            print('\n Lets reconstruct the SS for location {}!! \n'.format(
                sel_loc+1 # this is the shore            
            ))

            # perform the RBF interpolation, using MDA too
            self.real_ss_rbf = MDA_RBF_algorithm(
                self.slp_pcs_data[sel_loc], 
                self.ss_pcs_data[sel_loc], ss_pcs=ss_pcs, 
                ss_scaler=self.ss_scalers[sel_loc] if \
                    isinstance(self.ss_scalers,list) else None,
                try_all=try_all, append_extremes=append_extremes,
                percentage_pcs_ini=percentage_pcs_ini,
                num_samples_ini=num_samples_ini,
                plot=plot, verbose=verbose, 
                sel_loc=sel_loc if ss_pcs!=1 else \
                    self.sites_to_analyze[sel_loc] # which is just 0, 1... in the first case
            )

            print('\n Lets plot the SS reconstructions for location {}!! \n'.format(
                sel_loc+1 # this is the shore            
            )) if plot else None

            # iterate over all the shores / just one location
            for isite, site in enumerate(
                self.ss_real_data[sel_loc].site.values if ss_pcs==1 else
                    self.ss_real_data[sel_loc].site.values[::1] # TODO: Be careful with number of stations!!!
            ):  

                experiment_datasets = []

                # iterate over all the available experiments
                for exp in range(len(self.real_ss_rbf.experiment)):

                    # loop copy to avoid problems
                    experiment_dataset = self.real_ss_rbf\
                        .isel(site=isite,experiment=exp).copy()

                    # try for available nodes or continue
                    try:
                        title, stats = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site).values,
                            experiment_dataset.ss_interp.values.reshape(-1), # mda+rbf ss values
                            metrics=self.model_metrics,
                            ext_quantile=([0.9,0.95,0.99,0.999],0)
                        )
                    except:
                        title, stats = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site).values,
                            experiment_dataset.ss_interp.values.reshape(-1),
                            not_nan_idxs=np.where(
                                (~np.isnan(self.ss_real_data[sel_loc].sel(site=site).values) &
                                 ~np.isnan(experiment_dataset.ss_interp.values.reshape(-1)))
                            )[0],
                            metrics=self.model_metrics,
                            ext_quantile=([0.9,0.95,0.99,0.999],0)
                        )
                        continue

                    # save results in list
                    for metric in list(stats.keys()):
                        experiment_dataset[metric] = experiment_dataset.perpcs*0 + \
                            stats[metric]
                    experiment_datasets.append(
                        experiment_dataset # TODO: drop ss, .drop_vars('ss_interp')
                    )

                    if plot:
                        # figure spec-grid
                        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
                        gs = gridspec.GridSpec(nrows=1,ncols=3)
                        # time regular plot
                        ax_time = fig.add_subplot(gs[:,:2])
                        self.ss_real_data[sel_loc].sel(site=site).plot(
                            ax=ax_time,c='navy',label='Real SS measures',alpha=1.0,lw=2
                        )
                        ax_time.plot(
                            self.ss_real_data[sel_loc].sel(site=site).time.values,
                            self.real_ss_rbf.isel(site=isite,experiment=0)\
                                .ss_interp.values.reshape(-1),
                            c='royalblue',alpha=1.0,lw=2,label='Reconstructed SS -- MDA + RBF'
                        )
                        ax_time.set_xlim(
                            datetime(2010,1,1), # self.ss_real_data[sel_loc].sel(site=site).time.values[0],
                            datetime(2011,1,1)  # self.ss_real_data[sel_loc].sel(site=site).time.values[-1]
                        ) # delete white spaces
                        ax_time.legend(ncol=2,fontsize=20) # _fontsize_legend
                        # validation plot
                        ax_vali = fig.add_subplot(gs[:,2:])
                        ax_vali.set_xlabel('Observation')
                        ax_vali.set_ylabel('Prediction')
                        scatterplot(
                            self.ss_real_data[sel_loc].sel(site=site),
                            self.real_ss_rbf.isel(site=isite,experiment=0)\
                                .ss_interp.values.reshape(-1),
                            ax=ax_vali
                        )
                        qqplot(
                            self.ss_real_data[sel_loc].sel(site=site),
                            self.real_ss_rbf.isel(site=isite,experiment=0)\
                                .ss_interp.values.reshape(-1),
                            ax=ax_vali
                        )
                        # add title
                        fig.suptitle(
                            title+' -- SITE: {}'.format(site),fontsize=_fontsize_title,y=1.1
                        )
                        # show the results
                        plt.show()

                final_datasets.append(
                    xr.concat(experiment_datasets,dim='experiment')
                )

        return xr.concat(final_datasets,dim='site') if ss_pcs==1 \
            else xr.concat(final_datasets,dim='shore')


def MDA_RBF_algorithm(
    pcs_data, ss_data, ss_pcs: int = 2,
    ss_scaler = None, # to de-standarize the pcs
    percentage_pcs_ini: list = [0.6,0.9],
    num_samples_ini: list = [100,500],
    try_all: bool = False, append_extremes: int = 10,
    validate_rbf_kfold: bool = False,
    plot: bool = False, verbose: bool = True,
    sel_loc = None):

    """
    This out-of-class function performs the MDA+RBF interpolation given the
    pcs of the sea-level-pressure fields and some variable to interpolate, that
    might be the storm surge, but can also be the pcs of this spatial ss

    Args:
        pcs_data (xarray.Dataset): These are the pcs of the slp fields, that can
            be previously calculate using generate_slp_data
        ss_data (xarray.Dataset): This is the variable/s to interpolate, and might 
            be the storm surge or the pcs of the ss, in case the shore related
            analysis is done over NZ
        ss_pcs (int, optional): This is the number of pcs that is wanted to be
            used when the ss-pcs are calculated. Defaults to 1, which means
            no pcs are calculated.
        ss_scaler (sklearn.Scaler, optional): This is the scikit-learn scaler
            that will be used to recalculate the real ss. Defaults to None.
        percentage_pcs_ini (list, optional): List with all the percentages that
            will be used to crop the slp pcs. Defaults to [0.6,0.9].
        num_samples_ini (list, optional): List with all the integer numbers that
            represent the number of points the MDA algorithm will search for, and
            then the RBF interpolation will use. Defaults to [100,500].
        try_all (bool, optional): Whether to try or not all the possible combinations
            of the 2 previous defined parameters. Defaults to False.
        append_extremes (int, optional): Wheter to add or not the ss/pcs extreme values
            to the subset to interpolate, not extrapolate. Defaults to 10.
        validate_rbf_kfold (bool, optional): Wheter to call the rbf_validation function
            once the process is done. Defaults to False.
        plot (bool, optional): Whether to plot or not the results. Defaults to True.
        verbose (bool, optional): Wheter to plot or not logs. Defaults to True.
        sel_loc (int, None): This is the for loop iteration in calc_MDA_RBF

    Returns:
        [xarray.Datasets]: xarray datasets with the data reconstructed
    """

    outs, real_sss = [], []

    # try or not all different possibilities
    if try_all:
        percentage_pcs = percentage_pcs_ini * len(num_samples_ini)
        num_samples = []
        for num_sample in num_samples_ini:
            for _ in range(len(percentage_pcs_ini)):
                num_samples.append(num_sample)
    else:
        percentage_pcs = percentage_pcs_ini
        num_samples = num_samples_ini

    # perform the mda + rbf analysis over all the possibilities
    for i_exp,per_pcs,n_samp in zip(
        range(len(num_samples)),percentage_pcs,num_samples
    ):

        print('\n ------------------------------------------------------------- \n') \
            if verbose else None
        print('\n MDA + RBF with {} per of pcs and {} samples!! \n'.format(
            per_pcs, n_samp
        )) if verbose else None
        # number of pcs to use
        num_pcs = len(np.where(
            ((np.cumsum(pcs_data.variance)/float(pcs_data.total_variance)) < per_pcs).values==True
        )[0])
        print(' which means {} PCs... \n'.format(num_pcs)) if verbose else None
        print('\n ------------------------------------------------------------- \n') \
            if verbose else None

        predictor_dataset = pcs_data.PCs.isel(
            n_components=slice(0,num_pcs)
        ).dropna(dim='time',how='all') # mantain just num_pcs

        # get common times
        common_times = np.intersect1d(
            predictor_dataset.time.values,
            ss_data.time.values
        )

        # get predictor and target with common times
        # prepare datasets
        predictor_dataset = predictor_dataset.sel(time=common_times)
        target_dataset = ss_data.sel(time=common_times) if ss_pcs==1 else \
            ss_data.PCs.sel(time=common_times).isel(n_components=slice(0,ss_pcs))
        # use MDA to generate a demo dataset and subset for RBF interpolation
        pcs_to_mda = 4
        ix_scalar_pred_mda = list(np.arange(pcs_to_mda+ss_pcs))
        ix_directional_pred = []
        # perform the mda analysis (this is the mda input data)
        mda_dataset = np.concatenate(
            [predictor_dataset[:,:pcs_to_mda].values,
            target_dataset.values.reshape(-1,ss_pcs)
            ],axis=1
        )
        # MDA algorithm
        predictor_subset_red, subset_indexes = maxdiss_simplified_no_threshold(
            mda_dataset, n_samp, ix_scalar_pred_mda, ix_directional_pred, log=False
        )

        if plot:
            fig = Plot_MDA_Data(
                pd.DataFrame(
                    mda_dataset,columns=['PC'+str(i+1) for i in range(pcs_to_mda)] + \
                        ['SS'+str(i+1) for i in range(ss_pcs)]
                ),
                pd.DataFrame(
                    predictor_subset_red,columns=['PC'+str(i+1) for i in range(pcs_to_mda)] + \
                        ['SS'+str(i+1) for i in range(ss_pcs)]
                )
            )
            plt.show()

        # append max ss to subset_indexes
        if append_extremes:
            # get extremes location
            max_times_indexes = np.argsort(
                -target_dataset.values.reshape(-1)
            )[:append_extremes] if ss_pcs==1 else \
                np.argsort(-target_dataset.values[:,0])[:append_extremes]
            for max_indx in max_times_indexes:
                subset_indexes.append(max_indx) if max_indx not in subset_indexes else None
            print('We finally have {} points to interpolate with RBF!! \n\n'.format(
                len(subset_indexes)
            )) if verbose else None # appended extreme values to subset

        # get subsets with calculated indexes (pcs + ss)
        predictor_subset = predictor_dataset.sel(
            time=predictor_dataset.time.values[subset_indexes]
        )
        target_subset = target_dataset.sel(
            time=predictor_dataset.time.values[subset_indexes]
        )
        # crop ss PCs if the case
        idxs_not_in_subset = []
        for i in range(len(target_dataset.time.values)):
            idxs_not_in_subset.append(i) if i not in subset_indexes else None

        # plot ss subset and predictor
        if plot:
            fig, ax = plt.subplots(figsize=_figsize)
            if ss_pcs==1:
                target_dataset.plot(ax=ax,color='k',label='SS')
            else:
                for ss_pc in range(ss_pcs):
                    colors = ['k','b','g','c','m','y','k'][:ss_pcs]
                    target_dataset.sel(n_components=ss_pc).plot(
                        ax=ax,color=colors[ss_pc],label='SS'+str(ss_pc+1),alpha=1.0
                    ) # plot some ss pcs
            ax.plot(
                target_dataset.time.values[subset_indexes],
                target_subset.values.reshape(-1,ss_pcs)[:,0],'.',
                markersize=10,c='red'
            )
            ax.set(
                title='MDA cases to perform the RBF algorithm',
                xlabel='time',ylabel='Storm Surge'
            )
            ax.set_xlim(
                target_dataset.time.values[0],target_dataset.time.values[-1]
            )

        # RBF statistical interpolation allows us to solve calculations to big datasets 
        # that otherwise would be highly costly (time and/or computational resources)

        # mount RBF pred and target
        ix_scalar_pred_rbf = list(np.arange(num_pcs))
        ix_scalar_t = list(np.arange(ss_pcs))
        ix_directional_t = []

        # RBF reconstrution
        try:
            out = rbf_reconstruction(
                predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
                target_subset.values.reshape(-1,ss_pcs), ix_scalar_t, ix_directional_t,
                predictor_dataset.values
            )
            outs.append(xr.Dataset(
                {'ss_interp': (('time','experiment'),out), # n experiment in site
                 'perpcs': (('experiment'), [per_pcs]), 'nsamples': (('experiment'), [n_samp])
                }, coords={
                    'time': predictor_dataset.time.values, 'experiment': [i_exp]
                }).expand_dims({'site': [sel_loc]})
            ) if ss_pcs==1 else None
        except:
            print('\n the RBF reconstruction did not converge... \n')
            continue

        # TODO: add training metrics

        # RBF Validation: using k-fold mean squared error methodology
        if validate_rbf_kfold:
            test = rbf_validation(
                predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
                target_subset.values.reshape(-1,ss_pcs), ix_scalar_t, ix_directional_t,
                n_splits=3, shuffle=True
            )

        # plot output results
        if plot:
            for i_ss in range(1):
                # figure spec-grid
                fig = plt.figure(figsize=(_figsize_width*4.5,_figsize_height))
                gs = gridspec.GridSpec(nrows=1,ncols=3)
                # time regular plot
                ax_time = fig.add_subplot(gs[:,:2])
                target_dataset.isel(time=idxs_not_in_subset).plot(
                    ax=ax_time,c=real_obs_col,alpha=0.8,label='Real SS observations'
                ) if ss_pcs==1 else target_dataset.isel(
                    time=idxs_not_in_subset,n_components=i_ss
                ).plot(
                    ax=ax_time,alpha=0.8,c=real_obs_col,label='Real SS observations'
                )
                ax_time.plot(
                    common_times[idxs_not_in_subset],
                    out[idxs_not_in_subset,i_ss],
                    alpha=0.7,c='orange',linestyle='--',
                    label='RBF predictions'
                )
                ax_time.legend(ncol=2,fontsize=_fontsize_legend)
                ax_time.set_xlim(
                    common_times[idxs_not_in_subset][0],
                    common_times[idxs_not_in_subset][-1]
                )
                # validation plot
                ax_vali = fig.add_subplot(gs[:,2:])
                ax_vali.set_xlabel('Observation')
                ax_vali.set_ylabel('Prediction')
                scatterplot(
                    target_dataset.values.reshape(-1,ss_pcs)[idxs_not_in_subset,i_ss],
                    out[idxs_not_in_subset,i_ss],ax=ax_vali
                )
                qqplot(
                    target_dataset.values.reshape(-1,ss_pcs)[idxs_not_in_subset,i_ss],
                    out[idxs_not_in_subset,i_ss],ax=ax_vali
                )
                # add title
                title, ttl_stats = generate_stats(
                    target_dataset.values.reshape(-1,ss_pcs)[idxs_not_in_subset,i_ss],
                    out[idxs_not_in_subset,i_ss],
                )
                fig.suptitle(
                    'SS interpolation \n '+title,fontsize=_fontsize_title,y=1.1
                ) if ss_pcs==1 else fig.suptitle(
                    'PC1 interpolation',fontsize=_fontsize_title,y=1.1
                )
        # show the results
        plt.show()

        if ss_scaler:
            # reconstruct the ss values
            ss_stan = np.repeat(
                out[:,0],len(ss_data.n_features)
            ).reshape(
                len(ss_data.time),len(ss_data.n_features)
            ) * ss_data.EOFs.isel(n_components=0).values
            for i_comp in range(1,ss_pcs):
                ss_stan += np.repeat(
                    ss_data.PCs.isel(n_components=i_comp).values,
                    len(ss_data.n_features)).reshape(
                        len(ss_data.time),len(ss_data.n_features)
                    ) * ss_data.EOFs.isel(n_components=i_comp).values
            # get the real ss values
            real_ss = ss_scaler.inverse_transform(ss_stan)
            # and add the ss to final dataset
            real_sss.append(xr.Dataset( # save real storm surge
                {'ss_interp': (('time','site','experiment'),real_ss.reshape(
                    -1,real_ss.shape[1],1
                )), 'perpcs': (('experiment'), [per_pcs]), 
                    'nsamples': (('experiment'), [n_samp])
            }, coords={'time': predictor_dataset.time.values,
                'site': ss_data.site.values, 'experiment': [i_exp]
            }).expand_dims(
                {'shore':[sel_loc]}
            )) if ss_pcs!=1 else None

    # return real ss values
    real_ss_to_return = xr.concat(outs,dim='experiment') if ss_pcs==1 else \
        xr.concat(real_sss,dim='experiment') 

    return real_ss_to_return

