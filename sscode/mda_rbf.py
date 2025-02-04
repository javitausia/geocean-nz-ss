# arrays and time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs

# maths
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# custom (info, math and plotting)
from .config import default_region, default_region_reduced, \
    default_evaluation_metrics, default_ext_quantile
from .pca import PCA_DynamicPred
from .mda import maxdiss_simplified_no_threshold
from .rbf import rbf_reconstruction, rbf_validation
from .validation import generate_stats, calc_closest_data2_in_data1
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_legend, _fontsize_title, real_obs_col, pred_val_col
from .plotting.utils import plot_ccrs_nz, get_n_colors
from .plotting.mda import Plot_MDA_Data
from .plotting.validation import scatterplot, qqplot


class MDA_RBF_Model(object):
    """
    This MDA+RBF class allows the user to train a model which consists in
    reconstructing / interpolating the storm surge in different locations or
    shores around New Zealand.
    
    Two options are available for the reconstruction of the SS:
    
        - First, the storm surge might be directly reconstructed in some
        desired locations (this is what we will be doing to compare with
        the results obtained in the linear, knn and xgboost models)

        - Moreover, we add here an extra option which consists in giving
        the MDA+RBF interpolating machine the pcs extracted from the ss in 
        several locations from a desired shore. In this case, we will use
        the slp fields pcs to interpolate the non-linear surface of these
        ss-pcs, and then we will reconstruct the real ss in all the locations
    """

    def __init__(self, 
        ss_data, # this should be the loaded moana data with Loader class
        slp_data, # this will be used to calculate the predictors (Loader.predictor)
        dict_to_pca, # this is the parameters dictionary to calculate pca from the slp
        model_metrics: list = default_evaluation_metrics, # model metrics to validate performance
        ext_quantile: tuple = default_ext_quantile, # quantiles to be used in validation
        ss_data_attrs: tuple = ('lon','lat','site','ss','300m'), # ss attrs to use
        ss_pred_times: int = 0, # this is the number of future times that will be predicted
        ss_per_pcs: float = 0.95, # this is the percentage of PCs that will be saved/used
        sites_to_analyze = None, # sites to analyze in case individual analysis is performed
        ss_tuple_data: tuple = None, # give the class all the saved SS-PCs attrs
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
        ],
        lats=[-41.5, -43.3, -44.8, 
              -46.0, -47.1, 
              -50.8,
              -46.0, -44.1, -42.8, 
              -40.9, -39.9, -41.2,
              -41.1, -39.1, -37.4, 
              -37.8, -35.8, -34.5, 
              -35.6, -36.8, -38.3,
              -43.9
        ],
        min_dist_th=[110,100,100,
                     100,60,
                     110,
                     100,100,100,
                     90,90,100,
                     120,120,130,
                     100,120,110,
                     100,80,100,
                     110
        ],
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
        ], # be careful when playing with this tool
        time_resample: str = '1D',
        verbose: bool = True, 
        plot: bool = True,
        dask_PCA: bool = True):

        """
        As always, we have the initializator/constructor function, where the 
        instance of the class is created, giving some initial parameters

        Args:
            ss_data (xarray.Dataset): This is the previously loaded moana dataset.
                This might be just the coast moana dataset or one including more data,
                as the continental platform until 200m of depth. This xarray.Dataset 
                must have the dimensions site and time, and coordinates/variables lon, 
                lat and ss, although the names might change, so then we have to change
                the ss_data_attrs attribute
            slp_data (xarray.Dataset): This is the slp data, also loaded with the
                usually used, Loader class in data.py. Be sure this predictor dataset
                has all the variables specified in dict_to_pca
            dict_to_pca (dict): This is the parameters dictionary with all the
                requested information to calculate the slp principal components      
            model_metrics (list): This is a list with all the model metrics to 
                evaluate model performance 
            ext_quantile (tuple): These are the quantiles that will be calculated 
                during model validation
            ss_data_attrs (tuple): This is a tuple with the main info regarding
                the ss data, for posterior calculations:
                example:
                    - ('lon','lat','site','ss','300m')
            ss_pred_times (int): This is the number of "future" times that will be
                predicted
            ss_per_pcs (float): This is the percentage of PCs that will be used
                if shores/coast analysis is performed
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
        self.plot    = plot

        # and save also the resample time and the metrics
        self.time_resample = time_resample
        self.ss_pred_times = ss_pred_times
        self.ss_per_pcs    = ss_per_pcs
        self.model_metrics = model_metrics
        self.ext_quantile  = ext_quantile

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
                [ss_data_attrs[3]].load().resample(time=time_resample).max() \
                .dropna(dim='time',how='all') for site in sites_to_analyze
            ] # to mantain code workflow
            self.ss_pcs_data = self.ss_real_data.copy()
            # to mantain coding workflow, there are not PCs!!
            self.ss_scalers = None # no SS-pcs will be calculated in this case

        else: # shore analysis will be performed
             
            self.lons, self.lats = lons, lats # save lon/lat as class attrs
            print('\n with longitudes = {} \n \n and \n \n latitudes = {} !! \
                \n\n for the specified middle locations / shores !!'.format(
                lons, lats # output lon/lat values
            )) if verbose else None
            self.sites_to_analyze = None

            if len(self.lons)==len(self.lats):
                self.num_locs = len(self.lons) # TODO: add RaiseError

            # save data and data_attrs in the class attributes
            self.raw_ss_data = ss_data # this could be the moana dataset
            self.ss_attrs = ss_data_attrs # moana / other, vars and coords
            print('\n lets calculate the storm surge pcs... \n') \
                if True else None
            if ss_tuple_data is None:
                ss_tuple_data = self.generate_ss_data(
                    min_dist_th, extra_help, 
                    ss_pred_times=ss_pred_times, 
                    ss_per_pcs=ss_per_pcs,
                    dask_PCA=dask_PCA
                ) # generate the lists with the ss pcs...
            # save all the returned lists
            self.ss_pcs_data, self.ss_real_data, self.ss_scalers, \
                self.ss_pcs_recon_data, self.shore_sites = ss_tuple_data

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


    def generate_ss_data(self, min_dist_th, extra_help, 
                         ss_pred_times=0, ss_per_pcs=0.95,
                         dask_PCA=False):
        """
        This class function generates the ss pcs analysis, so the mda+rbf
        interpolation can be performed over all the different indicated
        shores, ex: west, south...

        For each site defined by self.lon/self.lat, the function finds all
        nodes in the raw storm surge data points that are located withing
        min_dist_th km and fullfill the extra_help condition (narrow node search
        to a band around the site's longitude or latitude).

        Args:
            min_dist_th (list): Distances in km to find the nearest points to
                the lons/lats specified
            extra_help (list): This is extra information, regarding the "dimension"
                in which the shore is projected, so fake nodes are not introduced
                in the pcs analysis
            ss_pred_times (int): This is the number of "future" times that will be
                predicted
            ss_per_pcs (float): This is the percentage of PCs that will be used
            dask_PCA (float): Whether to perform or not dask_PCA
        
        Returns:
            (lists): The function returns the list with the data for all the
                     shores involved in the calculation
        """

        # calculate closet points and select ss maximums
        # for each site (self.lat, self.lon), find all the sites in self.raw_ss_data
        # located within min_dist_th distance and also fullfill the extra_help condition
        # sites is a list that contains some lists of nodes (for each analyzed point)
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
        if self.ss_attrs[4]=='shores':
            sites_out_list = []  # these are the sites which are not in the shores
            for site in self.raw_ss_data.site.values:
                if site not in sites_in_list:
                    sites_out_list.append(site)
            # and calculate closest shores to new found nodes
            closest_shores = calc_closest_data2_in_data1(
                (self.lons,self.lats),
                (self.raw_ss_data.sel(site=sites_out_list)[self.ss_attrs[0]].values,
                 self.raw_ss_data.sel(site=sites_out_list)[self.ss_attrs[1]].values),
                min_dist_th=150, extra_help=('lon',1.5) # dont avoid any node
            )[0]
            for site_out, shore in zip(sites_out_list, closest_shores):
                if shore[0]==11:
                    sites[12].append(site_out)
                else:
                    sites[shore[0]].append(site_out)
            # check all lists are with unique elements
            sites = [
                list(np.unique(sites_in_shore)) for sites_in_shore in sites
            ]
            # save as list with all the data for the pcs
            raw_ss_locs = [self.raw_ss_data.sel(
                    {self.ss_attrs[2]:sites[i]}
                )[self.ss_attrs[3]].load().resample(time=self.time_resample).max()\
                    .dropna(dim='time',how='all') for i in range(self.num_locs)
            ] # num_locs==len(sites) if working
            
        else:
            
            # save as list with all the data for the pcs
            raw_ss_locs = [self.raw_ss_data.sel(
                    {self.ss_attrs[2]:sites[i]}
                )[self.ss_attrs[3]].resample(time=self.time_resample).max()\
                    for i in range(self.num_locs)
            ] # num_locs==len(sites) if working

        # calculate the pcs of all the shores
        ss_pcs_locs, ss_scalers_locs = [], [] # lists to save the data
        ss_pc_recon_data = []
        for loc in range(self.num_locs):
            # standarize the features
            ss_scaler = StandardScaler()
            ss_stan = ss_scaler.fit_transform(raw_ss_locs[loc].values.T) \
                if ss_pred_times==0 else ss_scaler.fit_transform(
                    np.concatenate(
                        [
                            raw_ss_locs[loc].values.T[sspti:-(ss_pred_times-sspti),:] \
                                for sspti in range(ss_pred_times)
                        ] + [
                            raw_ss_locs[loc].values.T[ss_pred_times:,:]
                        ], axis=1
                    )
                )
            # TODO: check .T!!
            ss_stan[np.isnan(ss_stan)] = 0.0 # check additional nans
            ss_scalers_locs.append(ss_scaler)
            # calculate de PCAs
            if dask_PCA:
                import dask.array as dask_array
                from dask_ml.decomposition import PCA
                from dask_ml.decomposition import IncrementalPCA
                # Turn numpy array into dask array
                print(f'\n Using DASK with {(ss_stan.shape[0]//128,ss_stan.shape[1])} chunks... \n')
                dask_pcs_stan = dask_array.from_array(ss_stan,
                                                      chunks=(ss_stan.shape[0]//128,
                                                              ss_stan.shape[1]))
                pca_fit = IncrementalPCA(n_components=100)
                PCs = pca_fit.fit_transform(dask_pcs_stan)
            else:
                from sklearn.decomposition import PCA
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
                        'time': raw_ss_locs[loc].time.values[:-ss_pred_times] \
                            if ss_pred_times!=0 else raw_ss_locs[loc].time.values
                    }
                )
            )
            # reconstruct the ss values
            num_ss_pcs = len(np.where((
                np.cumsum(
                    ss_pcs_locs[-1].variance)/ss_pcs_locs[-1].total_variance)<ss_per_pcs)[0])
            ss_stan = np.repeat(
                ss_pcs_locs[-1].PCs.values[:,0],len(ss_pcs_locs[-1].n_features)
            ).reshape(
                len(ss_pcs_locs[-1].time),len(ss_pcs_locs[-1].n_features)
            ) * ss_pcs_locs[-1].EOFs.isel(n_components=0).values
            for i_comp in range(1,num_ss_pcs+1):
                ss_stan += np.repeat(
                    ss_pcs_locs[-1].PCs.values[:,i_comp],len(ss_pcs_locs[-1].n_features)
                ).reshape(
                    len(ss_pcs_locs[-1].time),len(ss_pcs_locs[-1].n_features)
                ) * ss_pcs_locs[-1].EOFs.isel(n_components=i_comp).values
            # get the real ss values
            real_ss = ss_scalers_locs[-1].inverse_transform(ss_stan)
            number_sites = len(ss_pcs_locs[-1].site.values)
            # and add the ss to final dataset
            ss_interp_dict = dict(
                zip(
                    [f'ss_PCs_recon_{i}' for i in range(ss_pred_times+1)],
                    [(('time','site'),
                      real_ss[
                          :,i*number_sites:(i+1)*number_sites
                      ].reshape(-1,number_sites)
                     ) for i in range(ss_pred_times+1)
                    ]
                )
            )
            ss_pc_recon_data.append(xr.Dataset(
                ss_interp_dict, 
                coords={'time': ss_pcs_locs[-1].time.values,
                        'site': ss_pcs_locs[-1].site.values
                })
            )
        
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
                        self.raw_ss_data.isel(site=slice(None,None,1)).max(dim='time'),
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

        return ss_pcs_locs, raw_ss_locs, ss_scalers_locs, ss_pc_recon_data, sites


    def generate_slp_data(self):

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
            return PCA_DynamicPred(
                       self.raw_slp_data, # this is always the same
                       **self.dict_to_pca # extra arguments
                   ).pcs_get()[0] # [...] * self.num_locs # same PCs in all sites

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
                    site_id=self.sites_to_analyze[ipc] if self.sites_to_analyze \
                        else ipc, # TODO: check this!!
                    **dict_to_pca # extra arguments
                ).pcs_get()

            # append calculated pcs to list
            pcs_for_each_shore.append(pca_data)

        return pcs_for_each_shore


    def calc_MDA_RBF(self, selected_shores=None,
                     percentage_pcs_ini=[0.99],
                     num_samples_ini=[750], 
                     ss_pcs=1, ss_pred_times=0,
                     validate_rbf_kfold=(False,None),
                     try_all=False, append_extremes=None,
                     plot: bool = False, verbose: bool = True):

        """
        This is the calculation function, where given the wanted
        locations / shores, the MDA+RBF interpolation model is performed,
        plotting different results and statistics

        Args:
            selected_shores (list): List with ilocs for the shores, but could
                be range(n) being n the number of individual locations.
            percentage_pcs_ini (list): List with all the percentage of pcs
                that will be used to interpolate.
            num_samples_ini (list): List with all the number of samples that
                will be used to create the subset.
            ss_pcs (int): This indicates the number of SS pcs that will be used
                when interpolating, set this parameter to 1 if we will directly
                interpolate the SS in individual locations.
            ss_pred_times (int): This is the number of "future" times that will be
                predicted
            try_all (bool): Whether to try all the different combinations or not.
            append_extremes (int,None): This indicates the number of extremes that
                will be used to create the subset, but can be set to None.
            plot / verbose (bools): Whether to plot / log or not results.
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
            self.real_ss_rbf, common_times, experiments_idxs_in, experiments_idxs_out = \
                MDA_RBF_algorithm(
                    self.slp_pcs_data[sel_loc] if self.dict_to_pca['region'][0]=='local' else self.slp_pcs_data, 
                    self.ss_pcs_data[sel_loc], ss_pcs=ss_pcs, ss_pred_times=ss_pred_times,
                    ss_scaler=self.ss_scalers[sel_loc] if \
                        isinstance(self.ss_scalers,list) else None,
                    try_all=try_all, append_extremes=append_extremes,
                    percentage_pcs_ini=percentage_pcs_ini,
                    num_samples_ini=num_samples_ini,
                    validate_rbf_kfold=validate_rbf_kfold,
                    plot=plot, verbose=verbose, 
                    sel_loc=sel_loc if ss_pcs!=1 else \
                        self.sites_to_analyze[sel_loc] # which is just 0, 1... in the first case
                )

            print('\n Lets validate the SS reconstructions for location {}!! \n\n'.format(
                sel_loc+1 # this is the shore            
            )) if verbose else None

            # iterate over all the shores / just one location
            for isite, site in enumerate(self.ss_real_data[sel_loc].site.values):  
                
                print('Validating location {}!!'.format(
                    site # this is the site         
                )) if verbose else None

                experiment_datasets = []

                # iterate over all the available experiments
                for exp in range(len(self.real_ss_rbf.experiment)):

                    # loop copy to avoid problems
                    experiment_dataset = self.real_ss_rbf\
                        .isel(site=isite,experiment=exp).copy()
                    times_to_validate = experiment_dataset.sel(
                        time=common_times # to use same pred and moana times
                    ).time.values[
                        experiments_idxs_out[exp] # validate times not used in MDA+RBF model
                    ]

                    # try for available nodes or continue
                    try:
                        title, stats = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site).values \
                                if ss_pred_times==0 else \
                                self.ss_real_data[sel_loc].sel(site=site).values[:-ss_pred_times],
                            experiment_dataset.sel(
                                time=common_times # to use same pred and moana times
                            ).ss_interp_0.values.reshape(-1), # mda+rbf ss values
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_val, stats_val = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site,
                                time=times_to_validate).values,
                            experiment_dataset.ss_interp_0.sel(
                                time=times_to_validate
                            ).values.reshape(-1), # mda+rbf ss values
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_recon, stats_recon = generate_stats(
                            self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values \
                                if ss_pred_times==0 else \
                                self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values[:-ss_pred_times],
                            experiment_dataset.sel(
                                time=common_times # to use same pred and moana times
                            ).ss_interp_0.values.reshape(-1), # mda+rbf ss values
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_val_recon, stats_val_recon = generate_stats(
                            self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site,
                                time=times_to_validate).values,
                            experiment_dataset.ss_interp_0.sel(
                                time=times_to_validate
                            ).values.reshape(-1), # mda+rbf ss values
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                    except:
                        title, stats = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site).values \
                                if ss_pred_times==0 else \
                                self.ss_real_data[sel_loc].sel(site=site).values[:-ss_pred_times],
                            experiment_dataset.ss_interp_0.sel(
                                time=common_times # to use same pred and moana times
                            ).values.reshape(-1), # mda+rbf ss values
                            not_nan_idxs=np.where(
                                (~np.isnan(self.ss_real_data[sel_loc].sel(site=site).values \
                                              if ss_pred_times==0 else \
                                              self.ss_real_data[sel_loc].sel(site=site).values[:-ss_pred_times]) &
                                 ~np.isnan(experiment_dataset.ss_interp_0.sel(
                                      time=common_times # to use same pred and moana times
                                  ).values.reshape(-1)))
                            )[0],
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_val, stats_val = generate_stats(
                            self.ss_real_data[sel_loc].sel(site=site,
                                time=times_to_validate).values,
                            experiment_dataset.ss_interp_0.sel(
                                time=times_to_validate
                            ).values.reshape(-1), # mda+rbf ss values
                            not_nan_idxs=np.where(
                                (~np.isnan(self.ss_real_data[sel_loc].sel(
                                     site=site,time=times_to_validate).values) &
                                 ~np.isnan(experiment_dataset.ss_interp_0.sel(
                                     time=times_to_validate
                                ).values.reshape(-1)))
                            )[0],
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_recon, stats_recon = generate_stats(
                            self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values \
                                if ss_pred_times==0 else \
                                self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values[:-ss_pred_times],
                            experiment_dataset.ss_interp_0.sel(
                                time=common_times # to use same pred and moana times
                            ).values.reshape(-1), # mda+rbf ss values
                            not_nan_idxs=np.where(
                                (~np.isnan(self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values \
                                              if ss_pred_times==0 else \
                                              self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site).values[:-ss_pred_times]) &
                                 ~np.isnan(experiment_dataset.ss_interp_0.sel(
                                      time=common_times # to use same pred and moana times
                                  ).values.reshape(-1)))
                            )[0],
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )
                        title_val_recon, stats_val_recon = generate_stats(
                            self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(site=site,
                                time=times_to_validate).values,
                            experiment_dataset.ss_interp_0.sel(
                                time=times_to_validate
                            ).values.reshape(-1), # mda+rbf ss values
                            not_nan_idxs=np.where(
                                (~np.isnan(self.ss_pcs_recon_data[sel_loc].ss_PCs_recon_0.sel(
                                     site=site,time=times_to_validate).values) &
                                 ~np.isnan(experiment_dataset.ss_interp_0.sel(
                                     time=times_to_validate
                                ).values.reshape(-1)))
                            )[0],
                            metrics=self.model_metrics,
                            ext_quantile=self.ext_quantile
                        )

                    # save results in list
                    # print(experiment_dataset)
                    for metric in list(stats.keys()):
                        try:
                            experiment_dataset[metric] = experiment_dataset.perpcs*0 + \
                                stats[metric]
                            experiment_dataset[metric+'_val'] = experiment_dataset.perpcs*0 + \
                                stats_val[metric]
                            experiment_dataset[metric+'_recon'] = experiment_dataset.perpcs*0 + \
                                stats_recon[metric]
                            experiment_dataset[metric+'_val_recon'] = experiment_dataset.perpcs*0 + \
                                stats_val_recon[metric]
                        except:
                            experiment_dataset[metric] = experiment_dataset.perpcs*0 + \
                                stats[metric][0]
                            experiment_dataset[metric+'_val'] = experiment_dataset.perpcs*0 + \
                                stats_val[metric][0]
                            experiment_dataset[metric+'_recon'] = experiment_dataset.perpcs*0 + \
                                stats_recon[metric][0]
                            experiment_dataset[metric+'_val_recon'] = experiment_dataset.perpcs*0 + \
                                stats_val_recon[metric][0]
                    experiment_datasets.append(
                        experiment_dataset # TODO: drop ss, .drop_vars('ss_interp')
                    )

                final_datasets.append(
                    xr.concat(experiment_datasets,dim='experiment')
                )
                    
        return xr.concat(final_datasets,dim='site') if ss_pcs==1 \
            else xr.concat(final_datasets,dim='shore')


def MDA_RBF_algorithm(
    pcs_data, ss_data, ss_pcs: int = 1, ss_pred_times: int = 0,
    ss_scaler = None, # to de-standarize the pcs
    percentage_pcs_ini: list = [0.6,0.8],
    num_samples_ini: list = [100,200],
    try_all: bool = False, append_extremes: int = 10,
    validate_rbf_kfold: bool = False,
    plot: bool = True, verbose: bool = True,
    sel_loc = None):

    """
    This out-of-class function performs the MDA+RBF interpolation given the
    pcs of the sea-level-pressure fields and some variable to interpolate, that
    might be the storm surge, but can also be the pcs of this spatial ss

    Args:
        pcs_data (xarray.Dataset): These are the pcs of the slp fields, that can
            be previously calculate using generate_slp_data
        ss_data (xarray.Dataset): This is the variable/s to interpolate, and might 
            be the storm surge or the pcs of the SS, in case the shore related
            analysis is done over NZ
        ss_pcs (int, optional): This is the number of pcs that is wanted to be
            used when the ss-pcs are calculated. Defaults to 1, which means
            no pcs are calculated.
        ss_pred_times (int): This is the number of "future" times that will be
            predicted
        ss_scaler (sklearn.Scaler, optional): This is the scikit-learn scaler
            that will be used to recalculate the real ss. Defaults to None.
        percentage_pcs_ini (list, optional): List with all the percentages that
            will be used to crop the slp pcs. Defaults to [0.6,0.8].
        num_samples_ini (list, optional): List with all the integer numbers that
            represent the number of points the MDA algorithm will search for, and
            then the RBF interpolation will use. Defaults to [100,200].
        try_all (bool, optional): Whether to try or not all the possible combinations
            of the 2 previous defined parameters. Defaults to False.
        append_extremes (int, optional): Wheter to add or not the ss/pcs extreme values
            to the subset to interpolate, not extrapolate. Defaults to 10.
        previous_experiments_idxs_in (list, opt): Previous MDA points used.
            Defaults to None. ** It is a list of lists.
        validate_rbf_kfold (bool, optional): Wheter to call the rbf_validation function
            once the process is done. Defaults to False.
        plot (bool, optional): Whether to plot or not the results. Defaults to True.
        verbose (bool, optional): Wheter to plot or not logs. Defaults to True.
        sel_loc (int, None): This is the for loop iteration in calc_MDA_RBF.
            Defaults to None, but provided in calc_MDA_RBF().

    Returns:
        [xarray.Datasets]: xarray datasets with the data reconstructed
    """

    outs, real_sss = [], []
    experiments_idxs_in = []
    experiments_idxs_out = []

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

        predictor_dataset = (
            (pcs_data.PCs-pcs_data.PCs.mean(dim='time'))/np.sqrt(pcs_data.variance)
        ).isel(
            n_components=slice(0,num_pcs)
        ).dropna(dim='time',how='all') # mantain just num_pcs

        # get common times
        common_times, pred_ct, tar_ct = np.intersect1d(
            predictor_dataset.time.values,
            ss_data.time.values,
            return_indices=True
        )

        # get predictor and target with common times and prepare datasets
        predictor_dataset_ct = predictor_dataset.sel(time=common_times)
        target_dataset = ss_data.sel(time=common_times) if ss_pcs==1 else \
            ss_data.PCs.sel(time=common_times).isel(n_components=slice(0,ss_pcs))
        
        # use MDA to generate a demo dataset and subset for RBF interpolation
        pcs_to_mda = int(0.33*num_pcs) if int(0.2*num_pcs)<10 else 10 # used SLP pcs...
        ss_pcs_to_mda = ss_pcs if ss_pcs<8 else 8
        ix_scalar_pred_mda = list(np.arange(pcs_to_mda+ss_pcs_to_mda))
        ix_directional_pred = []
        # perform the mda analysis (this is the mda input data)
        mda_dataset = np.concatenate(
            [
                predictor_dataset_ct[:,:pcs_to_mda].values,
                target_dataset.values.reshape(-1,ss_pcs)[
                    :,:ss_pcs_to_mda
                ]
            ], axis=1
        )
        # MDA algorithm
        predictor_subset_red, subset_indexes = maxdiss_simplified_no_threshold(
            mda_dataset, n_samp, ix_scalar_pred_mda, ix_directional_pred, log=False
        )

        if plot:
            fig = Plot_MDA_Data(
                pd.DataFrame(
                    mda_dataset,columns=['PC'+str(i+1) for i in range(pcs_to_mda)] + \
                        ['SS'+str(i+1) for i in range(ss_pcs_to_mda)]
                ),
                pd.DataFrame(
                    predictor_subset_red,columns=['PC'+str(i+1) for i in range(pcs_to_mda)] + \
                        ['SS'+str(i+1) for i in range(ss_pcs_to_mda)]
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
        predictor_subset = predictor_dataset_ct.sel(
            time=predictor_dataset_ct.time.values[subset_indexes]
        )
        target_subset = target_dataset.sel(
            time=target_dataset.time.values[subset_indexes]
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
                    colors = ['k','b','g','c','m','y','k','b','g','c','m','y']*3
                    colors = colors[:ss_pcs]
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
                {
                    'ss_interp_0': (('time','experiment'),out), # n experiment in site
                    'perpcs': (('experiment'), [per_pcs]), 'nsamples': (('experiment'), [n_samp])
                }, coords={
                    'time': predictor_dataset.time.values, 'experiment': [i_exp],
                    # 'train_times': predictor_dataset.time.values[subset_indexes]
                }).expand_dims({'site': [sel_loc]})
            ) if ss_pcs==1 else None
            experiments_idxs_in.append(subset_indexes)
            experiments_idxs_out.append(idxs_not_in_subset)
        except:
            print('\n the RBF reconstruction did not converge... \n')
            continue

        # plot output results
        if plot: #and len(common_times)==len(predictor_dataset.time.values):
            for i_ss in range(ss_pcs):
                # figure spec-grid
                fig = plt.figure(figsize=(_figsize_width*4.5,_figsize_height))
                gs = gridspec.GridSpec(nrows=1,ncols=3)
                # time regular plot
                ax_time = fig.add_subplot(gs[:,:2])
                target_dataset.plot(
                    ax=ax_time,c=real_obs_col,alpha=0.8,label='Real SS observations'
                ) if ss_pcs==1 else target_dataset.isel(
                    n_components=i_ss
                ).plot(
                    ax=ax_time,alpha=0.8,c=real_obs_col,label='Storm Surge PCs'
                )
                ax_time.plot(
                    predictor_dataset.time.values,out[:,i_ss],
                    alpha=0.7,c='orange',linestyle='--',
                    label='RBF predictions'
                )
                ax_time.legend(ncol=2,fontsize=_fontsize_legend)
                ax_time.set_xlim(
                    predictor_dataset.time.values[0],predictor_dataset.time.values[-1]
                )
                # validation plot
                ax_vali = fig.add_subplot(gs[:,2:])
                ax_vali.set_xlabel('Observation')
                ax_vali.set_ylabel('Prediction')
                scatterplot(
                    target_dataset.values.reshape(-1,ss_pcs)[:,i_ss],
                    out[pred_ct,i_ss],ax=ax_vali
                )
                qqplot(
                    target_dataset.values.reshape(-1,ss_pcs)[:,i_ss],
                    out[pred_ct,i_ss],ax=ax_vali
                )
                # add title
                title, ttl_stats = generate_stats(
                    target_dataset.values.reshape(-1,ss_pcs)[:,i_ss],
                    out[pred_ct,i_ss],
                )
                fig.suptitle(
                    'SS interpolation \n '+title,fontsize=_fontsize_title,y=1.1
                ) if ss_pcs==1 else fig.suptitle(
                    f'PC{i_ss} interpolation',fontsize=_fontsize_title,y=1.1
                )
        # show the results
        plt.show()

        # RBF Validation: using k-fold mean squared error methodology
        if validate_rbf_kfold[0]:
            try:
                test = rbf_validation(
                    predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
                    target_subset.values.reshape(-1,ss_pcs), ix_scalar_t, ix_directional_t,
                    n_splits=validate_rbf_kfold[1], shuffle=True
                )
                print(test)
            except:
                print('This TEST did not converge!!') if verbose else None

        if ss_scaler:
            # reconstruct the ss values
            ss_stan = np.repeat(
                out[:,0],len(ss_data.n_features)
            ).reshape(
                len(predictor_dataset.time),len(ss_data.n_features)
            ) * ss_data.EOFs.isel(n_components=0).values
            for i_comp in range(1,ss_pcs):
                ss_stan += np.repeat(
                    out[:,i_comp],len(ss_data.n_features)
                ).reshape(
                    len(predictor_dataset.time),len(ss_data.n_features)
                ) * ss_data.EOFs.isel(n_components=i_comp).values
            # get the real ss values
            real_ss = ss_scaler.inverse_transform(ss_stan)
            number_sites = len(ss_data.site.values)
            # and add the ss to final dataset
            ss_interp_dict = dict(
                zip(
                    [f'ss_interp_{i}' for i in range(ss_pred_times+1)],
                    [(('time','site','experiment'),
                      real_ss[
                          :,i*number_sites:(i+1)*number_sites
                      ].reshape(-1,number_sites,1)
                     ) for i in range(ss_pred_times+1)
                    ]
                )
            )
            ss_interp_dict['perpcs']   = (('experiment'),[per_pcs])
            ss_interp_dict['nsamples'] = (('experiment'),[n_samp])
            #real_sss.append(xr.Dataset( # save real storm surge
            #    {'ss_interp': (('time','site','experiment'),real_ss.reshape(
            #        -1,real_ss.shape[1],1
            #    )), 'perpcs': (('experiment'), [per_pcs]), 
            #        'nsamples': (('experiment'), [n_samp])
            real_sss.append(xr.Dataset(
                ss_interp_dict, 
                coords={'time': predictor_dataset.time.values,
                    'site': ss_data.site.values, 'experiment': [i_exp]
                }).expand_dims(
                    {'shore':[sel_loc]}
                )) if ss_pcs!=1 else None

    # return real ss values
    real_ss_to_return = xr.concat(outs,dim='experiment') if ss_pcs==1 else \
        xr.concat(real_sss,dim='experiment') 

    return real_ss_to_return, common_times, experiments_idxs_in, experiments_idxs_out

