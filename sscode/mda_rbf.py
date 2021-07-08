# arrays
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
from .config import default_region
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
    This MDA+RBF model allows the user to perform a model which consists in
    reconstructing / interpolating the storm surge in different locations
    around New Zealand, but giving the pcs of both the sea-level-pressure
    fields and the pcs of this storm surge. This analysis will be performed
    over different locations / shores, that are indicated by the lon / lat of
    one point in the "middle" of this shore

    """

    def __init__(self, ss_data, # this could be the loaded moana with Loader class
        slp_data, # this will be used to calculate the local predictors
        dict_to_pca, # this is the parameters dictionary to calculate pca from the slp
        ss_data_attrs: tuple = ('lon','lat','site','ss'), # ss attrs
        sites_to_analyze = list(np.unique(
            [ 689,328,393,1327,393,480,999,116,224,1124,949,708, # UHSLC
              1296,378,1124,780,613,488,1442,1217,578,200,1177,1025,689,949,224,1146, # LINZ
              1174,1260,1217,744,1064,1214,803,999 # OTHER (ports...)
            ]
        )[::3]), # sites to analyze in case individual analysis is performed
        # lons / lats in case shore analysis is required
        lons: list = [168.1, 171.4, 167.5, 171.1, 173.5, 173.3, 175.1, 177.9, 172.6], 
        lats: list = [-44.2, -41.9, -47.2, -45.4, -42.5, -40.7, -40.2, -40.1, -34.3],
        min_dist_th: list = [130, 130, 140, 110, 110, 90, 90, 130, 400],
        extra_help: list = [ # this helps to stay in the wanted shore of NZ
            ('lon',1.5),('lon',1.5),('lat',1.5),('lon',1.5),('lon',1.5),
            ('lat',0.7),('lat',0.7),('lon',1.5),('lon',3.5)
        ], time_resample: str = '1D', # this is important, try 6H
        verbose: bool = True, plot: bool = True):

        """
        As always, we have the initializator/constructor function, where the 
        instance of the class is created, giving some initial parameters

        Args:
            ss_data (xarray.Dataset): This is the previously loaded moana datasets.
                This might be just the shore moana dataset or the one including
                the continental platform until 300m of depth, which is now saved
                in data/storm_surge_data/moana_hindcast_v2/moana_ss_300m_daily.nc
            slp_data (xarray.Dataset): This is the slp data, also loaded with the
                usually used, Loader class in data.py
            dict_to_pca (dict): This is the parameters dictionary with all the
                requested information to calculate the slp principal components                
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
            verbose, plot (bools): Whether to log / plot the results
        """

        # we first save the verbose and plot attrs to print/plot or not logs
        self.verbose = verbose
        self.plot = plot
        self.time_resample = time_resample

        print('\n Initializing the MDA + RBF constructor... \n') \
            if self.verbose else None

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
            print('\n in locations: {}, with coords: ({},{}) \n'.format(
                sites_to_analyze, self.lons, self.lats
            )) if self.verbose else None
            self.sites_to_analyze = sites_to_analyze

            # save data and data_attrs in the class attributes
            self.raw_ss_data = ss_data # this could be the moana dataset
            self.ss_attrs = ss_data_attrs # moana / other, vars and coords
            self.ss_pcs_data = [ss_data.sel({ss_data_attrs[2]:[site]}) \
                .ss.load().resample(time=time_resample).max() \
                .dropna(dim='time',how='all') for site in sites_to_analyze
            ] # to mantain coding workflow
            self.ss_real_data = [ss_data.sel({ss_data_attrs[2]:[site]}) \
                .ss.load().resample(time=time_resample).max() \
                .dropna(dim='time',how='all') for site in sites_to_analyze
            ] # to mantain code workflow 2.0
            self.ss_scalers = None # no SS-pcs will be calculated in this case

        else: # shore analysis will be performed
             
            self.lons, self.lats = lons, lats # save lon/lat as class attrs
            print('\n with longitudes = {} \n \n and \n \n latitudes = {} !! \
                \n\n for the specified middle locations / shores !!'.format(
                lons, lats # output lon/lat values
            )) if self.verbose else None

            if len(self.lons)==len(self.lats):
                self.num_locs = len(self.lons) # TODO: add RaiseError

            # save data and data_attrs in the class attributes
            self.raw_ss_data = ss_data # this could be the moana dataset
            self.ss_attrs = ss_data_attrs # moana / other, vars and coords
            print('\n lets calculate the storm surge pcs... \n') \
                if self.verbose else None
            ss_tuple_data = self.generate_ss_data(
                min_dist_th, extra_help
            ) # generate the lists with the ss pcs...
            # save all the returned lists
            self.ss_pcs_data, self.ss_real_data, self.ss_scalers = ss_tuple_data

        if len(self.lons)==len(self.lats):
            self.num_locs = len(self.lons) # TODO: add RaiseError

        # we now save the slp, slp-pcs...
        print('\n data will be resampled to {} \n\n and the PC analysis \
            will be calculated using the parameters \n\n {} \n'.format(
                time_resample,dict_to_pca
            )
        )
        self.time_resample = time_resample # save time resample
        self.dict_to_pca = dict_to_pca # save pca attrs
        self.raw_slp_data = slp_data # this could be the CFSR slp/winds
        print('\n lets calculate the slp pcs... \n') \
            if self.verbose else None
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
            print('\n All the SS pcs have been calculated, lets plot what we got!! \n')
            # plot selected points to check coherence
            fig, axes = plt.subplots(
                ncols=2,figsize=_figsize,
                subplot_kw={
                    'projection':ccrs.PlateCarree(central_longitude=180)
                }
            )
            cmap, loc_colors = get_n_colors('jet',self.num_locs)
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
                            transform=ccrs.PlateCarree(),s=10,zorder=111,
                            c=np.array(loc_colors[ishore],ndmin=2)
                        )
                        ax.scatter( # "middle" point as a star
                            x=self.lons[ishore],y=self.lats[ishore],s=100,zorder=112,
                            marker='*',c='k',transform=ccrs.PlateCarree(),edgecolors='yellow'
                        )
                        if ishore==0:
                            ax.set_title(
                                'Input locations and closest stations',
                                fontsize=_fontsize_title
                            )
                ax.set_facecolor('lightblue')
            plot_ccrs_nz(axes,plot_labels=(False,None,None))
            plt.show() # debug results as calculated
        else:
            print('\n All the SS pcs have been calculated!! \n')

        return ss_pcs_locs, raw_ss_locs, ss_scalers_locs


    def generate_slp_data(self, pres_vars: tuple = ('SLP','longitude','latitude'),
        calculate_gradient: bool = False, winds: tuple = (False,None),
        wind_vars: tuple = ('wind_proj_mask','lon','lat'),
        time_lapse: int = 1, # 1 equals to NO time delay 
        time_resample: str = '1D', region: tuple = (True,default_region),
        pca_plot: tuple = (True,False,2), verbose: bool = True,
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

        for ipc in range(self.num_locs):
            # save the dict in a local copy
            dict_to_pca = self.dict_to_pca.copy()

            if self.dict_to_pca['region'][0]=='local':
                region_coords = dict_to_pca.pop('region')[1]
                local_region = (True,(
                    self.lons[ipc]-region_coords[0], # new lon / lat region
                    self.lons[ipc]+region_coords[0],
                    self.lats[ipc]+region_coords[1],
                    self.lats[ipc]-region_coords[1],
                ))
                # lets first calculate the pcs
                pca_data, pca_scaler = PCA_DynamicPred(
                    self.raw_slp_data, # this is always the same
                    region=local_region, # pass the calculated local region
                    **dict_to_pca # extra arguments without the winds
                )
            else:
                # lets first calculate the pcs
                pca_data, pca_scaler = PCA_DynamicPred(
                    self.slp_data, # this is always the same
                    **dict_to_pca # extra arguments without the winds
                )
            # append calculated pcs to list
            pcs_for_each_shore.append(pca_data)

        return pcs_for_each_shore


    def calc_MDA_RBF(self, selected_shores=None,
                     percentage_pcs_ini=[0.9999],
                     num_samples_ini=[1500], ss_pcs=2,
                     try_all=False, append_extremes=None):

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
        """

        print('\n Lets interpolate the SS using all the calculated \
            pcs (slp + ss) and the MDA+RBF technique!! \
            This interpolation will be performed in all the locations!! \n'
        )

        final_datasets = []

        for sel_loc in selected_shores: # could be individual locations

            # perform the RBF interpolation, using MDA too
            real_ss_rbf = MDA_RBF_algorithm(
                self.slp_pcs_data[sel_loc], 
                self.ss_pcs_data[sel_loc], ss_pcs=ss_pcs, 
                ss_scaler=self.ss_scalers[sel_loc] if \
                    isinstance(self.ss_scalers,list) else None,
                try_all=try_all, append_extremes=append_extremes,
                percentage_pcs_ini=percentage_pcs_ini,
                num_samples_ini=num_samples_ini,
                plot=self.plot, verbose=self.verbose, 
                sel_loc=sel_loc if ss_pcs!=1 else \
                    self.sites_to_analyze[sel_loc] # which is just 0, 1... in the first case
            )
            final_datasets.append(real_ss_rbf)
            print('\n Lets plot the SS reconstructions for location {}!! \n'.format(
                sel_loc # this is the shore            
            ))

            for isite, site in enumerate(
                self.ss_real_data[sel_loc].site.values if ss_pcs==1 else
                    self.ss_real_data[sel_loc].site.values[::22]
            ):
                # try for available nodes or continue
                try:
                    title, stats = generate_stats(
                        self.ss_real_data[sel_loc].sel(site=site).values,
                        real_ss_rbf.isel(site=isite,experiment=0)\
                            .ss_interp.values.reshape(-1)
                    )
                except:
                    title, stats = generate_stats(
                        self.ss_real_data[sel_loc].sel(site=site).values,
                        real_ss_rbf.isel(site=isite,experiment=0)\
                            .ss_interp.values.reshape(-1),
                        not_nan_idxs=np.where(
                            (~np.isnan(self.ss_real_data[sel_loc].sel(site=site).values) &
                            (~np.isnan(real_ss_rbf.isel(site=isite,experiment=0)\
                                .ss_interp.values.reshape(-1))))
                        )[0]
                    )
                    continue

                # figure spec-grid
                fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
                gs = gridspec.GridSpec(nrows=1,ncols=3)
                # time regular plot
                ax_time = fig.add_subplot(gs[:,:2])
                self.ss_real_data[sel_loc].sel(site=site).plot(
                    ax=ax_time,c=real_obs_col,label='Real SS measures'
                )
                ax_time.plot(
                    self.ss_real_data[sel_loc].sel(site=site).time.values,
                    real_ss_rbf.isel(site=isite,experiment=0)\
                        .ss_interp.values.reshape(-1),
                    c='grey',linestyle='--',alpha=0.8,
                    label='Reconstructed SS -- MDA + RBF'
                )
                ax_time.set_xlim(
                    self.ss_real_data[sel_loc].sel(site=site).time.values[0],
                    self.ss_real_data[sel_loc].sel(site=site).time.values[-1]
                ) # delete white spaces
                ax_time.legend(ncol=2,fontsize=_fontsize_legend)
                # validation plot
                ax_vali = fig.add_subplot(gs[:,2:])
                ax_vali.set_xlabel('Observation')
                ax_vali.set_ylabel('Prediction')
                scatterplot(
                    self.ss_real_data[sel_loc].sel(site=site),
                    real_ss_rbf.isel(site=isite,experiment=0)\
                        .ss_interp.values.reshape(-1),
                    ax=ax_vali
                )
                qqplot(
                    self.ss_real_data[sel_loc].sel(site=site),
                    real_ss_rbf.isel(site=isite,experiment=0)\
                        .ss_interp.values.reshape(-1),
                    ax=ax_vali
                )
                # add title
                fig.suptitle(
                    title+' -- SITE: {}'.format(site),fontsize=_fontsize_title,y=1.1
                )
                # show the results
                plt.show()

        # TODO: add stats to returned xarray object

        return xr.concat(final_datasets,dim='site') if ss_pcs==1 \
            else xr.concat(final_datasets,dim='shore')


def MDA_RBF_algorithm(
    pcs_data, ss_data, ss_pcs: int = 2,
    ss_scaler = None, # to de-standarize the pcs
    percentage_pcs_ini: list = [0.6,0.9],
    num_samples_ini: list = [100,500],
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
            ((np.cumsum(pcs_data.variance)/np.sum(pcs_data.variance)) < per_pcs).values==True
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
            target_dataset.plot(ax=ax,c='k',alpha=0.8) if ss_pcs==1 else \
                target_dataset.plot(ax=ax,hue='n_components',alpha=0.7)
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
                    'SS interpolation',fontsize=_fontsize_title,y=1.1
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
            # get the real SLP values
            real_ss = ss_scaler.inverse_transform(ss_stan)

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

