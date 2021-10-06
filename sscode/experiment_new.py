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
from .config import data_path, default_location, default_region, \
    default_evaluation_metrics # get config params
from .data import datasets_attrs
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_title, _fontsize_legend
from .utils import calculate_relative_winds
from .pca_new import PCA_DynamicPred
from .linear import MultiLinear_Regression
from .knn import KNN_Regression
from .xgboost import XGBoost_Regression


# below some example dictionarys to use
pca_attrs_default = {
    'calculate_gradient': [False,True], 
    'winds': [False,True],
    'time_lapse': [1,2,3], # 1 equals to NO time delay 
    'time_resample': ['1D'],
    'region': [(True,default_region)]
}
linear_attrs_default = {
    'train_size': [0.8,0.9], 'percentage_PCs': [0.80,0.88,0.95]
}
knn_attrs_default = {
    'train_size': [0.8,0.9], 'percentage_PCs': [0.80,0.88,0.95],
    'k_neighbors': [4,8,12,16,None] # None calculates the optimum k-neighs
}
xgboost_attrs_default = {
    'train_size': [0.8,0.9], 'percentage_PCs': [0.80,0.88,0.95],
    'n_estimators': [40,50,60], 'max_depth': np.arange(7,11,2),
    'min_samples_split': np.linspace(0.01,0.4,5),
    'learning_rate': [0.1], 'loss': ['ls'] # more could be added
}

class Experiment(object):
    """
    This class Experiment summarizes all the previous work done with the linear and the
    knn models, as this class allows the user to perform a detailed analysis of one
    requested model given a set of parameters

    """

    def __init__(self, slp_data, wind_data, ss_data, # this must have several stations
                 sites_to_analyze: list = list(np.random.randint(10,1000,5)),
                 model: str = 'linear', # this is the model to analyze
                 model_metrics: list = [
                     'bias','si','rmse','pearson','spearman','rscore',
                     'mae', 'me', 'expl_var', # ...
                 ], # these are the metrics to evaluate the model
                 pca_attrs: dict = pca_attrs_default,
                 model_attrs: dict = linear_attrs_default,
                 pcs_folder: str = '/home/metocean/pcas',
                 verbose=False):
        """
        As the initializator, the __init__ function creates the instance of the class,
        given a set of parameters, which are described below

        Args:
            slp_data (xarray.Dataset): These are the sea-level-pressure fields, previously
                loaded with the Loader class, loader.predictor_slp!!
            wind_data (xarray.Dataset): These are the wind fields, previously
                loaded with the Loader class, loader.predictor_wind!!
            ss_data (xarray.Dataset): This is the storm surge from the moana hindcast, previously
                loaded with the Loader class, loader.predictand!!
            sites_to_analyze (list, optional): This is the list with all the moana v2
                hindcast locations to analyze. Defaults to random locations.
            model (str, optional): Type of model to analyze. Defaults to 'linear'.
            model_metrics (list, optional): These are all the evaluation metrics that might
                be used to evaluate the model performance. Defaults to simple list.
            pca_attrs (dict, optional): PCA dictionary with all the parameters to use in pca.
                Defaults to pca_attrs_default.
            model_attrs (dict, optional): Model dictionary with all the parameters to use. 
                Defaults to linear_attrs_default.
            pcs_folder (str, optional): Folder where the PCs are stored.
                Default to ''/home/metocean/pcas'
        """

        # lets build the experiment!! + with CFSR
        self.predictor_data = 'cfsr'

        # save class simplest attributes
        self.slp_data = slp_data
        self.wind_data = wind_data
        self.ss_data = ss_data
        self.ss_sites = sites_to_analyze
        self.model = model
        self.pca_attrs = pca_attrs
        self.model_attrs = model_attrs # save class attributes
        self.pcs_folder = pcs_folder
        self.verbose = verbose

        # add pre-defined metrics to existent ones
        for mta in default_evaluation_metrics: # defined in sscode/config.py
            model_metrics.append(mta) if mta not in model_metrics else None
        self.model_metrics = model_metrics

        # more complicated features to save in model
        model_iters = 1
        exp_attrs_shape = []
        for pca_key in [key for key in pca_attrs.keys()]:
            model_iters *= len(pca_attrs[pca_key])
            exp_attrs_shape.append(len(pca_attrs[pca_key]))
        for model_key in [key for key in model_attrs.keys()]:
            model_iters *= len(model_attrs[model_key])
            exp_attrs_shape.append(len(model_attrs[model_key]))
        self.model_iters = model_iters
        self.model_params_shape = exp_attrs_shape # saved as a list

        print(
            '\n The model has been correctly initialized with || model = {} || \
            \n\n and model evaluation metrics = {} \
            \n\n pca_params = {} \n\n model_params = {} \
            \n\n which makes a total of {} iterations as there are {} values for each parameter \
            \n\n the experiment will be performed in sites = {} \
            \n\n RUN CELL BELOW if this information is correct!!'.format(
                self.model,self.model_metrics,self.pca_attrs,self.model_attrs,
                self.model_iters,tuple(self.model_params_shape),
                self.ss_sites # saved list with the sites
            )
        ) # print experiment inputs


    def get_pcs(self, parameters, site_id=None, site_location=None,plot=False):
        
        dict_to_pca = dict(zip(list(self.pca_attrs.keys()),parameters[:5]))
        trash = dict_to_pca.pop('winds')
        # change region parameter if local area is required
        if parameters[4][1]==default_region and parameters[0]==True \
             and parameters[1]==False and parameters[2]==1 and parameters[3]=='1D':
            # we load the daily pcs
            pca_data = xr.open_dataset(
                            data_path+'/cfsr/cfsr_regional_daily_pcs.nc'
                        )
            return pca_data, None
        else:
            
            if parameters[4][0]=='local':
                local_region = (True,(
                            site_location[0]-parameters[4][1][0], # new lon / lat region
                            site_location[0]+parameters[4][1][0],
                            site_location[1]-parameters[4][1][1],
                            site_location[1]+parameters[4][1][1]
                        ))
                dict_to_pca['region'] = local_region
            if site_id:
                dict_to_pca['site_id'] = site_id

            # lets first calculate the pcs
            return PCA_DynamicPred(
                            self.slp_data,pres_vars=('SLP','longitude','latitude'),
                            wind=self.wind_data if parameters[1] else None,
                            #winds=(
                            #    parameters[1], # this is winds True or False
                            #    calculate_relative_winds(
                            #        location=site_location,
                            #        uw=self.wind_data[datasets_attrs[self.predictor_data][4]],
                            #        vw=self.wind_data[datasets_attrs[self.predictor_data][5]],
                            #        lat_name=datasets_attrs[self.predictor_data][1],
                            #        lon_name=datasets_attrs[self.predictor_data][0]
                            #    ) if parameters[1] else None 
                                # this are the winds projected in site
                            #),
                            #wind_vars=('wind_proj_mask','longitude','latitude'),
                            wind_vars=('wind_proj_mask',
                                       datasets_attrs[self.predictor_data][0],
                                       datasets_attrs[self.predictor_data][1],
                                       datasets_attrs[self.predictor_data][4],
                                       datasets_attrs[self.predictor_data][5]),
                            pca_plot=(plot,False,1),verbose=self.verbose,
                            site_location=site_location,
                            pcs_folder=self.pcs_folder,
                            **dict_to_pca # extra arguments without the winds
                        ).pcs_get()


    def execute_cross_model_calculations(self, verbose: bool = False, 
                                         plot: bool = False,
                                         save_ind_sites: bool = False):
        """
        This function performs all the cross-models with the input
        parameters given in the class initializator

        Args:
            verbose (bool, optional): Wheter to print logs or not. 
                Defaults to False.
            plot (bool, optional): Wheter to plot figs or not.
                Defaults to False.

        Returns:
            2 np.array matrices: The function returns two np.arrays with
            the statistical parameters calculated for the different models
        """

        # save model stats for the different sites studied
        model_params_for_sites = []

        for isite,site in enumerate(self.ss_sites):

            # we first load in memmory the selected site
            ss_site = self.ss_data.isel(site=site)[[
                'ss','lat','lon'
            ]].load() # load the ss and the location of the site
            site_location = (ss_site.lon.values,ss_site.lat.values)

            # create the array to save all the models stats
            model_params_for_site = np.zeros(
                tuple(self.model_params_shape+[len(self.model_metrics)])
            )

            # lets iterate over all the pca_attrs + model_attrs
            if self.model=='linear':
                # loop over all the combinations for the linear model
                model_counter = 0
                for i_parameters,parameters in zip(
                    [(icg,iw,itl,itr,ir,its,ipp) \
                        for icg in [i for i in range(len(list(self.pca_attrs.values())[0]))] \
                        for iw in [i for i in range(len(list(self.pca_attrs.values())[1]))] \
                        for itl in [i for i in range(len(list(self.pca_attrs.values())[2]))] \
                        for itr in [i for i in range(len(list(self.pca_attrs.values())[3]))] \
                        for ir in [i for i in range(len(list(self.pca_attrs.values())[4]))] \
                        for its in [i for i in range(len(list(self.model_attrs.values())[0]))] \
                        for ipp in [i for i in range(len(list(self.model_attrs.values())[1]))]]
                    , [(cg,w,tl,tr,r,ts,pp) \
                        for cg in list(self.pca_attrs.values())[0] \
                        for w in list(self.pca_attrs.values())[1] \
                        for tl in list(self.pca_attrs.values())[2] \
                        for tr in list(self.pca_attrs.values())[3] \
                        for r in list(self.pca_attrs.values())[4] \
                        for ts in list(self.model_attrs.values())[0] \
                        for pp in list(self.model_attrs.values())[1]]
                ):  

                    # perform all the individual experiments
                    print(
                        '\n --------------------------------------------------------- \
                        \n\n Experiment {} in site {}, coords = {} ...... \
                        \n\n pca_params = {} \n\n linear_model_params = {} \
                        \n\n and iteration with indexes = {} \
                        \n\n ---------------------------------------------------------'.format(
                            model_counter+1, # this is just the counter
                            site, # site to analyze in this loop
                            site_location, # site coordinates
                            dict(zip(self.pca_attrs.keys(),parameters[:5])),
                            dict(zip(self.model_attrs.keys(),parameters[5:])),
                            i_parameters # this are the parameters indexes
                        ), end='\r'
                    ) if verbose else None

                    pca_data, pca_scaler = self.get_pcs(parameters=parameters,
                                                        site_id=site,
                                                        site_location=site_location,
                                                        plot=plot)                    

                    # resample ss to time_resample parameter
                    ss_site_model = ss_site.resample(time=parameters[3]).max()\
                        .dropna(dim='time',how='all') # delete NaNs

                    # and lets now calculate the linear model
                    stats, model, t_train = MultiLinear_Regression(
                        pca_data,ss_site_model,pcs_scaler=None, # add to plot slp recon
                        model_metrics=self.model_metrics,
                        X_set_var='PCs',y_set_var='ss',
                        plot_results=plot,verbose=verbose,pca_ttls=None,
                        **dict(zip(self.model_attrs.keys(),parameters[5:]))
                    )

                    # save results in matrix
                    for istat,stat in enumerate(stats):
                        model_params_for_site[
                            tuple(list(i_parameters)+[istat])
                        ] = stats[stat] # append stat to each model / site

                    # sum 1 to counter
                    model_counter += 1
            
            elif self.model=='knn':
                # loop over all the combinations for the knn model
                model_counter = 0
                for i_parameters,parameters in zip(
                    [(icg,iw,itl,itr,ir,its,ipp,ikn) \
                        for icg in [i for i in range(len(list(self.pca_attrs.values())[0]))] \
                        for iw in [i for i in range(len(list(self.pca_attrs.values())[1]))] \
                        for itl in [i for i in range(len(list(self.pca_attrs.values())[2]))] \
                        for itr in [i for i in range(len(list(self.pca_attrs.values())[3]))] \
                        for ir in [i for i in range(len(list(self.pca_attrs.values())[4]))] \
                        for its in [i for i in range(len(list(self.model_attrs.values())[0]))] \
                        for ipp in [i for i in range(len(list(self.model_attrs.values())[1]))] \
                        for ikn in [i for i in range(len(list(self.model_attrs.values())[2]))]]
                    , [(cg,w,tl,tr,r,ts,pp,kn) \
                        for cg in list(self.pca_attrs.values())[0] \
                        for w in list(self.pca_attrs.values())[1] \
                        for tl in list(self.pca_attrs.values())[2] \
                        for tr in list(self.pca_attrs.values())[3] \
                        for r in list(self.pca_attrs.values())[4] \
                        for ts in list(self.model_attrs.values())[0] \
                        for pp in list(self.model_attrs.values())[1] \
                        for kn in list(self.model_attrs.values())[2]]
                ):  

                    # perform all the individual experiments
                    print(
                        '\n --------------------------------------------------------- \
                        \n\n Experiment {} in site {} ...... \
                        \n\n pca_params = {} \n\n knn_model_params = {} \
                        \n\n and iteration with indexes = {} \
                        \n\n ---------------------------------------------------------'.format(
                            model_counter+1, # this is just the counter
                            site, # site to analyze in this loop
                            dict(zip(self.pca_attrs.keys(),parameters[:5])),
                            dict(zip(self.model_attrs.keys(),parameters[5:])),
                            i_parameters # this are the parameters indexes
                        ), end='\r'
                    ) if verbose else None

                    # Get PCs
                    pca_data, pca_scaler = self.get_pcs(parameters=parameters,
                                                        site_id=site,
                                                        site_location=site_location,
                                                        plot=plot)    
 
                    # resample ss to time_resample parameter
                    ss_site_model = ss_site.resample(time=parameters[3]).max()\
                        .dropna(dim='time',how='all') # delete NaNs

                    # and lets now calculate the knn model
                    stats, model, t_train = KNN_Regression(
                        pca_data,ss_site_model,pcs_scaler=None, # add to plot slp recon
                        model_metrics=self.model_metrics,
                        X_set_var='PCs',y_set_var='ss',
                        plot_results=plot,verbose=verbose,pca_ttls=None,
                        **dict(zip(self.model_attrs.keys(),parameters[5:]))
                    )

                    # save results in matrix
                    for istat,stat in enumerate(stats):
                        model_params_for_site[
                            tuple(list(i_parameters)+[istat])
                        ] = stats[stat] # append stat to each model / site
                    
                    # sum 1 to counter
                    model_counter += 1

            elif self.model=='xgboost':
                # loop over all the combinations for the xgboost model
                model_counter = 0
                for i_parameters,parameters in zip(
                    [(icg,iw,itl,itr,ir,its,ipp,ine,imd,ims,ilr,ilf) \
                        for icg in [i for i in range(len(list(self.pca_attrs.values())[0]))] \
                        for iw in [i for i in range(len(list(self.pca_attrs.values())[1]))] \
                        for itl in [i for i in range(len(list(self.pca_attrs.values())[2]))] \
                        for itr in [i for i in range(len(list(self.pca_attrs.values())[3]))] \
                        for ir in [i for i in range(len(list(self.pca_attrs.values())[4]))] \
                        for its in [i for i in range(len(list(self.model_attrs.values())[0]))] \
                        for ipp in [i for i in range(len(list(self.model_attrs.values())[1]))] \
                        for ine in [i for i in range(len(list(self.model_attrs.values())[2]))] \
                        for imd in [i for i in range(len(list(self.model_attrs.values())[3]))] \
                        for ims in [i for i in range(len(list(self.model_attrs.values())[4]))] \
                        for ilr in [i for i in range(len(list(self.model_attrs.values())[5]))] \
                        for ilf in [i for i in range(len(list(self.model_attrs.values())[6]))]]
                    , [(cg,w,tl,tr,r,ts,pp,ne,md,ms,lr,lf) \
                        for cg in list(self.pca_attrs.values())[0] \
                        for w in list(self.pca_attrs.values())[1] \
                        for tl in list(self.pca_attrs.values())[2] \
                        for tr in list(self.pca_attrs.values())[3] \
                        for r in list(self.pca_attrs.values())[4] \
                        for ts in list(self.model_attrs.values())[0] \
                        for pp in list(self.model_attrs.values())[1] \
                        for ne in list(self.model_attrs.values())[2] \
                        for md in list(self.model_attrs.values())[3] \
                        for ms in list(self.model_attrs.values())[4] \
                        for lr in list(self.model_attrs.values())[5] \
                        for lf in list(self.model_attrs.values())[6]]
                ):  

                    # perform all the individual experiments
                    print(
                        '\n --------------------------------------------------------- \
                        \n\n Experiment {} in site {} ...... \
                        \n\n pca_params = {} \n\n knn_model_params = {} \
                        \n\n and iteration with indexes = {} \
                        \n\n ---------------------------------------------------------'.format(
                            model_counter+1, # this is just the counter
                            site, # site to analyze in this loop
                            dict(zip(self.pca_attrs.keys(),parameters[:5])),
                            dict(zip(self.model_attrs.keys(),parameters[5:])),
                            i_parameters # this are the parameters indexes
                        ), end='\r'
                    ) if verbose else None

                    pca_data, pca_scaler = self.get_pcs(parameters=parameters,
                                                        site_id=site,
                                                        site_location=site_location,
                                                        plot=plot)
                    
                    # resample ss to time_resample parameter
                    ss_site_model = ss_site.resample(time=parameters[3]).max()\
                        .dropna(dim='time',how='all') # delete NaNs

                    # and lets now calculate the xgboost model
                    stats, model, t_train = XGBoost_Regression(
                        pca_data,ss_site_model,pcs_scaler=None, # add to plot slp recon
                        model_metrics=self.model_metrics,
                        X_set_var='PCs',y_set_var='ss',
                        plot_results=plot,verbose=verbose,pca_ttls=None,
                        train_size=parameters[5],percentage_PCs=parameters[6],
                        xgboost_parameters=dict(zip(
                            list(self.model_attrs.keys())[2:],
                            parameters[7:]
                        )) # this is the way XGBoost_Regression will work!!
                    )

                    # save results in matrix
                    for istat,stat in enumerate(stats):
                        model_params_for_site[
                            tuple(list(i_parameters)+[istat])
                        ] = stats[stat] # append stat to each model / site
                    
                    # sum 1 to counter
                    model_counter += 1

            model_params_for_sites.append(model_params_for_site)

        # save model mean results
        model_mean_params = model_params_for_sites[0].copy()
        model_params_for_sites_rest = model_params_for_sites[1:].copy()
        for models_for_site in model_params_for_sites_rest:
            model_mean_params += models_for_site
        model_mean_params = model_mean_params / len(self.ss_sites)


        print(
            '\n ------------------------------------------------------------------- \
            \n\n All the models and in all the sites have been correctly calculated!! \
            \n\n with a final mean stats (in the sites) with the following shape... \
            \n\n BIAS, SI, RMSE, Pearson and Spearman coorelations and the R2 score... \n\n {} \
            \n\n -------------------------------------------------------------------'.format(
                model_mean_params
            )
        ) if verbose else None


        return model_params_for_sites, model_mean_params

