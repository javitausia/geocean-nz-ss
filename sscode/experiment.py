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
from .config import data_path, \
    default_region, default_region_reduced, \
    default_evaluation_metrics, default_ext_quantile # get config params
from .data import datasets_attrs
from .utils import calculate_relative_winds
from .pca import PCA_DynamicPred
from .linear import MultiLinear_Regression
from .knn import KNN_Regression
from .xgboost import XGBoost_Regression


sites_to_analyze = np.unique( # closest Moana v2 Hindcast to tidal gauges
    [ 689,328,393,1327,393,480,999,116,224,1124,949,708, # UHSLC
      1296,378,1124,780,613,488,1442,1217,578,200,1177,1025,689,949,224,1146, # LINZ
      1174,1260,1217,744,1064,1214,803,999 # OTHER (ports...)
    ]
) # these are the default sites to analyze
pca_attrs_exp = {
    'calculate_gradient': [False,True],
    'winds': [False,True],
    'time_lapse': [1,2,3], # 1 equals to NO time delay 
    'time_resample': ['6H','12H','1D'], # 6H and 12H available...
    'region': [('local',(1.5,1.5)),('local',(2.5,2.5)),(True,default_region_reduced)]
} # these are the pca attributes to analyze
linear_attrs_exp = {
    'train_size': [0.7], 'percentage_PCs': [0.98]
} # these are the basic linear regression attributes to analyze
knn_attrs_exp = {
    'train_size': [0.7], 'percentage_PCs': [0.98],
    'k_neighbors': np.arange(1,50,1) # None/0 calculates the optimum k-neighs
} # these are the basic knn attributes to analyze
xgboost_attrs_exp = {
    'train_size': [0.7], 'percentage_PCs': [0.98],
    'n_estimators': [50], 'max_depth': [6,12,18],
    'min_samples_split': [0.02,0.06,0.1],
    'learning_rate': [0.1], 'loss': ['ls'] # more could be added
} # these are the basic xgboost attributes to analyze


class Experiment(object):
    """
    This class Experiment summarizes all the previous work done with the linear, knn and the
    xgboost models, as this class allows the user to perform a detailed analysis of one
    requested model given a set of parameters
    """

    def __init__(self, slp_data, wind_data, # this is the predictor
                 ss_data, # this must have several stations
                 sites_to_analyze: list = sites_to_analyze,
                 model: str = 'linear', # this is the model to analyze
                 model_metrics: list = default_evaluation_metrics,
                 ext_quantile: tuple = default_ext_quantile,
                 pca_attrs: dict = pca_attrs_exp,
                 model_attrs: dict = linear_attrs_exp,
                 pcs_folder: str = data_path+'/pcs_experiments',
                 verbose: bool = True):
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
            sites_to_analyze (list/array, optional): This is the list with all the moana v2
                hindcast locations to analyze. Defaults to random locations.
            model (str, optional): Type of model to analyze. Defaults to 'linear'.
            model_metrics (list, optional): These are all the evaluation metrics that might
                be used to evaluate the model performance. Defaults to default_evaluation_metrics.
            ext_quantile (tuple, optional): These are the exterior quantiles to be used
                in the case extreme analysis will be performed when calculating the model
                performance metrics. Defaults to default_ext_quantile.
                    - see the config.py file for more information**
            pca_attrs (dict, optional): PCA dictionary with all the parameters to use in pca.
                Defaults to pca_attrs_exp.
            model_attrs (dict, optional): Model dictionary with all the parameters to use. 
                Defaults to linear_attrs_exp.
            pcs_folder (str, optional): Folder where the PCs are stored.
                Default to data_path+'/pcs'
            verbose (bool, optional): If True, prints the progress of the analysis.
                Defaults to True.
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
        self.ext_quantile = ext_quantile # quantiles analysis

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


    def get_pcs(self,
                parameters,
                site_id=None,
                site_location=None,
                plot=False):
        """
           Calculate or load the PCs corresponding to a specific configuration of
           the experiment.

           Args:
              (list)            parameters: The parameters of the experiment.
              (int)                site_id: The id of the site the pcs are required for.
              (tuple, float) site_location: The longitude and latitude of the site the pcs are required for.
              (boolean)               plot: Whether to plot the PCs.

           Returns:
              (xarray dataset): A xarray dataset containing the principal components.
              (object):         The scaler used to normalise the data before running the
                                PCA.
        """

        # these are the pca arguments to calculate the matrix
        dict_to_pca = dict(zip(list(self.pca_attrs.keys()),parameters[:5]))
        trash = dict_to_pca.pop('winds')
        print(trash)

        # change region parameter if local area is required
        if dict_to_pca['region'][0]=='local':
            local_region = (True,(
                site_location[0]-dict_to_pca['region'][1][0], # new lon / lat region
                site_location[0]+dict_to_pca['region'][1][0],
                site_location[1]-dict_to_pca['region'][1][1],
                site_location[1]+dict_to_pca['region'][1][1]
            ))
            dict_to_pca['region'] = local_region
            wind_var = 'wind_proj_mask'
            dict_to_pca['site_id'] = site_id if site_id else None
        else:
            wind_var = 'wind_magnitude'
            dict_to_pca['site_id'] = None

        # lets first calculate the pcs
        return PCA_DynamicPred(
            self.slp_data,pres_vars=('SLP','longitude','latitude'),
            wind=self.wind_data if trash else None,
            wind_vars=(wind_var, # might change
                datasets_attrs[self.predictor_data+'_winds'][0],
                datasets_attrs[self.predictor_data+'_winds'][1],
                datasets_attrs[self.predictor_data+'_winds'][4],
                datasets_attrs[self.predictor_data+'_winds'][5]),
            pca_plot=(plot,False,1),verbose=True,
            site_location=site_location,
            pcs_folder=self.pcs_folder,
            **dict_to_pca # extra arguments without the winds
        ).pcs_get()


    def execute_cross_model_calculations(self, verbose: bool = True, 
                                         plot: bool = True,
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
            
            print(f'\n Analyzing site {site}!! \n')

            # we first load in memmory the selected site
            ss_site = self.ss_data.isel(site=site)[[
                'ss','lat','lon'
            ]].load() # load the ss and the location of the site
            site_location = (ss_site.lon.values,ss_site.lat.values)

            # get ext metrics in metrics
            ext_counter = 0
            kges_counter = 0
            ext_kges_counter = 0
            for metric in self.model_metrics:
                if 'ext_' in metric and 'kge' not in metric:
                    ext_counter += 1
                if 'kge' in metric and 'ext_' not in metric:
                    kges_counter += 1
                if 'ext_' in metric and 'kge' in metric:
                    ext_kges_counter += 1

            # create the array to save all the models stats
            model_params_for_site = np.zeros(
                tuple(self.model_params_shape+\
                    [len(self.model_metrics)+\
                        ext_counter*(len(self.ext_quantile[0])-1)+\
                        kges_counter*3+\
                        ext_kges_counter*(4*len(self.ext_quantile[0])-1)
                    ]
                )
            )
            # print(f'\n Model params shape: {model_params_for_site.shape} \n')

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
                    
                    # get the pca and pcs scaler for this site
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
                        ext_quantile=self.ext_quantile,
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

                    # get the pca and pcs scaler for this site
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
                        ext_quantile=self.ext_quantile,
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

                    # get the pca and pcs scaler for this site
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
                        ext_quantile=self.ext_quantile,
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


        return model_params_for_sites, model_mean_params, stats.keys()

