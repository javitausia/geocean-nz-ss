# arrays and geomath
import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from scipy.stats import pearsonr, spearmanr

# sklearn metrics
from sklearn.metrics import explained_variance_score, max_error, \
    mean_absolute_error, mean_squared_error, mean_squared_log_error, \
    median_absolute_error, r2_score, mean_tweedie_deviance

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_evaluation_metrics
from .plotting.validation import qqplot, scatterplot
from .plotting.config import _fontsize_title, _fontsize_legend, \
    _figsize, _figsize_width, _figsize_height, _fontsize_label, \
        real_obs_col, pred_val_col

# some custom metrics and the metrics_dictionary

def si(targets, predictions):
    pred_mean = np.nanmean(predictions)
    tar_mean = np.nanmean(targets)
    return np.sqrt(np.nansum(((predictions-pred_mean)-(targets-tar_mean))**2)/((np.nansum(targets**2))))

def rmse(targets, predictions):
    return np.sqrt(np.nanmean((targets-predictions)**2))

def ext_rmse(targets, predictions, quant_threshold=0.75):
    quant_threshold_idxs = np.where(
        targets > np.nanquantile(targets,quant_threshold)
    )[0]
    return np.sqrt(np.nanmean((targets[quant_threshold_idxs] - \
        predictions[quant_threshold_idxs])**2))

def relative_rmse(targets, predictions):
    return np.sqrt(np.nanmean((targets-predictions)**2))/np.nanquantile(targets,0.95)

def bias(targets, predictions):
    return np.nanmean(targets-predictions)

def pocid(targets, predictions):
    updown = np.where(
        (predictions[1:]-predictions[:-1])*(targets[1:]-targets[:-1])>0,1,0
    )
    return np.sum(updown)/len(updown) * 100

def tu_test(targets, predictions):
    return np.sum([(targets[i]-predictions[i])**2 for i in range(1,len(targets))])/\
        np.sum([(targets[i]-targets[i-1])**2 for i in range(1,len(targets))])


metrics_dictionary = {
    'expl_var': explained_variance_score,
    'me': max_error, # max error in data
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'medae': median_absolute_error,
    'msle': mean_squared_log_error,
    'rscore': r2_score,
    'tweedie': mean_tweedie_deviance,
    'bias': bias, 
    'si': si,
    'rmse': rmse,
    'ext_rmse': ext_rmse,
    'rel_rmse': relative_rmse, 
    'pearson': pearsonr,
    'spearman': spearmanr,
    'pocid': pocid,
    'tu_test': tu_test
} # this is the metrics dictionary with all the possible metrics

# TODO: import DESIRED metric in line 9 and append to metrics_dictionary
#       if a new metric is wanted to be used !!


def compare_datasets(dataset1, dataset1_coords, 
                     dataset2, dataset2_coords, # these 4 are loaded in Loader
                     comparison_variables: list = [['ss','msea'],['ss','msea']],
                     time_resample = None):
    """
    Datasets quantitative and qualitative comparisons

    Args:
        dataset1 (xarray.Dataset): xarray data with lon/lat values, variables...
        dataset1_coords (tuple): this is a tuple with the name of the lon/lat variables,
            the location variable and the dataset name
            - example: ('lon','lat','site','Maoana v2 hindcast') for MOANA!!
        dataset2 / dataset2_coords: same as 1
        * Data in dataset1 and dataset2 is exactly the loaded data in data.py
        comparison_variables (list of lists): List of tuples with variables to analyze. 
            Defaults to [['ss','msea'],['ss','msea']] -- moana/uhslc
        time_resample (str, optional): Time resample, recommended if not already
            performed. Defaults to None.

    Returns:
        [xarray.Dataset, dict]: Closest sites in hindcast and statistics
    """

    print('\n Lets compare data in {} with {}!! \n'.format(
        dataset1_coords[3],dataset2_coords[3]
    ))
    # check name coord in hindcast dataset
    if not dataset1_coords[2]:
        dataset1 = xr.Dataset(
            {
                'ss': (('time','site'), dataset1.values.reshape(
                    -1,len(dataset1[dataset1_coords[0]])*len(dataset1[dataset1_coords[1]]))),
                dataset1_coords[0]: (('site'), list(dataset1[dataset1_coords[0]].values)*int(
                    (len(dataset1[dataset1_coords[0]])*len(dataset1[dataset1_coords[1]]))\
                        /len(dataset1[dataset1_coords[0]]))),
                dataset1_coords[1]: (('site'), np.repeat(dataset1[dataset1_coords[1]].values,
                    (len(dataset1[dataset1_coords[0]])*len(dataset1[dataset1_coords[1]]))\
                        /len(dataset1[dataset1_coords[1]])))
            }, coords={
                'site': np.arange(len(dataset1[dataset1_coords[0]])*len(dataset1[dataset1_coords[1]])),
                'time': dataset1.time.values
            }
        ).dropna(dim='site',how='all')
        # dataset1 = dataset1.expand_dims(dim='name').assign_coords(
        #     {'name':((dataset1_coords[1],dataset1_coords[0]),
        #      np.arange(len(dataset1[dataset1_coords[1]].values)*\
        #                len(dataset1[dataset1_coords[0]].values))\
        #         .reshape(len(dataset1[dataset1_coords[1]].values),
        #                  len(dataset1[dataset1_coords[0]].values)))}
        # )
        dataset1_coords = (dataset1_coords[0],dataset1_coords[1],'site',dataset1_coords[3])

    # resample if requested
    if time_resample:
        print('\n resampling to {}... \n'.format(time_resample))
        # check if lat/lon in coords
        if dataset1_coords[0] in [coords1 for coords1 in dataset1.coords]:
            dataset1 = dataset1[comparison_variables[0]].resample(time=time_resample).max()
        else:
            dataset1 = dataset1.assign_coords({
                dataset1_coords[0]:((dataset1_coords[2]),dataset1[dataset1_coords[0]].values),
                dataset1_coords[1]:((dataset1_coords[2]),dataset1[dataset1_coords[1]].values)
            })
            dataset1 = dataset1[comparison_variables[0]].resample(time=time_resample).max()
        if dataset2_coords[0] in [coords2 for coords2 in dataset2.coords]:
            dataset2 = dataset2[comparison_variables[1]].resample(time=time_resample).max()
        else:
            dataset2 = dataset2.assign_coords({
                dataset2_coords[0]:((dataset2_coords[2]),dataset2[dataset2_coords[0]].values),
                dataset2_coords[1]:((dataset2_coords[2]),dataset2[dataset2_coords[1]].values)
            })
            dataset2 = dataset2[comparison_variables[1]].resample(time=time_resample).max()
        print('\n resampled data: \n \n {} \n \n {}'.format(dataset1,dataset2))

    # we first extract closest stations (calculate data1 closest to data2)
    data1_clos_data2s, min_distss = calc_closest_data2_in_data1(
        (dataset1[dataset1_coords[0]].values,
         dataset1[dataset1_coords[1]].values),
        (dataset2[dataset2_coords[0]].values,
         dataset2[dataset2_coords[1]].values)
    )
    data1_clos_data2, min_dists = [], []
    data2_with_vali = []
    for i_site in range(len(data1_clos_data2s)):
        try:
            min_dist_pos = np.argmin(min_distss[i_site])
            data1_clos_data2.append(data1_clos_data2s[i_site][min_dist_pos])
            min_dists.append(min_distss[i_site][min_dist_pos])
            data2_with_vali.append(i_site)
        except:
            continue

    # chech resuts
    print('\n \n TGs to analyze are: \n {} \n'.format(
        dataset2[dataset2_coords[2]].values[
            data2_with_vali # maybe some tgs have not closest nodes
        ]
    ))
    print('\n which correspond to \n {} \n in {} \n'.format(
        data1_clos_data2,dataset1_coords[3]   
    ))
    print('\n calculated min distances (km) in {} to {}: \n {} \n'.format(
        dataset1_coords[3],dataset2_coords[3],min_dists
    ))
    # reduce stations to closest (in data1)
    dataset1 = dataset1.isel(
        {dataset1_coords[2]:data1_clos_data2}
    )
    dataset2 = dataset2.isel(
        {dataset2_coords[2]:data2_with_vali}
    )
    # list to save ss stats
    ss_stats = []
    # validate/plot data1 with data2 (inside the loop)
    for istat in range(len(data1_clos_data2)):
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*5.0,
            _figsize_height*len(comparison_variables[0])))
        gs = gridspec.GridSpec(nrows=len(comparison_variables[0]),ncols=3)
        # do the analysis for both variables
        for axi in range(len(comparison_variables[0])): # ss and msea
            # time regular plot
            ax_time = fig.add_subplot(gs[axi,0:2])
            dataset1[comparison_variables[0][axi]].isel({dataset1_coords[2]:istat}).plot(
                c=real_obs_col,alpha=0.7,label=dataset1_coords[3],ax=ax_time
            )
            dataset2[comparison_variables[1][axi]].isel({dataset2_coords[2]:istat}).plot(
                c=pred_val_col,alpha=0.7,linestyle='--',label=dataset2_coords[3],ax=ax_time
            )
            # calculate similar times for the scatter plot
            times_to_scatter = np.intersect1d(
                pd.to_datetime(dataset1.isel({dataset1_coords[2]:istat})\
                    .time.dropna(dim='time').values).round('H').values,
                pd.to_datetime(dataset2.isel({dataset2_coords[2]:istat})\
                    .time.dropna(dim='time').values).round('H').values,
                return_indices=True
            )
            # scatter plot for the data
            ax_vali = fig.add_subplot(gs[axi,2:])
            scatterplot(
                dataset1.isel({dataset1_coords[2]:istat,
                    'time':times_to_scatter[1]})[comparison_variables[0][axi]].values,
                dataset2.isel({dataset2_coords[2]:istat,
                    'time':times_to_scatter[2]})[comparison_variables[1][axi]].values,
                ax=ax_vali,alpha=0.7,s=5,label='Scatter plot -- {}'.format(
                    comparison_variables[0][axi].upper()
                ), density=False, c='grey'
            )
            # qqplot for the data
            qqplot(
                dataset1[comparison_variables[0][axi]].isel({dataset1_coords[2]:istat})\
                    .dropna(dim='time').values, 
                dataset2[comparison_variables[1][axi]].isel({dataset2_coords[2]:istat})\
                    .dropna(dim='time').values, 
                ax=ax_vali,alpha=0.6,rug=False,label='Q-Q plot -- {}'.format(
                    comparison_variables[0][axi].upper()
                )
            )
            # fig and axes decorators
            fig_title = dataset2[dataset2_coords[2]].values[istat].upper() + '\n'
            fig_title_last, stats = generate_stats(
                dataset1.isel({dataset1_coords[2]:istat,
                    'time':times_to_scatter[1]})[comparison_variables[0][axi]].values,
                dataset2.isel({dataset2_coords[2]:istat,
                    'time':times_to_scatter[2]})[comparison_variables[1][axi]].values,
                metrics=['bias','si','rmse','pearson','rscore'],
                not_nan_idxs = np.where(
                    (~np.isnan(dataset1.isel({dataset1_coords[2]:istat,
                        'time':times_to_scatter[1]})[comparison_variables[0][axi]].values)) &
                    (~np.isnan(dataset2.isel({dataset2_coords[2]:istat,
                        'time':times_to_scatter[2]})[comparison_variables[1][axi]].values)))[0]
            ) # generate title with stats
            fig_title += fig_title_last
            if axi==0:
                ss_stats.append(stats) # append site stats to global
                fig.suptitle(
                    fig_title,
                    fontsize=_fontsize_title,
                    y=1.01 if len(comparison_variables[0])>1 else 1.2
                )
            ax_time.set_title('')
            ax_time.legend(loc='upper right',fontsize=_fontsize_legend)
            ax_vali.legend(loc='lower right',fontsize=_fontsize_legend)
            ax_vali.set_xlabel('Observation',fontsize=_fontsize_label)
            ax_vali.set_ylabel('Prediction',fontsize=_fontsize_label)
        # plot current results
        plt.show()

    return dataset1.assign_coords({'tg_names':((dataset1_coords[2]),dataset2.name.values)})\
        , dataset2, ss_stats


def generate_stats(data1, data2, # these are just the 1-d numpy arrays
                   metrics: list = ['expl_var','mae','mse','me',
                       'medae','tweedie', # check theory
                       'ext_mae','ext_mse','ext_rmse','ext_pearson',
                       'bias','si','rmse','rel_rmse','pearson','spearman','rscore'
                   ], ext_quantile: tuple = (0.9,0),
                   not_nan_idxs=None):
    """
    Generates the title given two datasets
    - BIAS, SI, RMSE ... and correlations!!

    *** The ext_quantile parameter refers to the quantile to evaluate the
        models performance in the extremes, and the second element of the
        tuple refers to the dataset over the quantile will be calculated,
        0 refers to data1, and data2 otherwise
    ***

    Returns:
        [dict]: A python dictionary with all the metrics

    """

    metrics_dict = {} # empty dict for all the metrics

    # add bias, si, rmse, pearson and spearman...
    for mta in default_evaluation_metrics: # defined in sscode/config.py
        metrics.append(mta) if mta not in metrics else None

    try:
        data1 = data1[not_nan_idxs].reshape(-1)
        data2 = data2[not_nan_idxs].reshape(-1)
    except:
        print('\n Not NaN indexes not passed!! \n') if True else None

    if ext_quantile[0]:
        ext_idxs = np.where(
            data1>np.nanquantile(data1,ext_quantile[0])
        )[0] if ext_quantile[1]==0 else \
            np.where(
               data2>np.nanquantile(data2,ext_quantile[0])
            )[0]

    # calculate metrics
    for metric in metrics:
        if metric=='pearson' or metric=='spearman':
            metrics_dict[metric] = metrics_dictionary[metric](
                data1,data2
            )[0]
        elif metric[:4]=='ext_':
            metrics_dict[metric] = metrics_dictionary[metric[4:]](
                data1[ext_idxs],data2[ext_idxs]
            ) if metric[4:]!='pearson' and metric[4:]!='spearman' else \
                metrics_dictionary[metric[4:]](
                    data1[ext_idxs],data2[ext_idxs]
                )[0]
        else:
            metrics_dict[metric] = metrics_dictionary[metric](
                data1,data2
            )

    # customize title
    return_title = 'Data comparison is   --   BIAS: {:.2f}, SI: {:.2f}, Relative - RMSE: {:.2f}'.format(
        metrics_dict['bias'],metrics_dict['si'],metrics_dict['rel_rmse']
    )
    return_title += '\n and Correlations (Pearson, Rscore): ({:.2f}, {:.2f})'.format(
        metrics_dict['pearson'],metrics_dict['rscore']
    )

    return return_title, metrics_dict


def calc_closest_data2_in_data1(data1, data2, # data 1 is bigger than data2
                                min_dist_th: float = 100,
                                extra_help: tuple = ('lon',1.5)):
    """
    This function calculates the closest stations of data1 to data2

    Args:
        data1 (tuple): this is a tuple of numpy arrays with lon/lat
        data2 (tuple): this is a tuple of numpy arrays with lon/lat
        min_dist_th (float, optional): Max distance between stations. 
            Defaults to 20.

    Returns:
        [list]: Two lists with the closest stations in data1 to data2
    """

    # lists to save the results
    sites_list = [] # sites isel location in data1
    sites_dist = [] # sites distance between data1 and data2
    # check min_dist and helper format
    if type(min_dist_th)==int:
        min_dist_th = [min_dist_th]*len(data2[0])
    if type(extra_help)==tuple:
        extra_help = [extra_help]*len(data2[0])
    # loop over all the stations
    i_station = 0
    for d2lon,d2lat in zip(*data2):
        # site = -1
        # min_dist = min_dist_th
        site_counter = 0 # re-create counter
        closest_sites = []
        closets_dists = []
        d2lon = d2lon if d2lon<=180 else d2lon-360 # for geopy
        for d1lon,d1lat in zip(*data1):
            d1lon = d1lon if d1lon<=180 else d1lon-360 # for geopy
            dist = geodesic((d1lat,d1lon),(d2lat,d2lon)).km
            if extra_help[i_station][0]=='lon':
                cond = True if abs(d1lon-d2lon)<extra_help[i_station][1] else False
            elif extra_help[i_station][0]=='lat':
                cond = True if abs(d1lat-d2lat)<extra_help[i_station][1] else False
            else:
                cond = False # if no extra help is provided
            if dist<min_dist_th[i_station] and cond:
                # site = site_counter
                # min_dist = dist
                closest_sites.append(site_counter)
                closets_dists.append(dist)
            site_counter += 1 # add 1 to counter
        # save results
        sites_list.append(closest_sites), sites_dist.append(closets_dists)
        i_station += 1

    return sites_list, sites_dist
    

# function to extract the series to perform GEV analysis
def extract_time_series(moana_data,uhslc_data,lltype='list'):
    if lltype=='list':
        tot_lons = moana_data.lon.values
        tot_lats = moana_data.lat.values
    else:
        tot_lons = np.repeat(
            moana_data.lon.values.reshape(-1,1),
            len(moana_data.lat.values),
            axis=1
        ).T.reshape(-1)
        tot_lats = np.repeat(moana_data.lat.values,len(moana_data.lon.values))
    sites, dists = calc_closest_data2_in_data1(
        (tot_lons,tot_lats),
        (uhslc_data.longitude.values,
         uhslc_data.validator.latitude.values
        ),
        min_dist_th=15
    )
    sites = [site[np.argmin(dist)] for site,dist in zip(
        sites,dists
    )]
    
    return tot_lons[sites],tot_lats[sites],sites,dists


def validata_w_tgs(X,validator,model,tg_name,
                   plot_results: bool = True):
    """
    Validate linear regression with the TGs

    Args:
        X (pcs/predictor): This is an xarray with the predictor in the model
        validator (xarray): This is the validator that will be used
        model (sklearn.Model): This is the sklearn model used
    """

    # check time coherence
    common_times = np.intersect1d(
        pd.to_datetime(X.time.values).round('H'),
        pd.to_datetime(validator.time.values).round('H'),
        return_indices=True
    )
    
    # prepare X and y arrays
    X = X.isel(time=common_times[1]).values
    validator = validator.isel(time=common_times[2]).values

    # perform the prediction
    prediction = model.predict(X)

    # check model results
    title, stats = generate_stats(validator,prediction)
    title += '\n R score: {} -- UHSLC TGs -- at {}'.format(
        round(model.score(X,validator),2),
        tg_name # output tg name
    )

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
        gs = gridspec.GridSpec(nrows=1,ncols=3)
        # time regular plot
        ax_time = fig.add_subplot(gs[:,:2])
        ax_time.plot(
            common_times[0],validator,c=real_obs_col,
            label='UHSLC tgs validator'
        )
        ax_time.plot(
            common_times[0],prediction,
            label='Linear model predictions',
            c=pred_val_col,linestyle='--',alpha=0.5
        )
        ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[:,2:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(validator,prediction,ax=ax_vali)
        qqplot(validator,prediction,ax=ax_vali)
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.15
        )
        # show the results
        plt.show()

