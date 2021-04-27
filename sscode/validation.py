# arrays and geomath
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.stats import pearsonr, spearmanr


# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .plotting.validation import qqplot, scatterplot
from .plotting.config import _fontsize_title, _fontsize_legend, _figsize, _figsize_width, _figsize_height


def compare_datasets(dataset1, dataset1_coords,
                     dataset2, dataset2_coords,
                     comparison_variables: list = [['ss','msea'],['ss','msea']],
                     time_resample = None):
    """
    Datasets quantitative and qualitative comparisons

    Args:
        dataset1 (xarray.Dataset): xarray data with lon/lat values, variables...
        dataset1_coords (tuple): this is a tuple with the name of the lon/lat variables,
            the location variable and the dataset name
            - ecample: ('lon','lat','site','Maoana v2 hindcast) for MAOANA!!
        dataset2 / dataset2_coords: same as 1
        comparison_variables (list of lists): List of tuples with variables to analyze. 
            Defaults to [['ss','msea'],['ss','msl']] -- moana/geocean
        time_resample (str, optional): Time resample, recommended. Defaults to None.
    """

    print('\n Lets compare data in {} with {}!! \n'.format(
        dataset1_coords[3],dataset2_coords[3]
    ))
    # check name coord in hindcast dataset
    if not dataset1_coords[2]:
        dataset1 = dataset1.expand_dims(dim='name').assign_coords(
            {'name':((dataset1_coords[1],dataset1_coords[0]),
             np.arange(len(dataset1[dataset1_coords[1]].values)*\
                       len(dataset1[dataset1_coords[0]].values))\
                .reshape(len(dataset1[dataset1_coords[1]].values),
                         len(dataset1[dataset1_coords[0]].values)))}
        )
        dataset1_coords = (dataset1_coords[0],dataset1_coords[1],'name',dataset1_coords[3])
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
    data1_clos_data2, min_dists = calc_closest_data2_in_data1(
        (dataset1[dataset1_coords[0]].values,
         dataset1[dataset1_coords[1]].values),
        (dataset2[dataset2_coords[0]].values,
         dataset2[dataset2_coords[1]].values)
    )
    # chech resuts
    print('\n \n TGs to analyze are: \n {} \n'.format(
        dataset2[dataset2_coords[2]].values
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
    # list to save ss stats
    ss_stats = []
    # validate/plot data1 with data2 (inside the loop)
    for istat in range(len(data1_clos_data2)):
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*1.5,_figsize_height*2.2))
        fig.subplots_adjust(hspace=0.3)
        gs = gridspec.GridSpec(nrows=2,ncols=5)
        # do the analysis for both variables
        for axi in range(len(comparison_variables[0])): # ss and msea
            # time regular plot
            ax_time = fig.add_subplot(gs[axi,0:3])
            dataset1[comparison_variables[0][axi]].isel({dataset1_coords[2]:istat}).plot(
                c='k',alpha=0.7,label=dataset1_coords[3],ax=ax_time
            )
            dataset2[comparison_variables[1][axi]].isel({dataset2_coords[2]:istat}).plot(
                c='red',alpha=0.7,linestyle='--',label=dataset2_coords[3],ax=ax_time
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
            ax_scatter = fig.add_subplot(gs[axi,3])
            scatterplot(
                dataset1.isel({dataset1_coords[2]:istat,
                    'time':times_to_scatter[1]})[comparison_variables[0][axi]].values,
                dataset2.isel({dataset2_coords[2]:istat,
                    'time':times_to_scatter[2]})[comparison_variables[1][axi]].values,
                ax=ax_scatter,c='grey',alpha=0.7,edgecolor='k',
                s=5,label='Scatter plot -- {}'.format(comparison_variables[0][axi].upper())
            )
            # qqplot for the data
            ax_qq = fig.add_subplot(gs[axi,4])
            qqplot(
                dataset1[comparison_variables[0][axi]].isel({dataset1_coords[2]:istat})\
                    .dropna(dim='time').values, 
                dataset2[comparison_variables[1][axi]].isel({dataset2_coords[2]:istat})\
                    .dropna(dim='time').values, 
                ax=ax_qq,c='red',alpha=0.6,edgecolor='k',rug=False,
                label='Q-Q plot -- {}'.format(comparison_variables[0][axi].upper())
            )
            # fig and axes decorators
            fig_title = dataset2[dataset2_coords[2]].values[istat].upper() + '\n'
            fig_title_last, stats = generate_stats(
                dataset1.isel({dataset1_coords[2]:istat,
                    'time':times_to_scatter[1]})[comparison_variables[0][axi]].values,
                dataset2.isel({dataset2_coords[2]:istat,
                    'time':times_to_scatter[2]})[comparison_variables[1][axi]].values,
                not_nan_idxs = np.where(
                    (~np.isnan(dataset1.isel({dataset1_coords[2]:istat,
                        'time':times_to_scatter[1]})[comparison_variables[0][axi]].values)) &
                    (~np.isnan(dataset2.isel({dataset2_coords[2]:istat,
                        'time':times_to_scatter[2]})[comparison_variables[1][axi]].values)))[0]
            ) # generate title with stats
            fig_title += fig_title_last
            if axi==0:
                ss_stats.append(stats) # append site stats to global
            if axi==0:
                fig.suptitle(
                    fig_title,
                    fontsize=_fontsize_title
                )
            ax_time.set_title('')
            ax_time.legend(loc='upper right',fontsize=_fontsize_legend)
            ax_scatter.legend(loc='lower right',fontsize=_fontsize_legend)
            ax_qq.legend(loc='lower right',fontsize=_fontsize_legend)
        # plot current results
        plt.show()

    return dataset1.assign_coords({'tg_names':((dataset1_coords[2]),dataset2.name.values)})\
        .assign({'bias':((dataset1_coords[2]),np.array(ss_stats)[:,0]),
                 'si':((dataset1_coords[2]),np.array(ss_stats)[:,1]),
                 'rmse':((dataset1_coords[2]),np.array(ss_stats)[:,2]),
                 'pearson':((dataset1_coords[2]),np.array(ss_stats)[:,3]),
                 'spearman':((dataset1_coords[2]),np.array(ss_stats)[:,4])}), ss_stats


def generate_stats(data1, data2, not_nan_idxs=None):
    """
    Generates the title given two datasets
    - BIAS, SI and RMSE, and correlations!!

    """

    # calculate statistics
    biasd = bias(data1[not_nan_idxs],data2[not_nan_idxs])
    sid = si(data1[not_nan_idxs],data2[not_nan_idxs])
    rmsed = rmse(data1[not_nan_idxs],data2[not_nan_idxs])
    pearsond = pearsonr(data1[not_nan_idxs],data2[not_nan_idxs])[0]
    spearmand = spearmanr(data1[not_nan_idxs],data2[not_nan_idxs])[0]
    return_title = 'Data comparison is -- BIAS: {0:.2f}, SI: {0:.2f}, RMSE: {0:.2f}'.format(
        biasd,sid,rmsed
    )
    return_title += ' and Correlation (Pearson, Spearman): ({0:.2f}, {0:.2f})'.format(
        pearsond,spearmand
    )

    return return_title, [biasd,sid,rmsed,pearsond,spearmand]


def si(predictions,targets):
    pred_mean = np.nanmean(predictions)
    tar_mean = np.nanmean(targets)
    return np.sqrt(np.nansum(((predictions-pred_mean)-(targets-tar_mean))**2)/((np.nansum(targets**2))))

def rmse(predictions,targets):
    return np.sqrt(np.nanmean((targets-predictions)**2))

def bias(predictions,targets):
    return np.nanmean(targets-predictions)


def calc_closest_data2_in_data1(data1, data2, 
                                min_dist_th: float = 50):
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
    # loop over all the stations
    for d2lon,d2lat in zip(*data2):
        site = -1
        min_dist = min_dist_th
        site_counter = 0 # re-create counter
        d2lon = d2lon if d2lon<=180 else d2lon-360 # for geopy
        for d1lon,d1lat in zip(*data1):
            d1lon = d1lon if d1lon<=180 else d1lon-360 # for geopy
            dist = geodesic((d1lat,d1lon),(d2lat,d2lon)).km
            if dist<min_dist:
                site = site_counter
                min_dist = dist
            site_counter += 1 # add 1 to counter
        # save results
        sites_list.append(site), sites_dist.append(min_dist)

    return sites_list, sites_dist

