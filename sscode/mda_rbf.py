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
from .pca import PCA_DynamicPred
from .mda import maxdiss_simplified_no_threshold
from .rbf import rbf_reconstruction, rbf_validation
from .validation import generate_stats, calc_closest_data2_in_data1
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_legend, _fontsize_title
from .plotting.utils import plot_ccrs_nz, get_n_colors
from .plotting.mda import Plot_MDA_Data
from .plotting.validation import scatterplot, qqplot


class MDA_RBF_Model(object):
    """
    This MDA+RBF model allows the user to perform a model which consists in
    reconstructing / interpolating the storm surge in different locations
    around New Zealand, but giving the pcs of both the sea-level-pressure
    fields and the pcs of this storm surge. This analysis will be performed
    over different locations / shores, that are indicated by the lon/lat of
    one point in the "middle" of this shore

    """

    def __init__(self, ss_data, # this could be the loaded moana with Loader class
        slp_data, # this will be used to calculate the local predictors
        ss_data_attrs: tuple = ('lon','lat','site','ss'),
        lons: list = [168.1, 171.4, 167.5, 171.1, 173.5, 173.3, 175.1, 177.9, 172.6], 
        lats: list = [-44.2, -41.9, -47.2, -45.4, -42.5, -40.7, -40.2, -40.1, -34.3],
        min_dist_th: list = [100, 100, 120, 90, 90, 80, 80, 120, 400],
        extra_help: list = [ # this helps to stay in the wanted shore of NZ
            ('lon',1.0),('lon',1.0),('lat',1.5),('lon',1.0),('lon',1.0),
            ('lat',0.5),('lat',0.5),('lon',1.5),('lon',3.5)
        ], time_resample: str = '1D'):

        """
        As always, we have the initializator/constructor fucntion, where the 
        instance of the class is created, giving some initial parameters

        Args:
            ss_data (xarray.Dataset): This is the previously loaded moana datasets
                - TODO: add square-spatial Moana v2
            slp_data (xarray.Dataset): This is the slp data, also loaded with the
                usually used, Loader class in data.py
            ss_data_attrs (tuple): This is a tuple with the main info regarding
                the ss data, for posterior calculations
            lons (list): Longitudes of the "middle" location of the shores
            lats (list): Latitudes of the "middle" location of the shores
            min_dist_th (list): Distances in km to find the nearest points to
                the lons/lats specified
            extra_help (list): This is extra information, regarding the "dimension"
                in which the shore is projected, so fake nodes are not introduced
                in the pcs analysis
            time_resample (str, optional): As usually, time resample step
        """

        # we save the lons / lats to locate the areas
        print('\n Initializing the MDA + RBF constructor... \n')
        print('\n with longitudes = {} \n \n and \n \n latitudes = {} !! \n'.format(
            lons, lats # before saving the lons/lats in the object
        ))
        self.lons, self.lats = lons, lats # save lon/lat as class attrs
        if len(lons)==len(lats):
            self.num_locs = len(lons) # TODO: add RaiseError
        self.raw_ss_data = ss_data # this could be the moana dataset
        self.ss_attrs = ss_data_attrs # moana/other, vars and coords
        self.raw_slp_data = slp_data # this could be the CFSR slp/winds
        print('\n Lets calculate the storm surge pcs... \n')
        ss_tuple_data = self.generate_ss_data(
            min_dist_th, extra_help, time_resample=time_resample
        ) # generate the lists with the ss pcs...
        # save all the returned lists
        self.ss_pcs_data, self.ss_real_data, self.ss_scalers = ss_tuple_data
        print('\n Lets calculate the slp pcs... \n')
        self.slp_pcs_data = self.generate_slp_data(
            2.2, 2.2, # lon and lat deltas
            time_resample=time_resample
        ) # extra parameters will be added


    def generate_ss_data(self, min_dist_th, extra_help,
                         time_resample: str = '1D'):
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
            time_resample (str, optional): As usually, time resample step
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
                {self.ss_attrs[2]: sites[i]}
            ).ss.load().resample(time=time_resample).max() \
            .dropna(dim='time',how='all') for i in range(self.num_locs)
        ] # num_locs==len(sites) if working

        # calculate the pcs of all the pieces
        ss_pcs_locs, ss_scalers_locs = [], [] # lists to save the data
        for loc in range(self.num_locs):
            # standarize the features
            ss_scaler = StandardScaler()
            ss_stan = ss_scaler.fit_transform(raw_ss_locs[loc].values.T)
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
        
        # plot selected points to check coherence
        fig, axes = plt.subplots(
            ncols=self.num_locs,
            figsize=(_figsize_width*self.num_locs*1.4,_figsize_height*1.2),
            subplot_kw={
                'projection':ccrs.PlateCarree(central_longitude=180)
            }
        )
        cmap, loc_colors = get_n_colors('jet',self.num_locs)
        for axi,ax in enumerate(axes.flatten()):
            ax.scatter( # total selected points for each shore
                x=self.raw_ss_data.sel(
                    {self.ss_attrs[2]: ss_pcs_locs[axi][self.ss_attrs[2]].values}
                )[self.ss_attrs[0]].values,
                y=self.raw_ss_data.sel(
                    {self.ss_attrs[2]: ss_pcs_locs[axi][self.ss_attrs[2]].values}
                )[self.ss_attrs[1]].values,
                transform=ccrs.PlateCarree(),s=30,zorder=111,
                c=np.array(loc_colors[axi],ndmin=2)
            )
            ax.scatter( # "middle" point as a star
                x=self.lons[axi],y=self.lats[axi],s=100,zorder=112,
                marker='*',c='k',transform=ccrs.PlateCarree(),edgecolors='yellow'
            )
        plot_ccrs_nz(axes.flatten(),plot_labels=(False,None,None))

        plt.show() # debug results as calculated

        return ss_pcs_locs, raw_ss_locs, ss_scalers_locs


    def generate_slp_data(self, lon_d, lat_d,
                          time_lapse: int = 2, # time back-mind, 2=t,t-1
                          time_resample: str = '1D'): # TODO: add pcs parameters
        """
        This class function generates the slp pcs analysis, so the mda+rbf
        interpolation can be performed over all the different indicated
        shores, ex: west, south...

        Args:
            lon_d (float): Delta in the longitudes to calculate the local sea-
                level-pressure pcs
            lat_d (sloat): Delta in the latitudes to calculate the local sea-
                level-pressure pcs
            time_resample (str, optional): As usually, time resample step
        """

        return [
            PCA_DynamicPred(
                self.raw_slp_data, # TODO: add wind options
                calculate_gradient=True, time_lapse=time_lapse,
                time_resample=time_resample,
                region=(True,(
                    self.lons[ipc]-lon_d,self.lons[ipc]+lon_d,
                    self.lats[ipc]+lat_d,self.lats[ipc]-lat_d
                )),
                pca_plot=(True,False,1) # (plot,scale)
            )[0] if ipc <1 else \
            PCA_DynamicPred(
                self.raw_slp_data, # TODO: add wind options
                calculate_gradient=True, time_lapse=time_lapse,
                time_resample=time_resample,
                region=(True,(
                    self.lons[ipc]-lon_d,self.lons[ipc]+lon_d,
                    self.lats[ipc]+lat_d,self.lats[ipc]-lat_d
                )), verbose=False,
                pca_plot=(False,False,2) # (plot,scale)
            )[0] for ipc in range(self.num_locs)
        ]


    def calc_MDA_RBF(self, sel_locs=None):

        """
        This is the calculation function, where given the wanted
        locations / shores, the MDA+RBF interpolation model is performed,
        plotting different results and statistics

        Args:
            sel_sites (list): List with ilocs for the shores
        """

        for sel_loc in sel_locs:
            # perform the RBF interpolation, using MDA too
            rbf_prediction, real_ss_rbf = MDA_RBF_algorithm(
                self.slp_pcs_data[sel_loc], 
                self.ss_pcs_data[sel_loc], ss_pcs=8, 
                ss_scaler=self.ss_scalers[sel_loc],
                try_all=True, append_extremes=None,
                percentage_pcs_ini=[0.9999],
                num_samples_ini=[1500]
            )
            # reconstruct and plot
            for isite, site in enumerate(
                self.ss_real_data[sel_loc].site.values[::18]
            ):
                self.ss_real_data[sel_loc].sel(site=site).plot(
                    figsize=_figsize,c='k',label='Real SS measures'
                )
                plt.plot(
                    self.ss_real_data[sel_loc].sel(site=site).time.values,
                    real_ss_rbf[0][:,isite],c='seagreen',linestyle='--',
                    label='Reconstructed SS -- MDA + RBF'
                )
                plt.xlim(
                    self.ss_real_data[sel_loc].sel(site=site).time.values[0],
                    self.ss_real_data[sel_loc].sel(site=site).time.values[-1]
                ) # delete white spaces
                title, stats = generate_stats(
                    self.ss_real_data[sel_loc].sel(site=site).values,
                    real_ss_rbf[0][:,isite]
                )
                plt.title(
                    title+' -- SITE: {}'.format(site),fontsize=_fontsize_title,y=1.1
                )
                plt.legend(ncol=2,fontsize=_fontsize_legend)
            plt.show() # show final results

        # TODO: add xarray to return results and better plots


def MDA_RBF_algorithm(
    pcs_data, ss_data, ss_pcs: int = 3,
    ss_scaler = None, # to de-standarize the pcs
    percentage_pcs_ini: list = [0.6,0.9],
    num_samples_ini: list = [100,500],
    try_all: bool = False, append_extremes: int = 10,
    validate_rbf_kfold: bool = False,
    plot: bool = True, verbose: bool = True):

    """[summary]

    Args:
        pcs_data ([type]): [description]
        ss_data ([type]): [description]
        ss_pcs (int, optional): [description]. Defaults to 1.
        ss_scaler ([type], optional): [description]. Defaults to None.
        num_samples_ini (list, optional): [description]. Defaults to [100,500].
        try_all (bool, optional): [description]. Defaults to False.
        append_extremes (int, optional): [description]. Defaults to 10.
        validate_rbf_kfold (bool, optional): [description]. Defaults to False.
        plot (bool, optional): [description]. Defaults to True.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
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
    for per_pcs,n_samp in zip(percentage_pcs,num_samples):

        print('\n MDA + RBF with {} per of pcs and {} samples!! \n'.format(
            per_pcs, n_samp
        ))

        # number of pcs to use
        num_pcs = len(np.where(
            ((np.cumsum(pcs_data.variance)/np.sum(pcs_data.variance)) < per_pcs).values==True
        )[0])
        print(' which means {} PCs... \n'.format(num_pcs))
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
                -target_dataset.values
            )[:append_extremes]
            for max_indx in max_times_indexes:
                subset_indexes.append(max_indx) if max_indx not in subset_indexes else None
            print('We finally have {} points to interpolate with RBF'.format(
                len(subset_indexes)
            )) # TODO: add for ss-PCs!!!

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
        outs.append(out)

        # RBF Validation: using k-fold mean squared error methodology
        if validate_rbf_kfold:
            test = rbf_validation(
                predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
                target_subset.values.reshape(-1,ss_pcs), ix_scalar_t, ix_directional_t,
                n_splits=3, shuffle=True,
            )

        # plot output results
        if plot:
            for i_ss in range(2):
                # figure spec-grid
                fig = plt.figure(figsize=(_figsize_width*4.5,_figsize_height))
                gs = gridspec.GridSpec(nrows=1,ncols=3)
                # time regular plot
                ax_time = fig.add_subplot(gs[:,:2])
                target_dataset.isel(time=idxs_not_in_subset,n_components=i_ss).plot(
                    ax=ax_time,alpha=0.8,c='k',label='Real SS observations'
                )
                ax_time.plot(
                    common_times[idxs_not_in_subset],
                    out[idxs_not_in_subset,i_ss],
                    alpha=0.8,c='red',linestyle='--',
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
                    out[idxs_not_in_subset,i_ss],
                    ax=ax_vali,c='grey',edgecolor='k'
                )
                qqplot(
                    target_dataset.values.reshape(-1,ss_pcs)[idxs_not_in_subset,i_ss],
                    out[idxs_not_in_subset,i_ss],
                    ax=ax_vali,c='red',edgecolor='orange'
                )
                # add title
                title, ttl_stats = generate_stats(
                    target_dataset.values.reshape(-1,ss_pcs)[idxs_not_in_subset,i_ss],
                    out[idxs_not_in_subset,i_ss],
                )
                fig.suptitle(
                    title,fontsize=_fontsize_title,y=1.15
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
            real_sss.append(real_ss)

    return outs, real_sss

