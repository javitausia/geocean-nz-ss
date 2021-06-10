# arrays
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec

# custom
from .mda import maxdiss_simplified_no_threshold
from .plotting.mda import Plot_MDA_Data
from .rbf import rbf_reconstruction, rbf_validation
from .plotting.config import _figsize, _figsize_width, _figsize_height, \
    _fontsize_legend, _fontsize_title
from .validation import generate_stats
from .plotting.validation import scatterplot, qqplot


def MDA_RBF_Model(pcs_data, ss_data, # check the ipynb to understand
                  percentage_pcs_ini: list = [0.6,0.9],
                  num_samples_ini: list = [100,500],
                  try_all: bool = False, append_extremes: int = 10,
                  validate_rbf_kfold: bool = False,
                  plot: bool = True, verbose: bool = True):

    # TODO: add docstring

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
        predictor_dataset = pcs_data.PCs.isel(
            n_components=slice(0,num_pcs)
        ).dropna(dim='time',how='all')

        # get common times
        common_times = np.intersect1d(
            predictor_dataset.time.values,
            ss_data.time.values
        )

        # get predictor and target with common times
        # prepare datasets
        predictor_dataset = predictor_dataset.sel(time=common_times)
        target_dataset = ss_data.sel(time=common_times)
        # use MDA to generate a demo dataset and subset for RBF interpolation
        pcs_to_mda = 4
        ix_scalar_pred_mda = list(np.arange(pcs_to_mda+1))
        ix_directional_pred = []
        # perform the mda analysis (this is the mda input data)
        mda_dataset = np.concatenate(
            [predictor_dataset[:,:pcs_to_mda].values,
             target_dataset.values.reshape(-1,1)
            ],axis=1
        )
        # MDA algorithm
        predictor_subset_red, subset_indexes = maxdiss_simplified_no_threshold(
            mda_dataset, n_samp, ix_scalar_pred_mda, ix_directional_pred, log=verbose
        )

        if plot:
            fig = Plot_MDA_Data(
                pd.DataFrame(
                    mda_dataset,columns=['PC'+str(i+1) for i in range(pcs_to_mda)]+['SS']
                ),
                pd.DataFrame(
                    predictor_subset_red,columns=['PC'+str(i+1) for i in range(pcs_to_mda)]+['SS']
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
            ))

        # get subsets with calculated indexes (pcs + ss)
        predictor_subset = predictor_dataset.sel(
            time=predictor_dataset.time.values[subset_indexes]
        )
        target_subset = target_dataset.sel(
            time=predictor_dataset.time.values[subset_indexes]
        )
        idxs_not_in_subset = []
        for i in range(len(target_dataset.values)):
            idxs_not_in_subset.append(i) if i not in subset_indexes else None

        # plot ss subset and predictor
        if plot:
            fig, ax = plt.subplots(figsize=_figsize)
            target_dataset.plot(ax=ax,c='k',alpha=0.8)
            ax.plot(
                target_dataset.time.values[subset_indexes],
                target_subset.values,'.',
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
        ix_scalar_t = [0]
        ix_directional_t = []

        # RBF reconstrution
        out = rbf_reconstruction(
            predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
            target_subset.values.reshape(-1,1), ix_scalar_t, ix_directional_t,
            predictor_dataset.values
        )

        # RBF Validation: using k-fold mean squared error methodology
        if validate_rbf_kfold:
            test = rbf_validation(
                predictor_subset.values, ix_scalar_pred_rbf, ix_directional_pred,
                target_subset.values.reshape(-1,1), ix_scalar_t, ix_directional_t,
                n_splits=3, shuffle=True,
            )

        # plot output results
        if plot:
            # figure spec-grid
            fig = plt.figure(figsize=(_figsize_width*4.5,_figsize_height))
            gs = gridspec.GridSpec(nrows=1,ncols=3)
            # time regular plot
            ax_time = fig.add_subplot(gs[:,:2])
            target_dataset.isel(time=idxs_not_in_subset).plot(
                ax=ax_time,alpha=0.8,c='k',label='Real SS observations'
            )
            ax_time.plot(
                common_times[idxs_not_in_subset],
                out.reshape(-1)[idxs_not_in_subset],
                alpha=0.8,c='red',linestyle='--',
                label='RBF predictions'
            )
            ax_time.legend(fontsize=_fontsize_legend)
            # validation plot
            ax_vali = fig.add_subplot(gs[:,2:])
            ax_vali.set_xlabel('Observation')
            ax_vali.set_ylabel('Prediction')
            scatterplot(
                target_dataset.values[idxs_not_in_subset],
                out.reshape(-1)[idxs_not_in_subset],
                ax=ax_vali,c='grey',edgecolor='k'
            )
            qqplot(
                target_dataset.values[idxs_not_in_subset],
                out.reshape(-1)[idxs_not_in_subset],
                ax=ax_vali,c='red',edgecolor='orange'
            )
            # add title
            title, ttl_stats = generate_stats(
                target_dataset.values[idxs_not_in_subset],
                out.reshape(-1)[idxs_not_in_subset],
            )
            fig.suptitle(
                title,fontsize=_fontsize_title,y=1.15
            )

        # show the results
        plt.show()

    return out

