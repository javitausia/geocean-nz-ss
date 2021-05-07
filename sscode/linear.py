# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_region
from .plotting.config import _figsize, _fontsize_title, _figsize_width, _figsize_height, _fontsize_legend
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats


def MultiLinear_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,'ss'),
    X_set_var: str = 'PCs', y_set_var: str = 'ss',
    train_size: float = 0.8,
    percentage_PCs: float = 0.9,
    plot_results: bool = False,
    region: tuple = default_region):

    # TODO: add docstring
    
    # check nan existance
    X_data = X_set[X_set_var].dropna(dim='time')
    y_data = y_set[y_set_var].dropna(dim='time')

    # check time coherence
    common_times = np.intersect1d(
        pd.to_datetime(X_data.time.values),
        pd.to_datetime(y_data.time.values),
        return_indices=True
    )
    
    # prepare X and y arrays
    X = X_data.isel(time=common_times[1]).values
    y = y_data.isel(time=common_times[2]).values

    # number of PCs to use
    if percentage_PCs:
        num_pcs = int(
            np.where(np.cumsum(X_set.variance.values/\
                np.sum(X_set.variance.values))>percentage_PCs
                )[0][0]
        ) +1 # number of Pcs to use
        # print('\n {} PCs ({} expl. variance) will be used to train the model!! \n'.format(
        #    num_pcs,percentage_PCs)
        # )
        # plot the slp reconstruction
        if pcs_scaler:
            plot_recon_pcs(X_set,pcs_scaler,
                           n_pcs_recon=num_pcs,
                           return_slp=False,
                           region=region
            )
        # select pcs to train the model
        X = X[:,:num_pcs]
        
    # split predictors into train and test
    X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split(
        X, y, common_times[0], train_size=train_size, 
        random_state=88, shuffle=False
    )

    # perform the linear regression
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train)
    prediction = lm.predict(X_test)

    # check model results
    title, stats = generate_stats(y_test,prediction)
    title += '\n R score: {} -- in TEST data'.format(
        round(lm.score(X_test,y_test),2)
    )

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*4.6,_figsize_height*0.8))
        gs = gridspec.GridSpec(nrows=1,ncols=4)
        # time regular plot
        ax_time = fig.add_subplot(gs[0,0:3])
        ax_time.plot(t_test,y_test,label='Numerical model data',c='k')
        ax_time.plot(t_test,prediction,label='Linear model predictions',
                     c='red',linestyle='--')
        ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[0,3:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(y_test,prediction,ax=ax_vali,c='grey',edgecolor='k')
        qqplot(y_test,prediction,ax=ax_vali,c='red',edgecolor='orange')
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.15
        )
        # show the results
        plt.show()
    # else:
    #     print(title)

    # validate the model results
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs],
            validator[1][validator[2]].dropna(dim='time'),
            lm # pre-trained linear model
        )

    return stats


def validata_w_tgs(X,validator,lm,
                   plot_results: bool = True):

    # TODO: add docstring

    # check time coherence
    common_times = np.intersect1d(
        pd.to_datetime(X.time.values),
        pd.to_datetime(validator.time.values),
        return_indices=True
    )
    
    # prepare X and y arrays
    X = X.isel(time=common_times[1]).values
    validator = validator.isel(time=common_times[2]).values

    # perform the prediction
    prediction = lm.predict(X)

    # check model results
    title, stats = generate_stats(validator,prediction)
    title += '\n R score: {} -- UHSLC TGs'.format(
        round(lm.score(X,validator),2)
    )

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*4.6,_figsize_height*0.8))
        gs = gridspec.GridSpec(nrows=1,ncols=4)
        # time regular plot
        ax_time = fig.add_subplot(gs[0,0:3])
        ax_time.plot(common_times[0],validator,label='UHSLC tgs validators',c='k')
        ax_time.plot(common_times[0],prediction,label='Linear model predictions',
                     c='red',linestyle='--',alpha=0.5)
        ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[0,3:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(validator,prediction,ax=ax_vali,c='grey',edgecolor='k')
        qqplot(validator,prediction,ax=ax_vali,c='red',edgecolor='orange')
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.15
        )
        # show the results
        plt.show()

