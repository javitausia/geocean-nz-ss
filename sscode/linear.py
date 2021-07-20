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
from .plotting.config import _figsize, _fontsize_title, _figsize_width, \
    _figsize_height, _fontsize_legend, real_obs_col, pred_val_col
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats, validata_w_tgs


def MultiLinear_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,None),
    model_metrics: list = [
        'bias','si','rmse','pearson','spearman','rscore',
        'mae', 'me', 'expl_var', # ...
    ], X_set_var: str = 'PCs', y_set_var: str = 'ss',
    train_size: float = 0.9, percentage_PCs: float = 0.95,
    plot_results: bool = False, verbose: bool = True,
    pca_ttls = None):

    """
    Multilinear regression analysis to perform over the PCs,
    to predict the storm surge in all the requested locations

    Args:
        X_set (xarray.Predictor): This is the predictor, usually the PCs
        y_set (xarray.Predictand): This is the predictand, usually the SS
        pcs_scaler (sklearn.LinReg, optional): This is the pcs scaler to
            re-standarize the data. Defaults to None.
        validator (tuple, optional): This is the optional tuple to validate
            the data if required. Defaults to (False,None,None), but an
            example is (True,xarray.Validator(ss),'ss')
        X_set_var (str): This is the predictor var name. Defaults to 'PCs'.
        y_set_var (str): This is the predictand var name. Defaults to 'ss'.
        train_size (float, optional): Training set size out of 1. Defaults to 0.8.
        percentage_PCs (float, optional): Percentage of PCs to predict. Defaults to 0.9.
        plot_results (bool, optional): Wheter to plot the results or not. 
            Defaults to False.
        verbose (bool, optional): Indicator of prints. Defaults to True.
        pca_ttls: This is the title for the PCA plots.

    Returns:
        [list]: This is the list with the stats for each linear model
    """
    
    # check nan existance
    X_data = X_set[X_set_var].dropna(dim='time')
    y_data = y_set[y_set_var].dropna(dim='time')

    # check time coherence
    common_times = np.intersect1d(
        pd.to_datetime(X_data.time.values).round('H'),
        pd.to_datetime(y_data.time.values).round('H'),
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
        num_pcs = num_pcs if num_pcs>2 else 3
        print('\n {} PCs ({} expl. variance) will be used to train the model!! \n'.format(
            num_pcs,percentage_PCs)
        ) if verbose else None
        # plot the slp reconstruction
        if pcs_scaler:
            plot_recon_pcs(X_set,pcs_scaler, # check
                           n_pcs_recon=num_pcs,
                           return_slp=False,
                           region=default_region,
                           pca_ttls=pca_ttls
            )
        # select pcs to train the model
        X = X[:,:num_pcs]
    else:
        X = X[:,:50] # default number of PCs
        
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
    title, stats = generate_stats(y_test,prediction,metrics=model_metrics)
    stats['rscore'] = lm.score(X_test,y_test) \
        if 'rscore' not in list(stats.keys()) else stats['rscore']
    title += '\n R score: {} -- in TEST data'.format(
        round(stats['rscore'],2)
    )

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
        gs = gridspec.GridSpec(nrows=1,ncols=3)
        # time regular plot
        ax_time = fig.add_subplot(gs[:,:2])
        ax_time.plot(t_test,y_test,label='Numerical model data',c=real_obs_col)
        ax_time.plot(t_test,prediction,label='Linear model predictions',
                     c=pred_val_col,linestyle='--')
        ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[:,2:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(y_test,prediction,ax=ax_vali)
        qqplot(y_test,prediction,ax=ax_vali)
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.15
        )
        # show the results
        plt.show()
    # print results
    print(title) if verbose else None

    # validate the model results
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs],
            validator[1][validator[2]].dropna(dim='time'),
            lm, # pre-trained linear model
            str(validator[1].name.values)
        )

    return stats

