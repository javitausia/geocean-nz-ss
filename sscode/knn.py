# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_region, default_evaluation_metrics, default_ext_quantile
from .plotting.config import _figsize, _fontsize_title, _figsize_width, \
    _figsize_height, _fontsize_legend, real_obs_col, pred_val_col
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats, validata_w_tgs, metrics_dictionary


def KNN_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,None),
    model_metrics: list = default_evaluation_metrics,
    ext_quantile: tuple = default_ext_quantile,
    X_set_var: str = 'PCs', y_set_var: str = 'ss',
    percentage_PCs: float = 0.95,
    train_size = 0.8, # should be float -- (0,1)
    max_neighbors: int = 30, k_neighbors = None,
    cv_folds: int = 5, plot_results: bool = False,
    verbose: bool = False, pca_ttls = None):

    """
    KNN regression analysis to perform over the PCs,
    to predict the storm surge in all the requested locations

    Args:
        X_set (xarray.Dataset): This is the predictor, usually the PCs.
        y_set (xarray.Dataset): This is the predictand, usually the SS.
        pcs_scaler (sklearn.StandardScaler, optional): This is the pcs scaler to
            re-standarize the data. Defaults to None.
        validator (tuple, optional): This is the optional tuple to validate
            the data if required. Defaults to (False,None,None), but an
            example is (True,xarray.Validator(ss),'ss').
        model_metrics (list, optional): All the metrics to use.
            Defaults to default_evaluation_metrics.
        ext_quantile (tuple, optional): These are the exterior quantiles to be used
            in the case extreme analysis will be performed when calculating the model
            performance metrics.
        X_set_var (str, optional): This is the predictor var name. Defaults to 'PCs'.
        y_set_var (str, optional): This is the predictand var name. Defaults to 'ss'.
        train_size (float, optional): Training set size out of 1. Defaults to 0.8.
        percentage_PCs (float, optional): Percentage of PCs to predict. Defaults to 0.95.
        plot_results (bool, optional): Wheter to plot the results or not. 
            Defaults to False.
        verbose (bool, optional): Indicator of prints/logs. Defaults to False.
        pca_ttls (list, optional): This is the title for the PCA plots. Defaults to None.
        max_neighbors (int, optional): Maximum number of neighbors if grid search CV is
            wanted to be applied. Defaults to 20.
        k_neighbors (in, optional): Clearly specify the number of neighbors to use.
        cv_folds (int, optional): Number of foders to cross-validate.

    Returns:
        [list, model, train_time]: This is the list with the stats for each linear model,
            but we also return the model and the times used to train
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
                float(X_set.total_variance))>percentage_PCs
                )[0][0]
        ) # number of Pcs to use
        num_pcs = num_pcs if num_pcs>2 else 3
        print('\n {} PCs ({} expl. variance) will be used to train the model!! \n'.format(
            num_pcs,percentage_PCs)
        ) if verbose else None
        # plot the slp reconstruction
        if pcs_scaler:
            plot_recon_pcs(X_set,pcs_scaler,
                           n_pcs_recon=num_pcs,
                           return_slp=False,
                           region=default_region,
                           pca_ttls=pca_ttls
            )
        # select pcs to train the model
        X = X[:,:num_pcs]
    else:
        X = X[:,:100] # default PCs number

    # split predictors into train and test
    X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split(
        X, y, # StandardScaler().fit_transform(X) 
        common_times[0], train_size=train_size, 
        random_state=88, shuffle=False
    )

    # standarize the PCs (so the distance is correctly calculated)
    # pcs_scaler = StandardScaler()
    # X_train = pcs_scaler.fit_transform(X_train)
    # X_train[np.isnan(X_train)] = 0.0 # check additional nans
    # X_test = pcs_scaler.transform(X_test)
    # X_test[np.isnan(X_test)] = 0.0 # check additional nans

    # individual knn regressor
    if k_neighbors:
        print('\n KNN regression with {} neighbors... \n'.format(
            k_neighbors
        )) if verbose else None
        # perform the linear regression
        neigh = KNeighborsRegressor(
            n_neighbors=k_neighbors,weights='distance'
        ) # create the knn regressor
        neigh.fit(X_train,y_train)
        prediction = neigh.predict(X_test)
        # check model results
        title, stats = generate_stats(
            y_test,prediction,metrics=model_metrics,ext_quantile=ext_quantile
        ) # generate model stats
        stats['rscore'] = neigh.score(X_test,y_test) \
            if 'rscore' not in list(stats.keys()) else stats['rscore']
        title += '\n R score: {} -- in TEST data'.format(
            round(stats['rscore'],2)
        )
    # grid-search regressors
    else:
        print('\n KNN regression with {}-max neighbors... \n'.format(
            max_neighbors
        )) if verbose else None
        # perform the knn regression
        neigh = KNeighborsRegressor() # TODO: add params
        # specify parameters to test
        param_grid = {
            'n_neighbors': np.arange(1,max_neighbors,1),
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        # use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(
            neigh, param_grid, cv=cv_folds,
            scoring='r2', # check with custom callable function
            # make_scorer(metrics_dictionary['ext_rmse']),
            verbose=1 if verbose else 0
        )
        # fit model to data
        knn_gscv.fit(X_train,y_train) # input train data
        print('\n best model fitted with {} neighbors!! \n'.format(
            knn_gscv.best_params_['n_neighbors']
        )) if verbose else None
        prediction = knn_gscv.predict(X_test)
        # check model results
        title, stats = generate_stats(
            y_test,prediction,metrics=model_metrics,ext_quantile=ext_quantile
        ) # generate model stats
        stats['rscore'] = knn_gscv.score(X_test,y_test) \
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
        ax_time.plot(
            t_test,y_test,label='Numerical model data (Moana v2)',
            c=real_obs_col
        )
        ax_time.plot(
            t_test,prediction,label='KNN model predictions',
            c=pred_val_col,linestyle='--'
        )
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
    model_to_validate = neigh if k_neighbors else knn_gscv
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs] \
                if percentage_PCs else X_set[X_set_var].dropna(dim='time')[:,:100],
            validator[1][validator[2]].dropna(dim='time'),
            model_to_validate, # pre-trained knn model
            str(validator[1].name.values)
        )

    return stats, model_to_validate, t_train

