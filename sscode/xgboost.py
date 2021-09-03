# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import ensemble
from sklearn.inspection import permutation_importance

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_region, default_evaluation_metrics
from .plotting.config import _figsize, _fontsize_title, _figsize_width, \
    _figsize_height, _fontsize_legend, real_obs_col, pred_val_col
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats, validata_w_tgs, metrics_dictionary


def XGBoost_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,None),
    xgboost_parameters = {
        'n_estimators': 40,
        'max_depth': 6,
        'min_samples_split': 0.04,
        'learning_rate': 0.1,
        'loss': 'ls'
    }, # xgboost model parameters
    xgboost_gscv_parameters = {
        'n_estimators': [40],
        'max_depth': np.arange(3,12,2),
        'min_samples_split': np.linspace(0.01,0.1,9),
        'learning_rate': [0.1],
        'loss': ['ls'] # more could be added
    }, # xgboost to GridSearchCV model parameters
    model_metrics: list = default_evaluation_metrics,
    X_set_var: str = 'PCs', y_set_var: str = 'ss',
    train_size: float = 0.8, percentage_PCs: float = 0.95,
    plot_results: bool = False, verbose: bool = False,
    pca_ttls = None):

    """
    XGBoost - tree regression, analysis to perform over the PCs,
    to predict the storm surge in all the requested locations

    Args:
        X_set (xarray.Dataset): This is the predictor, usually the PCs.
        y_set (xarray.Dataset): This is the predictand, usually the SS.
        pcs_scaler (sklearn.StandardScaler, optional): This is the pcs scaler to
            re-standarize the data. Defaults to None.
        validator (tuple, optional): This is the optional tuple to validate
            the data if required. Defaults to (False,None,None), but an
            example is (True,xarray.Validator(ss),'ss').
        xgboost_parameters (dict, optional): These are the xgboost model
            parameters to use in the fit. When just one XGBoost model is tried.
        xgboost_gscv_parameters (dict, optional): These are the possible combinations
            that the GridSearchCV method will explore to find the optimal forest.
        model_metrics (list, optional): All the metrics to use.
            Defaults to default_evaluation_metrics.
        X_set_var (str, optional): This is the predictor var name. Defaults to 'PCs'.
        y_set_var (str, optional): This is the predictand var name. Defaults to 'ss'.
        train_size (float, optional): Training set size out of 1. Defaults to 0.8.
        percentage_PCs (float, optional): Percentage of PCs to predict. Defaults to 0.95.
        plot_results (bool, optional): Wheter to plot the results or not. 
            Defaults to False.
        verbose (bool, optional): Indicator of prints/logs. Defaults to True.
        pca_ttls (list, optional): This is the title for the PCA plots.
            Defaults to None.

    Returns:
        [list, model, pcs_to_use]: This is the list with the stats for each xgboost model,
            but also the trained model and the pcs used are returned
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

    # individual xgboost regressor
    if xgboost_parameters:
        print('\n XGBoost regression with {} parameters... \n'.format(
            xgboost_parameters
        )) if verbose else None
        # perform the xgboost regression
        xgboost = ensemble.GradientBoostingRegressor(
            **xgboost_parameters, verbose=0
        ) # create xgboost object
        xgboost.fit(X_train, y_train)
        prediction = xgboost.predict(X_test)
        # check model results
        title, stats = generate_stats(y_test,prediction,metrics=model_metrics)
        stats['rscore'] = xgboost.score(X_test,y_test) \
            if 'rscore' not in list(stats.keys()) else stats['rscore']
        title += '\n R score: {} -- in TEST data'.format(
            round(stats['rscore'],2)
        )
    # grid-search regressors
    else:
        print('\n XGBoost regression with {} grid parameters... \n'.format(
            xgboost_gscv_parameters
        )) if verbose else None
        # perform the xgboost regression
        xgboost = ensemble.GradientBoostingRegressor() # TODO: add params
        # use gridsearch to test all values in the dictionary
        xgb_grid = GridSearchCV(
            xgboost, xgboost_gscv_parameters, cv=3,
            scoring='r2', # TODO: check multiple scoring
            verbose=1 if verbose else 0
        ) # this is the GridSearchCV method
        # fit model to data
        xgb_grid.fit(X_train,y_train) # input train data
        print('\n best model fitted with {} parameters!! \n'.format(
            xgb_grid.best_params_
        )) if verbose else None
        prediction = xgb_grid.predict(X_test)
        # check model results
        title, stats = generate_stats(y_test,prediction,metrics=model_metrics)
        stats['rscore'] = xgb_grid.score(X_test,y_test) \
            if 'rscore' not in list(stats.keys()) else stats['rscore']
        title += '\n R score: {} -- in TEST data'.format(
            round(stats['rscore'],2)
        )

    # save final model to perform plotting and validation
    model_to_validate = xgboost if xgboost_parameters else xgb_grid.best_estimator_
    # extract feature importances
    feature_importance = model_to_validate.feature_importances_
    sorted_idx_fi = np.argsort(feature_importance)
    # and permutation importances
    permutacion_importancee = permutation_importance(
        model_to_validate,X_test,y_test,n_repeats=10,random_state=88,n_jobs=2
    ) # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    sorted_idx_pi = permutacion_importancee.importances_mean.argsort()

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
        gs = gridspec.GridSpec(nrows=1,ncols=3)
        # time regular plot
        ax_time = fig.add_subplot(gs[:,:2])
        ax_time.plot(t_test,y_test,label='Numerical model data',c=real_obs_col)
        ax_time.plot(t_test,prediction,label='XGBoost model predictions',
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
        # add features importance
        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height*2.0))
        plt.subplot(1, 2, 1) # add first plot
        plt.barh((np.arange(feature_importance.shape[0])+0.5)[-20:],
                 feature_importance[sorted_idx_fi][-20:],align='center')
        plt.yticks((np.arange(feature_importance.shape[0])+0.5)[-20:],
                   np.arange(feature_importance.shape[0])[sorted_idx_fi][-20:])
        plt.title('Feature Importance (MDI)',fontsize=_fontsize_title)
        # add permutation importances
        plt.subplot(1, 2, 2)
        plt.boxplot(
            permutacion_importancee.importances[sorted_idx_pi].T[:,-20:],
            vert=False, labels=np.arange(feature_importance.shape[0])[sorted_idx_pi][-20:]
        )
        plt.title('Permutation Importance (test set)',fontsize=_fontsize_title)
        plt.show() # show the results

    # print results
    print(title) if verbose else None

    # validate the model results
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs] \
                if percentage_PCs else X_set[X_set_var].dropna(dim='time')[:,:50],
            validator[1][validator[2]].dropna(dim='time'),
            model_to_validate, # pre-trained linear model
            str(validator[1].name.values)
        )

    return stats, model_to_validate, np.arange(feature_importance.shape[0])[sorted_idx_pi]

