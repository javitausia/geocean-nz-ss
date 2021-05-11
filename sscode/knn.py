# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_region
from .plotting.config import _figsize, _fontsize_title, _figsize_width, _figsize_height, _fontsize_legend
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats, validata_w_tgs


def KNN_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,None),
    X_set_var: str = 'PCs', y_set_var: str = 'ss',
    percentage_PCs: float = 0.9,
    train_size = None, # can be float
    max_neighbors: int = 50,
    k_neighbors: int = 8,
    cv_folds: int = 5,
    plot_results: bool = True,
    verbose: bool = False):

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
        ) # number of Pcs to use
        if verbose:
            print('\n {} PCs ({} expl. variance) will be used to train the model!! \n'.format(
                num_pcs,percentage_PCs)
            )
        # plot the slp reconstruction
        if pcs_scaler:
            plot_recon_pcs(X_set,pcs_scaler,
                           n_pcs_recon=num_pcs,
                           return_slp=False,
                           region=default_region
            )
        # select pcs to train the model
        X = X[:,:num_pcs]

    # individual knn regressor
    if train_size:
        print('\n KNN regression with {} neighbors \n'.format(
            k_neighbors
        ))
        # split predictors into train and test
        X_train, X_test, y_train, y_test, t_train, t_test = \
        train_test_split(
            X, y, common_times[0], train_size=train_size, 
            random_state=88, shuffle=False
        )
        # perform the linear regression
        neigh = KNeighborsRegressor(n_neighbors=k_neighbors)
        neigh.fit(X_train, y_train)
        prediction = neigh.predict(X_test)

        # check model results
        title, stats = generate_stats(y_test,prediction)
        r_score = neigh.score(X_test,y_test)
        stats.append(r_score)
        title += '\n R score: {} -- in TEST data'.format(
            round(r_score,2)
        )
        # change vars for plotting
        X, y, t_plot = X_test, y_test, t_test
    else:
        print('\n KNN regression with {}-max neighbors \n'.format(
            max_neighbors
        ))
        # perform the knn regression
        neigh = KNeighborsRegressor() # TODO: add params
        # specify parameters to test
        param_grid = {
            'n_neighbors': np.arange(1,k_neighbors,5),
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        # use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(
            neigh, param_grid, cv=cv_folds,
            scoring='explained_variance',
            verbose=1
        ) 
        # TODO: add verbose and cv, DONE!!
        # fit model to data
        knn_gscv.fit(X, y) # input total data
        prediction = knn_gscv.predict(X)
        # check model results
        title, stats = generate_stats(y,prediction)
        r_score = knn_gscv.score(X,y)
        stats.append(r_score)
        title += '\n R score: {} -- in TEST data'.format(
            round(r_score,2)
        )
        t_plot = common_times[0]

    # plot results
    if plot_results:
        # figure spec-grid
        fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
        gs = gridspec.GridSpec(nrows=1,ncols=3)
        # time regular plot
        ax_time = fig.add_subplot(gs[:,:2])
        ax_time.plot(t_plot,y,label='Numerical model data (Moana v2)',c='k')
        ax_time.plot(t_plot,prediction,label='Linear model predictions',
                     c='red',linestyle='--')
        ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[:,2:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(y,prediction,ax=ax_vali,c='grey',edgecolor='k')
        qqplot(y,prediction,ax=ax_vali,c='red',edgecolor='orange')
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.15
        )
        # show the results
        plt.show()
    # print results
    if verbose:
        print(title)

    # validate the model results
    model_to_validate = neigh if train_size else knn_gscv
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs],
            validator[1][validator[2]].dropna(dim='time'),
            model_to_validate # pre-trained linear model
        )

    return stats #, model_to_validate

