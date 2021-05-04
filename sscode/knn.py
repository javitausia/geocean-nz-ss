# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# plotting
import matplotlib.pyplot as plt

# custom
from .plotting.config import _figsize


def KNN_Regression(X_set, y_set,
                   X_set_var: str = 'PCs',
                   y_set_var: str = 'ss',
                   train_size: float = 0.7,
                   percentage_PCs: float = 0.9,
                   k_neighbors: int = 50,
                   cv_folds: int = 5):

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
        X = X[:,:int(
            np.where(np.cumsum(X_set.variance.values/\
                np.sum(X_set.variance.values))>percentage_PCs
                )[0][0]
        )]
    # split predictors into train and test
    X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split(
        X, y, common_times[0], train_size=train_size, 
        random_state=88, shuffle=False
    )

    # perform the knn regression
    neigh = KNeighborsRegressor()
    # specify parameters to test
    param_grid = {
        'n_neighbors': np.arange(1,k_neighbors,5)
    }
    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(neigh, param_grid, cv=5)
    # fit model to data
    knn_gscv.fit(X_train, y_train)
    prediction = knn_gscv.predict(X_test)

    # check model results
    fig, ax = plt.subplots(figsize=_figsize)
    ax.plot(t_test,y_test,c='k')
    ax.plot(t_test,prediction,c='red',linestyle='--')
    fig.suptitle('R score: {}, k-neighbors: {}'.format(
        round(knn_gscv.score(X_test,y_test),2),
        knn_gscv.best_params_['n_neighbors']),
        fontsize=_fontsize_title
    )
    fig, axes = plt.subplots(ncols=2,figsize=_figsize)
    scatterplot(y_test,prediction,ax=axes[0],c='grey',edgecolor='k')
    qqplot(y_test,prediction,ax=axes[1],c='red',edgecolor='k')
    title, stats = generate_stats(y_test,prediction)
    # add title
    fig.suptitle(
        title,fontsize=_fontsize_title,y=1.1
    )

    return knn_gscv

