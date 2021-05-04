# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# plotting
import matplotlib.pyplot as plt

# custom
from .plotting.config import _figsize, _fontsize_title
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats


def MultiLinear_Regression(X_set, y_set,
                           X_set_var: str = 'PCs',
                           y_set_var: str = 'ss',
                           train_size: float = 0.7,
                           percentage_PCs: float = 0.9,
                           plot_results: bool = False):

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

    # perform the linear regression
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train)
    prediction = lm.predict(X_test)

    # check model results
    title, stats = generate_stats(y_test,prediction)
    # plot results
    if plot_results:
        fig, ax = plt.subplots(figsize=_figsize)
        ax.plot(t_test,y_test,c='k')
        ax.plot(t_test,prediction,c='red',linestyle='--')
        fig.suptitle('R score: {}'.format(
            round(lm.score(X_test,y_test),2)),
            fontsize=_fontsize_title
        )
        fig, axes = plt.subplots(ncols=2,figsize=_figsize)
        scatterplot(y_test,prediction,ax=axes[0],c='grey',edgecolor='k')
        qqplot(y_test,prediction,ax=axes[1],c='red',edgecolor='k')
        # add title
        fig.suptitle(
            title,fontsize=_fontsize_title,y=1.1
        )
    else:
        print(title)

    return stats

