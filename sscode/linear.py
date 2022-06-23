# arrays and math
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.feature_selection import RFE, f_regression

# statsmodels
import statsmodels.formula.api as smf

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# custom
from .config import default_region, default_evaluation_metrics, default_ext_quantile
from .plotting.config import _figsize, _fontsize_title, _figsize_width, \
    _figsize_height, _fontsize_legend, real_obs_col, pred_val_col
from .plotting.pca import plot_recon_pcs
from .plotting.validation import qqplot, scatterplot
from .validation import generate_stats, validata_w_tgs


def MultiLinear_Regression(
    X_set, y_set, pcs_scaler = None,
    validator: tuple = (False,None,None),
    linear_model = LinearRegression, # this is the model to use
    linear_model_parameters: dict = {
        'fit_intercept':True, 'normalize':False,
        'n_jobs':None, 'positive':False
    }, # linear model parameters
    forward_selection: bool = False,
    model_metrics: list = default_evaluation_metrics, 
    ext_quantile: tuple = default_ext_quantile,
    X_set_var: str = 'PCs', y_set_var: str = 'ss',
    train_size: float = 0.8, percentage_PCs: float = 0.95,
    plot_results: bool = False, verbose: bool = False,
    pca_ttls = None):

    """
    Multilinear regression analysis to perform over the PCs,
    to predict the storm surge in all the requested locations

    Args:
        X_set (xarray.Dataset): This is the predictor, usually the PCs.
        y_set (xarray.Dataset): This is the predictand, usually the SS.
        pcs_scaler (sklearn.StandardScaler, optional): This is the pcs scaler to
            re-standarize the data. Defaults to None.
        validator (tuple, optional): This is the optional tuple to validate
            the data if required. Defaults to (False,None,None), but an
            example is (True,xarray.Validator(ss),'ss').
        linear_model (sklearn.linear_model, optional): This is the linear model to use
            where it can be any linear model in the statistical sklearn
            module located at sklearn.linear_model. The default linear model is
            the sklearn.linear_model.LinearRegression.
        linear_model_parameters (dict, optional): These are the linear model
            parameters to use in the fit. Defaults to basic dictionary.
        forward_selection (bool, optional): Perform forward feature selection to choose
            optimal pcs. Defaults to True. TODO: check https://scikit-learn.org/stable/
            modules/generated/sklearn.feature_selection.f_regression.html.
        model_metrics (list, optional): All the metrics to use.
            Defaults to default_evaluation_metrics. All the metrics avilable in
            sklearn could be added, just go to validation.py.
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

    Returns:
        [list, model, pcs]: This is the list with the metrics for each linear model,
            the model trained with data and the pcs used if the forward_selection 
            parameter is set to True
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

    # perform the forward selection
    if forward_selection:
        print('\n A forward selection will be performed... \n') if verbose else None
        f_scores, p_values = f_regression(X_train, y_train)
        pcs_to_use = f_scores.argsort()[::-1][
            :len(np.where(p_values<0.1)[0])
        ] # choose pcs below p-value<=0.1 and highest f-score
        print(' where {} PCs : {}, \n will be finally used to train the model!! \n'.format(
            len(pcs_to_use),pcs_to_use)
        ) if verbose else None
        # change X_train/X_test values to important pcs
        X_train = X_train[:,pcs_to_use]
        X_test = X_test[:,pcs_to_use]
    else:
        pcs_to_use = num_pcs # add pcs to avoid return errors

    # perform the linear regression
    lm = linear_model(**linear_model_parameters)
    # lm = RFE(lm,n_features_to_select=0.6,step=0.05)
    lm.fit(X_train, y_train)
    prediction = lm.predict(X_test)

    # check model results
    title, stats = generate_stats(
        y_test,prediction,metrics=model_metrics,ext_quantile=ext_quantile
    ) # calculate model metrics
    stats['rscore'] = lm.score(X_test,y_test) \
        if 'rscore' not in list(stats.keys()) else stats['rscore']
    title += '\n R score: {} -- in TEST data'.format(
        round(stats['rscore'],2)
    )
    props = dict(boxstyle='round', facecolor='w', edgecolor='grey', linewidth=0.8, alpha=0.5)
    textstr = '\n'.join((
        r'Pearson = %.2f' % (stats['pearson'], ),
        r'RMSE = %.2f' % (stats['rmse'], ),
        r'SI = %.2f' % (stats['si'], ),
        r'KGE = %.2f' % (stats['kge'], )))

    # plot results
    if plot_results:
        # figure spec-grid
        # fig = plt.figure(figsize=(_figsize_width*5.0,_figsize_height))
        fig = plt.figure(figsize=(16,4))
        gs = gridspec.GridSpec(nrows=1,ncols=3)
        # time regular plot
        ax_time = fig.add_subplot(gs[:,:2])
        ax_time.plot(t_test[150:350],y_test[150:350],label='Numerical model data',c=real_obs_col)
        ax_time.plot(t_test[150:350],prediction[150:350],label='Linear model predictions',
                     c=pred_val_col,linestyle='--')
        ax_time.set_xlim(t_test[150],t_test[350])
        #ax_time.legend(fontsize=_fontsize_legend)
        # validation plot
        ax_vali = fig.add_subplot(gs[:,2:])
        ax_vali.set_xlabel('Observation')
        ax_vali.set_ylabel('Prediction')
        scatterplot(y_test,prediction,ax=ax_vali)
        qqplot(y_test,prediction,ax=ax_vali)
        # place a text box in upper left in axes coords
        ax_vali.text(0.05, 0.95, textstr, transform=ax_vali.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        ax_vali.set_xlim(-0.3,0.5)
        ax_vali.set_ylim(-0.3,0.5)
        # add title
        #fig.suptitle(
        #    title,fontsize=_fontsize_title,y=1.15
        #)
        # show the results
        plt.show()

    # print results
    print(title) if verbose else None

    # validate the model results
    if validator[0]:
        validata_w_tgs(
            X_set[X_set_var].dropna(dim='time')[:,:num_pcs][:,pcs_to_use] \
                if forward_selection else \
                X_set[X_set_var].dropna(dim='time')[:,:num_pcs],
            validator[1][validator[2]].dropna(dim='time'),
            lm, # pre-trained linear model
            str(validator[1].name.values)
        )

    return stats, lm, pcs_to_use


# https://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selected(data, response):
    """
    Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()

    return model

